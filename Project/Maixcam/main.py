from maix import camera, display, app, time, image, nn, touchscreen
import cv2
import numpy as np
import socket
import threading, queue

# 新增一个全局变量用于标志是否接收到 "rst"
should_exit = False

# ======================
# 棋盘格标定参数
# ======================
Nx_cor = 7          # 棋盘格横向角点数
Ny_cor = 5          # 棋盘格纵向角点数
square_size = 3.4    # 单个格子边长（cm）

# 角点亚像素优化参数
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 坐标系原点及轴索引（基于棋盘格角点顺序）
origin_index = (Ny_cor // 2) * Nx_cor + (Nx_cor // 2)    # 原点索引
x_axis_index = origin_index + 1                          # X轴正方向点索引
y_axis_index = origin_index - Nx_cor                      # Y轴正方向点索引

# ======================
# 网络通信参数（需根据实际环境修改）
# ======================
# MaixCam 本地IP（若为动态IP，可注释掉或设为"0.0.0.0"）
# local_ip = "192.168.1.1"   # 示例：MaixCam 的本地IP地址
local_ip = "0.0.0.0"           # 通用监听所有接口（推荐）

local_port = 8080               # 本地监听端口（需与ESP32一致）
esp32_ip = "192.168.1.101"     # ESP32 的IP地址（请按实际 WiFi 环境修改）
esp32_port = 8080               # ESP32 的监听端口

# ======================
# 模型与摄像头初始化
# ======================
detector = nn.YOLOv5(model="/root/models/maixhub/195485/model_195485.mud")

# 摄像头分辨率需与模型输入尺寸匹配（示例：320x240）
cam = camera.Camera(320, 240, detector.input_format())
disp = display.Display()
ts = touchscreen.TouchScreen()

# 存储最新检测结果的队列（线程安全）
detection_queue = queue.Queue(maxsize=1)

# 记录上一次的透视变换矩阵，用于异常时恢复
last_M = None

# draw exit button
exit_label = "< Exit"
size = image.string_size(exit_label)
exit_btn_pos = [0, 0, 8*2 + size.width(), 12 * 2 + size.height()]

def is_in_button(x, y, btn_pos):
    return x > btn_pos[0] and x < btn_pos[0] + btn_pos[2] and y > btn_pos[1] and y < btn_pos[1] + btn_pos[3]

# ======================
# 目标检测线程
# ======================
def target_detection():
    global last_M
    # 创建 UDP 套接字
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    while not app.need_exit() and not should_exit:
        try:
            # 读取摄像头图像
            img = cam.read()
            # 镜头畸变校正（根据实际情况调整强度）
            img = img.lens_corr(strength=1.65)

            # 转换为OpenCV格式（BGR转GRAY）
            frame = image.image2cv(img, ensure_bgr=False, copy=False)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 查找棋盘格角点（用于透视变换标定）
            ret, corners = cv2.findChessboardCorners(gray, (Nx_cor, Ny_cor), None)

            if ret:
                # 亚像素级角点优化
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                # 定义世界坐标系与图像坐标系的对应点（以棋盘格中心为原点）
                half_x = (Nx_cor - 1) * square_size / 2
                half_y = (Ny_cor - 1) * square_size / 2
                world_corners = np.array([
                    [-half_x, half_y],    # 左上
                    [half_x, half_y],     # 右上
                    [half_x, -half_y],    # 右下
                    [-half_x, -half_y]     # 左下
                ], dtype=np.float32)

                # 图像坐标系四角点（按顺时针顺序）
                image_corners = np.array([
                    corners[0].ravel(),         # 左上
                    corners[Nx_cor - 1].ravel(),  # 右上
                    corners[-1].ravel(),        # 右下
                    corners[-Nx_cor].ravel()     # 左下
                ], dtype=np.float32)

                # 计算透视变换矩阵（用于坐标转换）
                try:
                    M = cv2.getPerspectiveTransform(world_corners, image_corners)
                    if np.linalg.cond(M) > 1e6:  # 检查矩阵条件数，若过大则使用上一次的矩阵
                        if last_M is not None:
                            M = last_M
                        else:
                            continue
                    last_M = M
                except Exception as e:
                    if last_M is not None:
                        M = last_M
                    else:
                        continue

                # 目标检测（YOLOv5模型推理）
                objs = detector.detect(img, conf_th=0.4, iou_th=0.45)

                # 存储目标世界坐标
                centers_world = []
                for obj in objs:
                    # 计算图像坐标中心
                    center_x = obj.x + obj.w // 2
                    center_y = obj.y + obj.h // 2
                    img_point = np.array([[center_x, center_y]], dtype=np.float32).reshape(-1, 1, 2)

                    # 转换为世界坐标（使用逆透视矩阵）
                    try:
                        world_point = cv2.perspectiveTransform(img_point, np.linalg.inv(M)).reshape(-1).astype(float)
                        # 可以根据实际情况添加坐标范围检查和修正逻辑
                        if abs(world_point[0]) > 1000 or abs(world_point[1]) > 1000:
                            continue
                        centers_world.append(world_point)
                    except Exception as e:
                        continue

                    # 在图像上绘制检测框
                    img.draw_rect(obj.x, obj.y, obj.w, obj.h, color=image.COLOR_RED)
                    # 显示模型发出的标志信息
                    msg = f'{detector.labels[obj.class_id]}: {obj.score:.2f}'
                    img.draw_string(obj.x, obj.y, msg, color=image.COLOR_RED)

                # 更新最新检测结果（队列仅保留最新一帧）
                if not detection_queue.empty():
                    try:
                        detection_queue.get_nowait()
                    except queue.Empty:
                        pass
                detection_queue.put(centers_world)

            # 绘制退出按钮
            img.draw_string(8, 12, exit_label, image.COLOR_WHITE)
            img.draw_rect(exit_btn_pos[0], exit_btn_pos[1], exit_btn_pos[2], exit_btn_pos[3],  image.COLOR_WHITE, 2)

            # 显示图像（调试用，可注释提升性能）
            img_show = image.cv2image(frame, bgr=False, copy=False)
            disp.show(img_show)
        except Exception as e:
            print(f"[ERROR] 目标检测异常：{str(e)}")

    sock.close()

# ======================
# UDP通信线程
# ======================
def udp_communication():
    global should_exit
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.bind((local_ip, local_port))
            print(f"[INFO] UDP服务器启动，监听 {local_ip}:{local_port}")

            while not app.need_exit() and not should_exit:
                # 阻塞等待接收请求（仅在收到数据时执行后续逻辑）
                data, addr = sock.recvfrom(1024)
                request = data.decode().strip()

                if request == "请求数据":
                    # 处理单次请求
                    centers_world = detection_queue.get() if not detection_queue.empty() else []
                    valid_center = None
                    # 查找第一个y坐标在[-2.0, 2.0]范围内的目标
                    for point in centers_world:
                        x, y = point
                        if y <= 2.0:
                            valid_center = (x, y)
                            break

                    if valid_center:
                        x, y = valid_center
                        response = f"{x:.2f},{y:.2f},0,1"
                        # 发送单次响应
                        sock.sendto(response.encode(), (esp32_ip, esp32_port))
                        print(f"[SEND] 响应已发送：{response} 到 {esp32_ip}:{esp32_port}")
                    else:
                        response = "rst"
                        sock.sendto(response.encode(), (esp32_ip, esp32_port))
                        print(f"[SEND] 响应已发送：{response} 到 {esp32_ip}:{esp32_port}")
                elif request == "rst":
                    print("[INFO] 收到 rst，准备退出程序")
                    should_exit = True
                    sock.close()
                else:
                    print(f"[INFO] 忽略无效请求：{request}")

    except Exception as e:
        print(f"[ERROR] UDP通信异常：{str(e)}")


# ======================
# 程序入口
# ======================
if __name__ == "__main__":
    # 启动检测线程和通信线程
    detection_thread = threading.Thread(target=target_detection, daemon=True)
    udp_thread = threading.Thread(target=udp_communication, daemon=True)

    detection_thread.start()
    udp_thread.start()

    # 主循环保持运行
    while not app.need_exit() and not should_exit:
        x, y, pressed = ts.read()
        if is_in_button(x, y, exit_btn_pos):
            app.set_exit_flag(True)
            should_exit = True
        time.sleep(0.1)

    # 程序退出清理
    pass