from maix import camera, display, app, time, image, nn
import cv2
import numpy as np
import socket
import threading, queue

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
detector = nn.YOLOv5(model="/root/models/maixhub/194993/model_194993.mud")

# 摄像头分辨率需与模型输入尺寸匹配（示例：320x240）
cam = camera.Camera(320, 240, detector.input_format())
disp = display.Display()

# 存储最新检测结果的队列（线程安全）
detection_queue = queue.Queue(maxsize=1)

# ======================
# 目标检测线程
# ======================
def target_detection():
    while not app.need_exit():
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
                corners[Nx_cor-1].ravel(),  # 右上
                corners[-1].ravel(),        # 右下
                corners[-Nx_cor].ravel()     # 左下
            ], dtype=np.float32)
            
            # 计算透视变换矩阵（用于坐标转换）
            M = cv2.getPerspectiveTransform(world_corners, image_corners)
            
            # 目标检测（YOLOv5模型推理）
            objs = detector.detect(img, conf_th=0.5, iou_th=0.45)
            
            # 存储目标世界坐标
            centers_world = []
            for obj in objs:
                # 计算图像坐标中心
                center_x = obj.x + obj.w // 2
                center_y = obj.y + obj.h // 2
                img_point = np.array([[center_x, center_y]], dtype=np.float32).reshape(-1, 1, 2)
                
                # 转换为世界坐标（使用逆透视矩阵）
                world_point = cv2.perspectiveTransform(img_point, np.linalg.inv(M)).reshape(-1).astype(float)
                centers_world.append(world_point)
                
                # 在图像上绘制检测框
                img.draw_rect(obj.x, obj.y, obj.w, obj.h, color=image.COLOR_RED)
                img.draw_string(obj.x, obj.y, "paperball", color=image.COLOR_RED)
            
            # 更新最新检测结果（队列仅保留最新一帧）
            if not detection_queue.empty():
                detection_queue.get()
            detection_queue.put(centers_world)
        
        # 显示图像（调试用，可注释提升性能）
        img_show = image.cv2image(frame, bgr=False, copy=False)
        disp.show(img_show)

# ======================
# UDP通信线程
# ======================
def udp_communication():
    try:
        # 创建UDP套接字
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            # 绑定本地IP和端口（若local_ip设为"0.0.0.0"，则监听所有接口）
            sock.bind((local_ip, local_port))
            print(f"[INFO] UDP服务器启动，监听 {local_ip}:{local_port}")
            
            while not app.need_exit():
                # 接收ESP32的请求
                data, addr = sock.recvfrom(1024)
                request = data.decode().strip()
                
                if request == "GET_POS":  # 自定义请求指令（可改为"请求数据"）
                    # 获取最新检测结果
                    centers_world = detection_queue.get() if not detection_queue.empty() else []
                    
                    if len(centers_world) > 0:
                        # 取第一个目标的坐标（可根据需求调整）
                        x, y = centers_world[0]
                        response = f"{x:.2f},{y:.2f},5.0,1"  # 格式：x,y,z,flag（z固定为5.0）
                    else:
                        response = "NO_TARGET"  # 未检测到目标时的响应
                
                # 发送响应给ESP32
                sock.sendto(response.encode(), (esp32_ip, esp32_port))
                print(f"[SEND] 数据发送至 {esp32_ip}:{esp32_port}，内容：{response}")
                
                # 发送复位指令（可选）
                sock.sendto("RST".encode(), (esp32_ip, esp32_port))
                
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
    while not app.need_exit():
        time.sleep(0.1)

    # 程序退出清理
    cam.deinit()
    disp.deinit()