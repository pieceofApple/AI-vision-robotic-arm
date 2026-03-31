import cv2
import numpy as np

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
y_axis_index = origin_index - Nx_cor                      # Y轴正方向点索引，确保Y轴正方向向上

def draw_axis(frame, origin, x_axis, y_axis, scale=50):
    """
    绘制坐标轴
    :param frame: 图像帧
    :param origin: 原点坐标
    :param x_axis: X轴正方向点坐标
    :param y_axis: Y轴正方向点坐标
    :param scale: 坐标轴长度缩放比例
    """
    x_end = origin + (x_axis - origin) * scale
    y_end = origin + (y_axis - origin) * scale
    # 绘制 X 轴
    cv2.arrowedLine(frame, tuple(origin.astype(int)), tuple(x_end.astype(int)), (0, 255, 0), 2)
    # 在 X 轴箭头旁边添加 X 标识
    cv2.putText(frame, 'X', tuple((x_end + np.array([10, 0])).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    # 绘制 Y 轴
    cv2.arrowedLine(frame, tuple(origin.astype(int)), tuple(y_end.astype(int)), (0, 0, 255), 2)
    # 在 Y 轴箭头旁边添加 Y 标识
    cv2.putText(frame, 'Y', tuple((y_end + np.array([0, 10])).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    # 在原点添加 O 标识
    cv2.putText(frame, 'O', tuple((origin + np.array([-10, 10])).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    return frame

def draw_scale(frame, M, origin, x_axis, y_axis, interval=1):
    """
    绘制坐标轴刻度
    :param frame: 图像帧
    :param M: 透视变换矩阵
    :param origin: 原点坐标
    :param x_axis: X轴正方向点坐标
    :param y_axis: Y轴正方向点坐标
    :param interval: 刻度间隔（cm）
    """
    # 计算 X 轴方向向量
    x_direction = x_axis - origin
    x_direction = x_direction / np.linalg.norm(x_direction)

    # 计算 Y 轴方向向量
    y_direction = y_axis - origin
    y_direction = y_direction / np.linalg.norm(y_direction)

    # 绘制 X 轴刻度
    for i in range(-17, 17):  # 假设刻度范围为 -10 到 10 cm
        world_x = np.array([[i * interval, 0]], dtype=np.float32).reshape(-1, 1, 2)
        img_x = cv2.perspectiveTransform(world_x, M).reshape(-1).astype(int)
        start_point = tuple((img_x - x_direction * 5).astype(int))
        end_point = tuple((img_x + x_direction * 5).astype(int))
        cv2.line(frame, start_point, end_point, (0, 255, 0), 1)
        text_position = tuple((img_x + np.array([0, 10])).astype(int))
        cv2.putText(frame, str(i * interval), text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 绘制 Y 轴刻度
    for i in range(-10, 11):  # 假设刻度范围为 -10 到 10 cm
        world_y = np.array([[0, i * interval]], dtype=np.float32).reshape(-1, 1, 2)
        img_y = cv2.perspectiveTransform(world_y, M).reshape(-1).astype(int)
        start_point = tuple((img_y - y_direction * 5).astype(int))
        end_point = tuple((img_y + y_direction * 5).astype(int))
        cv2.line(frame, start_point, end_point, (0, 0, 255), 1)
        text_position = tuple((img_y + np.array([-10, 0])).astype(int))
        cv2.putText(frame, str(i * interval), text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return frame

def draw_points_and_lines(frame, M, origin, x_axis, y_axis):
    """
    绘制坐标轴和指定点
    :param frame: 图像帧
    :param M: 透视变换矩阵
    :param origin: 原点坐标
    :param x_axis: X轴正方向点坐标
    :param y_axis: Y轴正方向点坐标
    """
    # 绘制坐标轴
    frame = draw_axis(frame, origin, x_axis, y_axis)

    # 在世界坐标 (3.4, 3.4) 处画一个点
    world_point = np.array([[3.4, 3.4]], dtype=np.float32).reshape(-1, 1, 2)
    img_point = cv2.perspectiveTransform(world_point, M).reshape(-1).astype(int)
    cv2.circle(frame, tuple(img_point), 5, (255, 255, 0), -1)

    # 绘制刻度
    frame = draw_scale(frame, M, origin, x_axis, y_axis)

    return frame

def main():
    cap = cv2.VideoCapture(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 转换为灰度图像
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
            M = cv2.getPerspectiveTransform(world_corners, image_corners)

            # 确定原点、X轴正方向点、Y轴正方向点
            origin = corners[origin_index].ravel()
            x_axis = corners[x_axis_index].ravel()
            y_axis = corners[y_axis_index].ravel()

            # 防止坐标翻转：检查 X 轴和 Y 轴方向
            # 检查坐标方向是否正确
            x_axis_correct = x_axis[0] >= origin[0]
            y_axis_correct = y_axis[1] <= origin[1]

            # 若坐标方向正确，进行画点画线和其他任务
            if x_axis_correct and y_axis_correct:
                # 调用画点画线函数
                frame = draw_points_and_lines(frame, M, origin, x_axis, y_axis)

        # 显示图像
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    