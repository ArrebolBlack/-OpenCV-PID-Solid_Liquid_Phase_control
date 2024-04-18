import cv2
import numpy as np
import os
#hue min,hue max,sat min,sat max,val min,val max
Phase_Color1 = []
Phase_Color2 = []

# 初始化摄像头
#cap = cv2.VideoCapture(0)
video_path = 'rawdata\BV17d4y1171Y\BV17d4y1171Y-1.mp4'
# 获取视频文件所在的目录
base_dir = os.path.dirname(video_path)
# 截图保存目录
screenshots_dir = os.path.join(base_dir, 'screenshots')
# 确保截图目录存在
if not os.path.exists(screenshots_dir):
    os.makedirs(screenshots_dir)
cap = cv2.VideoCapture(video_path)

if  not cap.isOpened():
    print("Failed to open the video file.")
    exit()

# 确保有一个目录用于保存截图
if not os.path.exists('screenshots'):
    os.makedirs('screenshots')

# 颜色范围设置，示例颜色
color1_range = {'lower': np.array([95, 48, 94]), 'upper': np.array([131, 195, 151])}
color2_range = {'lower': np.array([119, 6, 67]), 'upper': np.array([179, 255, 181])}

# PID控制器参数
Kp, Ki, Kd = 0.1, 0.01, 0.05
previous_error = 0
integral = 0
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver
def calculate_area(mask):
    return cv2.countNonZero(mask)

def pid_control(set_point, measured_value):
    global previous_error, integral
    error = set_point - measured_value
    integral += error
    derivative = error - previous_error
    output = Kp * error + Ki * integral + Kd * derivative
    previous_error = error
    return output

desired_ratio = 1.0  # 目标固液相变面积比例
snapshot_count = 0  # 用于保存截图的计数器
print("Press '+' or '-' to adjust the desired ratio. Press 's' to take a snapshot. Press 'ESC' to exit.")


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(frame_hsv, color1_range['lower'], color1_range['upper'])
    mask2 = cv2.inRange(frame_hsv, color2_range['lower'], color2_range['upper'])

    area1 = calculate_area(mask1)
    area2 = calculate_area(mask2)

    if area2 != 0:
        current_ratio = area1 / area2
    else:
        current_ratio = float('inf')  # 防止除零错误

    control_signal = pid_control(desired_ratio, current_ratio)
    print(f"Control Signal: {control_signal}, Current Ratio: {current_ratio}, Desired Ratio: {desired_ratio}")

    # cv2.imshow('Frame', frame)
    # cv2.imshow('Mask1', mask1)
    # cv2.imshow('Mask2', mask2)
    imgStack = stackImages(0.6,([frame,mask1,mask2]))
    cv2.imshow('imgStack',imgStack)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Press 'ESC' to exit
        break
    elif key == ord('+'):  # Increase desired ratio
        desired_ratio += 0.1
    elif key == ord('-'):  # Decrease desired ratio
        desired_ratio -= 0.1
    elif key == ord('s'):  # 按 's' 截图保存当前帧
        snapshot_path = os.path.join(screenshots_dir, f'snapshot_{snapshot_count}.png')
        cv2.imwrite(snapshot_path, frame)
        print(f"Snapshot taken and saved as {snapshot_path}")
        snapshot_count += 1

cap.release()
cv2.destroyAllWindows()

