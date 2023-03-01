import numpy as np


def deal(data):
    reach, block, overturn, warning = 0, 0, 0, 0
    velocity = 0
    distance = []
    posture = []

    data = data.split(',')
    MaxCheckSize = float(data[-3])  # 起终点距离
    MaxCheckSize = MaxCheckSize
    # print(MaxCheckSize)
    detection_dis = float(data[-2])

    # 距离终点距离
    distance_terminal = np.float64(data[20]) / MaxCheckSize
    # 倾覆检测
    data[21], data[22], data[23], data[24] = np.float64(data[21]), np.float64(data[22]), np.float64(
        data[23]), np.float64(data[24])
    R, P, Y, destination_angle = data[21], data[22], data[23], data[24]
    R = (R + 180) / (180 + 180)  # 归一化
    P = (P + 180) / (180 + 180)
    # Y = (Y + 180) / (180 + 180)

    destination_angle = (destination_angle + 180) / (180 + 180) - 0.5  # 航向角
    overturn = 0
    if R > (30 + 180) / 360 or R < (-30 + 180) / 360:
        overturn = 1
    if P > (30 + 180) / 360 or P < (-30 + 180) / 360:
        overturn = 1
    posture.append(R)
    posture.append(P)

    for i in range(len(data)):
        if i < 19:
            data[i] = np.float64(data[i]) / detection_dis  # 归一化

            if data[i] == 0:
                data[i] = 1
            if i < 19:  # 碰撞检测
                if 500 / detection_dis < data[i] < 3000 / detection_dis:  # 进入警告区域
                    warning = 1
                elif data[i] < 400 / detection_dis:
                    block = 1
            distance.append(data[i])
        if i == 19:  # 速度
            velocity = float(data[i]) / 1000  # 以***进行归一化
        elif i == len(data) - 1:
            reach = float(data[-1])
            break

    distance = np.array(distance)
    distance_terminal = np.array(float(distance_terminal))
    posture = np.array(posture)
    velocity = np.array(velocity)

    return distance_terminal, posture, distance, warning, block, overturn, reach, velocity, destination_angle
