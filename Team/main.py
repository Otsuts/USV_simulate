import math
import time
import numpy as np
from socket import *

host = '127.0.0.1'
port = 8085
bufsiz = 2048 * 32  # 2048 * 16
addr = (host, port)


def deal(data):
    reach1, block1, overturn1, warning1 = 0, 0, 0, 0
    velocity1 = 0
    distance1 = []
    reach2, block2, overturn2, warning2 = 0, 0, 0, 0
    velocity2 = 0
    distance2 = []

    posture1 = []
    posture2 = []

    data = data.split(',')
    # 处理绝对世界坐标
    pos1 = data[-2].split(' ')
    pos2 = data[-1].split(' ')
    position1 = {
        'x': float(pos1[0].split('=')[-1]),
        'y': float(pos1[1].split('=')[-1]),
        'z': float(pos1[2].split('=')[-1]),
    }
    position2 = {
        'x': float(pos2[0].split('=')[-1]),
        'y': float(pos2[1].split('=')[-1]),
        'z': float(pos2[2].split('=')[-1]),
    }

    MaxCheckSize1 = float(data[25])  # 起终点距离
    MaxCheckSize2 = float(data[25 + 28])

    detection_dis = float(data[26])

    # 距离终点距离
    distance_terminal1 = np.float64(data[20]) / MaxCheckSize1
    distance_terminal2 = np.float64(data[20 + 28]) / MaxCheckSize2

    # 倾覆检测
    data[21], data[22], data[23], data[24] = np.float64(data[21]), np.float64(data[22]), np.float64(
        data[23]), np.float64(data[24])
    R1, P1, Y1, destination_angle1 = data[21], data[22], data[23], data[24]
    if destination_angle1 <= -90:
        destination_angle1 += 180
    R1 = (R1 + 180) / (180 + 180)  # 归一化
    P1 = (P1 + 180) / (180 + 180)
    overturn1 = 0
    if R1 > (30 + 180) / 360 or R1 < (-30 + 180) / 360:
        overturn1 = 1
    if P1 > (30 + 180) / 360 or P1 < (-30 + 180) / 360:
        overturn1 = 1
    posture1.append(R1)
    posture1.append(P1)

    data[21 + 28], data[22 + 28], data[23 + 28], data[24 + 28] = np.float64(data[21 + 28]), np.float64(
        data[22 + 28]), np.float64(
        data[23 + 28]), np.float64(data[24 + 28])
    R2, P2, Y2, destination_angle2 = data[21 + 28], data[22 + 28], data[23 + 28], data[24 + 28]
    if destination_angle2 <= -90:
        destination_angle2 += 180
    R2 = (R2 + 180) / (180 + 180)  # 归一化
    P2 = (P2 + 180) / (180 + 180)
    overturn2 = 0
    if R2 > (30 + 180) / 360 or R2 < (-30 + 180) / 360:
        overturn2 = 1
    if P2 > (30 + 180) / 360 or P2 < (-30 + 180) / 360:
        overturn2 = 1
    posture2.append(R2)
    posture2.append(P2)

    for i in range(0, 28):
        if i < 19:
            data[i] = np.float64(data[i]) / detection_dis  # 归一化

            if data[i] == 0:
                data[i] = 1
            if i < 19:  # 碰撞检测
                if 500 / detection_dis < data[i] < 3000 / detection_dis:  # 进入警告区域
                    warning1 = 1
                elif data[i] < 400 / detection_dis:
                    block1 = 1
            distance1.append(data[i])
        if i == 19:  # 速度
            velocity1 = float(data[i]) / 1000  # 以***进行归一化
        elif i == len(data) - 1:
            reach1 = float(data[-1])
            break

    for i in range(28, 56):
        if i < 19 + 28:
            data[i] = np.float64(data[i]) / detection_dis  # 归一化

            if data[i] == 0:
                data[i] = 1
            if i < 19 + 28:  # 碰撞检测
                if 500 / detection_dis < data[i] < 3000 / detection_dis:  # 进入警告区域
                    warning2 = 1
                elif data[i] < 400 / detection_dis:
                    block2 = 1
            distance2.append(data[i])
        if i == 19 + 28:  # 速度
            velocity2 = float(data[i]) / 1000  # 以***进行归一化
        elif i == 27 + 28:
            reach2 = float(data[i])
            break

    return distance_terminal1, posture1, distance1, warning1, block1, overturn1, reach1, velocity1, destination_angle1, distance_terminal2, posture2, distance2, warning2, block2, overturn2, reach2, velocity2, destination_angle2, position1, position2


def act(get_tcp_state1, get_tcp_state2):
    left1 = right1 = left2 = right2 = 50000
    print(get_tcp_state1['position'], get_tcp_state2['position'])
    for index, dist in enumerate(get_tcp_state1['distance']):
        if index < 9:
            left1 += index * (1 - dist) * 5000
            right1 += -index * (1 - dist) * 5000
        if 10 <= index:
            left1 += -(19 - index) * (1 - dist) * 5000
            right1 += (19 - index) * (1 - dist) * 5000
    left1 -= get_tcp_state1['angle'] * 1000
    right1 += get_tcp_state1['angle'] * 1000

    for index, dist in enumerate(get_tcp_state2['distance']):
        if index < 9:
            left2 += index * (1 - dist) * 5000
            right2 += -index * (1 - dist) * 5000
        if 10 <= index:
            left2 += -(19 - index) * (1 - dist) * 5000
            right2 += (19 - index) * (1 - dist) * 5000
    left2 -= get_tcp_state2['angle'] * 1000
    right2 += get_tcp_state2['angle'] * 1000
    clear1, clear2 = all([x == 1 for x in get_tcp_state1['distance']]), all(
        [x == 1 for x in get_tcp_state2['distance']])
    print(clear1, clear2)
    if clear1 and clear2:
        if get_tcp_state1['position']['y'] - get_tcp_state2['position']['y'] > 400:
            left2 = right2 = 0
        elif get_tcp_state1['position']['y'] - get_tcp_state2['position']['y'] < -400:
            left1 = right1 = 0



    if not clear1 or not clear2:
        if math.sqrt((get_tcp_state1['position']['x'] - get_tcp_state2['position']['x']) ** 2 + (
                get_tcp_state1['position']['y'] - get_tcp_state2['position']['y']) ** 2) <= 5000:
            left1 = right1 = 0
    if left1 > 50000:
        left1 = 50000
    if left1 < -50000:
        left1 = -50000

    if left2 > 50000:
        left2 = 50000
    if left2 < -50000:
        left2 = -50000
    return left1, right1, left2, right2


def TCPcommuition():
    # 创建tcp套接字，绑定，监听
    tcpServerSock = socket(AF_INET, SOCK_STREAM)  # 创建TCP Socket
    # AF_INET 服务器之间网络通信
    # socket.SOCK_STREAM 流式socket , for TCP
    tcpServerSock.bind(addr)  # 将套接字绑定到地址,
    # 在AF_INET下,以元组（host,port）的形式表示地址.
    tcpServerSock.listen(5)  # 操作系统可以挂起的最大连接数量，至少为1，大部分为5

    while True:
        print('-' * 10 + 'waiting for connection' + '-' * 10)
        # tcp这里接收到的是客户端的sock对象，后面接受数据时使用socket.recv()
        tcpClientSock, addr2 = tcpServerSock.accept()  # 接受客户的连接
        # 接受TCP连接并返回（conn,address）,其中conn是新的套接字对象，
        # 可以用来接收和发送数据。
        # address是连接客户端的地址。
        print('-' * 10 + f'connected from :{addr2}' + '-' * 10)

        while True:
            data = tcpClientSock.recv(bufsiz)  # 接收客户端发来的数据
            if not data:
                break
            # 接收数据
            Receive_Data = data.decode().replace('\x00', '')
            # print(Receive_Data)
            distance_terminal1, posture1, distance1, warning1, block1, overturn1, reach1, velocity1, destination_angle1, \
            distance_terminal2, posture2, distance2, warning2, block2, overturn2, reach2, velocity2, destination_angle2, position1, position2 = deal(
                Receive_Data)
            # 是否要回到起点重新开始
            reset = block1 or block2 or overturn1 or overturn2 or reach1 or reach2
            # 获取两艘船的状态
            get_tcp_state1 = {
                'velocity': velocity1,
                'angle': destination_angle1,
                'distance': distance1,
                'position': position1,
                'warning': warning1,
                'block': block1,
                'reach': reach1
            }

            get_tcp_state2 = {
                'velocity': velocity2,
                'angle': destination_angle2,
                'distance': distance2,
                'position': position2,
                'warning': warning2,
                'block': block2,
                'reach': reach2
            }
            # 发送数据
            left1, right1, left2, right2 = act(get_tcp_state1, get_tcp_state2)
            msg = str(left1) + ',' + str(right1) + ',' + str(left2) + ',' + str(right2) + ',' + str(reset)
            tcpClientSock.send(msg.encode())  # 返回给客户端数据
            # 给船一些喘息的时间
            time.sleep(1)
        tcpClientSock.close()
    tcpServerSock.close()


if __name__ == '__main__':
    TCPcommuition()
