import time
import numpy as np
import threading

import torch
from tqdm import tqdm
from socket import *
from argparser import parse_args
from model import DQNAgent

global left, right, get_tcp_state, reset
host = '127.0.0.1'
port = 8085
bufsiz = 2048 * 32  # 2048 * 16
addr = (host, port)
get_tcp_state = []


def Next_state(action, get_tcp_state):
    global left
    global right
    left, right = int(action[0]), int(action[1])
    # print(action[0],action[1])
    time.sleep(1)
    distance_terminal, posture, distance, warning, block, overturn, reach, velocity, destination_angle = deal_main_data(
        get_tcp_state)
    nextstate = np.append(distance, posture)
    nextstate = np.append(nextstate, destination_angle)
    nextstate = np.append(nextstate, velocity)
    nextstate = np.append(nextstate, distance_terminal).astype(np.float)
    return torch.from_numpy(
        nextstate), distance_terminal, destination_angle, posture, warning, block, overturn, reach, velocity


def Reward(difference_distance_terminal, distance_terminal, destination_angle, posture, warning, block, overturn, reach,
           velocity):
    global left, right
    restart = 0
    reward = 0
    block_reward, overturn_reward, reach_reward, warning_reward, velocity_reward, posture_reward, difference_distance_terminal_reward, distance_terminal_reward = 0, 0, 0, 0, 0, 0, 0, 0
    if block == 1:
        block_reward = -2
        restart = 1
    if overturn == 1:
        overturn_reward = -1
        restart = 1
    if reach == 1:
        reach_reward = 20
        restart = 1
    if warning == 1:
        warning_reward = -0.5
    if velocity < 200 / 1000:
        if -1 / ((velocity + 0.00001) * 10) < -2:
            velocity_reward = -1
    elif velocity >= 200 / 1000:
        velocity_reward = 1

    reward += 0
    # distance_terminal_reward = -10 * distance_terminal
    distance_terminal_reward = 0.1 * (1 / distance_terminal)
    # print(distance_terminal_reward)
    # posture_reward = - 0.1*((posture[0] - 0.5) + (posture[1] - 0.5))
    difference_distance_terminal_reward = difference_distance_terminal * 200  # ????????????0.00X??????
    if difference_distance_terminal_reward > 0.9:
        difference_distance_terminal_reward = 0.9
    # print(difference_distance_terminal_reward)
    destination_angle_reward = - abs(destination_angle) * 10
    if destination_angle_reward <= -2:
        destination_angle_reward = -2
    # print('block_reward:{} overturn_reward:{} reach_reward:{} warning_reward:{} difference_distance_terminal_reward:{} distance_terminal_reward:{} destination_angle_reward:{} velocity_reward:{} reward:{}\n\n'
    #       .format(block_reward, overturn_reward, reach_reward, warning_reward, difference_distance_terminal_reward, distance_terminal_reward, destination_angle_reward, velocity_reward, reward))
    reward = block_reward + overturn_reward + reach_reward + warning_reward + difference_distance_terminal_reward + distance_terminal_reward + destination_angle_reward + velocity_reward + reward
    # print(reward)
    return reward, restart


def deal_main_data(get_tcp_state):
    get_state = get_tcp_state
    distance_terminal = get_state[0]
    posture = get_state[1]
    distance = get_state[2]
    warning = get_state[3]
    block = get_state[4]
    overturn = get_state[5]
    reach = get_state[6]
    velocity = get_state[7]
    destination_angle = get_state[8]
    return distance_terminal, posture, distance, warning, block, overturn, reach, velocity, destination_angle


def deal(data):
    reach, block, overturn, warning = 0, 0, 0, 0
    velocity = 0
    distance = []

    posture = []

    data = data.split(',')
    MaxCheckSize = float(data[-3])  # ???????????????
    MaxCheckSize = MaxCheckSize
    # print(MaxCheckSize)
    detection_dis = float(data[-2])

    # ??????????????????
    distance_terminal = np.float64(data[20]) / MaxCheckSize
    # ????????????
    data[21], data[22], data[23], data[24] = np.float64(data[21]), np.float64(data[22]), np.float64(
        data[23]), np.float64(data[24])
    R, P, Y, destination_angle = data[21], data[22], data[23], data[24]
    R = (R + 180) / (180 + 180)  # ?????????
    P = (P + 180) / (180 + 180)
    # Y = (Y + 180) / (180 + 180)

    destination_angle = (destination_angle + 180) / (180 + 180) - 0.5  # ?????????
    overturn = 0
    if R > (30 + 180) / 360 or R < (-30 + 180) / 360:
        overturn = 1
    if P > (30 + 180) / 360 or P < (-30 + 180) / 360:
        overturn = 1
    posture.append(R)
    posture.append(P)

    for i in range(len(data)):
        if i < 19:
            data[i] = np.float64(data[i]) / detection_dis  # ?????????

            if data[i] == 0:
                data[i] = 1
            if i < 19:  # ????????????
                if 500 / detection_dis < data[i] < 3000 / detection_dis:  # ??????????????????
                    warning = 1
                elif data[i] < 400 / detection_dis:
                    block = 1
            distance.append(data[i])
        if i == 19:  # ??????
            velocity = float(data[i]) / 1000  # ???***???????????????
        elif i == len(data) - 1:
            reach = float(data[-1])
            break

    distance = np.array(distance)
    distance_terminal = np.array(float(distance_terminal))
    posture = np.array(posture)
    velocity = np.array(velocity)

    return distance_terminal, posture, distance, warning, block, overturn, reach, velocity, destination_angle


def TCPcommuition():
    # ??????tcp???????????????????????????
    tcpServerSock = socket(AF_INET, SOCK_STREAM)  # ??????TCP Socket
    # AF_INET ???????????????????????????
    # socket.SOCK_STREAM ??????socket , for TCP
    tcpServerSock.bind(addr)  # ???????????????????????????,
    # ???AF_INET???,????????????host,port????????????????????????.
    tcpServerSock.listen(5)  # ?????????????????????????????????????????????????????????1???????????????5

    while True:
        global left, right, get_tcp_state, reset
        reset = 0
        print('waiting for connection')
        # tcp?????????????????????????????????sock????????????????????????????????????socket.recv()
        tcpClientSock, addr2 = tcpServerSock.accept()  # ?????????????????????
        # ??????TCP??????????????????conn,address???,??????conn???????????????????????????
        # ????????????????????????????????????
        # address??????????????????????????????
        print('connected from :', addr2)

        left = 0
        right = 0
        # t1 = threading.Thread(target=input_control, name='T1')  # ??????????????????
        # t1.start()
        while True:
            data = tcpClientSock.recv(bufsiz)  # ??????????????????????????????
            if not data:
                break
            # ????????????
            # time.sleep(0.01)
            # start = time.time()
            Receive_Data = data.decode().replace('\x00', '')

            distance_terminal, posture, distance, warning, block, overturn, reach, velocity, destination_angle = deal(
                Receive_Data)
            get_tcp_state = [distance_terminal, posture, distance, warning, block, overturn, reach, velocity,
                             destination_angle]
            # ????????????

            msg = str(left) + ',' + str(right) + ',' + str(reset)
            tcpClientSock.send(msg.encode())  # ????????????????????????
            # end = time.time()
            # print(end - start)
        tcpClientSock.close()
    tcpServerSock.close()


t = threading.Thread(target=TCPcommuition, name='TCP communicate')
t.start()


def main(args):
    state_dim = 24  # 24?????????
    global left, right, get_tcp_state, reset
    # ??????????????????????????????????????????
    while not get_tcp_state:
        pass

    model = DQNAgent(state_dim, args.hidden_dim, 8)

    if args.mode == 'train':
        print('-' * 15 + 'trianing starts' + '-' * 15)
        for iter in tqdm(range(args.epoch)):
            # reset the env and get cur state
            step = 0
            reset = 1
            time.sleep(0.2)  # ??????????????????????????????????????????????????????
            start_time = time.time()
            reset = 0
            difference_distance_terminal = 0
            distance_terminal, posture, distance, warning, block, overturn, reach, velocity, destination_angle = deal_main_data(
                get_tcp_state)
            state = np.append(distance, posture)
            state = np.append(state, destination_angle)
            state = np.append(state, velocity)
            state = np.append(state, distance_terminal)
            state = torch.from_numpy(state)
            # state:torch.size([24])

            while step <= 1000:
                action,action_index = model.act(state)  # action?????????2*1???numpy??????,???????????????????????????
                # get reward
                reward, restart = Reward(
                    difference_distance_terminal,
                    distance_terminal,
                    destination_angle,
                    posture,
                    warning,
                    block,
                    overturn,
                    reach,
                    velocity
                )
                # get next state
                next_state, next_distance_terminal, next_destination_angle, posture, warning, block, overturn, reach, next_velocity = Next_state(
                    action, get_tcp_state)
                # put next state into pool
                model.exp_pool(state, next_state, action_index, reward, restart)
                difference_distance_terminal = distance_terminal - next_distance_terminal  # ?????????????????????????????????????????????
                # next
                destination_angle = next_destination_angle
                velocity = next_velocity
                distance_terminal = next_distance_terminal
                q_, loss = model.learn()
                print(q_, loss)
                state = next_state
                step += 1
                if restart:
                    break
    if args.mode == 'test':
        print('-' * 15 + 'testing starts' + '-' * 15)
        reset = 1
        time.sleep(0.2)  # ??????????????????????????????????????????????????????
        reset = 0
        difference_distance_terminal = 0
        distance_terminal, posture, distance, warning, block, overturn, reach, velocity, destination_angle = deal_main_data(
            get_tcp_state)
        state = np.append(distance, posture)
        state = np.append(state, destination_angle)
        state = np.append(state, velocity)
        state = np.append(state, distance_terminal)
        state = torch.from_numpy(state)
        # state:torch.size([24])

        while True:
            action,action_index = model.act(state)  # action?????????2*1???numpy??????,???????????????????????????
            # get reward
            reward, restart = Reward(
                difference_distance_terminal,
                distance_terminal,
                destination_angle,
                posture,
                warning,
                block,
                overturn,
                reach,
                velocity
            )
            # get next state
            next_state, next_distance_terminal, next_destination_angle, posture, warning, block, overturn, reach, next_velocity = Next_state(
                action, get_tcp_state)
            # put next state into pool
            model.exp_pool(state, next_state, action_index, reward, restart)
            difference_distance_terminal = distance_terminal - next_distance_terminal  # ?????????????????????????????????????????????
            # next
            destination_angle = next_destination_angle
            velocity = next_velocity
            distance_terminal = next_distance_terminal
            q_, loss = model.learn()
            print(q_, loss)
            state = next_state
            if restart:
                break


if __name__ == '__main__':
    args = parse_args()
    main(args)
