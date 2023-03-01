from utils.TCP import deal
import threading
import random
import pandas as pd
from matplotlib import pyplot as plt
from utils.argparser import parse_args
from itertools import count
from models.TD3 import TD3
from models.SAC import SAC
from world.environment import *
import time
from socket import *
from tqdm import tqdm

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
    time.sleep(0.5)
    distance_terminal, posture, distance, warning, block, overturn, reach, velocity, destination_angle = deal_main_data(
        get_tcp_state)
    nextstate = np.append(distance, posture)
    nextstate = np.append(nextstate, destination_angle)
    nextstate = np.append(nextstate, velocity)
    nextstate = np.append(nextstate, distance_terminal)
    return nextstate, distance_terminal, destination_angle, posture, warning, block, overturn, reach, velocity


def TCPcommuition():
    # 创建tcp套接字，绑定，监听
    tcpServerSock = socket(AF_INET, SOCK_STREAM)  # 创建TCP Socket
    # AF_INET 服务器之间网络通信
    # socket.SOCK_STREAM 流式socket , for TCP
    tcpServerSock.bind(addr)  # 将套接字绑定到地址,
    # 在AF_INET下,以元组（host,port）的形式表示地址.
    tcpServerSock.listen(5)  # 操作系统可以挂起的最大连接数量，至少为1，大部分为5

    while True:
        global left, right, get_tcp_state, reset
        reset = 0
        print('waiting for connection')
        # tcp这里接收到的是客户端的sock对象，后面接受数据时使用socket.recv()
        tcpClientSock, addr2 = tcpServerSock.accept()  # 接受客户的连接
        # 接受TCP连接并返回（conn,address）,其中conn是新的套接字对象，
        # 可以用来接收和发送数据。
        # address是连接客户端的地址。
        print('connected from :', addr2)

        left = 0
        right = 0
        # t1 = threading.Thread(target=input_control, name='T1')  # 输入推力控制
        # t1.start()
        while True:
            data = tcpClientSock.recv(bufsiz)  # 接收客户端发来的数据
            if not data:
                break
            # 接收数据
            # time.sleep(0.01)
            # start = time.time()
            Receive_Data = data.decode().replace('\x00', '')

            distance_terminal, posture, distance, warning, block, overturn, reach, velocity, destination_angle = deal(
                Receive_Data)
            get_tcp_state = [distance_terminal, posture, distance, warning, block, overturn, reach, velocity,
                             destination_angle]
            # 发送数据

            msg = str(left) + ',' + str(right) + ',' + str(reset)
            tcpClientSock.send(msg.encode())  # 返回给客户端数据
            # end = time.time()
            # print(end - start)
        tcpClientSock.close()
    tcpServerSock.close()


t = threading.Thread(target=TCPcommuition, name='TCP communicate')
t.start()


def main(args):
    state_dim = 24  # 24维数据

    action_dim = 2
    max_action = 50000
    global left, right, get_tcp_state, reset
    # 在虚幻引擎没有开始运行时等待
    while not get_tcp_state:
        pass
    if args.model == 'TD3':
        agent = TD3(state_dim, action_dim, max_action, args)
    elif args.model == 'SAC':
        agent = SAC(state_dim, action_dim, 256, args)
    else:
        raise f'model {args.model} not available'
    print('-' * 10 + '开始' + '-' * 10)
    time.sleep(5)
    if args.mode == 'train':
        total_step = 0
        explore_num = 0  # 探索的次数
        return_list = []
        mean_return_list = []
        each_step = []
        last_100_reward = []
        explore_min = 20000
        episode = 5000  # 原2500
        if args.model == 'TD3':
            for i in tqdm(range(episode)):
                flag = 0
                reset = 1
                time.sleep(0.2)
                start = time.time()
                reset = 0
                total_reward = 0
                step = 0
                difference_distance_terminal = 0
                distance_terminal, posture, distance, warning, block, overturn, reach, velocity, destination_angle = deal_main_data(
                    get_tcp_state)
                state = np.append(distance, posture)
                state = np.append(state, destination_angle)
                state = np.append(state, velocity)
                state = np.append(state, distance_terminal)  # 24

                for t in range(500):
                    action = agent.select_action(state)  # action 2维连续数值
                    # action[0], action[1] = 100000, 100000
                    # print('action:', action)

                    # exploration
                    select_explore = [
                        action + max(explore_min - i * 400 * (explore_min / max_action), max_action - i * 400),
                        action + np.random.normal(0, max(1000, args.exploration_noise - i * 400), size=2),
                        action - max(explore_min - i * 400 * (explore_min / max_action), max_action - i * 400)]
                    if len(agent.memory.storage) <= (args.capacity - 1) / 20:  # 探索
                        # if t < 4000:
                        flag = 1
                        j = random.random()
                        if j < 0.6:
                            action = select_explore[0]
                            # print(0)
                        elif 0.6 <= j < 0.9:
                            action = select_explore[1]
                            # print(1)
                        else:
                            action = select_explore[2]
                            # print(2)

                    action = (action + np.random.normal(0, 2000, size=2)).clip(-50000, 50000)  # 截断（-x,x）之间
                    reward, restart = Reward(difference_distance_terminal, distance_terminal, destination_angle,
                                             posture,
                                             warning, block, overturn, reach, velocity)
                    next_state, next_distance_terminal, next_destination_angle, posture, warning, block, overturn, reach, next_velocity = Next_state(
                        action, get_tcp_state)
                    agent.memory.push((state, next_state, action, reward,
                                       np.float(restart)))  # x是state，y是next_state，u是action，r是reward，d是done
                    difference_distance_terminal = distance_terminal - next_distance_terminal  # 此时刻与下一时刻距离终点的差值
                    # next
                    destination_angle = next_destination_angle
                    velocity = next_velocity
                    distance_terminal = next_distance_terminal
                    state = next_state
                    # last_100_reward只维护最近的100个奖励函数
                    if len(last_100_reward) > 100:
                        last_100_reward.pop(0)
                    step += 1
                    total_reward += reward

                    if restart:
                        break

                if flag == 1: explore_num += 1  # flag记录是否在进行探索
                left, right = 0, 0
                # 平均奖励
                mean_reward = total_reward / (step + 1)
                return_list.append(total_reward)
                mean_return_list.append(mean_reward)  # 保存到list
                last_100_reward.append(total_reward)
                # 保存每个episode的步数
                each_step.append(step)
                total_step += step + 1
                print("Total T:{} Episode: \t{} Total Reward: \t{}".format(total_step, i, total_reward))
                print("Total T:{} Episode: \t{} Mean Reward: \t{}".format(total_step, i, mean_reward))
                if len(agent.memory.storage) >= (args.capacity - 1) / 20:
                    agent.update(100)
                # else:
                #     print(len(agent.memory.storage), args.capacity)

                if i % args.log_interval == 0:
                    agent.save()
                last100_mean_reward = last_100_mean_reward(last_100_reward)
                if len(last_100_reward) == 100:
                    print("Total T:{} Episode: \t{} Last 100 Reward: \t{}".format(total_step, i, last100_mean_reward))
                end = time.time()
                print('预计训练结束还有{}小时'.format(((end - start) * (episode - i)) / 3600))

        elif args.model == 'SAC':
            for epis in tqdm(range(episode)):  # 玩episode次,
                flag = 0
                reset = 1
                total_reward = 0
                time.sleep(0.2)
                start_time = time.time()
                reset = 0
                step = 0
                difference_distance_terminal = 0
                distance_terminal, posture, distance, warning, block, overturn, reach, velocity, destination_angle = deal_main_data(
                    get_tcp_state)
                state = np.append(distance, posture)
                state = np.append(state, destination_angle)
                state = np.append(state, velocity)
                state = np.append(state, distance_terminal)  # 24

                for step in range(50000):  # 每次最多走50000步
                    action = agent.sample_action(state) * max_action
                    if len(agent.memory.storage) <= (args.capacity - 1) / 20:  # 探索
                        flag = 1
                        select_explore = [
                            action + max(explore_min - epis * 400 * (explore_min / max_action),
                                         max_action - epis * 400),
                            action + np.random.normal(0, max(1000, args.exploration_noise - epis * 400), size=2),
                            action - max(explore_min - epis * 400 * (explore_min / max_action),
                                         max_action - epis * 400)]
                        j = random.random()
                        if j < 0.6:
                            action = select_explore[0]
                            # print(0)
                        elif 0.6 <= j < 0.9:
                            action = select_explore[1]
                            # print(1)
                        else:
                            action = select_explore[2]
                    action = action.clip(-50000, 50000)  # 使action保持在阈值
                    reward, restart = Reward(difference_distance_terminal, distance_terminal, destination_angle,
                                             posture,
                                             warning, block, overturn, reach, velocity)
                    next_state, next_distance_terminal, next_destination_angle, posture, warning, block, overturn, reach, next_velocity = Next_state(
                        action, get_tcp_state)
                    agent.memory.push((state, next_state, action, reward,
                                       np.float(restart)))  # x是state，y是next_state，u是action，r是reward，d是done
                    difference_distance_terminal = distance_terminal - next_distance_terminal  # 此时刻与下一时刻距离终点的差值
                    # next
                    destination_angle = next_destination_angle
                    velocity = next_velocity
                    distance_terminal = next_distance_terminal
                    state = next_state
                    if len(last_100_reward) > 100:
                        last_100_reward.pop(0)
                    step += 1
                    total_reward += reward

                    if restart:
                        break
                if flag == 1: explore_num += 1  # 记录探索的次数
                # 平均奖励
                mean_reward = total_reward / (step + 1)
                return_list.append(total_reward)
                mean_return_list.append(mean_reward)
                last_100_reward.append(total_reward)

                each_step.append(step)
                total_step += step + 1
                print("Total T:{} Episode: \t{} Total Reward: \t{}".format(total_step, epis, total_reward))
                print("Total T:{} Episode: \t{} Mean Reward: \t{}".format(total_step, epis, mean_reward))
                if len(agent.memory.storage) > (args.capacity - 1) / 20:
                    agent.update(100)
                if not epis % args.log_interval:
                    agent.save()
                last100_mean_reward = last_100_mean_reward(last_100_reward)
                if len(last_100_reward) == 100:
                    print(
                        "Total T:{} Episode: \t{} Last 100 Reward: \t{}".format(total_step, epis, last100_mean_reward))
                print('预计训练结束还有{}小时'.format(((time.time() - start_time) * (episode - epis)) / 3600))
        else:
            raise f'model {args.model} not available'
        print('探索%d次' % explore_num)
        print('-' * 10 + "结束" + '-' * 10)
        return_list = pd.DataFrame(data=return_list)
        return_list.to_csv('./data/total_return_list2.csv', encoding='utf-8')
        mean_return_list = pd.DataFrame(data=mean_return_list)
        mean_return_list.to_csv('./data/mean_return_list2.csv', encoding='utf-8')
        each_step = pd.DataFrame(data=each_step)
        each_step.to_csv('./data/each_step2.csv', encoding='utf-8')

        dic = {'gamma': args.gamma, 'capacity': args.capacity, 'sample_frequency': args.sample_frequency,
               'policy_noise': args.policy_noise,
               'noise_clip': args.noise_clip, 'policy_delay': args.policy_delay,
               'exploration_noise': args.exploration_noise, 'actor_lr': 1e-4,
               'critic_1_lr': 1e-3, 'critic_2_lr': 1e-3, 'episode': episode, 'steps': 500, 'time interval': 0.5}
        key = list(dic.keys())
        value = list(dic.values())
        result_excel = pd.DataFrame()
        result_excel['参数名'] = key
        result_excel['参数值'] = value
        result_excel.to_excel('chaocanshu.xls')

        plt.title("Total_Reward")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.plot(range(len(return_list)), return_list, color="red")
        plt.show()
    if args.mode == 'test':
        agent.load()
        for i in range(args.iteration):
            time.sleep(0.3)
            total_reward = 0
            step = 0
            difference_distance_terminal = 0
            distance_terminal, posture, distance, warning, block, overturn, reach, velocity, destination_angle = deal_main_data(
                get_tcp_state)
            state = np.append(distance, posture)  # 22
            state = np.append(state, destination_angle)
            state = np.append(state, velocity)
            state = np.append(state, distance_terminal)
            for t in count():
                action = agent.select_action(state) * max_action
                print(action)
                reward, restart = Reward(difference_distance_terminal, distance_terminal, destination_angle,
                                         posture,
                                         warning, block, overturn, reach, velocity)
                next_state, next_distance_terminal, next_destination_angle, posture, warning, block, overturn, reach, next_velocity = Next_state(
                    action, get_tcp_state)

                difference_distance_terminal = distance_terminal - next_distance_terminal  # 此时刻与下一时刻距离终点的差值
                step += 1
                total_reward += reward

                reset = restart
                if reset:
                    left, right = 0, 0
                    time.sleep(0.3)
                    print(
                        "Total reward \t{}, the episode is \t{:0.2f}, the step is \t{}".format(total_reward, i, t))
                    break
                # next
                destination_angle = next_destination_angle
                velocity = next_velocity
                distance_terminal = next_distance_terminal
                state = next_state
    else:
        raise NameError("mode wrong!!!")


if __name__ == '__main__':
    args = parse_args()
    main(args)
