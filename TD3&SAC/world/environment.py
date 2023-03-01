import time
import numpy as np
from utils.util import deal_main_data



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
    difference_distance_terminal_reward = difference_distance_terminal * 200  # 正常都是0.00X的数
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


def last_100_mean_reward(last_100_reward):
    sum = 0
    for item in last_100_reward:
        sum += item
    return sum / len(last_100_reward)
