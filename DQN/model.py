import copy
import random
from random import choice, sample
import torch
import torch.nn as nn
import os
from collections import deque
import numpy as np


class DQNNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.online = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Softmax()
        )

        self.target = copy.deepcopy(self.online)

        # Q_target parameters are frozen
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, x, mode):
        if mode == 'online':
            return self.online(x)
        elif mode == 'target':
            return self.target(x)
        else:
            raise 'mode error'


class DQNAgent():
    def __init__(self, state_dim, hidden_dim, action_dim, save_dir='./model'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.save_path = save_dir
        self.action_dim = action_dim
        self.net = DQNNet(state_dim, hidden_dim, action_dim)
        self.net.to(self.device)
        self.memory = deque(maxlen=1000000)
        self.batch_size = 32
        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.cur_step = 0
        self.save_interval = 5e4
        self.gamma = 0.9
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()
        self.burnin = 1e4  # min. experiences before training 1e4
        self.learn_interval = 3  # 更新网络的间隔
        self.sync_interval = 1e4  # 赋值 target网络的间隔
        self.actions = [
            np.array([40000, 40000]),
            np.array([30000, 30000]),
            np.array([20000, 20000]),
            np.array([10000, 10000]),
            np.array([0, 0]),
            # np.array([-10000, -10000]),
            # np.array([-20000, -20000]),
            # np.array([-30000, -30000]),
            np.array([-40000, -40000]),
            np.array([40000, -40000]),
            # np.array([30000, -30000]),
            # np.array([20000, -20000]),
            # np.array([10000, -10000]),
            np.array([-40000, 40000]),
            # np.array([-30000, 30000]),
            # np.array([-20000, 20000]),
            # np.array([-10000, 10000])
        ]

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            action_idx = random.randint(0, self.action_dim - 1)
        else:
            state = state.to(self.device)
            action_values = self.net(state, mode='online')
            action_idx = torch.argmax(action_values, axis=1).item()

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.cur_step += 1

        action = self.actions[action_idx]
        return action, action_idx

    def exp_pool(self, state, next_state, action, reward, done):
        """
                Store the experience to self.memory (replay buffer)
                Inputs:
                state (LazyFrame),
                next_state (LazyFrame),
                action (int),
                reward (float),
                done(bool))
                """

        state = torch.tensor(state).to(self.device).to(torch.float)
        next_state = torch.tensor(next_state).to(self.device).to(torch.float)
        action = torch.tensor([action]).to(self.device).to(torch.float)
        reward = torch.tensor([reward]).to(self.device).to(torch.float)
        done = torch.tensor([done]).to(self.device).to(torch.float)

        self.memory.append((state, next_state, action, reward, done,))

    def recall(self):
        batch = sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def td_estimate(self, state, action):
        state = state.squeeze(-1)

        current_Q = self.net(state, mode="online")[
            np.arange(0, self.batch_size), action.to(torch.long)
        ]  # Q_online(s,a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state = next_state.squeeze(-1)
        next_state_Q = self.net(next_state, mode="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, mode="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def learn(self):
        if self.cur_step % self.sync_interval == 0:
            self.sync_Q_target()

        if self.cur_step % self.save_interval == 0:
            self.save()

        if self.cur_step < self.burnin:
            return None, None

        if self.cur_step % self.learn_interval != 0:
            return None, None

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)
        return (td_est.mean().item(), loss)

    def save(self):
        save_path = os.path.join(self.save_path, f"DQN{int(self.cur_step // self.save_interval)}.pth")
        torch.save(self.net.state_dict(), save_path)
        print(f'model saved at{save_path}')

    def load(self):
        load_path = os.path.join(self.save_path, 'DQN.pth')
        self.net.load_state_dict(torch.load(load_path))
        print(f'model loaded from {load_path}')
