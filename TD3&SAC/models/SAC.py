import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import itertools
import numpy as np
from utils.util import Replay_buffer, soft_update_params
import copy


class SAC(object):
    def __init__(self, state_dim, act_dim, hidden_dim, args, path='./trained_models/SAC'):
        self.memory = Replay_buffer()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_path = path
        self.args = args
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.init_temperature = 0.1
        self.discount = args.discount
        self.num_training = 0
        self.actor_update_freq = args.actor_update_freq
        self.critic_target_update_freq = args.critic_target_update_freq
        self.critic_tau = args.critic_tau

        self.actor = Actor(
            self.state_dim, self.act_dim, self.hidden_dim, log_std_min=-10, log_std_max=2
        ).to(self.device)
        self.critic = Critic(
            self.state_dim, self.hidden_dim, self.act_dim
        ).to(self.device)

        self.critic_target = Critic(
            self.state_dim, self.hidden_dim, self.act_dim
        ).to(self.device)

        # 目标网络初始化为critic网络的参数
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True

        self.target_entropy = -np.prod(self.act_dim)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=1e-3, betas=(0.9, 0.999)
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=1e-3, betas=(0.9, 0.999)
        )
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=1e-4, betas=(0.5, 0.999)
        )
        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            state = state.unsqueeze(0)
            mu, _, _, _ = self.actor(
                state, compute_pi=False, compute_log_pi=False
            )
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            state = state.unsqueeze(0)
            mu, pi, _, _ = self.actor(state, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()

    def update_critic(self, state, action, reward, next_state, not_done):
        _, policy_action, log_pi, _ = self.actor(next_state)
        target_Q1, target_Q2 = self.critic_target(next_state, policy_action)
        # 这里采用了double Q-learning防止其中一个Q网络的估值过高
        # 计算有熵的形式的V和Q
        target_V = torch.min(target_Q1,
                             target_Q2) - self.alpha.detach() * log_pi
        target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        # 两个Q估计，双保险

        current_Q1, current_Q2 = self.critic(state, action)
        # J(Q)
        critic_loss = F.mse_loss(current_Q1,
                                 target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor_and_alpha(self, state):
        _, pi, log_pi, log_std = self.actor(state)
        actor_Q1, actor_Q2 = self.critic(state, pi)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        # V(s_t)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)
                                            ) + log_std.sum(dim=-1)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_pi - self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update(self, num_iteration):
        if self.num_training % 10 == 0:
            print('-' * 15)
            print("training steps:{} ".format(self.num_training))
            print('-' * 15)
        for i in range(num_iteration):
            x, y, u, r, d = self.memory.sample(self.args.batch_size)

            state = torch.FloatTensor(x).to(self.device)
            action = torch.FloatTensor(u).to(self.device)
            next_state = torch.FloatTensor(y).to(self.device)
            done = torch.FloatTensor(d).to(self.device)
            reward = torch.FloatTensor(r).to(self.device)

            self.update_critic(state, action, reward, next_state, 1 - done)
            if i % self.actor_update_freq == 0:
                self.update_actor_and_alpha(state)

            if i % self.critic_target_update_freq == 0:
                # 软更新
                soft_update_params(
                    self.critic.Q1, self.critic_target.Q1, self.critic_tau
                )
                soft_update_params(
                    self.critic.Q2, self.critic_target.Q2, self.critic_tau
                )

        self.num_training += 1

    def save(self):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (self.model_path, self.num_training)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (self.model_path, self.num_training)
        )

    def load(self):
        print('load started')
        self.actor.load_state_dict(
            torch.load('%s/actor.pt' % (self.model_path))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic.pt' % (self.model_path))
        )
        print("====================================")
        print("model has been loaded...")
        print("====================================")


class QFunction(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)
        return self.fc(state_action)


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        super().__init__()
        self.Q1 = QFunction(
            self.state_dim, self.action_dim, self.hidden_dim
        )
        self.Q2 = QFunction(
            self.state_dim, self.action_dim, self.hidden_dim
        )

    def forward(self, state, action):
        q1 = self.Q1(state, action)
        q2 = self.Q2(state, action)
        return q1, q2


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, log_std_min, log_std_max):
        super().__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.fc = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, 2 * action_dim)
        )

    def forward(self, state, compute_pi=True, compute_log_pi=True):
        mu, log_std = self.fc(state).chunk(2, dim=-1)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)

        if compute_pi:
            std = log_std.exp()
            noice = torch.randn_like(mu)
            pi = mu + noice * std

        else:
            pi = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noice, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)
        return mu, pi, log_pi, log_std


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi
