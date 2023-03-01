import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from utils.util import Replay_buffer


class TD3():
    def __init__(self, state_dim, action_dim, max_action, args, path='./trained_models/'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_path = path
        self.args = args
        self.replay_buffer = Replay_buffer()
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.critic_1 = Critic(state_dim, action_dim).to(self.device)
        self.critic_1_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_2 = Critic(state_dim, action_dim).to(self.device)
        self.critic_2_target = Critic(state_dim, action_dim).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=1e-3)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=1e-3)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.max_action = max_action
        self.memory = Replay_buffer(args.capacity)

        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def select_action(self, state):
        state = torch.tensor(state.reshape(1, -1)).float().to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

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

            # Select next action according to target policy:
            noise = torch.ones_like(action).data.normal_(0, self.args.policy_noise).to(self.device)
            noise = noise.clamp(-self.args.noise_clip, self.args.noise_clip)

            smoothed_target_a = (self.actor_target(next_state) + noise)
            smoothed_target_a = smoothed_target_a.clamp(-self.max_action, self.max_action)

            # Compute target Q-value:
            target_Q1 = self.critic_1_target(next_state, smoothed_target_a)
            target_Q2 = self.critic_2_target(next_state, smoothed_target_a)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1 - done) * self.args.gamma * target_Q).detach()

            # Optimize Critic 1:
            current_Q1 = self.critic_1(state, action)
            loss_Q1 = F.mse_loss(current_Q1, target_Q)
            self.critic_1_optimizer.zero_grad()
            loss_Q1.backward()
            self.critic_1_optimizer.step()

            # Optimize Critic 2:
            current_Q2 = self.critic_2(state, action)
            loss_Q2 = F.mse_loss(current_Q2, target_Q)
            self.critic_2_optimizer.zero_grad()
            loss_Q2.backward()
            self.critic_2_optimizer.step()
            # Delayed policy updates:
            if i % self.args.policy_delay == 0:# critic 更新两轮，actor更新一轮
                # Compute actor loss:
                actor_loss = - self.critic_1(state, self.actor(state)).mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                # self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(((1 - self.args.tau) * target_param.data) + self.args.tau * param.data)

                for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                    target_param.data.copy_(((1 - self.args.tau) * target_param.data) + self.args.tau * param.data)

                for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                    target_param.data.copy_(((1 - self.args.tau) * target_param.data) + self.args.tau * param.data)

                self.num_actor_update_iteration += 1
        self.num_critic_update_iteration += 1
        self.num_training += 1

    def save(self):
        if os.path.exists(self.model_path) :
            torch.save(self.actor.state_dict(), self.model_path + 'actor.pth')
            torch.save(self.actor_target.state_dict(), self.model_path + 'actor_target.pth')
            torch.save(self.critic_1.state_dict(), self.model_path + 'critic_1.pth')
            torch.save(self.critic_1_target.state_dict(), self.model_path + 'critic_1_target.pth')
            torch.save(self.critic_2.state_dict(), self.model_path + 'critic_2.pth')
            torch.save(self.critic_2_target.state_dict(), self.model_path + 'critic_2_target.pth')
        else:
            os.mkdir(self.model_path)

    def load(self):
        print('load started')
        self.actor.load_state_dict(torch.load(self.model_path + 'actor.pth'))
        self.actor_target.load_state_dict(torch.load(self.model_path + 'actor_target.pth'))
        self.critic_1.load_state_dict(torch.load(self.model_path + 'critic_1.pth'))
        self.critic_1_target.load_state_dict(torch.load(self.model_path + 'critic_1_target.pth'))
        self.critic_2.load_state_dict(torch.load(self.model_path + 'critic_2.pth'))
        self.critic_2_target.load_state_dict(torch.load(self.model_path + 'critic_2_target.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")


class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_dim, 400)
        # self.fc2=nn.LSTM(400,300)
        self.fc2 = nn.Linear(400, 256)
        self.fc3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.fc1(state))
        a = F.relu(self.fc2(a))
        a = torch.tanh(self.fc3(a)) * self.max_action
        return a


class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)

        q = F.relu(self.fc1(state_action))
        q = F.relu(self.fc2(q))
        q = self.fc3(q)
        return q
