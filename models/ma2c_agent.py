import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, action_dim)

    def forward(self, s):
        x = torch.relu(self.fc1(s))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.out(x), dim=-1)

class Critic(nn.Module):
    def __init__(self, global_state_dim):
        super().__init__()
        self.fc1 = nn.Linear(global_state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.out = nn.Linear(128, 1)

    def forward(self, g):
        x = torch.relu(self.fc1(g))
        x = torch.relu(self.fc2(x))
        return self.out(x)

class MA2C:
    def __init__(self, agent_num, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.agent_num = agent_num
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        self.actors = [Actor(state_dim, action_dim) for _ in range(agent_num)]
        self.critic = Critic(state_dim * agent_num)

        self.actor_optims = [optim.Adam(a.parameters(), lr=lr) for a in self.actors]
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)

    def get_action(self, idx, state):
        s = torch.FloatTensor(state)
        prob = self.actors[idx](s)
        return torch.multinomial(prob, 1).item()

    def update(self, experiences):
        for idx, (s, a, r, s_next, done) in enumerate(experiences):
            s = torch.FloatTensor(s)
            s_next = torch.FloatTensor(s_next)
            r = torch.FloatTensor([r])

            prob = self.actors[idx](s)
            log_prob = torch.log(prob[a])

            g_s = torch.FloatTensor(np.hstack([s for s, _, _, _, _ in experiences]))
            g_n = torch.FloatTensor(np.hstack([s_next for _, _, _, s_next, _ in experiences]))

            v = self.critic(g_s)
            vn = self.critic(g_n)
            target = r + self.gamma * vn * (1 - done)
            adv = target - v

            loss_c = F.mse_loss(v, target.detach())
            loss_a = -(log_prob * adv.detach())

            self.actor_optims[idx].zero_grad()
            loss_a.backward()
            self.actor_optims[idx].step()

            self.critic_optim.zero_grad()
            loss_c.backward()
            self.critic_optim.step()