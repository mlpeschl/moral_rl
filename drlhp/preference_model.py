import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Use GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class PreferenceModelConv(nn.Module):
    def __init__(self, state_shape, in_channels=6, n_actions=9):
        super(PreferenceModelConv, self).__init__()

        # General Parameters
        self.state_shape = state_shape
        self.in_channels = in_channels
        self.n_actions = n_actions

        self.action_embedding = nn.Linear(n_actions, in_channels*state_shape[0]*state_shape[1])
        self.reward_conv1 = nn.Conv2d(in_channels=self.in_channels*2, out_channels=128, kernel_size=2)
        self.reward_conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=2)
        self.reward_conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=2)
        self.reward_out = nn.Linear(32*(state_shape[0]-3)*(state_shape[1]-3), 1)

        # Dropout
        #self.dropout = nn.Dropout(p=0)

        # Activation
        self.relu = nn.LeakyReLU(0.01)

    def forward(self, state, action):
        action_embedded = F.one_hot(torch.tensor(action).long(), self.n_actions).float().to(device)
        action_embedded = self.relu(self.action_embedding(action_embedded)).view(-1, self.in_channels,
                                                                                 self.state_shape[0],
                                                                                 self.state_shape[1])
        state = state.view(-1, self.in_channels, self.state_shape[0], self.state_shape[1])
        x = torch.cat([state, action_embedded], dim=1)
        x = self.relu(self.reward_conv1(x))
        x = self.relu(self.reward_conv2(x))
        x = self.relu(self.reward_conv3(x))
        x = x.view(x.shape[0], -1)
        x = self.reward_out(x)

        return x

    def evaluate_trajectory(self, tau):
        trajectory_states = torch.tensor(tau['states']).float().to(device)
        trajectory_actions = torch.tensor(tau['actions']).to(device)
        predicted_rewards = self.forward(trajectory_states, trajectory_actions)

        return predicted_rewards.sum(dim=0).squeeze(0)

    def compare_trajectory(self, tau_1, tau_2):
        # Returns log P(tau_1 > tau_2) using the Bradley-Terry model

        returns_1 = self.evaluate_trajectory(tau_1)
        returns_2 = self.evaluate_trajectory(tau_2)

        return returns_1 - torch.log(torch.exp(returns_1) + torch.exp(returns_2) + 1e-6)


class PreferenceModelMLP(nn.Module):
    def __init__(self, state_shape, in_channels=6, n_actions=9):
        super(PreferenceModelMLP, self).__init__()

        # General Parameters
        self.state_shape = state_shape
        self.in_channels = in_channels
        self.n_actions = n_actions

        # Layers
        self.action_embedding = nn.Linear(n_actions, 512)
        self.reward_l1 = nn.Linear(self.in_channels*self.state_shape[0]*self.state_shape[1], 512)
        self.reward_l2 = nn.Linear(1024, 2056)
        self.reward_l3 = nn.Linear(2056, 1024)
        self.reward_out = nn.Linear(1024, 1)

        # Dropout
        #self.dropout = nn.Dropout(p=0)

        # Activation
        self.relu = nn.LeakyReLU(0.01)


    def forward(self, state, action):
        action_embedded = F.one_hot(torch.tensor(action).long(), self.n_actions).float().to(device)
        action_embedded = self.relu(self.action_embedding(action_embedded))

        state = state.view(state.shape[0], -1)
        state_embedded = self.relu(self.reward_l1(state))

        x = torch.cat([state_embedded, action_embedded], dim=-1)
        x = self.relu(self.reward_l2(x))
        #x = self.dropout(x)
        x = self.relu(self.reward_l3(x))
        #x = self.dropout(x)
        x = self.reward_out(x)

        return x

    def evaluate_trajectory(self, tau):
        trajectory_states = torch.tensor(tau['states']).float().to(device)
        trajectory_actions = torch.tensor(tau['actions']).to(device)
        predicted_rewards = self.forward(trajectory_states, trajectory_actions)

        return predicted_rewards.sum(dim=0).squeeze(0)

    def compare_trajectory(self, tau_1, tau_2):
        # Returns log P(tau_1 > tau_2) using the Bradley-Terry model

        returns_1 = self.evaluate_trajectory(tau_1)
        returns_2 = self.evaluate_trajectory(tau_2)

        return returns_1 - torch.log(torch.exp(returns_1) + torch.exp(returns_2) + 1e-6)

class PreferenceBuffer:
    def __init__(self):
        self.storage = []

    def add_preference(self, tau_1, tau_2, mu):
        self.storage.append((tau_1, tau_2, mu))


def update_preference_model(preference_model, preference_buffer, preference_optimizer, batch_size):
    overall_loss = torch.tensor(0.).to(device)
    for i in range(batch_size):

        # Sample random preference
        rand_idx = np.random.randint(len(preference_buffer.storage))
        rand_tau_1, rand_tau_2, rand_mu = preference_buffer.storage[rand_idx]

        # Add to Loss
        superior_log_prob = preference_model.compare_trajectory(rand_tau_1, rand_tau_2)
        inferior_log_prob = preference_model.compare_trajectory(rand_tau_2, rand_tau_1)
        overall_loss -= (rand_mu[0]*superior_log_prob + rand_mu[1]*inferior_log_prob)

    overall_loss = overall_loss/batch_size
    preference_optimizer.zero_grad()
    overall_loss.backward()
    preference_optimizer.step()

    return overall_loss.item()
