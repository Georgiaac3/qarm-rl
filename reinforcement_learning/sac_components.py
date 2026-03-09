import collections
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from core.config import settings


class ReplayBuffer:
    """Circular buffer for storing transitions."""

    def __init__(self):
        self.buffer = collections.deque(maxlen=settings.buffer_limit)

    def put(self, transition: Tuple):
        """
        Add a transition (state, action, reward, next_state, done) to the buffer.
        """
        self.buffer.append(transition)

    def sample(self, n: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample n random transitions from the buffer.

        Args:
            n: Number of transitions to sample
        Returns:
            Tuple of tensors (states, actions, rewards, next_states, done_masks)
        """
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_mask_lst.append([done_mask])

        return (
            torch.tensor(s_lst, dtype=torch.float),
            torch.tensor(a_lst, dtype=torch.float),
            torch.tensor(r_lst, dtype=torch.float),
            torch.tensor(s_prime_lst, dtype=torch.float),
            torch.tensor(done_mask_lst, dtype=torch.float),
        )

    def size(self) -> int:
        return len(self.buffer)


class PolicyNetwork(nn.Module):
    """
    Policy network (Actor).

    Takes a state (7D) and produces a Gaussian distribution
    over actions (4D).
    """

    def __init__(self, learning_rate: float, state_dim: int = 7, action_dim: int = 4):
        """
        Initialize the policy network.

        Args:
            learning_rate: Learning rate for the optimizer
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
        """
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc_mu = nn.Linear(128, action_dim)  # Mean
        self.fc_std = nn.Linear(128, action_dim)  # Standard deviation
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # Alpha (temperature) as a learnable parameter
        self.log_alpha = torch.tensor(np.log(settings.init_alpha))
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=settings.lr_alpha)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: produce an action from the state.

        Args:
            x: State tensor

        Returns:
            real_action: Action bounded in [-1, 1]
            real_log_prob: Log probability of the action
        """
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))  # Ensure std > 0
        dist = Normal(mu, std)
        action = dist.rsample()  # Reparameterization trick
        log_prob = dist.log_prob(action)

        # Apply tanh to bound action in [-1, 1]
        real_action = torch.tanh(action)

        # Correct log_prob for tanh transformation
        real_log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        real_log_prob = real_log_prob.sum(1, keepdim=True)  # Sum over dimensions

        return real_action, real_log_prob

    def train_net(self, q1: nn.Module, q2: nn.Module, mini_batch: Tuple):
        """
        Train the policy to maximize Q - alpha*log_prob.

        Args:
            q1: First Q-network
            q2: Second Q-network
            mini_batch: Batch of transitions
        """
        s, _, _, _, _ = mini_batch
        a, log_prob = self.forward(s)
        entropy = -self.log_alpha.exp() * log_prob

        # Take the minimum of the two Q-values: critic values
        q1_val, q2_val = q1(s, a), q2(s, a)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]

        # Objective: maximize Q and entropy
        loss = -min_q - entropy  # Gradient ascent
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        # Train alpha to reach target entropy
        self.log_alpha_optimizer.zero_grad()
        alpha_loss = -(self.log_alpha.exp() * (log_prob + settings.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()


class QNet(nn.Module):
    """
    Q-Network (Critic).
    Takes a state (7D) and an action (4D) and predicts the Q-value.
    """

    def __init__(self, learning_rate: float, state_dim: int = 7, action_dim: int = 4):
        """
        Initialize the Q-network.

        Args:
            learning_rate: Learning rate for the optimizer
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
        """
        super(QNet, self).__init__()
        self.fc_s = nn.Linear(state_dim, 64)
        self.fc_a = nn.Linear(action_dim, 64)
        self.fc_cat = nn.Linear(128, 32)
        self.fc_out = nn.Linear(32, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Compute Q(s, a).
        x: State tensor
        a: Action tensor
        """
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1, h2], dim=1)
        q = F.relu(self.fc_cat(cat))
        q = self.fc_out(q)
        return q

    def train_net(self, target: torch.Tensor, mini_batch: Tuple):
        """
        Train the Q-network to minimize TD error.

        Args:
            target: Target Q-values
            mini_batch: Batch of transitions
        """
        s, a, r, s_prime, done = mini_batch
        loss = F.smooth_l1_loss(self.forward(s, a), target)
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

    def soft_update(self, net_target: nn.Module):
        """
        Soft update of the target network.
        It is called soft because the target network parameters are slowly updated
        towards the main network parameters.
        Args:
            net_target: Target network to update
        """
        for param_target, param in zip(net_target.parameters(), self.parameters()):
            param_target.data.copy_(
                param_target.data * (1.0 - settings.tau) + param.data * settings.tau
            )
