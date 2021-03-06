import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from pydantic import BaseModel


class NetworkConfig(BaseModel):
    """A class used to configure an actor or critic"""
    observation_dim: int
    action_dim: int
    layers: Dict[str, int]


class Actor(nn.Module):
    """A class representing the actor"""

    def __init__(self, config: NetworkConfig):
        """Initialize parameters and build model."""
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(config.observation_dim, config.layers['fc1'])
        self.fc2 = nn.Linear(config.layers['fc1'], config.layers['fc2'])
        self.fc3 = nn.Linear(config.layers['fc2'], config.layers['fc3'])
        self.fc4 = nn.Linear(config.layers['fc3'], config.action_dim)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Build a network that maps state -> action values."""
        cache_ = F.relu(self.fc1(observation))
        cache_ = F.relu(self.fc2(cache_))
        cache_ = F.relu(self.fc3(cache_))
        return torch.tanh(self.fc4(cache_))


class Critic(nn.Module):
    """A class representing the critic of a DDPG-Agent"""

    def __init__(self, config: NetworkConfig):
        """Initialize parameters and build model."""
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(config.observation_dim + config.action_dim, config.layers['fc1'])
        self.fc2 = nn.Linear(config.layers['fc1'], config.layers['fc2'])
        self.fc3 = nn.Linear(config.layers['fc2'], 1)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Build a network that maps state + action -> state-action values."""
        cache_ = torch.cat([observations, actions], axis=1)
        cache_ = F.relu(self.fc1(cache_))
        cache_ = F.relu(self.fc2(cache_))
        return self.fc3(cache_)


class TwinCritic(nn.Module):
    """A class representing the critic of a TD4-Agent"""

    def __init__(self, config: NetworkConfig):
        """Initialize parameters and build model."""
        super(TwinCritic, self).__init__()
        self.fc1 = nn.Linear(config.observation_dim + config.action_dim, config.layers['fc1'])
        self.fc2 = nn.Linear(config.layers['fc1'], config.layers['fc2'])
        self.fc3 = nn.Linear(config.layers['fc2'], 1)

        self.fc4 = nn.Linear(config.observation_dim + config.action_dim, config.layers['fc1'])
        self.fc5 = nn.Linear(config.layers['fc1'], config.layers['fc2'])
        self.fc6 = nn.Linear(config.layers['fc2'], 1)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Build a network that maps state + action -> state-action values."""
        x = torch.cat([observations, actions], axis=1)
        cache_ = F.relu(self.fc1(x))
        cache_ = F.relu(self.fc2(cache_))
        y_1 = self.fc3(cache_)

        cache_ = F.relu(self.fc4(x))
        cache_ = F.relu(self.fc5(cache_))
        y_2 = self.fc6(cache_)

        return y_1, y_2
