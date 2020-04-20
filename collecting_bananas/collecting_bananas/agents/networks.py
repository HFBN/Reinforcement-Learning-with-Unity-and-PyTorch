import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from pydantic import BaseModel


class NetworkConfig(BaseModel):
    """A class used to configure a Deep Q-Network"""
    observation_dim: int
    action_dim: int
    layers: Dict[str, int]


class DQN(nn.Module):
    """A class representing a Deep Q-Network"""

    def __init__(self, config: NetworkConfig):
        """Initialize parameters and build model."""
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(config.observation_dim, config.layers['fc1'])
        self.fc2 = nn.Linear(config.layers['fc1'], config.layers['fc2'])
        self.fc3 = nn.Linear(config.layers['fc2'], config.action_dim)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Build a network that maps state -> action values."""
        cache_ = F.relu(self.fc1(observation))
        cache_ = F.relu(self.fc2(cache_))
        return self.fc3(cache_)


class DuelingQNetwork(nn.Module):
    """A class representing a Dueling Q-Network"""

    def __init__(self, config: NetworkConfig):
        """Initialize parameters and build model."""
        super(DuelingQNetwork, self).__init__()
        # Feature Layer
        self.fc1 = nn.Linear(config.observation_dim, config.layers['fc1'])

        # Value Stream
        self.fc2 = nn.Linear(config.layers['fc1'], config.layers['fc2'])
        self.fc3 = nn.Linear(config.layers['fc2'], 1)

        # Advantage Stream
        self.fc4 = nn.Linear(config.layers['fc1'], config.layers['fc2'])
        self.fc5 = nn.Linear(config.layers['fc2'], config.action_dim)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Build a network that maps state -> action values."""
        feature_layer = F.relu(self.fc1(observation))

        # Compute Value
        value_cache_ = F.relu(self.fc2(feature_layer))
        value = F.relu(self.fc3(value_cache_))

        # Compute Advantages
        advantage_cache_ = F.relu(self.fc4(feature_layer))
        advantages = F.relu(self.fc5(advantage_cache_))

        return value + (advantages - advantages.mean())
