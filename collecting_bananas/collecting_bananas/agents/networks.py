import math
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
    std: float


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
        value = self.fc3(value_cache_)

        # Compute Advantages
        advantage_cache_ = F.relu(self.fc4(feature_layer))
        advantages = self.fc5(advantage_cache_)
        return value + (advantages - advantages.mean(dim=1, keepdim=True))


class NoisyLinear(nn.Module):
    """A class used to represent a Noisy Linear Layer inside the Noisy Networks"""
    def __init__(self, input_dim: int, output_dim: int, std: float):
        super(NoisyLinear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.std = std

        # Build the weights
        self.weight_mu = nn.Parameter(torch.empty(output_dim, input_dim))
        self.weight_sigma = nn.Parameter(torch.empty(output_dim, input_dim))
        self.register_buffer('weight_epsilon', torch.empty(output_dim, input_dim))
        # Build the biases
        self.bias_mu = nn.Parameter(torch.empty(output_dim))
        self.bias_sigma = nn.Parameter(torch.empty(output_dim))
        self.register_buffer('bias_epsilon', torch.empty(output_dim))
        # Initialize
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.input_dim)
        # Initialize parameters
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std / math.sqrt(self.out_features))

    def _scale_noise(self, size: int):
        cache = torch.randn(size)
        return cache.sign().mul_(cache.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.input_dim)
        epsilon_out = self._scale_noise(self.output_dim)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, observation: torch.tensor, train_mode):
        if train_mode:
            return F.linear(observation,
                            self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(observation, self.weight_mu, self.bias_mu)


class NoisyDQN(DQN):
    """A class representing a Noisy Deep Q-Network"""
    def __init__(self, config: NetworkConfig):
        """Initialize parameters and build model."""
        super().__init__(config)
        self.fc1 = nn.NoisyLinear(config.observation_dim, config.layers['fc1'], config.std)
        self.fc2 = nn.NoisyLinear(config.layers['fc1'], config.layers['fc2'], config.std)
        self.fc3 = nn.NoisyLinear(config.layers['fc2'], config.action_dim, config.std)


class NoisyDuelingQNetwork(DuelingQNetwork):
    """A class representing a Noisy Dueling Q-Network"""
    def __init__(self, config: NetworkConfig):
        """Initialize parameters and build model."""
        # Feature Layer
        super().__init__(config)
        self.fc1 = nn.NoisyLinear(config.observation_dim, config.layers['fc1'], config.std)

        # Value Stream
        self.fc2 = nn.NoisyLinear(config.layers['fc1'], config.layers['fc2'], config.std)
        self.fc3 = nn.NoisyLinear(config.layers['fc2'], 1, config.std)

        # Advantage Stream
        self.fc4 = nn.NoisyLinear(config.layers['fc1'], config.layers['fc2'], config.std)
        self.fc5 = nn.NoisyLinear(config.layers['fc2'], config.action_dim, config.std)
