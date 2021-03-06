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
    sigma: float
    alpha: float


class DeepQNetwork(nn.Module):
    """A class representing a Deep Q-Network"""
    def __init__(self, config: NetworkConfig):
        """Initialize parameters and build model."""
        super(DeepQNetwork, self).__init__()
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
    def __init__(self, input_dim: int, output_dim: int, sigma: float, alpha: float):
        super(NoisyLinear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sigma = sigma
        self.alpha = alpha

        # Build the weights
        self.weights = nn.Parameter(torch.randn(output_dim, input_dim) / self.input_dim)
        self.register_buffer('weight_noise', torch.empty(output_dim, input_dim))
        # Build the biases
        self.bias = nn.Parameter(torch.randn(output_dim) / self.input_dim)
        self.register_buffer('bias_noise', torch.empty(output_dim))

    def _create_noise(self):
        weight_noise = torch.randn(self.output_dim, self.input_dim) * self.sigma
        bias_noise = torch.randn(self.output_dim) * self.sigma
        self.weight_noise.copy_(weight_noise)
        self.bias_noise.copy_(bias_noise)

    def forward(self, observation: torch.tensor, noise=False):
        if noise:
            self._create_noise()
            # Decrease sigma
            self.sigma = self.alpha * self.sigma
            return F.linear(observation, self.weights + self.weight_noise, self.bias + self.bias_noise)
        else:
            return F.linear(observation, self.weights, self.bias)


class NoisyDeepQNetwork(nn.Module):
    """A class representing a Noisy Deep Q-Network"""
    def __init__(self, config: NetworkConfig):
        """Initialize parameters and build model."""
        super(NoisyDeepQNetwork, self).__init__()
        self.fc1 = NoisyLinear(config.observation_dim, config.layers['fc1'], config.sigma, config.alpha)
        self.fc2 = NoisyLinear(config.layers['fc1'], config.layers['fc2'], config.sigma, config.alpha)
        self.fc3 = NoisyLinear(config.layers['fc2'], config.action_dim, config.sigma, config.alpha)

    def forward(self, observation: torch.Tensor, noise=False) -> torch.Tensor:
        """Build a network that maps state -> action values."""
        cache_ = F.relu(self.fc1.forward(observation, noise))
        cache_ = F.relu(self.fc2.forward(cache_, noise))
        return self.fc3.forward(cache_, noise)


class NoisyDuelingQNetwork(nn.Module):
    """A class representing a Noisy Dueling Q-Network"""
    def __init__(self, config: NetworkConfig):
        """Initialize parameters and build model."""
        # Feature Layer
        super(NoisyDuelingQNetwork, self).__init__()
        self.fc1 = NoisyLinear(config.observation_dim, config.layers['fc1'], config.sigma, config.alpha)

        # Value Stream
        self.fc2 = NoisyLinear(config.layers['fc1'], config.layers['fc2'], config.sigma, config.alpha)
        self.fc3 = NoisyLinear(config.layers['fc2'], 1, config.sigma, config.alpha)

        # Advantage Stream
        self.fc4 = NoisyLinear(config.layers['fc1'], config.layers['fc2'], config.sigma, config.alpha)
        self.fc5 = NoisyLinear(config.layers['fc2'], config.action_dim, config.sigma, config.alpha)

    def forward(self, observation: torch.Tensor, noise=False) -> torch.Tensor:
        """Build a network that maps state -> action values."""
        feature_layer = F.relu(self.fc1.forward(observation, noise))

        # Compute Value
        value_cache_ = F.relu(self.fc2.forward(feature_layer, noise))
        value = self.fc3.forward(value_cache_, noise)

        # Compute Advantages
        advantage_cache_ = F.relu(self.fc4.forward(feature_layer, noise))
        advantages = self.fc5.forward(advantage_cache_, noise)
        return value + (advantages - advantages.mean(dim=1, keepdim=True))
