import numpy as np
import copy
from pydantic import BaseModel
from .buffer import BufferConfig, ReplayBuffer
from .networks import NetworkConfig, DQN
import torch
import torch.nn.functional as F
import torch.optim as optim


class AgentConfig(BaseModel):
    """A class used to configure an Agent"""
    observation_dim: int
    action_dim: int
    batch_size: int
    epsilon: float
    epsilon_decay_period: int
    min_epsilon: float
    gamma: float
    learning_rate: float
    update_period: int
    tau: float
    buffer_config: BufferConfig
    network_config: NetworkConfig


class BaseAgent:
    """ A basis class for several agents """
    def __init__(self, config: AgentConfig):
        """ Initialize the ReplayBuffer as well as some attributes"""
        # Save config
        self.config = config

        # Initialize the parts:
        self.memory = ReplayBuffer(config.buffer_config)
        self.main_network = None

        # Initialize starting epsilon:
        self.epsilon = config.epsilon
        self.epsilon_decay_rate = self.epsilon / config.epsilon_decay_period

    def memento(self, observation: np.ndarray, action: np.ndarray, reward: np.float,
                next_observation: np.ndarray, done: bool):
        """Store a (observation, action, reward, next_observation, done) tuple"""
        self.memory.store(observation, action, reward, next_observation, done)

    def _predict(self, observations: np.ndarray) -> np.ndarray:
        """ Predict the state-action-values using the main network given one or several observation(s) """
        observations = torch.from_numpy(observations.reshape(-1, self.config.observation_dim)).float()
        return self.main_network(observations).detach().numpy()

    def act(self, observation: np.ndarray) -> np.int32:
        """ Makes the Agent choose an action based on the observation and its current estimator"""
        # Decrease exploration
        self.epsilon = np.max([self.config.min_epsilon, self.epsilon - self.epsilon_decay_rate])

        if np.random.rand() < self.epsilon:
            # Explore
            return np.int32(np.random.choice(self.config.action_dim))

        # Exploit
        estimates = self._predict(observation)
        # Casting necessary for environment
        return np.argmax(estimates[0]).astype(np.int32)

    def save(self, path: str):
        torch.save(self.main_network, path)

    def load(self, path: str):
        self.main_network = torch.load(path)


class DeepQAgent(BaseAgent):
    """A class representing an agent using Deep-Q-Learning"""
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        """ Initialize the components Estimator and Target Network as well as some further attributes"""

        # Initialize the additional parts:
        self.main_network = DQN(config.network_config)
        # Copy the main network as the target network:
        self.target_network = copy.copy(self.main_network)

        # Initialize optimizer:
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=config.learning_rate)

        # Initialize Update Counter:
        self.num_updates = 0

    def _predict_targets(self, observations: np.ndarray) -> np.ndarray:
        """ Predict the state-action-values using the target network given one or several observation(s) """
        observations = torch.from_numpy(observations.reshape(-1, self.config.observation_dim)).float()
        return self.target_network(observations).detach().numpy()

    def learn(self):
        """Perform a one step gradient update with a batch samples from experience"""
        experience = self.memory.sample_batch(self.config.batch_size)
        observations = experience.observations
        actions = experience.actions
        rewards = experience.rewards
        next_observations = experience.next_observations
        # We are going to use the dones to adjust target values once a terminal observation is reached.
        dones = experience.dones

        # Calculate update targets:
        next_Qs = self._predict_targets(next_observations)
        targets = rewards.reshape(-1) + self.config.gamma * (1-dones.reshape(-1)) * \
            np.amax(next_Qs, axis=1).reshape(-1)

        # Calculate loss
        observations = torch.from_numpy(observations).float()
        actions = torch.from_numpy(actions.reshape(len(actions), 1)).long()
        targets = torch.from_numpy(targets.reshape(len(targets), 1)).float()
        predictions = self.main_network(observations).gather(dim=1, index=actions)
        loss = F.mse_loss(predictions, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Increment number of updates done so far and update target network if necessary
        self.num_updates += 1
        self._soft_update(self.main_network, self.target_network, self.config.tau)
        if self.num_updates % self.config.update_period == 0:
            pass
            # self.target_network = copy.copy(self.main_network)

    def _soft_update(self, main_network: DQN, target_network: DQN, tau: float):
        """Soft update model parameters"""
        for target_param, main_param in zip(target_network.parameters(), main_network.parameters()):
            target_param.data.copy_(tau * main_param.data + (1.0 - tau) * target_param.data)

    def load(self, path: str):
        self.main_network = torch.load(path)
        self.target_network = torch.load(path)
