import numpy as np
import copy
from pydantic import BaseModel
from .buffer import BufferConfig, ReplayBuffer, PrioritisingReplayBuffer
from .networks import NetworkConfig, DeepQNetwork, DuelingQNetwork, NoisyDeepQNetwork, NoisyDuelingQNetwork
import torch
import torch.optim as optim


class AgentConfig(BaseModel):
    """A class used to configure an Agent"""
    observation_dim: int
    action_dim: int
    epsilon: float
    epsilon_decay_period: int
    min_epsilon: float
    gamma: float
    learning_rate: float
    tau: float
    buffer_config: BufferConfig
    network_config: NetworkConfig
    prioritising: bool
    noisy: bool


class BaseAgent:
    """ A basis class for several agents """
    def __init__(self, config: AgentConfig):
        """ Initialize the ReplayBuffer as well as some attributes"""
        # Save config
        self.config = config

        # Initialize the parts:
        if not config.prioritising:
            self.memory = ReplayBuffer(config.buffer_config)
        else:
            self.memory = PrioritisingReplayBuffer(config.buffer_config)
        self.main_network = None
        self.target_network = None

        # Initialize starting epsilon:
        self.epsilon = config.epsilon
        self.epsilon_decay_rate = self.epsilon / config.epsilon_decay_period

    def memento(self, observation: np.ndarray, action: np.int32, reward: np.float,
                next_observation: np.ndarray, done: bool):
        """Store a (observation, action, reward, next_observation, done) tuple"""
        self.memory.store(observation, action, reward, next_observation, done)

    def _predict(self, observations: np.ndarray) -> np.ndarray:
        """ Predict the state-action-values using the main network given one or several observation(s) """
        observations = torch.from_numpy(observations.reshape(-1, self.config.observation_dim)).float()
        if self.config.noisy:
            return self.main_network.forward(observations, True).detach().numpy()
        else:
            return self.main_network(observations).detach().numpy()

    def act(self, observation: np.ndarray) -> np.int32:
        """ Makes the Agent choose an action based on the observation and its current estimator"""
        if not self.config.noisy:
            # Decrease exploration
            self.epsilon = np.max([self.config.min_epsilon, self.epsilon - self.epsilon_decay_rate])
            if np.random.rand() < self.epsilon:
                # Explore
                return np.int32(np.random.choice(self.config.action_dim))
        # Exploit
        estimates = self._predict(observation)
        # Casting necessary for environment
        return np.argmax(estimates[0]).astype(np.int32)

    def _predict_targets(self, observations: np.ndarray) -> np.ndarray:
        """ Predict the state-action-values using the target network given one or several observation(s) """
        observations = torch.from_numpy(observations.reshape(-1, self.config.observation_dim)).float()
        return self.target_network.forward(observations).detach().numpy()

    def _soft_update(self, main_network: DeepQNetwork, target_network: DeepQNetwork, tau: float):
        """Soft update model parameters"""
        for target_param, main_param in zip(target_network.parameters(), main_network.parameters()):
            target_param.data.copy_(tau * main_param.data + (1.0 - tau) * target_param.data)

    def save(self, path: str):
        torch.save(self.main_network, path)

    def load(self, path: str):
        self.main_network = torch.load(path)
        self.target_network = torch.load(path)


class DeepQAgent(BaseAgent):
    """A class representing an agent using Deep-Q-Learning"""
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        """ Initialize the components Estimator and Target Network as well as some further attributes"""

        # Initialize the additional parts:
        if not config.noisy:
            self.main_network = DeepQNetwork(config.network_config)
        else:
            self.main_network = NoisyDeepQNetwork(config.network_config)
        # Copy the main network as the target network:
        self.target_network = copy.copy(self.main_network)

        # Initialize optimizer:
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=config.learning_rate)

    def learn(self):
        """Perform a one step gradient update with a batch samples from experience"""
        experience = self.memory.sample_batch()
        observations = experience.observations
        actions = experience.actions
        rewards = experience.rewards
        next_observations = experience.next_observations
        # We are going to use the dones to adjust target values once a terminal observation is reached.
        dones = experience.dones
        importance = experience.importance

        # Calculate update targets:
        next_Qs = self._predict_targets(next_observations)
        targets = rewards.reshape(-1) + self.config.gamma * (1-dones.reshape(-1)) * \
            np.amax(next_Qs, axis=1).reshape(-1)

        # Calculate loss
        observations = torch.from_numpy(observations).float()
        actions = torch.from_numpy(actions.reshape(len(actions), 1)).long()
        targets = torch.from_numpy(targets.reshape(len(targets), 1)).float()
        importance = torch.from_numpy(importance.reshape(len(importance), 1)).float()
        predictions = self.main_network.forward(observations).gather(dim=1, index=actions)
        # For a non-prioritizing replay buffer, importance has only entries equal to one and this is equal to MSE
        loss = torch.sum(importance * (targets - predictions) ** 2)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update Prioritising Replay Buffer (if applicable)
        if self.config.prioritising:
            # Compute TD-targets and store them in the buffer
            deltas = (targets - predictions).detach().numpy()
            self.memory.update_deltas(deltas)

        # Increment number of updates done so far and update target network if necessary
        self._soft_update(self.main_network, self.target_network, self.config.tau)


class DoubleDeepQAgent(DeepQAgent):
    """A class representing an agent using Double-Deep-Q-Learning"""
    def __init__(self, config: AgentConfig):
        super().__init__(config)

    def learn(self):
        """Perform a one step gradient update with a batch samples from experience"""
        experience = self.memory.sample_batch()
        observations = experience.observations
        actions = experience.actions
        rewards = experience.rewards
        next_observations = experience.next_observations
        # We are going to use the dones to adjust target values once a terminal observation is reached.
        dones = experience.dones
        importance = experience.importance

        # Calculate update targets:
        next_actions = np.argmax(self._predict(next_observations), axis=1).reshape(-1)
        next_Qs = self._predict_targets(next_observations)[np.arange(len(next_actions)), next_actions].reshape(-1)
        targets = rewards.reshape(-1) + self.config.gamma * (1-dones.reshape(-1)) * next_Qs

        # Calculate loss
        observations = torch.from_numpy(observations).float()
        actions = torch.from_numpy(actions.reshape(len(actions), 1)).long()
        targets = torch.from_numpy(targets.reshape(len(targets), 1)).float()
        importance = torch.from_numpy(importance.reshape(len(targets), 1)).float()
        predictions = self.main_network.forward(observations).gather(dim=1, index=actions)
        # For a non-prioritizing replay buffer, importance has only entries equal to one and this is equal to MSE
        loss = torch.sum(importance * (targets - predictions) ** 2)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update Prioritising Replay Buffer (if applicable)
        if self.config.prioritising:
            # Compute TD-targets and store them in the buffer
            deltas = (targets - predictions).detach().numpy()
            self.memory.update_deltas(deltas)

        # Increment number of updates done so far and update target network if necessary
        self._soft_update(self.main_network, self.target_network, self.config.tau)


class DuelingDeepQAgent(DeepQAgent):
    """A class representing an agent using Dueling-Deep-Q-Learning"""
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        """ Initialize the components Estimator and Target Network as well as some further attributes"""

        # Initialize the additional parts:
        if not config.noisy:
            self.main_network = DuelingQNetwork(config.network_config)
        else:
            self.main_network = NoisyDuelingQNetwork(config.network_config)
        # Copy the main network as the target network:
        self.target_network = copy.copy(self.main_network)

        # Initialize optimizer:
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=config.learning_rate)


class DuelingDoubleDeepQAgent(DoubleDeepQAgent):
    """A class representing an agent using Dueling-Double-Deep-Q-Learning"""
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        """ Initialize the components Estimator and Target Network as well as some further attributes"""

        # Initialize the additional parts:
        if not config.noisy:
            self.main_network = DuelingQNetwork(config.network_config)
        else:
            self.main_network = NoisyDuelingQNetwork(config.network_config)
        # Copy the main network as the target network:
        self.target_network = copy.copy(self.main_network)

        # Initialize optimizer:
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=config.learning_rate)


class NoisyDeepQAgent(DeepQAgent):
    """A class representing an agent using Deep-Q-Learning with Noisy Exploration"""
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        """ Initialize the Noisy Estimator and Target Network as well as the optimizer"""

        # Initialize the additional parts:
        self.main_network = NoisyDeepQNetwork(config.network_config)
        # Copy the main network as the target network:
        self.target_network = copy.copy(self.main_network)

        # Initialize optimizer:
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=config.learning_rate)

    def _predict(self, observations: np.ndarray) -> np.ndarray:
        """ Predict the state-action-values using the main network given one or several observation(s) """
        observations = torch.from_numpy(observations.reshape(-1, self.config.observation_dim)).float()
        return self.main_network.forward(observations, True).detach().numpy()

    def act(self, observation: np.ndarray) -> np.int32:
        """ Makes the Agent choose an action based on the observation and its current estimator"""
        estimates = self._predict(observation)
        # Casting necessary for environment
        return np.argmax(estimates[0]).astype(np.int32)
