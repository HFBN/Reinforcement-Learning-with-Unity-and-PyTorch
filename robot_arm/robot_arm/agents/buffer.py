import numpy as np
from typing import Dict
from pydantic import BaseModel


class Batch:
    """A class that represents a batch used in training"""
    def __init__(self, observations: np.ndarray, actions: np.ndarray, rewards: np.float,
                 next_observations: np.ndarray, dones: bool):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.next_observations = next_observations
        self.dones = dones


class BufferConfig(BaseModel):
    """A class used to configure a Replay Buffer"""
    observation_dim: int
    action_dim: int
    buffer_size: int
    min_buffer_size: int
    layers: Dict[str, int]


class ReplayBuffer:
    """A class representing the Replay Buffer"""
    def __init__(self, config: BufferConfig):
        self.config = config
        self.observations = np.zeros([config.observation_dim, config.buffer_size], dtype=np.float32)
        self.next_observations = np.zeros([config.observation_dim, config.buffer_size], dtype=np.float32)
        self.actions = np.zeros([config.action_dim, config.buffer_size], dtype=np.int32)
        self.rewards = np.zeros(config.buffer_size, dtype=np.float32)
        self.dones = np.zeros(config.buffer_size, dtype=np.float32)
        self.pointer, self.size, self.buffer_size = 0, 0, config.buffer_size

    def store(self, observation: np.ndarray, action: np.ndarray, reward: np.float,
              next_observation: np.ndarray, done: bool):
        self.observations[:, self.pointer] = observation
        self.next_observations[:, self.pointer] = next_observation
        self.actions[:, self.pointer] = action
        self.rewards[self.pointer] = reward
        self.dones[self.pointer] = done
        self.pointer = (self.pointer + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def is_ready(self) -> bool:
        """Returns whether more than (observation, action, reward, next_observation, done) tuples are stored"""
        return self.size > self.config.min_buffer_size

    def sample_batch(self, batch_size: int) -> Batch:
        idxs = np.random.randint(0, self.size, size=batch_size)
        return Batch(self.observations[:, idxs].T, self.actions[:, idxs], self.rewards[idxs],
                     self.next_observations[:, idxs].T, self.dones[idxs])
