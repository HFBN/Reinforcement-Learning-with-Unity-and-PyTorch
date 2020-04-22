import numpy as np
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
    buffer_size: int
    min_buffer_size: int
    batch_size: int
    alpha: float
    beta: float


class ReplayBuffer:
    """A class representing the Replay Buffer"""
    def __init__(self, config: BufferConfig):
        self.config = config
        self.observations = np.zeros([config.observation_dim, config.buffer_size], dtype=np.float32)
        self.next_observations = np.zeros([config.observation_dim, config.buffer_size], dtype=np.float32)
        self.actions = np.zeros(config.buffer_size, dtype=np.int32)
        self.rewards = np.zeros(config.buffer_size, dtype=np.float32)
        self.dones = np.zeros(config.buffer_size, dtype=np.float32)
        self.pointer, self.size, self.buffer_size = 0, 0, config.buffer_size

    def store(self, observation: np.ndarray, action: np.int32, reward: np.float,
              next_observation: np.ndarray, done: bool):
        self.observations[:, self.pointer] = observation
        self.next_observations[:, self.pointer] = next_observation
        self.actions[self.pointer] = action
        self.rewards[self.pointer] = reward
        self.dones[self.pointer] = done
        self.pointer = (self.pointer + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def is_ready(self) -> bool:
        """Returns whether enough (observation, action, reward, next_observation, done) tuples have been stored"""
        return self.size > self.config.min_buffer_size

    def sample_batch(self) -> Batch:
        idxs = np.random.randint(0, self.size, size=self.config.batch_size)
        return Batch(self.observations[:, idxs].T, self.actions[idxs], self.rewards[idxs],
                     self.next_observations[:, idxs].T, self.dones[idxs])


class PrioritisingReplayBuffer(ReplayBuffer):
    """A class enhancing the ReplayBuffer by Prioritised Experience Replay Sampling"""
    def __init__(self, config: BufferConfig):
        super().__init__(config)
        self.last_idxs = np.zeros(config.batch_size, dtype=np.int32)
        self.deltas = np.zeros(config.buffer_size, dtype=np.float32)
        print(self.deltas)
        self.probabilities = np.zeros(config.buffer_size, dtype=np.float32)

    def update_deltas(self, deltas: np.ndarray):
        np.put(self.deltas, self.last_idxs, deltas)

    def _update_probabilities(self):
        """
        Update the probabilities dependent on the individual td-errors - samples with high td-error will be more likely
        to be sampled.
        """
        # Robust computation of power
        powered_deltas = np.power(np.absolute(self.deltas - np.max(self.deltas)), self.config.alpha)
        # Set unexplored deltas to the max, such that they are picked with highest probability possible
        indicator = [powered_deltas == 0]
        indicator[self.size:] = False
        len(powered_deltas)
        powered_deltas[indicator] = np.max(powered_deltas)
        len(powered_deltas)
        # Normalize to one and store
        self.probabilities = powered_deltas / np.sum(powered_deltas)

    def sample_batch(self) -> Batch:
        # Robustness check:
        if np.sum(np.absolute(self.deltas)) > 0:
            self._update_probabilities()
            idxs = np.random.choice(np.arange(self.config.buffer_size), size=self.config.batch_size,
                                    replace=False, p=self.probabilities)
        else:
            idxs = np.random.choice(np.arange(self.config.buffer_size), size=self.config.batch_size,
                                    replace=False)
        self.last_idxs = idxs
        return Batch(self.observations[:, idxs].T, self.actions[idxs], self.rewards[idxs],
                     self.next_observations[:, idxs].T, self.dones[idxs])
