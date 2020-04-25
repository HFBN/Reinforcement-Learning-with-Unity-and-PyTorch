import pytest
import numpy as np
from collecting_bananas.agents.utils import get_config
from collecting_bananas.agents.buffer import Batch


config = get_config()


@pytest.fixture()
def observation():
    # Get a sample observation
    observation = np.ones([1, config.observation_dim])
    return observation


@pytest.fixture()
def batch():
    # Create a sample Batch
    observations = np.random.randn(100, config.observation_dim).astype(np.float32)
    actions = np.random.choice(config.action_dim, size=100).reshape(-1)
    rewards = np.random.choice(1, size=100).reshape(-1)
    next_observations = np.random.randn(100, config.observation_dim).astype(np.float32)
    dones = np.random.choice(1, size=100).astype(bool).reshape(-1)

    return Batch(observations, actions, rewards, next_observations, dones)


@pytest.fixture()
def deltas():
    # Creates a number of td-errors as basis for probability calculation inside an prioritising replay buffer
    return np.random.randn(config.buffer_config.batch_size)
