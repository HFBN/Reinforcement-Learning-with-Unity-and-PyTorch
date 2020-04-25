import numpy as np
from collecting_bananas.agents.buffer import ReplayBuffer
from collecting_bananas.agents.utils import get_config


config = get_config()


def test_sample_batch_concurrency(batch):
    """
    Tests if (observation, action, reward, next_observation, done)-tuples sampled from the buffer are
    equal to (observation, action, reward, next_observation, done)-tuples fed into the buffer
    (i.e. samples remain intact)
    """
    observations = batch.observations
    actions = batch.actions
    rewards = batch.rewards
    next_observations = batch.next_observations
    dones = batch.dones
    buffer = ReplayBuffer(config.buffer_config)
    for i in range(2):
        buffer.store(observations[i, :], actions[i], rewards[i], next_observations[i], dones[i])
    two_instance_sample = buffer.sample_batch(batch_size=1)

    assert (((two_instance_sample.observations[0] == observations[0]).all() and
             (two_instance_sample.actions[0] == actions[0]) and
             (two_instance_sample.rewards[0] == rewards[0]) and
             (two_instance_sample.next_observations[0] == next_observations[0]).all() and
             (two_instance_sample.dones[0] == dones[0])) or
            ((two_instance_sample.observations[0] == observations[1]).all() and
             (two_instance_sample.actions[0] == actions[1]) and
             (two_instance_sample.rewards[0] == rewards[1]) and
             (two_instance_sample.next_observations[0] == next_observations[1]).all() and
             (two_instance_sample.dones[0] == dones[1])))


def test_sample_batch_shape(batch):
    """ Tests the sampled batch's shape"""
    observations = batch.observations
    actions = batch.actions
    rewards = batch.rewards
    next_observations = batch.next_observations
    dones = batch.dones
    buffer = ReplayBuffer(config.buffer_config)
    for i in range(np.shape(observations)[0]):
        buffer.store(observations[i, :], actions[i], rewards[i], next_observations[i], dones[i])
    batch_sample = buffer.sample_batch()

    assert np.shape(batch_sample.observations) == (config.batch_size, config.observation_dim)
    assert np.shape(batch_sample.actions) == (config.batch_size,)
    assert np.shape(batch_sample.rewards) == (config.batch_size,)
    assert np.shape(batch_sample.next_observations) == (config.batch_size, config.observation_dim)
    assert np.shape(batch_sample.dones) == (config.batch_size, )
