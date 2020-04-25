import numpy as np
from collecting_bananas.agents.agents import DeepQAgent
from collecting_bananas.agents.utils import get_config


config = get_config()


def test_act(observation):
    config.noisy = False
    agent = DeepQAgent(config)

    # Exploration
    agent.epsilon = 1
    random_action = agent.act(observation)
    # Exploitation
    agent.epsilon = 0
    chosen_action = agent.act(observation)

    assert np.shape(random_action) == ()
    assert np.shape(chosen_action) == ()
    assert random_action < config.action_dim
    assert chosen_action < config.action_dim
    assert type(random_action) == np.int32
    assert type(chosen_action) == np.int32


def test_noisy_act(observation):
    config.noisy = True
    agent = DeepQAgent(config)

    # Exploitation
    agent.epsilon = 0
    chosen_action = agent.act(observation)

    assert np.shape(chosen_action) == ()
    assert chosen_action < config.action_dim
    assert type(chosen_action) == np.int32
