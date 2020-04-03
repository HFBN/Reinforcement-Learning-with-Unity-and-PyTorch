import numpy as np
from collecting_bananas.agents.agents import DoubleQAgent
from collecting_bananas.agents.utils import get_config


config = get_config()


def test_act(observation):
    agent = DoubleQAgent(config)

    # Exploration
    agent.epsilon = 1
    random_action = agent.act(observation)
    # Exploitation
    agent.epsilon = 0
    chosen_action = agent.act(observation)

    assert random_action < config.action_dim
    assert chosen_action < config.action_dim
    assert type(random_action) == np.int32
    assert type(chosen_action) == np.int32
