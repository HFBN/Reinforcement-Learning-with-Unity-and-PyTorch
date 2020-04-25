import numpy as np
import torch
from collecting_bananas.agents.networks import (DeepQNetwork, DuelingQNetwork,
                                                NoisyDeepQNetwork, NoisyDuelingQNetwork)
from collecting_bananas.agents.utils import get_config


config = get_config()


def test_Deep_Q_prediction(observation):
    """ Test if the network's output has the right dimension """
    observation = torch.from_numpy(observation).float()
    network = DeepQNetwork(config.network_config)
    prediction = network(observation).detach().numpy()

    assert np.shape(prediction) == (1, config.action_dim)


def test_Dueling_Q_prediction(observation):
    """ Test if the network's output has the right dimension """
    observation = torch.from_numpy(observation).float()
    network = DuelingQNetwork(config.network_config)
    prediction = network(observation).detach().numpy()

    assert np.shape(prediction) == (1, config.action_dim)


def test_Noisy_Deep_Q_prediction(observation):
    """ Test if the network's output has the right dimension """
    observation = torch.from_numpy(observation).float()
    network = NoisyDeepQNetwork(config.network_config)
    prediction = network(observation).detach().numpy()

    assert np.shape(prediction) == (1, config.action_dim)


def test_Noisy_Dueling_Q_prediction(observation):
    """ Test if the network's output has the right dimension """
    observation = torch.from_numpy(observation).float()
    network = NoisyDuelingQNetwork(config.network_config)
    prediction = network(observation).detach().numpy()

    assert np.shape(prediction) == (1, config.action_dim)
