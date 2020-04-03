import numpy as np
import torch
from collecting_bananas.agents.networks import DQN
from collecting_bananas.agents.utils import get_config


config = get_config()


def test_prediction(observation):
    """ Test if the network's output has the right dimension """
    observation = torch.from_numpy(observation).float()
    dqn = DQN(config.network_config)
    prediction = dqn(observation).detach().numpy()

    assert np.shape(prediction) == (1, config.action_dim)