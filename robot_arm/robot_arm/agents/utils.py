import numpy as np
from strictyaml import load, YAML
from .agents import AgentConfig
from .buffer import BufferConfig
from .networks import NetworkConfig
from .noise import OUConfig
from pathlib import Path


FILE_PATH = Path(__file__)
ROOT = FILE_PATH.parent.parent
CONFIG_PATH = ROOT / 'config.yml'


def get_config() -> AgentConfig:
    """A helper that loads the config"""
    if CONFIG_PATH.is_file():
        with open(CONFIG_PATH, 'r') as conf_file:
            parsed_config = load(conf_file.read()).data
        buffer_config = BufferConfig(**parsed_config['shared_config'],
                                     **parsed_config['buffer_config'])
        actor_config = NetworkConfig(**parsed_config['shared_config'],
                                     **parsed_config['actor_config'])
        critic_config = NetworkConfig(**parsed_config['shared_config'],
                                      **parsed_config['critic_config'])
        ou_config = OUConfig(**parsed_config['shared_config'],
                             **parsed_config['ou_config'])
        config = AgentConfig(**parsed_config['shared_config'],
                             **parsed_config['agent_config'],
                             buffer_config=buffer_config,
                             actor_config=actor_config,
                             critic_config=critic_config,
                             noise_config=ou_config)
        return config


# A helper for evaluation and smooth plotting:
def smooth(x: np.ndarray) -> np.ndarray:
    n = len(x)
    y = np.zeros(n)
    for i in range(n):
        start = max(0, i-99)
        y[i] = float(x[start:(i+1)].sum()) / (i - start + 1)
    return y
