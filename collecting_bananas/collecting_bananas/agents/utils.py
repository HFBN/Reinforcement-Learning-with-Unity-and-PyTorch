import numpy as np
from strictyaml import load, YAML
from .buffer import BufferConfig
from .networks import NetworkConfig
from .agents import AgentConfig
from pathlib import Path


FILE_PATH = Path(__file__)
ROOT = FILE_PATH.parent.parent
CONFIG_PATH = ROOT / 'config.yml'


def get_config() -> AgentConfig:
    """A helper that loads the config"""
    if CONFIG_PATH.is_file():
        with open(CONFIG_PATH, 'r') as conf_file:
            parsed_config = load(conf_file.read())
        buffer_config = BufferConfig(**parsed_config.data)
        network_config = NetworkConfig(**parsed_config.data)
        config = AgentConfig(**parsed_config.data,
                             buffer_config=buffer_config,
                             network_config=network_config)
        return config


# A helper for evaluation and smooth plotting:
def smooth(x: np.ndarray) -> np.ndarray:
    n = len(x)
    y = np.zeros(n)
    for i in range(n):
        start = max(0, i-99)
        y[i] = float(x[start:(i+1)].sum()) / (i - start + 1)
    return y
