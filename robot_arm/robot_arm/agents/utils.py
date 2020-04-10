import numpy as np
from pydantic import BaseModel
from strictyaml import load, YAML
from .agents import AgentConfig
from .buffer import BufferConfig
from .networks import NetworkConfig
from pathlib import Path


FILE_PATH = Path(__file__)
ROOT = FILE_PATH.parent.parent
CONFIG_PATH = ROOT / 'config.yml'


class OUConfig(BaseModel):
    """A class used to configure an OUNoise"""
    mu: np.float32
    theta: np.float32
    sigma: np.float32
    max_sigma: np.float32
    min_sigma: np.float32
    decay_period: int
    action_dim: int


class OUNoise:
    """A class that can be used to create Ornstein-Uhlenbeck-Noise"""

    def __init__(self, config: OUConfig):
        """Initialize the class"""
        if config.min_sigma is None:
            config.min_sigma = config.max_sigma
        self.config = config
        self.state = np.ones(self.config.action_dim) * self.config.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.config.action_dim) * self.config.mu

    def _evolve_state(self) -> np.ndarray:
        x = self.state
        dx = self.config.theta * (self.config.mu - x) + self.config.sigma * np.random.randn(len(x))
        self.state = x + dx

    def noise(self, t: int) -> np.ndarray:
        self._evolve_state()
        self.config.sigma = self.config.max_sigma - (self.config.max_sigma -
                                                     self.config.min_sigma) * min(1.0, t / self.config.decay_period)
        return self.state


def get_config() -> AgentConfig:
    """A helper that loads the config"""
    if CONFIG_PATH.is_file():
        with open(CONFIG_PATH, 'r') as conf_file:
            parsed_config = load(conf_file.read())
        buffer_config = BufferConfig(**parsed_config.shared_config.data,
                                     **parsed_config.buffer_config.data)
        actor_config = NetworkConfig(**parsed_config.shared_config.data,
                                     **parsed_config.actor_config.data)
        critic_config = NetworkConfig(**parsed_config.shared_config.data,
                                      **parsed_config.critic_config.data)
        ou_config = OUConfig(**parsed_config.shared_config.data,
                             **parsed_config.ou_config.data)
        config = AgentConfig(**parsed_config.shared_config.data,
                             **parsed_config.actor_config.data,
                             buffer_config=buffer_config,
                             actor_config=actor_config,
                             critic_config=critic_config,
                             ou_config=ou_config)
        return config


# A helper for evaluation and smooth plotting:
def smooth(x: np.ndarray) -> np.ndarray:
    n = len(x)
    y = np.zeros(n)
    for i in range(n):
        start = max(0, i-99)
        y[i] = float(x[start:(i+1)].sum()) / (i - start + 1)
    return y