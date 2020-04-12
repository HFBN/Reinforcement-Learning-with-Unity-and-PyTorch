import numpy as np
from pydantic import BaseModel


class OUConfig(BaseModel):
    """A class used to configure an OUNoise"""
    mu: float
    theta: float
    max_sigma: float
    min_sigma: float
    decay_period: int
    action_dim: int


class OUNoise:
    """A class that can be used to create Ornstein-Uhlenbeck-Noise"""

    def __init__(self, config: OUConfig):
        """Initialize the class"""
        self.config = config
        self.sigma = config.max_sigma
        self.state = np.ones(self.config.action_dim) * self.config.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.config.action_dim) * self.config.mu

    def _evolve_state(self) -> np.ndarray:
        x = self.state
        dx = self.config.theta * (self.config.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx

    def noise(self, t: int) -> np.ndarray:
        self._evolve_state()
        self.sigma = self.config.max_sigma - (self.config.max_sigma -
                                              self.config.min_sigma) * min(1.0, t / self.config.decay_period)
        return self.state
