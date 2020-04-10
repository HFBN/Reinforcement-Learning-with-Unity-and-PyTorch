import numpy as np
import copy
from pydantic import BaseModel
from .buffer import BufferConfig, ReplayBuffer
from .networks import NetworkConfig, Actor, Critic
from .utils import OUConfig, OUNoise
import torch
import torch.nn.functional as F
import torch.optim as optim


class AgentConfig(BaseModel):
    """A class used to configure an Agent"""
    observation_dim: int
    action_dim: int
    action_high: np.float32
    action_low: np.float32
    batch_size: int
    gamma: np.float32
    actor_learning_rate: np.float32
    critic_learning_rate: np.float32
    tau: np.float32
    buffer_config: BufferConfig
    actor_config: NetworkConfig
    critic_config: NetworkConfig
    noise_config: OUConfig


class BaseAgent:
    """ A basis class for several agents """
    def __init__(self, config: AgentConfig):
        """ Initialize the ReplayBuffer as well as some attributes"""
        # Save config
        self.config = config

        # Initialize the parts:
        self.memory = ReplayBuffer(config.buffer_config)
        self.main_actor = Actor(config.actor_config)
        self.main_critic = Critic(config.critic_config)
        self.target_actor = copy.copy(self.main_actor)
        self.target_critic = copy.copy(self.main_critic)
        self.ou_noise = OUNoise(config.noise_config)

    def memento(self, observation: np.ndarray, action: np.ndarray, reward: np.float,
                next_observation: np.ndarray, done: bool):
        """Store a (observation, action, reward, next_observation, done) tuple"""
        self.memory.store(observation, action, reward, next_observation, done)

    def _predict_action(self, observations: np.ndarray) -> np.ndarray:
        """ Predict the actions using the main actor given one or several observation(s) """
        observations = torch.from_numpy(observations.reshape(-1, self.config.observation_dim)).float()
        return self.main_actor.forward(observations).detach().numpy()

    def _predict_targets(self, observations: np.ndarray) -> np.ndarray:
        """ Predict the state-actions-values given one or several state-action-pairs"""
        observations = torch.from_numpy(observations.reshape(-1, self.config.observation_dim)).float()
        actions = self.target_actor.forward(observations)
        values = self.target_critic.forward(observations, actions).detach().numpy()
        return values

    def act(self, observation: np.ndarray, t, reset_noise=False) -> np.int32:
        """ Makes the Agent choose an action based on the observation and its current estimator"""
        noise = self.ou_noise.noise(t)
        if reset_noise:
            self.ou_noise.reset()
        action = self._predict_action(observation)
        return np.clip(action + noise, self.config.action_low, self.config.action_high)

    def _soft_update(self, main_actor: Actor, target_actor: Actor,
                     main_critic: Critic, target_critic: Critic, tau: float):
        """Soft update model parameters"""
        for target_param, main_param in zip(target_actor.parameters(), main_actor.parameters()):
            target_param.data.copy_(tau * main_param.data + (1.0 - tau) * target_param.data)
        for target_param, main_param in zip(target_critic.parameters(), main_critic.parameters()):
            target_param.data.copy_(tau * main_param.data + (1.0 - tau) * target_param.data)

    def save(self, actor_path: str, critic_path: str):
        torch.save(self.main_actor, actor_path)
        torch.save(self.main_critic, critic_path)

    def load(self, actor_path: str, critic_path):
        self.main_actor = torch.load(actor_path)
        self.main_critic = torch.load(critic_path)
        self.target_actor = torch.load(actor_path)
        self.target_critic = torch.load(critic_path)


class DDPGAgent(BaseAgent):
    """A class representing an agent using Deep-Q-Learning"""
    def __init__(self, config: AgentConfig):
        super().__init__(config)

        # Initialize optimizer:
        self.actor_optimizer = optim.Adam(self.main_actor.parameters(), lr=config.action_learning_rate)
        self.critic_optimizer = optim.Adam(self.main_critic.parameters(), lr=config.critic_learning_rate)

    def learn(self):
        """Perform a one step gradient update with a batch samples from experience"""
        experience = self.memory.sample_batch(self.config.batch_size)
        observations = experience.observations
        actions = experience.actions
        rewards = experience.rewards
        next_observations = experience.next_observations
        # We are going to use the dones to adjust target values once a terminal observation is reached.
        dones = experience.dones

        # Calculate update targets:
        values = self._predict_targets(next_observations)
        targets = rewards.reshape(-1) + self.config.gamma * (1-dones.reshape(-1)) * values.reshape(-1)

        # Update Critic
        observations = torch.from_numpy(observations).float()
        actions = torch.from_numpy(actions.reshape(len(actions), 1)).long()
        targets = torch.from_numpy(targets.reshape(len(targets), 1)).float()
        predictions = self.main_critic.forward(observations, actions)
        critic_loss = F.mse_loss(predictions, targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor
        actor_loss = -self.main_critic.forward(observations, self.main_actor.forward(observations)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self._soft_update(self.main_actor, self.target_actor,
                          self.main_critic, self.target_critic, self.config.tau)
