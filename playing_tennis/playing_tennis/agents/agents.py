import numpy as np
import copy
from pydantic import BaseModel
from .buffer import BufferConfig, ReplayBuffer
from .networks import NetworkConfig, Actor, Critic, TwinCritic
from .noise import OUConfig, OUNoise
import torch
import torch.nn.functional as F
import torch.optim as optim


class AgentConfig(BaseModel):
    """A class used to configure an Agent"""
    observation_dim: int
    action_dim: int
    action_high: float
    action_low: float
    batch_size: int
    gamma: float
    actor_learning_rate: float
    critic_learning_rate: float
    action_noise_std: float
    tau: float
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

        # Set train mode (enables exploration)
        self.train_mode = False

    def memento(self, observation: np.ndarray, action: np.ndarray, reward: np.float,
                next_observation: np.ndarray, done: bool):
        """Store a (observation, action, reward, next_observation, done) tuple"""
        self.memory.store(observation, action, reward, next_observation, done)

    def set_buffer(self, buffer: ReplayBuffer):
        """Replaces the internal buffer with a given external buffer as an easy way to incorporate shared buffer"""
        self.memory = buffer

    def _predict_action(self, observations: np.ndarray) -> np.ndarray:
        """Predict the actions using the main actor given one or several observation(s) """
        observations = torch.from_numpy(observations.reshape(-1, self.config.observation_dim)).float()
        return self.main_actor.forward(observations).detach().numpy()

    def _predict_targets(self, observations: np.ndarray) -> np.ndarray:
        """Predict the state-actions-values given one or several state-action-pairs"""
        observations = torch.from_numpy(observations.reshape(-1, self.config.observation_dim)).float()
        actions = self.target_actor.forward(observations)
        values = self.target_critic.forward(observations, actions).detach().numpy()
        return values

    def act(self, observation: np.ndarray, t=0) -> np.ndarray:
        """Makes the Agent choose an action based on the observation and its current estimator"""
        action = self._predict_action(observation)
        if self.train_mode:
            noise = self.ou_noise.noise(t)
            action = np.clip(action + noise, self.config.action_low, self.config.action_high)
        return action

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
    """A class representing an agent using Deep Deterministic Policy Gradient"""
    def __init__(self, config: AgentConfig):
        super().__init__(config)

        # Initialize optimizer:
        self.actor_optimizer = optim.Adam(self.main_actor.parameters(), lr=config.actor_learning_rate)
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
        actions = torch.from_numpy(actions).float()
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


class TD3Agent(BaseAgent):
    """A class representing an agent using Twin Delayed Deep Deterministic Policy Gradient"""
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.main_critic = TwinCritic(self.config.critic_config)
        self.target_critic = copy.copy(self.main_critic)

        # Initialize optimizer:
        self.actor_optimizer = optim.Adam(self.main_actor.parameters(), lr=config.actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.main_critic.parameters(), lr=config.critic_learning_rate)

        # We will need to count the update steps so far:
        self.step_count = 0

    def _predict_targets(self, observations: np.ndarray) -> np.ndarray:
        """ Predict the state-actions-values given one or several state-action-pairs"""
        observations = torch.from_numpy(observations.reshape(-1, self.config.observation_dim)).float()
        actions = self.target_actor.forward(observations)
        noise = torch.empty(actions.size()).normal_(mean=0, std=self.config.action_noise_std)
        actions = torch.clamp(actions + noise, min=self.config.action_low, max=self.config.action_high)
        values, alternative_values = self.target_critic.forward(observations, actions)
        return np.minimum(values.detach().numpy(), alternative_values.detach().numpy())

    def learn(self):
        """Perform a one step gradient update with a batch samples from experience"""
        experience = self.memory.sample_batch(self.config.batch_size)
        observations = experience.observations
        actions = experience.actions
        rewards = experience.rewards
        next_observations = experience.next_observations
        # We are going to use the dones to adjust target values once a terminal observation is reached.
        dones = experience.dones

        # Calculate update targets
        values = self._predict_targets(next_observations)
        targets = rewards.reshape(-1) + self.config.gamma * (1-dones.reshape(-1)) * values.reshape(-1)

        # Update Critic
        observations = torch.from_numpy(observations).float()
        actions = torch.from_numpy(actions).float()
        targets = torch.from_numpy(targets.reshape(len(targets), 1)).float()
        predictions, alternative_predictions = self.main_critic.forward(observations, actions)
        critic_loss = F.mse_loss(predictions, targets) + F.mse_loss(alternative_predictions, targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor (delayed)
        if self.step_count % 2 == 0:
            actor_loss, _ = self.main_critic.forward(observations, self.main_actor.forward(observations))
            actor_loss = -1 * actor_loss.mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self._soft_update(self.main_actor, self.target_actor,
                              self.main_critic, self.target_critic, self.config.tau)

        self.step_count += 1
