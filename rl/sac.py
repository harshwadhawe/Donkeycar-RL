"""
Soft Actor-Critic (SAC) with VAE encoder.

Architecture from L2D paper (arXiv:2008.00715):
- Actor: MLP(35 -> 64 -> 64 -> 4) with tanh squashing
- Twin Critics: MLP(37 -> 64 -> 64 -> 1)
- Automatic entropy tuning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as cfg
from .vae import VAE
from .mlp import MLP
from .buffer import ReplayBuffer


class GaussianActor(nn.Module):
    """
    Gaussian policy with tanh squashing.
    Input: state (35D) -> Output: mean, log_std for 2 actions (steering, throttle)
    """

    LOG_STD_MIN = -20
    LOG_STD_MAX = 2

    def __init__(self, state_dim, action_dim=2, hidden_dim=64):
        super().__init__()
        self.net = MLP(state_dim, action_dim * 2, hidden_dim)

    def forward(self, state):
        out = self.net(state)
        mean, log_std = out.chunk(2, dim=-1)
        log_std = log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x = normal.rsample()  # reparameterization trick
        action = torch.tanh(x)
        # Log prob with tanh squashing correction
        log_prob = normal.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def get_action(self, state):
        """Get action for inference (no grad)."""
        with torch.no_grad():
            action, _ = self.sample(state)
        return action


class TwinCritic(nn.Module):
    """Twin Q-networks for SAC."""

    def __init__(self, state_dim, action_dim=2, hidden_dim=64):
        super().__init__()
        self.q1 = MLP(state_dim + action_dim, 1, hidden_dim)
        self.q2 = MLP(state_dim + action_dim, 1, hidden_dim)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)


class SAC_VAE:
    """
    SAC agent with VAE encoder.

    State = [VAE_latent(20), command_history(15)] = 35D
    Action = [steering, throttle] = 2D
    """

    def __init__(self, device='cpu'):
        self.device = device
        state_dim = cfg.VAE_LATENT_DIM + cfg.COMMAND_HISTORY_LENGTH * 3  # 35
        action_dim = 2

        # VAE
        channels = 1 if not cfg.RGB else 3
        self.vae = VAE(
            in_channels=channels,
            latent_dim=cfg.VAE_LATENT_DIM,
            image_size=cfg.IMAGE_SIZE,
            lr=cfg.VAE_LR,
            kl_weight=cfg.VAE_KL_WEIGHT,
            device=device
        )

        # Actor
        self.actor = GaussianActor(state_dim, action_dim, cfg.SAC_HIDDEN_SIZE).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.SAC_LR)

        # Twin critics + targets
        self.critic = TwinCritic(state_dim, action_dim, cfg.SAC_HIDDEN_SIZE).to(device)
        self.critic_target = TwinCritic(state_dim, action_dim, cfg.SAC_HIDDEN_SIZE).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic.parameters()) + list(self.vae.encoder.parameters()) +
            list(self.vae.decoder.parameters()),
            lr=cfg.SAC_LR
        )

        # Automatic entropy tuning
        self.target_entropy = cfg.SAC_TARGET_ENTROPY
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=cfg.SAC_LR)

        # Replay buffer
        self.buffer = ReplayBuffer(cfg.SAC_BUFFER_SIZE)

        self.gamma = cfg.SAC_GAMMA
        self.tau = cfg.SAC_TAU

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, image_tensor, cmd_history_tensor):
        """
        Select action given preprocessed image tensor and command history.

        Args:
            image_tensor: 1xCxHxW float tensor
            cmd_history_tensor: 1x(COMMAND_HISTORY_LENGTH*3) float tensor

        Returns:
            action: numpy array [steering, throttle] in [-1, 1]
        """
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            cmd_history_tensor = cmd_history_tensor.to(self.device)
            z = self.vae.embed(image_tensor)
            state = torch.cat([z, cmd_history_tensor], dim=-1)
            action = self.actor.get_action(state)
        return action.cpu().numpy().flatten()

    def random_action(self):
        """Random action for exploration."""
        return torch.rand(2).numpy() * 2 - 1  # uniform [-1, 1]

    def update(self, gradient_steps=600):
        """Run SAC training for gradient_steps on replay buffer."""
        if len(self.buffer) < cfg.SAC_BATCH_SIZE:
            return {}

        total_critic_loss = 0
        total_actor_loss = 0
        total_alpha_loss = 0

        for _ in range(gradient_steps):
            images, cmds, actions, rewards, next_images, next_cmds, not_dones = \
                self.buffer.sample(cfg.SAC_BATCH_SIZE)

            images = images.to(self.device)
            cmds = cmds.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            next_images = next_images.to(self.device)
            next_cmds = next_cmds.to(self.device)
            not_dones = not_dones.to(self.device)

            # Encode images
            z = self.vae.embed(images)
            z_next = self.vae.embed_target(next_images)
            state = torch.cat([z, cmds], dim=-1)
            next_state = torch.cat([z_next, next_cmds], dim=-1)

            # --- Critic update ---
            with torch.no_grad():
                next_action, next_log_prob = self.actor.sample(next_state)
                q1_next, q2_next = self.critic_target(next_state, next_action)
                q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_prob
                q_target = rewards + self.gamma * not_dones * q_next

            q1, q2 = self.critic(state, actions)
            critic_loss = 0.5 * F.mse_loss(q1, q_target) + 0.5 * F.mse_loss(q2, q_target)

            # Add VAE loss to critic (joint encoder optimization)
            vae_loss, _, _ = self.vae.loss(images)
            total_loss = critic_loss + vae_loss

            self.critic_optimizer.zero_grad()
            total_loss.backward()
            self.critic_optimizer.step()

            # Soft update targets
            self._soft_update(self.critic, self.critic_target, self.tau)
            self.vae.soft_update_target(self.tau)

            # --- Actor update ---
            new_action, log_prob = self.actor.sample(state.detach())
            q1_new, q2_new = self.critic(state.detach(), new_action)
            q_new = torch.min(q1_new, q2_new)
            actor_loss = (self.alpha.detach() * log_prob - q_new).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # --- Alpha update ---
            alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            total_critic_loss += critic_loss.item()
            total_actor_loss += actor_loss.item()
            total_alpha_loss += alpha_loss.item()

        n = gradient_steps
        return {
            'critic_loss': total_critic_loss / n,
            'actor_loss': total_actor_loss / n,
            'alpha_loss': total_alpha_loss / n,
            'alpha': self.alpha.item(),
            'buffer_size': len(self.buffer),
        }

    def _soft_update(self, source, target, tau):
        for p, tp in zip(source.parameters(), target.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'encoder': self.vae.encoder.state_dict(),
            'encoder_target': self.vae.encoder_target.state_dict(),
            'decoder': self.vae.decoder.state_dict(),
            'log_alpha': self.log_alpha.data,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.vae.encoder.load_state_dict(checkpoint['encoder'])
        self.vae.encoder_target.load_state_dict(checkpoint['encoder_target'])
        self.vae.decoder.load_state_dict(checkpoint['decoder'])
        self.log_alpha.data = checkpoint['log_alpha']
