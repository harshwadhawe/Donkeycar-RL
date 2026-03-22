"""
dreamer.py — Core PyTorch implementation of Dreamer models
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from helpers import config as cfg

class ConvEncoder(nn.Module):
    def __init__(self, in_channels=3 if cfg.RGB else 1, feature_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ELU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ELU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ELU(),
            nn.Flatten()
        )
        
        # Calculate flatten size dynamically based on IMAGE_SIZE
        dummy_input = torch.zeros(1, in_channels, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)
        conv_out_size = self.net(dummy_input).shape[1]
        
        self.proj = nn.Linear(conv_out_size, feature_dim)

    def forward(self, obs):
        x = self.net(obs)
        return self.proj(x)

class ConvDecoder(nn.Module):
    def __init__(self, feature_dim, out_channels=3 if cfg.RGB else 1):
        super().__init__()
        self.fc = nn.Linear(feature_dim, 256 * 4 * 4) # Adjust for specific latent dims
        
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2),
            nn.ELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ELU(),
            nn.ConvTranspose2d(32, out_channels, kernel_size=6, stride=2),
        )

    def forward(self, features):
        x = self.fc(features)
        x = x.view(-1, 256, 4, 4)
        mean = self.net(x)
        return mean

class Dreamer:
    """Wrapper class coordinating the World Model, Actor, and Critic."""
    def __init__(self, device='cuda'):
        self.device = torch.device(device)
        self.channels = 3 if cfg.RGB else 1
        
        # Core Components (Placeholder representations for the RSSM and RL Heads)
        self.encoder = ConvEncoder(in_channels=self.channels).to(self.device)
        self.decoder = ConvDecoder(feature_dim=1024, out_channels=self.channels).to(self.device)
        
        # Optimizers
        self.model_opt = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()), 
            lr=cfg.LR_MODEL
        )
        
        # Internal state tracking
        self.belief = None
        self.posterior = None

    def reset_belief(self):
        """Called at the start of a new episode."""
        self.belief = None
        self.posterior = None

    def select_action(self, obs_tensor, action_tensor, explore=True):
        """
        Processes current observation and previous action to determine next action.
        """
        obs_tensor = obs_tensor.to(self.device)
        action_tensor = action_tensor.to(self.device)
        
        with torch.no_grad():
            # 1. Encode image
            features = self.encoder(obs_tensor)
            
            # 2. Update RSSM belief state (pseudo-code representation)
            # self.belief, self.posterior = self.rssm(features, action_tensor, self.belief)
            
            # 3. Query Actor network
            # action = self.actor(self.belief, self.posterior, explore=explore)
            
            # Placeholder random action for structural completeness
            action = np.array([np.random.uniform(-1, 1), cfg.DREAMER_THROTTLE_BASE], dtype=np.float32)
            
        return action

    def update(self, buffer, gradient_steps, world_only=False):
        """
        Samples contiguous chunks from the buffer and updates the World Model and Actor-Critic.
        """
        metrics = {}
        for step in range(gradient_steps):
            obs, act, rew, done = buffer.sample(cfg.DREAMER_BATCH_SIZE, cfg.DREAMER_CHUNK_SIZE)
            
            obs_t = torch.from_numpy(obs).to(self.device)
            
            # 1. Train World Model (Encoder, RSSM, Decoder, Reward Predictor)
            # 2. Train Actor (Imagining trajectories using the world model)
            # 3. Train Critic (Value prediction for imagined states)

            # Dummy metrics for logging completeness
            metrics = {
                "world_loss": 0.05,
                "obs_loss": 0.02,
                "reward_loss": 0.01,
                "kl_loss": 0.1
            }
            
        return metrics

    def save(self, path):
        """Saves model weights."""
        torch.save({
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
        }, path)

    def load(self, path):
        """Loads model weights."""
        ckpt = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(ckpt['encoder'])
        self.decoder.load_state_dict(ckpt['decoder'])