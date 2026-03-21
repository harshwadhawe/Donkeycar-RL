"""
Variational Autoencoder for Donkey Car image encoding.

Encodes 120x160x3 camera images to a latent space (default 64-dim).
Used by SAC to get compact observation vectors instead of raw pixels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """CNN encoder: 3x120x160 -> z_dim."""

    def __init__(self, in_channels=3, z_dim=64):
        super().__init__()
        # 120x160 -> 60x80 -> 30x40 -> 15x20 -> 7x10 -> 3x5
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.SiLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.SiLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.SiLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.SiLU(),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512), nn.SiLU(),
        )
        # 512 * 3 * 5 = 7680
        self.fc_mu = nn.Linear(512 * 3 * 5, z_dim)
        self.fc_logvar = nn.Linear(512 * 3 * 5, z_dim)

    def forward(self, x):
        h = self.conv(x).flatten(start_dim=1)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """Transposed CNN decoder: z_dim -> 3x120x160."""

    def __init__(self, in_channels=3, z_dim=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(z_dim, 512 * 3 * 5),
            nn.SiLU(),
        )
        # 3x5 -> 7x10 -> 15x20 -> 30x40 -> 60x80 -> 120x160
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, output_padding=(1, 0)),
            nn.BatchNorm2d(256), nn.SiLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, output_padding=(1, 0)),
            nn.BatchNorm2d(128), nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.SiLU(),
            nn.ConvTranspose2d(32, in_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, 512, 3, 5)
        return self.deconv(h)


class VAE(nn.Module):
    """Full VAE: encode images to latent z, decode back."""

    def __init__(self, in_channels=3, z_dim=64):
        super().__init__()
        self.z_dim = z_dim
        self.encoder = Encoder(in_channels, z_dim)
        self.decoder = Decoder(in_channels, z_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def encode(self, x):
        """Deterministic encoding (mu only) for policy input."""
        mu, _ = self.encoder(x)
        return mu

    @staticmethod
    def loss(recon, x, mu, logvar, kl_weight=1.0):
        # L1 + L2 reconstruction (L1 preserves sharpness, L2 smooths)
        l1 = F.l1_loss(recon, x, reduction='sum') / x.size(0)
        l2 = F.mse_loss(recon, x, reduction='sum') / x.size(0)
        recon_loss = l1 + l2
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        return recon_loss + kl_weight * kl_loss, recon_loss, kl_loss
