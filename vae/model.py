"""
Variational Autoencoder for Donkey Car image encoding.

Encodes 120x160x3 camera images to a 32-dim latent space.
Used by SAC to get compact observation vectors instead of raw pixels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """CNN encoder: 3x120x160 -> z_dim."""

    def __init__(self, in_channels=3, z_dim=32):
        super().__init__()
        # 120x160 -> 60x80 -> 30x40 -> 15x20 -> 7x10
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(),
        )
        # 256 * 7 * 10 = 17920
        self.fc_mu = nn.Linear(256 * 7 * 10, z_dim)
        self.fc_logvar = nn.Linear(256 * 7 * 10, z_dim)

    def forward(self, x):
        h = self.conv(x)
        h = h.flatten(start_dim=1)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """Transposed CNN decoder: z_dim -> 3x120x160."""

    def __init__(self, in_channels=3, z_dim=32):
        super().__init__()
        self.fc = nn.Linear(z_dim, 256 * 7 * 10)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),           # -> 14x20
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),            # -> 28x40
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),             # -> 56x80
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, 4, stride=2, padding=1),    # -> 112x160
            nn.Sigmoid(),
        )

    def forward(self, z):
        h = F.relu(self.fc(z))
        h = h.view(-1, 256, 7, 10)
        out = self.deconv(h)
        # Crop to exact 120x160 (deconv gives 112x160, so pad height)
        # Actually 7*2=14, 14*2=28, 28*2=56, 56*2=112 — need to handle this
        return F.interpolate(out, size=(120, 160), mode='bilinear', align_corners=False)


class VAE(nn.Module):
    """Full VAE: encode images to latent z, decode back."""

    def __init__(self, in_channels=3, z_dim=32):
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
        recon_loss = F.mse_loss(recon, x, reduction='sum') / x.size(0)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        return recon_loss + kl_weight * kl_loss, recon_loss, kl_loss
