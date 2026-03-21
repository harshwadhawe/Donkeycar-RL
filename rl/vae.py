"""
Variational Autoencoder for image encoding.
Encodes 40x40 grayscale images to a 20-dim latent space.

Architecture from L2D paper (arXiv:2008.00715).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


class Encoder(nn.Module):
    """CNN encoder: 1x40x40 -> latent_dim (mu, log_var)."""

    def __init__(self, in_channels=1, latent_dim=20, image_size=40):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        # After conv1 with stride=2: 40->20, so flat_size = 32 * 20 * 20
        flat_size = 32 * (image_size // 2) * (image_size // 2)
        self.fc_mu = nn.Linear(flat_size, latent_dim)
        self.fc_log_var = nn.Linear(flat_size, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.flatten(start_dim=1)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var


class Decoder(nn.Module):
    """Transposed CNN decoder: latent_dim -> 1x40x40."""

    def __init__(self, in_channels=1, latent_dim=20, image_size=40):
        super().__init__()
        self.image_size = image_size
        half = image_size // 2
        flat_size = 32 * half * half

        self.fc = nn.Linear(latent_dim, flat_size)
        self.half = half
        self.deconv1 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose2d(
            16, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1
        )

    def forward(self, z):
        x = F.relu(self.fc(z))
        x = x.view(-1, 32, self.half, self.half)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))
        return x


class VAE(nn.Module):
    """
    Full VAE with encoder, decoder, and target encoder.

    Loss = BCE reconstruction + KL_WEIGHT * KL divergence
    """

    def __init__(self, in_channels=1, latent_dim=20, image_size=40,
                 lr=1e-5, kl_weight=3.0, device='cpu'):
        super().__init__()
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.device = device

        self.encoder = Encoder(in_channels, latent_dim, image_size).to(device)
        self.decoder = Decoder(in_channels, latent_dim, image_size).to(device)
        # Target encoder for stable critic training (momentum updated)
        self.encoder_target = Encoder(in_channels, latent_dim, image_size).to(device)
        self.encoder_target.load_state_dict(self.encoder.state_dict())

        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=lr
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decoder(z)
        return recon, mu, log_var

    def embed(self, x):
        """Deterministic embedding (mu only) for policy input."""
        mu, _ = self.encoder(x)
        return mu

    def embed_target(self, x):
        """Embedding using target encoder."""
        with torch.no_grad():
            mu, _ = self.encoder_target(x)
        return mu

    def loss(self, x):
        recon, mu, log_var = self.forward(x)
        bce = F.binary_cross_entropy(recon, x, reduction='sum') / x.size(0)
        kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.size(0)
        return bce + self.kl_weight * kl, bce, kl

    def soft_update_target(self, tau=0.005):
        for p, tp in zip(self.encoder.parameters(), self.encoder_target.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)


def preprocess_image(image, crop_top=40, target_size=40, grayscale=True):
    """
    Preprocess raw camera image for VAE input.

    Args:
        image: HxWx3 uint8 numpy array (120x160x3 from camera)
        crop_top: pixels to crop from top
        target_size: resize to target_size x target_size
        grayscale: convert to grayscale

    Returns:
        1xHxW float32 tensor normalized to [0,1]
    """
    img = image[crop_top:, :, :]  # crop top (removes irrelevant scenery)

    if grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (target_size, target_size))
        img = img.astype(np.float32) / 255.0
        img = img[np.newaxis, :, :]  # 1xHxW
    else:
        img = cv2.resize(img, (target_size, target_size))
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # CxHxW

    return torch.from_numpy(img)
