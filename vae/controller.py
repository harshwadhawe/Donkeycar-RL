"""
VAE Controller — wraps a trained VAE for use in RL observation encoding.

Loads a trained VAE checkpoint and provides encode_observation() to convert
raw camera images (120x160x3 uint8) into latent vectors (z_dim float32).
"""

import numpy as np
import torch
from .model import VAE


class VAEController:
    """Encodes camera images to VAE latent vectors."""

    def __init__(self, model_path, z_dim=32, device=None):
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        self.device = device
        self.z_dim = z_dim

        self.vae = VAE(in_channels=3, z_dim=z_dim).to(device)
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.vae.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.vae.load_state_dict(checkpoint)
        self.vae.eval()

    @torch.no_grad()
    def encode_observation(self, obs):
        """
        Encode a raw camera image to a latent vector.

        Args:
            obs: (H, W, 3) uint8 numpy array (120x160x3)

        Returns:
            (z_dim,) float32 numpy array
        """
        # Normalize to [0, 1] and convert to (1, 3, H, W) tensor
        img = obs.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img_tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)
        z = self.vae.encode(img_tensor)
        return z.squeeze(0).cpu().numpy()
