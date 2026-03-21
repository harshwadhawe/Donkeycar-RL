#!/usr/bin/env python3
"""
Train a VAE on Donkey Car tub data.

Usage:
    python -m vae.train_vae --tub=data/tub_sim --epochs=100
    python -m vae.train_vae --tub=data/tub_sim --epochs=100 --z-dim=32

The trained model is saved to logs/vae/vae_z{z_dim}_best.pth
"""

import os
import json
import glob
import argparse

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.utils import make_grid

from .model import VAE


class TubImageDataset(Dataset):
    """Load images from a donkeycar tub directory."""

    def __init__(self, tub_path):
        self.images_dir = os.path.join(tub_path, 'images')
        self.image_files = sorted(glob.glob(os.path.join(self.images_dir, '*.jpg')))
        if not self.image_files:
            raise ValueError(f'No images found in {self.images_dir}')
        print(f'Found {len(self.image_files)} images in {tub_path}')

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx]).convert('RGB')
        img = np.array(img, dtype=np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        return torch.from_numpy(img)


def save_reconstruction_samples(vae, dataset, epoch, save_dir, device, n=5):
    """Save side-by-side original vs reconstruction for n random images."""
    vae.eval()
    indices = np.random.choice(len(dataset), size=n, replace=False)
    originals = torch.stack([dataset[i] for i in indices]).to(device)

    with torch.no_grad():
        recons, _, _ = vae(originals)

    # Interleave: orig1, recon1, orig2, recon2, ...
    pairs = torch.stack([originals, recons], dim=1).flatten(0, 1)  # (2*n, C, H, W)
    grid = make_grid(pairs.cpu(), nrow=2, padding=2, pad_value=1.0)

    # Convert to PIL and save
    grid_np = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img = Image.fromarray(grid_np)
    path = os.path.join(save_dir, f'recon_epoch_{epoch:03d}.png')
    img.save(path)
    return path


def train(args):
    # Device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f'Device: {device}')

    # Data
    dataset = TubImageDataset(args.tub)
    val_size = max(1, int(len(dataset) * 0.1))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    pin = device == 'cuda'
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0)

    # Model
    vae = VAE(in_channels=3, z_dim=args.z_dim).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    # Output
    save_dir = os.path.join('logs', 'vae')
    samples_dir = os.path.join(save_dir, 'samples')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    best_path = os.path.join(save_dir, f'vae_z{args.z_dim}_best.pth')
    best_val_loss = float('inf')

    print(f'Training VAE (z_dim={args.z_dim}) for {args.epochs} epochs')
    print(f'Train: {train_size}, Val: {val_size}, Batch: {args.batch_size}')
    print(f'Reconstruction samples saved to: {samples_dir}/')

    for epoch in range(1, args.epochs + 1):
        # Train
        vae.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            recon, mu, logvar = vae(batch)
            loss, recon_l, kl_l = VAE.loss(recon, batch, mu, logvar,
                                           kl_weight=args.kl_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.size(0)
        train_loss /= train_size

        # Validate
        vae.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                recon, mu, logvar = vae(batch)
                loss, _, _ = VAE.loss(recon, batch, mu, logvar,
                                      kl_weight=args.kl_weight)
                val_loss += loss.item() * batch.size(0)
        val_loss /= val_size
        scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch:3d}/{args.epochs} | '
                  f'train={train_loss:.2f} val={val_loss:.2f} lr={lr:.1e}')

        # Save 5 random reconstruction samples every epoch
        sample_path = save_reconstruction_samples(
            vae, dataset, epoch, samples_dir, device, n=5
        )
        if epoch % 10 == 0 or epoch == 1:
            print(f'  Samples: {sample_path}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': vae.state_dict(),
                'z_dim': args.z_dim,
                'epoch': epoch,
                'val_loss': val_loss,
            }, best_path)

    print(f'Best val loss: {best_val_loss:.2f}')
    print(f'Model saved to: {best_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train VAE on tub data')
    parser.add_argument('--tub', type=str, required=True, help='Path to tub directory')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--z-dim', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--kl-weight', type=float, default=1.0)
    args = parser.parse_args()
    train(args)
