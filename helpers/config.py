"""
config.py — Hyperparameters and settings for Dreamer v3 Donkey Car
"""

# Image Processing
RGB = True                    # Set to True to use 3-channel color images
IMAGE_SIZE = 64               # Downscaled dimensions for the CNN
IMAGE_CROP_TOP = 40           # Remove the top 40 pixels (usually sky/background)

# Dreamer Training Hyperparameters
DREAMER_BATCH_SIZE = 16       # Number of sequences per training batch (Reduced for RGB)
DREAMER_CHUNK_SIZE = 50       # Length of each sequence (Sequence length for the RSSM)
DREAMER_TRAIN_RATIO = 1.0     # Gradient steps per environment step
DREAMER_BUFFER_SIZE = 200000  # Total transitions stored in replay memory

# Environment & Episode Logic
DREAMER_SEED_EPISODES = 5     # Initial random/P-controller episodes to populate buffer
DREAMER_MAX_EPISODE_STEPS = 1000
DREAMER_THROTTLE_BASE = 0.3   # Default forward throttle

# Neural Network Settings
LR_ACTOR = 3e-5
LR_CRITIC = 3e-5
LR_MODEL = 1e-4               # Slightly higher for the world model to learn dynamics quickly