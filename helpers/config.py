"""
RL configuration for SAC+VAE and Dreamer v3 agents.
"""

# ─── Shared ───────────────────────────────────────────────

# Action limits
STEER_LIMIT_LEFT = -1.0
STEER_LIMIT_RIGHT = 1.0
THROTTLE_MIN = 0.25
THROTTLE_MAX = 0.6
MAX_STEERING_DIFF = 0.2  # max steering change per step

# Observation
IMAGE_SIZE = 64           # resize to 64x64 (more detail for world model)
IMAGE_CROP_TOP = 40       # crop top 40px from 120x160 raw image
FRAME_STACK = 1
RGB = True                # RGB (4060 can handle it, more info for world model)

# Episode
MAX_EPISODE_STEPS = 500
STEP_LENGTH = 0.05        # 20Hz control loop

# Reward
REWARD_ALIVE = 1.0
REWARD_CRASH = -10.0

# Death detection
DEAD_ZONE_CROP_BOTTOM = 20  # bottom N pixels for track detection
DEAD_ZONE_THRESHOLD = 10    # max bright pixels to count as off-track

# Command history (SAC only, but kept shared for convenience)
COMMAND_HISTORY_LENGTH = 5  # last 5 (steering, throttle, speed) triples

# ─── SAC+VAE ─────────────────────────────────────────────

SAC_RANDOM_EPISODES = 1       # pure random exploration before training
SAC_GRADIENT_STEPS = 600      # training steps per episode

# SAC hyperparameters
SAC_LR = 1e-4
SAC_GAMMA = 0.99
SAC_TAU = 0.005
SAC_TARGET_ENTROPY = -2.0
SAC_BATCH_SIZE = 128
SAC_HIDDEN_SIZE = 64
SAC_BUFFER_SIZE = 1_000_000

# VAE hyperparameters
VAE_LATENT_DIM = 20
VAE_LR = 1e-5
VAE_BATCH_SIZE = 64
VAE_KL_WEIGHT = 3.0
VAE_L2_REG = True

# ─── Dreamer v3 ──────────────────────────────────────────
# Based on: Mastering Diverse Domains through World Models (arXiv:2301.04104)

DREAMER_SEED_EPISODES = 20      # P-controller episodes to seed buffer with full-track data
DREAMER_GRADIENT_STEPS = 50     # fallback if train_ratio can't be computed
DREAMER_TRAIN_RATIO = 512       # paper default for DMC continuous control

# RSSM — categorical latent state
DREAMER_BELIEF_SIZE = 256       # GRU hidden (deterministic)
DREAMER_NUM_CLASSES = 16        # number of categorical distributions
DREAMER_NUM_CATEGORIES = 16     # categories per distribution
DREAMER_STATE_SIZE = 16 * 16    # flattened one-hot = 256
DREAMER_HIDDEN_SIZE = 512       # MLP hidden layers
DREAMER_EMBEDDING_SIZE = 1024   # visual encoder output
DREAMER_UNIMIX = 0.01           # uniform noise mixed into categoricals

# World model training
DREAMER_WORLD_LR = 1e-4
DREAMER_ACTOR_LR = 8e-5
DREAMER_VALUE_LR = 8e-5
DREAMER_ADAM_EPS = 1e-8
DREAMER_BATCH_SIZE = 64
DREAMER_CHUNK_SIZE = 16         # sequence length for BPTT (reduced: short episodes need short chunks)
DREAMER_GRAD_CLIP = 100.0

# KL balancing (replaces free nats)
DREAMER_KL_FREE_NATS = 1.0     # minimum KL (free nats threshold)
DREAMER_KL_REP_WEIGHT = 0.1    # representation loss: trains encoder
DREAMER_KL_DYN_WEIGHT = 0.5    # dynamics loss: trains transition model

# Behavior learning
DREAMER_PLANNING_HORIZON = 8
DREAMER_GAMMA = 0.997
DREAMER_DISCLAM = 0.95         # lambda for GAE / lambda-returns
DREAMER_PCONT = True           # use learned continue predictor

# Return normalization (percentile-based, replaces reward scaling)
DREAMER_RETURN_NORM_LOW = 5     # lower percentile
DREAMER_RETURN_NORM_HIGH = 95   # upper percentile

# Actor
DREAMER_ACTOR_ENTROPY = 1e-4   # entropy reg (reduced: EXPL_AMOUNT provides exploration)
DREAMER_ACTOR_MIN_STD = 0.2    # minimum action std (prevents premature collapse)
DREAMER_ACTOR_INIT_STD = 0.1   # initial action std — low so tanh outputs stay near zero initially
DREAMER_EXPL_AMOUNT = 0.3      # exploration noise std (added to actions during training)

# Slow critic (EMA target, replaces hard target updates)
DREAMER_SLOW_TARGET_FRACTION = 0.02  # EMA rate per gradient step

# Buffer
DREAMER_BUFFER_SIZE = 100_000     # ~4.7GB for RGB 64x64 (was 1M = 46GB OOM)

# Throttle control
DREAMER_FIX_SPEED = True        # fix throttle so agent only learns steering first
DREAMER_THROTTLE_BASE = 0.35   # fixed throttle value (moderate speed)

# Episode limits
DREAMER_MAX_EPISODE_STEPS = 1000  # truncate episodes to prevent aimless wandering