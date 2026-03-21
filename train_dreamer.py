#!/usr/bin/env python3
"""
train_dreamer.py — Dreamer v3 training for Donkey Car simulator.

Trains directly against the gym environment (no donkeycar vehicle loop).
Uses the categorical RSSM world model to learn to drive from scratch.

Setup:
    conda activate donkey
    pip install -r requirements.txt

Usage:
    python train_dreamer.py                          # new training
    python train_dreamer.py --model=models/dreamer.pth --resume
    python train_dreamer.py --episodes=200
    python train_dreamer.py --track=donkey-warehouse-v0

Monitor:
    tensorboard --logdir ./logs/tb_logs/ --port 6006
"""

import os
import time
import argparse
import logging

import numpy as np
import torch
import gymnasium as gym
import gym_donkeycar  # noqa: F401
import cv2

# Bridge old gym_donkeycar versions that register with gym (not gymnasium)
try:
    gym.spec('donkey-generated-track-v0')
    _USE_SHIMMY = False
except gym.error.NameNotFound:
    import shimmy
    gym.register_envs(shimmy)
    _USE_SHIMMY = True

import donkeycar as dk

from rl.dreamer import Dreamer
from rl.buffer import EpisodeBuffer
from rl import config as cfg

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


# ==============================================================================
# Device
# ==============================================================================
def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


# ==============================================================================
# Image Preprocessing Wrapper
# ==============================================================================
class DonkeyPreprocessWrapper(gym.ObservationWrapper):
    """Preprocess donkey images for Dreamer: crop, resize, grayscale, normalize."""

    def __init__(self, env, crop_top=40, target_size=40, grayscale=True):
        super().__init__(env)
        self.crop_top = crop_top
        self.target_size = target_size
        self.grayscale = grayscale
        channels = 1 if grayscale else 3
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(channels, target_size, target_size),
            dtype=np.float32
        )

    def observation(self, obs):
        img = obs[self.crop_top:, :, :]
        if self.grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, (self.target_size, self.target_size))
            img = img.astype(np.float32) / 255.0
            return img[np.newaxis, :, :]  # (1, H, W)
        else:
            img = cv2.resize(img, (self.target_size, self.target_size))
            img = img.astype(np.float32) / 255.0
            return img.transpose(2, 0, 1)  # (3, H, W)


# ==============================================================================
# Reward Shaping Wrapper
# ==============================================================================
class RewardShapingWrapper(gym.Wrapper):
    """
    Reward shaping for Dreamer: wider CTE bell, moderate penalties.

    Key design: rewards should be dense, informative, and NOT dominated
    by a single penalty. The agent needs gradient signal for "slightly
    better" trajectories, not just "everything is equally bad."
    """

    def __init__(self, env, max_episode_steps=None):
        super().__init__(env)
        self.episode_step = 0
        self.max_episode_steps = max_episode_steps or cfg.DREAMER_MAX_EPISODE_STEPS
        self.last_steering = 0.0

    def reset(self, **kwargs):
        self.episode_step = 0
        self.last_steering = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, _reward, terminated, truncated, info = self.env.step(action)
        self.episode_step += 1

        speed = float(info.get("speed", 0.0))
        cte = float(info.get("cte", 0.0))

        # CTE bell curve — WIDER sigma (1.0) so agent gets gradient
        # even when moderately off-center. sigma=0.35 was too tight.
        cte_reward = 3.0 * np.exp(-(cte / 1.0) ** 2)

        # Speed reward — only forward progress matters
        speed_reward = 0.5 * max(speed, 0.0)

        # Survival bonus (constant, not ramped — we want immediate signal)
        survival = 0.1

        reward = cte_reward + speed_reward + survival

        # Steering smoothness penalty — punish back-and-forth oscillation
        steering = float(np.asarray(action)[0]) if hasattr(action, '__len__') else 0.0
        steer_diff = abs(steering - self.last_steering)
        reward -= 0.3 * steer_diff ** 2
        self.last_steering = steering

        # Standstill penalty (mild)
        if speed < 0.1:
            reward -= 0.5

        # Crash penalty — moderate (10, not 120!)
        if terminated or truncated:
            reward -= 10.0

        # Episode step limit
        if self.episode_step >= self.max_episode_steps:
            truncated = True

        return obs, reward, terminated, truncated, info


# ==============================================================================
# Smooth Action Wrapper
# ==============================================================================
class SmoothActionWrapper(gym.Wrapper):
    """Exponential moving average on actions."""

    def __init__(self, env, alpha=0.5):
        super().__init__(env)
        self.alpha = alpha
        self._prev = None

    def reset(self, **kwargs):
        self._prev = None
        return self.env.reset(**kwargs)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if self._prev is not None:
            action = self.alpha * action + (1 - self.alpha) * self._prev
        self._prev = action
        return self.env.step(action)


# ==============================================================================
# Env Factory
# ==============================================================================
def make_env(env_id, conf):
    if _USE_SHIMMY:
        env = gym.make(
            "GymV21Environment-v0",
            env_id=env_id,
            make_kwargs={"conf": conf},
        )
    else:
        env = gym.make(env_id, conf=conf)
    env = DonkeyPreprocessWrapper(
        env, crop_top=cfg.IMAGE_CROP_TOP,
        target_size=cfg.IMAGE_SIZE, grayscale=not cfg.RGB
    )
    env = SmoothActionWrapper(env, alpha=0.5)
    env = RewardShapingWrapper(env)
    return env


# ==============================================================================
# Training Loop
# ==============================================================================
def train(args):
    dk_cfg = dk.load_config(myconfig=args.myconfig)
    device = get_device()
    logger.info(f'Device: {device}')

    track = args.track or getattr(dk_cfg, 'DONKEY_GYM_ENV_NAME', 'donkey-generated-track-v0')
    sim_path = getattr(dk_cfg, 'DONKEY_SIM_PATH', 'manual')

    conf = {
        "exe_path": sim_path,
        "host": getattr(dk_cfg, 'SIM_HOST', '127.0.0.1'),
        "port": 9091,
        "frame_skip": 2,
        "cam_resolution": (
            getattr(dk_cfg, 'IMAGE_H', 120),
            getattr(dk_cfg, 'IMAGE_W', 160),
            getattr(dk_cfg, 'IMAGE_DEPTH', 3),
        ),
        "log_level": 20,
        "throttle_max": 1.0,
        "max_cte": 3.0,        # terminate early when hopelessly off-track
        "start_delay": 5.0,    # give sim time to launch before connecting
    }

    logger.info(f'Track: {track}')
    logger.info(f'Sim: {sim_path}')

    env = make_env(track, conf)

    # Dreamer agent
    dreamer = Dreamer(device=device)
    model_path = args.model

    if args.resume and os.path.exists(model_path):
        dreamer.load(model_path)
        logger.info(f'Resumed from {model_path}')

    # Replay buffer
    channels = 1 if not cfg.RGB else 3
    buffer = EpisodeBuffer(
        max_steps=cfg.DREAMER_BUFFER_SIZE,
        obs_shape=(channels, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
        action_dim=2
    )

    os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)
    os.makedirs('logs/tb_logs', exist_ok=True)

    # Optional: TensorBoard
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter('logs/tb_logs/dreamer')
    except ImportError:
        writer = None

    total_steps = 0
    best_reward = -float('inf')

    logger.info("=" * 60)
    logger.info(f"DREAMER v3 TRAINING")
    logger.info(f"  Episodes: {args.episodes}")
    logger.info(f"  Model: {model_path}")
    logger.info("=" * 60)

    try:
        for episode in range(args.episodes):
            obs, info = env.reset()
            dreamer.reset_belief()
            last_action = np.zeros(2, dtype=np.float32)

            episode_reward = 0.0
            episode_steps = 0
            t0 = time.time()

            done = False
            while not done:
                # Select action
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)  # (1, C, H, W)
                action_tensor = torch.from_numpy(last_action).unsqueeze(0)

                explore = episode < cfg.DREAMER_SEED_EPISODES
                if explore:
                    # Forward-biased exploration: small random steering,
                    # fixed throttle. Pure uniform [-1,1] creates garbage data.
                    steer = np.random.uniform(-0.5, 0.5)
                    throttle = cfg.DREAMER_THROTTLE_BASE + np.random.uniform(-0.1, 0.1)
                    action = np.array([steer, throttle], dtype=np.float32)
                else:
                    action = dreamer.select_action(
                        obs_tensor, action_tensor, explore=True
                    )

                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Store in buffer
                buffer.add_step(obs, action, reward, done)

                episode_reward += reward
                episode_steps += 1
                total_steps += 1

                obs = next_obs
                last_action = action

            dt = time.time() - t0
            avg_reward = episode_reward / max(episode_steps, 1)
            logger.info(
                f'Episode {episode}: steps={episode_steps}, '
                f'reward={episode_reward:.1f} (avg={avg_reward:.2f}/step), '
                f'buffer={buffer.total_steps}, time={dt:.1f}s'
            )

            if writer:
                writer.add_scalar('episode/reward', episode_reward, episode)
                writer.add_scalar('episode/steps', episode_steps, episode)
                writer.add_scalar('episode/total_steps', total_steps, episode)

            # Train after seed episodes
            if episode >= cfg.DREAMER_SEED_EPISODES:
                logger.info(f'Training for {cfg.DREAMER_GRADIENT_STEPS} steps...')
                t0 = time.time()
                metrics = dreamer.update(buffer)
                dt = time.time() - t0

                if metrics:
                    logger.info(
                        f'Training done in {dt:.1f}s | '
                        f'world={metrics["world_loss"]:.4f} '
                        f'actor={metrics["actor_loss"]:.4f} '
                        f'value={metrics["value_loss"]:.4f} '
                        f'kl={metrics["kl_loss"]:.4f}'
                    )
                    if writer:
                        for k, v in metrics.items():
                            writer.add_scalar(f'train/{k}', v, episode)

            # Save checkpoint
            if episode_reward > best_reward:
                best_reward = episode_reward
                dreamer.save(model_path)
                logger.info(f'New best reward {best_reward:.1f}, saved to {model_path}')

            # Periodic save
            if (episode + 1) % 10 == 0:
                checkpoint_path = model_path.replace(
                    '.pth', f'_ep{episode + 1}.pth'
                )
                dreamer.save(checkpoint_path)

    except KeyboardInterrupt:
        logger.info('Interrupted by user')
        dreamer.save(model_path.replace('.pth', '_interrupted.pth'))

    finally:
        env.close()
        if writer:
            writer.close()

    logger.info(f'Training complete. Best reward: {best_reward:.1f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dreamer v3 training')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of episodes to train')
    parser.add_argument('--model', type=str, default='models/dreamer_v3.pth',
                        help='Model save path')
    parser.add_argument('--track', type=str, default=None,
                        help='Gym env name (default: from myconfig.py)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing model')
    parser.add_argument('--myconfig', type=str, default='myconfig.py')
    args = parser.parse_args()
    train(args)
