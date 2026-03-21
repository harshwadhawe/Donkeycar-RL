#!/usr/bin/env python3
"""
train_dreamer_v1.py — Dreamer v1 baseline training for Donkey Car simulator.

Faithful port of AaltoVision/donkeycar-dreamer (known to converge on donkeycar).
Uses Gaussian RSSM, 40x40 grayscale input, MSE losses, twin critics.

Usage:
    python train_dreamer_v1.py                                  # new training
    python train_dreamer_v1.py --resume                         # resume
    python train_dreamer_v1.py --episodes=200                   # custom count
    python train_dreamer_v1.py --track=donkey-warehouse-v0      # different track

Monitor:
    tensorboard --logdir ./logs/tb_logs/ --port 6006
"""

import os
import time
import argparse
import logging

import cv2
import numpy as np
import torch
import gymnasium as gym
import gym_donkeycar  # noqa: F401

try:
    gym.spec('donkey-generated-track-v0')
    _USE_SHIMMY = False
except gym.error.NameNotFound:
    import shimmy
    gym.register_envs(shimmy)
    _USE_SHIMMY = True

import donkeycar as dk

from rl.dreamer_v1 import DreamerV1

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
# Image Preprocessing (matches original: crop top 40, 40x40, grayscale, [-0.5, 0.5])
# ==============================================================================
def preprocess_obs(raw_obs):
    """Convert raw 120x160x3 uint8 image to (1, 1, 40, 40) tensor in [-0.5, 0.5]."""
    img = raw_obs[40:, :, :]  # crop top 40 rows
    img = cv2.resize(img, (40, 40))
    gray = np.dot(img, [0.299, 0.587, 0.114])  # luminance
    obs = torch.tensor(gray, dtype=torch.float32).div_(255.).sub_(0.5)
    return obs.unsqueeze(0).unsqueeze(0)  # (1, 1, 40, 40)


# ==============================================================================
# CTE Estimator (reused from train_dreamer.py, for reward shaping)
# ==============================================================================
class CTEEstimator:
    """Estimate CTE from raw camera image when sim doesn't provide it."""

    def __init__(self, smooth_alpha=0.3):
        self.smooth_alpha = smooth_alpha
        self._smooth_cte = 0.0
        self._available = None

    def reset(self):
        self._smooth_cte = 0.0

    def update(self, raw_obs, info):
        """Returns updated CTE. Modifies info dict in place."""
        sim_cte = float(info.get("cte", 0.0))

        if self._available is None and abs(sim_cte) > 1e-6:
            self._available = True

        if self._available:
            return sim_cte

        raw_cte = self._estimate(raw_obs)
        self._smooth_cte = (self.smooth_alpha * raw_cte +
                            (1 - self.smooth_alpha) * self._smooth_cte)
        info["cte"] = self._smooth_cte
        info["cte_source"] = "image"
        return self._smooth_cte

    def _estimate(self, obs):
        h, w = obs.shape[:2]
        strips = [
            obs[h - 20:h - 10, :, :],
            obs[h - 35:h - 25, :, :],
            obs[h - 50:h - 40, :, :],
        ]
        weights = [0.5, 0.3, 0.2]
        total_offset = 0.0
        total_weight = 0.0

        for strip, sw in zip(strips, weights):
            gray = cv2.cvtColor(strip, cv2.COLOR_RGB2GRAY)
            mean_val = np.mean(gray)
            road_mask = gray < mean_val
            cols = np.where(road_mask)
            if len(cols[1]) < 5:
                continue
            w_vals = mean_val - gray[road_mask]
            centroid = np.average(cols[1], weights=w_vals)
            offset = (centroid - w / 2.0) / (w / 2.0)
            total_offset += offset * sw
            total_weight += sw

        if total_weight < 0.1:
            return 0.0
        return float((total_offset / total_weight) * 3.0)


# ==============================================================================
# Reward Shaping (simple, matching reference repo style)
# ==============================================================================
def compute_reward(info, speed, cte, terminated, last_steering, steering):
    """Simple reward: speed-proportional + CTE bell curve + crash penalty."""
    # Speed reward (reference: reward = 1 + normalized_speed)
    speed_reward = 1.0 + max(speed, 0.0)

    # CTE bell curve
    cte_reward = 2.0 * np.exp(-(cte / 1.0) ** 2)

    # Steering smoothness
    steer_penalty = 0.1 * (steering - last_steering) ** 2

    reward = speed_reward + cte_reward - steer_penalty

    if terminated:
        reward = -10.0

    return reward


# ==============================================================================
# Training Loop
# ==============================================================================
def train(args):
    dk_cfg = dk.load_config(myconfig=args.myconfig)
    device = get_device()
    logger.info(f'Device: {device}')

    track = args.track or getattr(
        dk_cfg, 'DONKEY_GYM_ENV_NAME', 'donkey-generated-track-v0')
    sim_path = getattr(dk_cfg, 'DONKEY_SIM_PATH', 'manual')

    conf = {
        "exe_path": sim_path,
        "host": getattr(dk_cfg, 'SIM_HOST', '127.0.0.1'),
        "port": 9091,
        "frame_skip": 1,      # reference uses action_repeat=1
        "cam_resolution": (
            getattr(dk_cfg, 'IMAGE_H', 120),
            getattr(dk_cfg, 'IMAGE_W', 160),
            getattr(dk_cfg, 'IMAGE_DEPTH', 3),
        ),
        "log_level": 20,
        "throttle_max": 1.0,
        "max_cte": 4.0,       # reference uses max_cte=4
        "start_delay": 5.0,
    }

    logger.info(f'Track: {track}')
    logger.info(f'Sim: {sim_path}')

    # Create env
    if _USE_SHIMMY:
        env = gym.make(
            "GymV21Environment-v0",
            env_id=track,
            make_kwargs={"conf": conf},
        )
    else:
        env = gym.make(track, conf=conf)

    # Agent
    agent = DreamerV1(
        device=device,
        fix_speed=True,
        throttle_base=args.throttle_base,
        seed_episodes=args.seed_episodes,
        gradient_steps=args.gradient_steps,
        max_episode_steps=args.max_steps,
    )

    model_path = args.model
    if args.resume and os.path.exists(model_path):
        agent.load(model_path)
        logger.info(f'Resumed from {model_path}')

    os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)
    os.makedirs('logs/tb_logs', exist_ok=True)

    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter('logs/tb_logs/dreamer_v1')
    except ImportError:
        writer = None

    cte_estimator = CTEEstimator()
    total_steps = 0
    best_reward = -float('inf')

    logger.info("=" * 60)
    logger.info("DREAMER v1 BASELINE TRAINING")
    logger.info(f"  Episodes: {args.episodes}")
    logger.info(f"  Seed episodes: {args.seed_episodes}")
    logger.info(f"  Gradient steps: {args.gradient_steps}")
    logger.info(f"  Throttle: {args.throttle_base}")
    logger.info(f"  Model: {model_path}")
    logger.info("=" * 60)

    try:
        for episode in range(args.episodes):
            raw_obs, info = env.reset()
            cte_estimator.reset()
            agent.reset_belief()

            last_action = np.zeros(2, dtype=np.float32)
            last_action_tensor = torch.zeros(1, 2, device=device)

            episode_reward = 0.0
            episode_steps = 0
            last_steering = 0.0
            t0 = time.time()

            done = False
            while not done and episode_steps < args.max_steps:
                # Preprocess for the model
                obs_tensor = preprocess_obs(raw_obs)  # (1, 1, 40, 40)

                is_seed = episode < args.seed_episodes
                if is_seed:
                    # Random exploration (forward-biased)
                    steer = np.random.uniform(-0.5, 0.5)
                    action = np.array(
                        [steer, args.throttle_base], dtype=np.float32)
                else:
                    action = agent.select_action(
                        obs_tensor, last_action_tensor, explore=True)

                # Step environment
                next_raw_obs, sim_reward, terminated, truncated, info = \
                    env.step(action)
                done = terminated or truncated

                # CTE estimation
                speed = float(info.get("speed", 0.0))
                cte = cte_estimator.update(next_raw_obs, info)

                # Reward shaping
                reward = compute_reward(
                    info, speed, cte, terminated,
                    last_steering, float(action[0]))

                # Store in replay buffer (preprocessed observation)
                next_obs_tensor = preprocess_obs(next_raw_obs)
                agent.D.append(
                    next_obs_tensor.squeeze(0),  # (1, 40, 40)
                    action,
                    reward,
                    done,
                )

                episode_reward += reward
                episode_steps += 1
                total_steps += 1

                raw_obs = next_raw_obs
                last_action = action
                last_action_tensor = torch.tensor(
                    action, dtype=torch.float32, device=device).unsqueeze(0)
                last_steering = float(action[0])

            dt = time.time() - t0
            avg_reward = episode_reward / max(episode_steps, 1)
            logger.info(
                f'Episode {episode}: steps={episode_steps}, '
                f'reward={episode_reward:.1f} (avg={avg_reward:.2f}/step), '
                f'buffer={agent.D.steps}, time={dt:.1f}s'
            )

            if writer:
                writer.add_scalar('episode/reward', episode_reward, episode)
                writer.add_scalar('episode/steps', episode_steps, episode)
                writer.add_scalar('episode/total_steps', total_steps, episode)

            # Train after seed episodes
            if episode >= args.seed_episodes:
                logger.info(
                    f'Training for {args.gradient_steps} gradient steps...')
                t0 = time.time()
                metrics = agent.update_parameters()
                dt = time.time() - t0

                if metrics:
                    logger.info(
                        f'Training done in {dt:.1f}s | '
                        f'obs={metrics["obs_loss"]:.2f} '
                        f'rew={metrics["reward_loss"]:.2f} '
                        f'kl={metrics["kl_loss"]:.2f} '
                        f'actor={metrics["actor_loss"]:.2f} '
                        f'value={metrics["value_loss"]:.2f}'
                    )
                    if writer:
                        for k, v in metrics.items():
                            writer.add_scalar(f'train/{k}', v, episode)

            # Save checkpoint
            if episode_reward > best_reward:
                best_reward = episode_reward
                agent.save(model_path)
                logger.info(
                    f'New best reward {best_reward:.1f}, saved to {model_path}')

            if (episode + 1) % 10 == 0:
                checkpoint = model_path.replace(
                    '.pth', f'_ep{episode + 1}.pth')
                agent.save(checkpoint)

    except KeyboardInterrupt:
        logger.info('Interrupted by user')
        agent.save(model_path.replace('.pth', '_interrupted.pth'))

    finally:
        env.close()
        if writer:
            writer.close()

    logger.info(f'Training complete. Best reward: {best_reward:.1f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dreamer v1 baseline')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--model', type=str, default='models/dreamer_v1.pth')
    parser.add_argument('--track', type=str, default=None)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--myconfig', type=str, default='myconfig.py')
    parser.add_argument('--seed-episodes', type=int, default=5,
                        help='Random exploration episodes before training')
    parser.add_argument('--gradient-steps', type=int, default=100,
                        help='World model updates per data collection')
    parser.add_argument('--throttle-base', type=float, default=0.35,
                        help='Fixed throttle value')
    parser.add_argument('--max-steps', type=int, default=1000,
                        help='Max steps per episode')
    args = parser.parse_args()
    train(args)
