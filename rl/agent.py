"""
Donkey Car Vehicle Parts for RL-based driving.

Two pilot implementations:
- SACPilot:     Model-free SAC+VAE (fast to start, needs more episodes)
- DreamerPilot: Model-based Dreamer (learns faster, heavier compute)

Both plug into the donkeycar Vehicle loop via V.add().
Based on L2D paper (arXiv:2008.00715).
"""

import os
import time
import logging
import numpy as np
import torch

from . import config as cfg
from .vae import preprocess_image

logger = logging.getLogger(__name__)


# ─── Shared Utilities ───────────────────────────────────

def is_dead(image):
    """
    Detect if car is off-track by checking bottom strip of image.
    On-track: bottom has bright track markings.
    Off-track: bottom is mostly dark.
    """
    bottom_strip = image[-cfg.DEAD_ZONE_CROP_BOTTOM:, :, :]
    gray = np.mean(bottom_strip, axis=2)
    bright_pixels = np.sum(gray > 100)
    return bright_pixels < cfg.DEAD_ZONE_THRESHOLD


def apply_limits(steering, throttle, prev_steering):
    """Apply action limits: rate-limit steering, scale throttle to [min, max]."""
    steering = float(np.clip(steering, -1.0, 1.0))
    diff = np.clip(steering - prev_steering,
                   -cfg.MAX_STEERING_DIFF, cfg.MAX_STEERING_DIFF)
    steering = np.clip(prev_steering + diff,
                       cfg.STEER_LIMIT_LEFT, cfg.STEER_LIMIT_RIGHT)
    throttle = float(np.clip(throttle, -1.0, 1.0))
    throttle = cfg.THROTTLE_MIN + (throttle + 1) * 0.5 * (cfg.THROTTLE_MAX - cfg.THROTTLE_MIN)
    return steering, throttle


def make_image_tensor(image_array):
    """Preprocess raw camera image to tensor."""
    return preprocess_image(
        image_array,
        crop_top=cfg.IMAGE_CROP_TOP,
        target_size=cfg.IMAGE_SIZE,
        grayscale=not cfg.RGB
    )


# ─── SAC+VAE Pilot ──────────────────────────────────────

class SACPilot:
    """
    Donkey Car part: drives using SAC+VAE reinforcement learning.

    Inputs:  cam/image_array, (optional) speed
    Outputs: pilot/angle, pilot/throttle

    Usage:
        pilot = SACPilot(model_path='models/sac_pilot.pth')
        V.add(pilot, inputs=['cam/image_array'], outputs=['pilot/angle', 'pilot/throttle'])
    """

    def __init__(self, model_path=None, train_mode=True, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.train_mode = train_mode
        self.model_path = model_path or 'models/sac_pilot.pth'

        from .sac import SAC_VAE
        self.agent = SAC_VAE(device=device)

        if os.path.exists(self.model_path):
            logger.info(f'Loading SAC model from {self.model_path}')
            self.agent.load(self.model_path)

        self.cmd_history = np.zeros(cfg.COMMAND_HISTORY_LENGTH * 3, dtype=np.float32)
        self.episode_num = 0
        self.step_num = 0
        self.episode_step = 0
        self.episode_reward = 0.0
        self.last_image = None
        self.last_cmd = None
        self.last_action = None
        self.is_training = False
        self.prev_steering = 0.0

        logger.info(f'SACPilot initialized on {device}, train={train_mode}')

    def run(self, image_array, speed=None):
        if image_array is None:
            return 0.0, 0.0
        if speed is None:
            speed = 0.0

        image_tensor = make_image_tensor(image_array).unsqueeze(0)
        dead = is_dead(image_array) if self.train_mode else False
        cmd_tensor = torch.from_numpy(self.cmd_history).unsqueeze(0)

        if not self.train_mode:
            action = self.agent.select_action(image_tensor, cmd_tensor)
            steering, throttle = apply_limits(action[0], action[1], self.prev_steering)
            self.prev_steering = steering
            self._update_cmd_history(steering, throttle, speed)
            return float(steering), float(throttle)

        self.step_num += 1
        self.episode_step += 1

        # Store transition from previous step
        if self.last_image is not None and self.last_action is not None:
            reward = cfg.REWARD_CRASH if dead else cfg.REWARD_ALIVE
            self.episode_reward += reward
            self.agent.buffer.push(
                self.last_image.squeeze(0), self.last_cmd.squeeze(0),
                torch.from_numpy(self.last_action),
                reward,
                image_tensor.squeeze(0), cmd_tensor.squeeze(0),
                dead
            )

        if dead or self.episode_step >= cfg.MAX_EPISODE_STEPS:
            self._end_episode()
            return 0.0, 0.0

        if self.episode_num < cfg.SAC_RANDOM_EPISODES:
            action = self.agent.random_action()
        else:
            action = self.agent.select_action(image_tensor, cmd_tensor)

        steering, throttle = apply_limits(action[0], action[1], self.prev_steering)
        self.prev_steering = steering

        self.last_image = image_tensor
        self.last_cmd = cmd_tensor
        # Store raw action (pre-limits) so buffer matches actor output space [-1, 1]
        self.last_action = np.array(action, dtype=np.float32)
        self._update_cmd_history(steering, throttle, speed)

        return float(steering), float(throttle)

    def _end_episode(self):
        logger.info(
            f'SAC Episode {self.episode_num}: '
            f'steps={self.episode_step}, reward={self.episode_reward:.1f}, '
            f'buffer={len(self.agent.buffer)}'
        )

        if self.episode_num >= cfg.SAC_RANDOM_EPISODES and not self.is_training:
            self.is_training = True
            logger.info(f'SAC training for {cfg.SAC_GRADIENT_STEPS} steps...')
            t0 = time.time()
            metrics = self.agent.update(gradient_steps=cfg.SAC_GRADIENT_STEPS)
            dt = time.time() - t0
            logger.info(
                f'SAC training done in {dt:.1f}s | '
                f'critic={metrics.get("critic_loss", 0):.4f} '
                f'actor={metrics.get("actor_loss", 0):.4f} '
                f'alpha={metrics.get("alpha", 0):.4f}'
            )
            os.makedirs(os.path.dirname(self.model_path) or '.', exist_ok=True)
            self.agent.save(self.model_path)
            logger.info(f'SAC model saved to {self.model_path}')
            self.is_training = False

        self.episode_num += 1
        self.episode_step = 0
        self.episode_reward = 0.0
        self.last_image = None
        self.last_cmd = None
        self.last_action = None
        self.prev_steering = 0.0
        self.cmd_history = np.zeros(cfg.COMMAND_HISTORY_LENGTH * 3, dtype=np.float32)

    def _update_cmd_history(self, steering, throttle, speed):
        self.cmd_history = np.roll(self.cmd_history, -3)
        self.cmd_history[-3] = steering
        self.cmd_history[-2] = throttle
        self.cmd_history[-1] = speed

    def shutdown(self):
        if self.train_mode and self.episode_step > 0:
            self._end_episode()
        logger.info('SACPilot shutdown')


# ─── Dreamer Pilot ──────────────────────────────────────

class DreamerPilot:
    """
    Donkey Car part: drives using Dreamer model-based RL.

    Learns a world model (RSSM) from images, then plans in imagination.
    Converges faster than SAC (<5 min real interaction) but uses more compute.

    Inputs:  cam/image_array
    Outputs: pilot/angle, pilot/throttle

    Usage:
        pilot = DreamerPilot(model_path='models/dreamer_pilot.pth')
        V.add(pilot, inputs=['cam/image_array'], outputs=['pilot/angle', 'pilot/throttle'])
    """

    def __init__(self, model_path=None, train_mode=True, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.train_mode = train_mode
        self.model_path = model_path or 'models/dreamer_pilot.pth'

        from .dreamer import Dreamer
        from .buffer import EpisodeBuffer

        channels = 1 if not cfg.RGB else 3
        self.agent = Dreamer(device=device)
        self.buffer = EpisodeBuffer(
            max_steps=cfg.DREAMER_BUFFER_SIZE,
            obs_shape=(channels, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
            action_dim=2
        )

        if os.path.exists(self.model_path):
            logger.info(f'Loading Dreamer model from {self.model_path}')
            self.agent.load(self.model_path)

        self.episode_num = 0
        self.step_num = 0
        self.episode_step = 0
        self.episode_reward = 0.0
        self.is_training = False
        self.prev_steering = 0.0

        # Last action for RSSM belief update
        self.last_action = np.zeros(2, dtype=np.float32)
        # Last preprocessed image for buffer storage
        self.last_obs = None

        logger.info(f'DreamerPilot initialized on {device}, train={train_mode}')

    def run(self, image_array):
        if image_array is None:
            return 0.0, 0.0

        obs = make_image_tensor(image_array)            # (C, H, W)
        obs_batch = obs.unsqueeze(0)                    # (1, C, H, W)
        dead = is_dead(image_array) if self.train_mode else False
        action_tensor = torch.from_numpy(self.last_action).unsqueeze(0)  # (1, 2)

        if not self.train_mode:
            raw_action = self.agent.select_action(obs_batch, action_tensor, explore=False)
            steering, throttle = apply_limits(raw_action[0], raw_action[1], self.prev_steering)
            self.prev_steering = steering
            # Store raw action for RSSM belief update (matches training distribution)
            self.last_action = np.array(raw_action, dtype=np.float32)
            return float(steering), float(throttle)

        self.step_num += 1
        self.episode_step += 1

        # Store previous transition
        if self.last_obs is not None:
            reward = cfg.REWARD_CRASH if dead else cfg.REWARD_ALIVE
            self.episode_reward += reward
            self.buffer.add_step(
                self.last_obs, self.last_action, reward, dead
            )

        if dead or self.episode_step >= cfg.MAX_EPISODE_STEPS:
            # Store terminal observation
            if self.last_obs is not None:
                self.buffer.add_step(
                    obs.numpy(), np.zeros(2, dtype=np.float32), 0.0, True
                )
            self._end_episode()
            return 0.0, 0.0

        # Select action
        explore = self.episode_num < cfg.DREAMER_SEED_EPISODES
        if explore:
            raw_action = np.random.uniform(-1, 1, size=2).astype(np.float32)
        else:
            raw_action = self.agent.select_action(
                obs_batch, action_tensor, explore=True
            )

        steering, throttle = apply_limits(raw_action[0], raw_action[1], self.prev_steering)
        self.prev_steering = steering

        self.last_obs = obs.numpy()
        # Store raw action (pre-limits) so buffer matches actor output space [-1, 1]
        self.last_action = np.array(raw_action, dtype=np.float32)

        return float(steering), float(throttle)

    def _end_episode(self):
        logger.info(
            f'Dreamer Episode {self.episode_num}: '
            f'steps={self.episode_step}, reward={self.episode_reward:.1f}, '
            f'buffer_steps={self.buffer.total_steps}, episodes={self.buffer.num_episodes}'
        )

        if self.episode_num >= cfg.DREAMER_SEED_EPISODES and not self.is_training:
            self.is_training = True
            steps = cfg.DREAMER_GRADIENT_STEPS
            logger.info(f'Dreamer training for {steps} steps...')
            t0 = time.time()
            metrics = self.agent.update(self.buffer, gradient_steps=steps)
            dt = time.time() - t0
            if metrics:
                logger.info(
                    f'Dreamer training done in {dt:.1f}s | '
                    f'world={metrics.get("world_loss", 0):.4f} '
                    f'actor={metrics.get("actor_loss", 0):.4f} '
                    f'value={metrics.get("value_loss", 0):.4f} '
                    f'kl={metrics.get("kl_loss", 0):.4f}'
                )
            os.makedirs(os.path.dirname(self.model_path) or '.', exist_ok=True)
            self.agent.save(self.model_path)
            logger.info(f'Dreamer model saved to {self.model_path}')
            self.is_training = False

        self.episode_num += 1
        self.episode_step = 0
        self.episode_reward = 0.0
        self.prev_steering = 0.0
        self.last_action = np.zeros(2, dtype=np.float32)
        self.last_obs = None
        self.agent.reset_belief()

    def shutdown(self):
        if self.train_mode and self.episode_step > 0:
            self._end_episode()
        logger.info('DreamerPilot shutdown')


# ─── Helper Parts ───────────────────────────────────────

class RLTrainToggle:
    """Toggle RL training on/off via web UI button (rising-edge only)."""

    def __init__(self, rl_pilot):
        self.pilot = rl_pilot
        self._prev_button = False

    def run(self, button):
        pressed = bool(button)
        if pressed and not self._prev_button:
            self.pilot.train_mode = not self.pilot.train_mode
            mode = 'TRAIN' if self.pilot.train_mode else 'INFERENCE'
            logger.info(f'RL mode toggled to: {mode}')
        self._prev_button = pressed
