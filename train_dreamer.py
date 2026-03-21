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
# Image-Based CTE Estimator
# ==============================================================================
class CTEEstimatorWrapper(gym.Wrapper):
    """
    Estimate cross-track error from camera image when the sim doesn't provide it.

    Looks at the bottom strip of the raw image (road area visible to the car).
    The road is darker than the grass/shoulder. We find the road centroid and
    compute its offset from image center → estimated CTE.

    Must be placed BEFORE DonkeyPreprocessWrapper (needs raw 120x160x3 image).
    """

    def __init__(self, env, bottom_rows=30, smooth_alpha=0.3):
        super().__init__(env)
        self.bottom_rows = bottom_rows
        self.smooth_alpha = smooth_alpha  # EMA smoothing (lower = smoother)
        self._cte_available = None  # auto-detect on first step
        self._smooth_cte = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._smooth_cte = 0.0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        sim_cte = float(info.get("cte", 0.0))

        # Auto-detect: if sim provides CTE, don't override
        if self._cte_available is None and abs(sim_cte) > 1e-6:
            self._cte_available = True

        if self._cte_available:
            return obs, reward, terminated, truncated, info

        # Estimate CTE from image with temporal smoothing
        raw_cte = self._estimate_cte(obs)
        self._smooth_cte = (self.smooth_alpha * raw_cte +
                            (1 - self.smooth_alpha) * self._smooth_cte)
        info["cte"] = self._smooth_cte
        info["cte_source"] = "image"

        return obs, reward, terminated, truncated, info

    def _estimate_cte(self, obs):
        """
        Estimate CTE from raw camera image (H, W, 3) uint8.

        Uses multiple bottom strips at different heights for robustness.
        Road is darker than grass — find road centroid vs image center.
        """
        h, w = obs.shape[:2]

        # Sample 3 strips at different heights for robustness
        strips = [
            obs[h - 20:h - 10, :, :],  # near (most reliable)
            obs[h - 35:h - 25, :, :],  # mid
            obs[h - 50:h - 40, :, :],  # far
        ]
        strip_weights = [0.5, 0.3, 0.2]  # trust near strips more

        total_offset = 0.0
        total_weight = 0.0

        for strip, sw in zip(strips, strip_weights):
            gray = cv2.cvtColor(strip, cv2.COLOR_RGB2GRAY)

            # Adaptive threshold: road is darker than the mean
            mean_val = np.mean(gray)
            road_mask = gray < mean_val

            road_cols = np.where(road_mask)
            if len(road_cols[1]) < 5:
                continue

            # Weighted centroid (darker = more confident it's road)
            weights = mean_val - gray[road_mask]
            centroid = np.average(road_cols[1], weights=weights)
            center = w / 2.0

            offset = (centroid - center) / center  # [-1, 1]
            total_offset += offset * sw
            total_weight += sw

        if total_weight < 0.1:
            return 0.0

        # Scale to CTE range [-3, 3]
        cte = (total_offset / total_weight) * 3.0
        return float(cte)


# ==============================================================================
# Reward Shaping Wrapper
# ==============================================================================
class RewardShapingWrapper(gym.Wrapper):
    """
    Reward shaping for Dreamer: anti-circling using ONLY working telemetry.

    Working fields:  speed, car (roll/pitch/yaw), gyro, accel, hit
    BROKEN fields:   cte=0, forward_vel=0, pos=(0,0,0), vel=(0,0,0)

    Anti-circling strategy:
    - Dead-reckon position from speed + yaw → compute real displacement
    - Penalize high yaw rate (gyro) → directly detects spinning
    - Penalize high absolute steering → taxes circle commands
    - CTE from CTEEstimatorWrapper (image-based, placed before this wrapper)
    """

    def __init__(self, env, max_episode_steps=None):
        super().__init__(env)
        self.episode_step = 0
        self.max_episode_steps = max_episode_steps or cfg.DREAMER_MAX_EPISODE_STEPS
        self.last_steering = 0.0
        self.stuck_count = 0
        self.stuck_threshold = 10
        # Dead reckoning state
        self._dr_x = 0.0
        self._dr_z = 0.0
        self._last_yaw = None
        self._cumulative_yaw = 0.0  # total absolute heading change

    def reset(self, **kwargs):
        self.episode_step = 0
        self.last_steering = 0.0
        self.stuck_count = 0
        self._dr_x = 0.0
        self._dr_z = 0.0
        self._last_yaw = None
        self._cumulative_yaw = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, _sim_reward, terminated, truncated, info = self.env.step(action)
        self.episode_step += 1

        speed = float(info.get("speed", 0.0))
        cte = float(info.get("cte", 0.0))  # from CTEEstimatorWrapper (image-based)
        steering = float(np.asarray(action)[0]) if hasattr(action, '__len__') else 0.0

        # Dead-reckon position from speed + yaw
        car = info.get("car", (0, 0, 0))
        yaw_deg = float(car[1])  # car = (roll, pitch, yaw) in degrees
        yaw_rad = np.radians(yaw_deg)

        if self._last_yaw is not None:
            # Track cumulative heading change (detects circling)
            dyaw = yaw_deg - self._last_yaw
            # Normalize to [-180, 180]
            if dyaw > 180:
                dyaw -= 360
            elif dyaw < -180:
                dyaw += 360
            self._cumulative_yaw += abs(dyaw)
        self._last_yaw = yaw_deg

        # Dead reckon: integrate speed along heading direction
        # dt ~ 1 step, speed is in sim units per step
        self._dr_x += speed * np.sin(yaw_rad) * 0.05  # ~20Hz
        self._dr_z += speed * np.cos(yaw_rad) * 0.05
        displacement = (self._dr_x**2 + self._dr_z**2) ** 0.5

        # Inject dead-reckoned values into info for logging
        info["dr_displacement"] = displacement
        info["cumulative_yaw"] = self._cumulative_yaw

        # ── Reward components ──

        # Speed reward (only have magnitude, but yaw penalty handles circling)
        speed_reward = 1.0 + max(speed, 0.0)

        # CTE bell curve (from image-based estimator)
        cte_reward = 2.0 * np.exp(-(cte / 1.0) ** 2)

        # Steering smoothness
        steer_change_penalty = 0.2 * (steering - self.last_steering) ** 2

        # Steering magnitude penalty (circles need constant high steering)
        steer_mag_penalty = 0.5 * steering ** 2

        # Yaw rate penalty: penalize high turning rate from gyro
        # gyro = (gx, gy, gz) — gz is yaw rate in rad/s
        gyro = info.get("gyro", (0, 0, 0))
        yaw_rate = abs(float(gyro[1]))  # yaw component
        yaw_rate_penalty = 0.3 * min(yaw_rate, 2.0)  # cap to avoid dominating

        reward = (speed_reward + cte_reward
                  - steer_change_penalty - steer_mag_penalty
                  - yaw_rate_penalty)
        self.last_steering = steering

        # Standstill detection
        if speed < 0.1:
            reward -= 0.5
            self.stuck_count += 1
        else:
            self.stuck_count = 0

        if self.stuck_count >= self.stuck_threshold:
            terminated = True

        if abs(cte) > 2.5:
            terminated = True

        if terminated or truncated:
            reward -= 10.0

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
    # CTE estimator MUST come before preprocessing (needs raw image)
    env = CTEEstimatorWrapper(env)
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
            episode_speeds = []
            episode_ctes = []
            episode_steers = []
            episode_yaw_rates = []
            t0 = time.time()

            is_seed = episode < cfg.DREAMER_SEED_EPISODES

            done = False
            while not done:
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                action_tensor = torch.from_numpy(last_action).unsqueeze(0)

                if is_seed:
                    steer = np.random.uniform(-0.5, 0.5)
                    action = np.array([steer, cfg.DREAMER_THROTTLE_BASE],
                                      dtype=np.float32)
                else:
                    action = dreamer.select_action(
                        obs_tensor, action_tensor, explore=True
                    )

                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                step_speed = float(info.get("speed", 0.0))
                step_cte = float(info.get("cte", 0.0))
                episode_speeds.append(step_speed)
                episode_ctes.append(step_cte)
                episode_steers.append(abs(float(np.asarray(action)[0])))
                gyro = info.get("gyro", (0, 0, 0))
                episode_yaw_rates.append(abs(float(gyro[1])))

                # Diagnostic: first episode
                if episode == 0 and episode_steps == 1:
                    logger.info(f'FIRST STEP INFO KEYS: {sorted(info.keys())}')
                    logger.info(
                        f'FIRST STEP INFO: cte={info.get("cte")}, '
                        f'speed={info.get("speed")}, '
                        f'forward_vel={info.get("forward_vel")}, '
                        f'pos={info.get("pos")}, '
                        f'car={info.get("car")}, '
                        f'gyro={info.get("gyro")}')

                buffer.add_step(obs, action, reward, done)

                episode_reward += reward
                episode_steps += 1
                total_steps += 1

                obs = next_obs
                last_action = action

            dt = time.time() - t0
            avg_reward = episode_reward / max(episode_steps, 1)
            avg_speed = np.mean(episode_speeds) if episode_speeds else 0.0
            avg_cte = np.mean(np.abs(episode_ctes)) if episode_ctes else 0.0
            avg_steer = np.mean(episode_steers) if episode_steers else 0.0
            avg_yaw_rate = np.mean(episode_yaw_rates) if episode_yaw_rates else 0.0
            dr_disp = float(info.get("dr_displacement", 0.0))
            cum_yaw = float(info.get("cumulative_yaw", 0.0))
            logger.info(
                f'Episode {episode}: steps={episode_steps}, '
                f'reward={episode_reward:.1f} (avg={avg_reward:.2f}/step), '
                f'speed={avg_speed:.2f}, |steer|={avg_steer:.2f}, '
                f'|cte|={avg_cte:.2f}, yaw_rate={avg_yaw_rate:.3f}, '
                f'dr_disp={dr_disp:.1f}m, cum_yaw={cum_yaw:.0f}deg, '
                f'buffer={buffer.total_steps}, '
                f'{"seed" if is_seed else "policy"}, '
                f'time={dt:.1f}s'
            )

            if writer:
                writer.add_scalar('episode/reward', episode_reward, episode)
                writer.add_scalar('episode/steps', episode_steps, episode)
                writer.add_scalar('episode/total_steps', total_steps, episode)
                writer.add_scalar('episode/avg_speed', avg_speed, episode)
                writer.add_scalar('episode/avg_abs_steer', avg_steer, episode)
                writer.add_scalar('episode/avg_abs_cte', avg_cte, episode)
                writer.add_scalar('episode/avg_yaw_rate', avg_yaw_rate, episode)
                writer.add_scalar('episode/dr_displacement', dr_disp, episode)
                writer.add_scalar('episode/cumulative_yaw', cum_yaw, episode)

            # Train after seed episodes
            if episode >= cfg.DREAMER_SEED_EPISODES:
                gradient_steps = max(1, int(
                    episode_steps * cfg.DREAMER_TRAIN_RATIO /
                    (cfg.DREAMER_BATCH_SIZE * cfg.DREAMER_CHUNK_SIZE)
                ))
                logger.info(f'Training for {gradient_steps} steps '
                            f'(episode_steps={episode_steps}, '
                            f'ratio={cfg.DREAMER_TRAIN_RATIO})...')
                t0 = time.time()
                metrics = dreamer.update(buffer, gradient_steps=gradient_steps)
                dt = time.time() - t0

                if metrics:
                    logger.info(
                        f'Training done in {dt:.1f}s | '
                        f'world={metrics["world_loss"]:.4f} '
                        f'obs={metrics["obs_loss"]:.4f} '
                        f'rew={metrics["reward_loss"]:.4f} '
                        f'actor={metrics["actor_loss"]:.4f} '
                        f'value={metrics["value_loss"]:.4f} '
                        f'kl={metrics["kl_loss"]:.4f}'
                    )
                    if writer:
                        for k, v in metrics.items():
                            writer.add_scalar(f'train/{k}', v, episode)

            # ── Test Episode (deterministic, every 10 eps) ──
            if (episode >= cfg.DREAMER_SEED_EPISODES and
                    (episode + 1) % 10 == 0):
                test_obs, test_info = env.reset()
                dreamer.reset_belief()
                test_action = np.zeros(2, dtype=np.float32)
                test_reward = 0.0
                test_steps = 0
                test_done = False
                test_steers = []

                with torch.no_grad():
                    while not test_done and test_steps < cfg.DREAMER_MAX_EPISODE_STEPS:
                        test_obs_t = torch.from_numpy(test_obs).unsqueeze(0)
                        test_act_t = torch.from_numpy(test_action).unsqueeze(0)
                        test_action = dreamer.select_action(
                            test_obs_t, test_act_t, explore=False)
                        test_obs, r, term, trunc, test_info = env.step(test_action)
                        test_done = term or trunc
                        test_reward += r
                        test_steps += 1
                        test_steers.append(abs(float(test_action[0])))

                test_avg = test_reward / max(test_steps, 1)
                test_avg_steer = np.mean(test_steers) if test_steers else 0
                test_dr_disp = float(test_info.get("dr_displacement", 0.0))
                test_cum_yaw = float(test_info.get("cumulative_yaw", 0.0))
                logger.info(
                    f'  TEST ep {episode}: steps={test_steps}, '
                    f'reward={test_reward:.1f} (avg={test_avg:.2f}/step), '
                    f'dr_disp={test_dr_disp:.1f}m, cum_yaw={test_cum_yaw:.0f}deg, '
                    f'|steer|={test_avg_steer:.2f}')
                if writer:
                    writer.add_scalar('test/reward', test_reward, episode)
                    writer.add_scalar('test/steps', test_steps, episode)
                    writer.add_scalar('test/dr_displacement', test_dr_disp, episode)
                    writer.add_scalar('test/cumulative_yaw', test_cum_yaw, episode)

            # Save checkpoint
            if episode_reward > best_reward:
                best_reward = episode_reward
                dreamer.save(model_path)
                logger.info(f'New best reward {best_reward:.1f}, saved to {model_path}')

            if (episode + 1) % 20 == 0:
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
    parser.add_argument('--episodes', type=int, default=500,
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
