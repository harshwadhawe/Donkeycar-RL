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

from helpers.dreamer import Dreamer
from helpers.buffer import EpisodeBuffer
from helpers import config as cfg

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

    def __init__(self, env, crop_top=40, target_size=cfg.IMAGE_SIZE, grayscale=True):
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
    """

    def __init__(self, env, bottom_rows=30, smooth_alpha=0.3):
        super().__init__(env)
        self.bottom_rows = bottom_rows
        self.smooth_alpha = smooth_alpha  
        self._cte_available = None  
        self._smooth_cte = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._smooth_cte = 0.0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        sim_cte = float(info.get("cte", 0.0))

        if self._cte_available is None and abs(sim_cte) > 1e-6:
            self._cte_available = True

        if self._cte_available:
            return obs, reward, terminated, truncated, info

        raw_cte = self._estimate_cte(obs)
        self._smooth_cte = (self.smooth_alpha * raw_cte +
                            (1 - self.smooth_alpha) * self._smooth_cte)
        info["cte"] = self._smooth_cte
        info["cte_source"] = "image"

        return obs, reward, terminated, truncated, info

    def _estimate_cte(self, obs):
        h, w = obs.shape[:2]
        strips = [
            obs[h - 20:h - 10, :, :],  
            obs[h - 35:h - 25, :, :],  
            obs[h - 50:h - 40, :, :],  
        ]
        strip_weights = [0.5, 0.3, 0.2]  

        total_offset = 0.0
        total_weight = 0.0

        for strip, sw in zip(strips, strip_weights):
            gray = cv2.cvtColor(strip, cv2.COLOR_RGB2GRAY)
            mean_val = np.mean(gray)
            road_mask = gray < mean_val
            road_cols = np.where(road_mask)
            if len(road_cols[1]) < 5:
                continue

            weights = mean_val - gray[road_mask]
            centroid = np.average(road_cols[1], weights=weights)
            center = w / 2.0

            offset = (centroid - center) / center  
            total_offset += offset * sw
            total_weight += sw

        if total_weight < 0.1:
            return 0.0

        cte = (total_offset / total_weight) * 3.0
        return float(cte)


# ==============================================================================
# Reward Shaping Wrapper (Pure Progression + Calibrated Breadcrumbs)
# ==============================================================================
class RewardShapingWrapper(gym.Wrapper):
    """
    Rewards ONLY true forward progression down the track.
    Uses a 10Hz calibrated breadcrumb trail to instantly kill donuts.
    """

    def __init__(self, env, max_episode_steps=None):
        super().__init__(env)
        self.episode_step = 0
        self.max_episode_steps = max_episode_steps or cfg.DREAMER_MAX_EPISODE_STEPS
        
        self.max_dist_from_start = 0.0
        self.start_pos = None
        self.breadcrumbs = []
        self.stuck_count = 0
        self.stuck_threshold = 15  # 1.5 seconds at 10Hz

    def reset(self, **kwargs):
        self.episode_step = 0
        self.max_dist_from_start = 0.0
        self.start_pos = None
        self.breadcrumbs = []
        self.stuck_count = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, _sim_reward, terminated, truncated, info = self.env.step(action)
        self.episode_step += 1

        cte = float(info.get("cte", 0.0))
        speed = float(info.get("speed", 0.0))
        cur_pos = info.get("pos", None)
        
        reward = 0.0

        # 1. Base Reward: Stay on the track (mild bell curve)
        cte_reward = 2.0 * np.exp(-(cte / 1.0) ** 2)
        reward += cte_reward

        # 2. True Progression Reward
        if cur_pos is not None:
            if self.start_pos is None:
                self.start_pos = cur_pos
            
            dist_from_start = ((cur_pos[0]-self.start_pos[0])**2 + (cur_pos[2]-self.start_pos[2])**2)**0.5
            
            if dist_from_start > self.max_dist_from_start:
                progress = dist_from_start - self.max_dist_from_start
                reward += (progress * 10.0) 
                self.max_dist_from_start = dist_from_start

            # 3. Calibrated Breadcrumb Anti-Looping (10Hz assumption)
            if len(cur_pos) >= 3:
                x, z = cur_pos[0], cur_pos[2]
                
                # Drop a breadcrumb every 5 steps (0.5 seconds)
                if self.episode_step % 5 == 0:
                    self.breadcrumbs.append((self.episode_step, x, z))
                
                # Check for path crossings from 2 to 7 seconds ago (20 to 70 steps)
                # This ignores start-line crossings during lap completions (>120 steps)
                for step_num, bx, bz in self.breadcrumbs:
                    age_steps = self.episode_step - step_num
                    
                    if 20 <= age_steps <= 70:
                        dist = ((x - bx)**2 + (z - bz)**2)**0.5
                        if dist < 1.5:
                            terminated = True
                            info["circle_terminated"] = True
                            break

        # Standstill detection
        if speed < 0.1 and self.episode_step > 10:
            reward -= 0.5
            self.stuck_count += 1
        else:
            self.stuck_count = 0

        if self.stuck_count >= self.stuck_threshold:
            terminated = True

        # Track bounds detection
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
    track_short = track.replace('donkey-', '').replace('_', '-')

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
        "max_cte": 3.0,        
        "start_delay": 5.0,    
    }

    logger.info(f'Track: {track}')
    logger.info(f'Sim: {sim_path}')

    env = make_env(track, conf)

    dreamer = Dreamer(device=device)
    if args.model == 'models/dreamer_v3.pth':
        model_path = f'models/{track_short}/dreamer.pth'
    else:
        model_path = args.model

    if args.resume and os.path.exists(model_path):
        dreamer.load(model_path)
        logger.info(f'Resumed from {model_path}')

    channels = 1 if not cfg.RGB else 3
    buffer = EpisodeBuffer(
        max_steps=cfg.DREAMER_BUFFER_SIZE,
        obs_shape=(channels, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
        action_dim=2
    )

    os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)
    os.makedirs('logs/tb_logs', exist_ok=True)

    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(f'logs/tb_logs/dreamer_{track_short}')
    except ImportError:
        writer = None

    total_steps = 0
    best_reward = -float('inf')
    best_test_score = -float('inf')  

    real_lap_count = 0
    fake_lap_count = 0

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
            episode_fwd_vels = []
            episode_ctes = []
            episode_steers = []  
            start_pos = None
            max_displacement = 0.0
            ep_real_laps = 0
            ep_fake_laps = 0
            last_lap_time = None  
            lap_window_ctes = []
            lap_window_fwd_vels = []
            lap_window_max_disp = 0.0
            lap_window_start_pos = None
            positions = []  
            t0 = time.time()

            is_seed = episode < cfg.DREAMER_SEED_EPISODES

            done = False
            while not done:
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                action_tensor = torch.from_numpy(last_action).unsqueeze(0)

                if is_seed:
                    seed_cte = float(info.get("cte", 0.0))
                    steer = np.clip(-0.7 * seed_cte, -1.0, 1.0)
                    steer += np.random.normal(0, 0.15)
                    steer = float(np.clip(steer, -1.0, 1.0))
                    action = np.array([steer, cfg.DREAMER_THROTTLE_BASE],
                                      dtype=np.float32)
                else:
                    action = dreamer.select_action(
                        obs_tensor, action_tensor, explore=True
                    )

                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                fwd_vel = float(info.get("forward_vel", 0.0))
                step_cte = float(info.get("cte", 0.0))
                steer_val = float(np.asarray(action)[0])
                episode_fwd_vels.append(fwd_vel)
                episode_ctes.append(step_cte)
                episode_steers.append(steer_val)

                cur_pos = info.get("pos", (0, 0, 0))

                if start_pos is None:
                    start_pos = cur_pos
                dx = cur_pos[0] - start_pos[0]
                dz = cur_pos[2] - start_pos[2]
                max_displacement = max(max_displacement, (dx**2 + dz**2)**0.5)

                lap_window_ctes.append(abs(step_cte))
                lap_window_fwd_vels.append(fwd_vel)
                if lap_window_start_pos is None:
                    lap_window_start_pos = cur_pos
                ldx = cur_pos[0] - lap_window_start_pos[0]
                ldz = cur_pos[2] - lap_window_start_pos[2]
                lap_window_max_disp = max(lap_window_max_disp,
                                          (ldx**2 + ldz**2)**0.5)

                cur_lap_time = info.get("last_lap_time", None)
                if cur_lap_time is not None:
                    cur_lap_time = float(cur_lap_time)
                    # Filter out fake 0.0s lap times that trigger at spawn
                    if cur_lap_time > 5.0 and (last_lap_time is None or abs(cur_lap_time - last_lap_time) > 0.1):
                        avg_lap_cte = (np.mean(lap_window_ctes) if lap_window_ctes else 99.0)
                        avg_lap_fwd = (np.mean(lap_window_fwd_vels) if lap_window_fwd_vels else 0.0)
                        lap_disp = lap_window_max_disp

                        is_real = (avg_lap_cte < 2.0 and lap_disp > 5.0 and avg_lap_fwd > 0.3)

                        if is_real:
                            ep_real_laps += 1
                            real_lap_count += 1
                            logger.info(
                                f'  REAL LAP {cur_lap_time:.1f}s '
                                f'(|cte|={avg_lap_cte:.2f}, '
                                f'disp={lap_disp:.1f}m, '
                                f'fwd={avg_lap_fwd:.2f}) '
                                f'ep {episode} step {episode_steps}')
                        elif cur_lap_time > 0.5:
                            ep_fake_laps += 1
                            fake_lap_count += 1

                        last_lap_time = cur_lap_time
                        lap_window_ctes = []
                        lap_window_fwd_vels = []
                        lap_window_max_disp = 0.0
                        lap_window_start_pos = cur_pos

                buffer.add_step(obs, action, reward, done)

                episode_reward += reward
                episode_steps += 1
                total_steps += 1

                obs = next_obs
                last_action = action

            dt = time.time() - t0
            avg_reward = episode_reward / max(episode_steps, 1)
            avg_fwd = np.mean(episode_fwd_vels) if episode_fwd_vels else 0.0
            avg_cte = np.mean(np.abs(episode_ctes)) if episode_ctes else 0.0
            avg_steer = np.mean(np.abs(episode_steers)) if episode_steers else 0.0

            steers_arr = np.array(episode_steers) if episode_steers else np.zeros(1)
            left_pct = np.mean(steers_arr < -0.05) * 100  
            right_pct = np.mean(steers_arr > 0.05) * 100   
            steer_bias = max(left_pct, right_pct)           
            mean_steer = np.mean(steers_arr)                
            
            end_disp = 0.0
            if start_pos and cur_pos:
                end_disp = ((cur_pos[0]-start_pos[0])**2 +
                            (cur_pos[2]-start_pos[2])**2)**0.5

            lap_str = ''
            if ep_real_laps or ep_fake_laps:
                lap_str = f', laps={ep_real_laps}real/{ep_fake_laps}fake'
            circle_str = ''
            if info.get("circle_terminated"):
                circle_str = ' [CIRCLE KILLED]'

            logger.info(
                f'Episode {episode}: steps={episode_steps}, '
                f'reward={episode_reward:.1f} (avg={avg_reward:.2f}/step), '
                f'fwd_vel={avg_fwd:.2f}, |steer|={avg_steer:.2f}, '
                f'|cte|={avg_cte:.2f}, disp={max_displacement:.1f}m, '
                f'end_disp={end_disp:.1f}m, '
                f'steer_bias={steer_bias:.0f}%{"L" if mean_steer < 0 else "R"} '
                f'buffer={buffer.total_steps}, '
                f'{"seed" if is_seed else "policy"}'
                f'{lap_str}{circle_str}, '
                f'time={dt:.1f}s'
            )

            if writer:
                writer.add_scalar('episode/reward', episode_reward, episode)
                writer.add_scalar('episode/steps', episode_steps, episode)
                writer.add_scalar('episode/avg_fwd_vel', avg_fwd, episode)
                writer.add_scalar('episode/avg_abs_steer', avg_steer, episode)
                writer.add_scalar('episode/avg_abs_cte', avg_cte, episode)
                writer.add_scalar('episode/max_displacement',
                                  max_displacement, episode)

            if episode == cfg.DREAMER_SEED_EPISODES:
                warmup_steps = 100
                logger.info(f'World model warmup: {warmup_steps} steps '
                            f'(buffer={buffer.total_steps})...')
                t0_warmup = time.time()
                warmup_metrics = dreamer.update(
                    buffer, gradient_steps=warmup_steps, world_only=True
                )
                dt_warmup = time.time() - t0_warmup
                if warmup_metrics:
                    logger.info(
                        f'Warmup done in {dt_warmup:.1f}s | '
                        f'world={warmup_metrics.get("world_loss", 0):.4f} '
                        f'obs={warmup_metrics.get("obs_loss", 0):.4f} '
                        f'rew={warmup_metrics.get("reward_loss", 0):.4f} '
                        f'kl={warmup_metrics.get("kl_loss", 0):.4f}'
                    )

            if episode >= cfg.DREAMER_SEED_EPISODES:
                gradient_steps = max(1, int(
                    episode_steps * cfg.DREAMER_TRAIN_RATIO /
                    (cfg.DREAMER_BATCH_SIZE * cfg.DREAMER_CHUNK_SIZE)
                ))
                t0 = time.time()
                metrics = dreamer.update(buffer, gradient_steps=gradient_steps)
                dt = time.time() - t0

                if metrics and writer:
                    for k, v in metrics.items():
                        writer.add_scalar(f'train/{k}', v, episode)

            # ── Test Episode ──
            if (episode >= cfg.DREAMER_SEED_EPISODES and
                    (episode + 1) % 10 == 0):
                test_obs, test_info = env.reset()
                dreamer.reset_belief()
                test_action = np.zeros(2, dtype=np.float32)
                test_reward = 0.0
                test_steps = 0
                test_done = False
                test_start = None
                test_max_disp = 0.0
                test_real_laps = 0
                test_fake_laps = 0
                test_last_lap = None
                test_lap_ctes = []
                test_lap_fwds = []
                test_lap_max_disp_w = 0.0
                test_lap_start = None

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

                        tp = test_info.get("pos", (0, 0, 0))
                        if test_start is None:
                            test_start = tp
                        td = ((tp[0]-test_start[0])**2 +
                              (tp[2]-test_start[2])**2)**0.5
                        test_max_disp = max(test_max_disp, td)

                        t_cte = abs(float(test_info.get("cte", 0.0)))
                        t_fwd = float(test_info.get("forward_vel", 0.0))
                        test_lap_ctes.append(t_cte)
                        test_lap_fwds.append(t_fwd)
                        if test_lap_start is None:
                            test_lap_start = tp
                        tld = ((tp[0]-test_lap_start[0])**2 +
                               (tp[2]-test_lap_start[2])**2)**0.5
                        test_lap_max_disp_w = max(test_lap_max_disp_w, tld)

                        tlt = test_info.get("last_lap_time", None)
                        if tlt is not None:
                            tlt = float(tlt)
                            if tlt > 5.0 and (test_last_lap is None or abs(tlt - test_last_lap) > 0.1):
                                avg_c = np.mean(test_lap_ctes) if test_lap_ctes else 99.0
                                avg_f = np.mean(test_lap_fwds) if test_lap_fwds else 0.0
                                is_real = (avg_c < 2.0 and test_lap_max_disp_w > 5.0 and avg_f > 0.3)
                                if is_real:
                                    test_real_laps += 1
                                elif tlt > 0.5:
                                    test_fake_laps += 1
                                test_last_lap = tlt
                                test_lap_ctes = []
                                test_lap_fwds = []
                                test_lap_max_disp_w = 0.0
                                test_lap_start = tp

                test_avg = test_reward / max(test_steps, 1)
                logger.info(
                    f'  TEST ep {episode}: steps={test_steps}, '
                    f'reward={test_reward:.1f} (avg={test_avg:.2f}/step), '
                    f'disp={test_max_disp:.1f}m')
                if writer:
                    writer.add_scalar('test/reward', test_reward, episode)
                    writer.add_scalar('test/steps', test_steps, episode)
                    writer.add_scalar('test/displacement', test_max_disp, episode)

                test_score = test_max_disp + test_real_laps * 50.0
                if test_score > best_test_score:
                    best_test_score = test_score
                    best_path = model_path.replace('.pth', '_best_test.pth')
                    dreamer.save(best_path)
                    logger.info(
                        f'  New best test score {test_score:.1f} '
                        f'(disp={test_max_disp:.1f}, real_laps={test_real_laps}), '
                        f'saved to {best_path}')

            if episode_reward > best_reward:
                best_reward = episode_reward
                dreamer.save(model_path)
                logger.info(f'New best reward {best_reward:.1f}, saved to {model_path}')

            if (episode + 1) % 20 == 0:
                checkpoint_path = model_path.replace(
                    '.pth', f'_ep{episode + 1}.pth'
                )
                dreamer.save(checkpoint_path)
                logger.info(
                    f'Checkpoint ep {episode + 1} | '
                    f'total real laps: {real_lap_count}, '
                    f'total fake laps: {fake_lap_count}'
                )

    except KeyboardInterrupt:
        logger.info('Interrupted by user')
        dreamer.save(model_path.replace('.pth', '_interrupted.pth'))

    finally:
        env.close()
        if writer:
            writer.close()

    logger.info(f'Training complete. Best reward: {best_reward:.1f}')


def evaluate(args):
    """Run trained model in inference mode — no training, no exploration noise."""
    dk_cfg = dk.load_config(myconfig=args.myconfig)
    device = get_device()
    logger.info(f'Device: {device}')

    track = args.track or getattr(dk_cfg, 'DONKEY_GYM_ENV_NAME', 'donkey-generated-track-v0')
    track_short = track.replace('donkey-', '').replace('_', '-')
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
        "max_cte": 3.0,
        "start_delay": 5.0,
    }

    env = make_env(track, conf)

    dreamer = Dreamer(device=device)
    model_path = args.model if args.model != 'models/dreamer_v3.pth' else f'models/{track_short}/dreamer.pth'
    
    if not os.path.exists(model_path):
        logger.error(f'Model not found: {model_path}')
        return
    dreamer.load(model_path)
    logger.info(f'Loaded model from {model_path}')

    try:
        for episode in range(args.episodes):
            obs, info = env.reset()
            dreamer.reset_belief()
            last_action = np.zeros(2, dtype=np.float32)

            episode_reward = 0.0
            episode_steps = 0
            real_laps = 0
            last_lap_time = None

            done = False
            while not done:
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                action_tensor = torch.from_numpy(last_action).unsqueeze(0)

                action = dreamer.select_action(
                    obs_tensor, action_tensor, explore=False
                )

                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                episode_reward += reward
                episode_steps += 1

                lap_time = info.get("last_lap_time", None)
                if lap_time is not None and episode_steps > 1:
                    lap_time = float(lap_time)
                    # Suppress fake spawn laps by enforcing > 5.0s
                    if lap_time > 5.0 and (last_lap_time is None or abs(lap_time - last_lap_time) > 0.1):
                        real_laps += 1
                        logger.info(
                            f'  LAP {lap_time:.1f}s '
                            f'(cte={abs(float(info.get("cte", 0))):.2f}, '
                            f'fwd={float(info.get("forward_vel", 0)):.2f}) '
                            f'ep {episode} step {episode_steps}')
                        last_lap_time = lap_time

                obs = next_obs
                last_action = np.array(action, dtype=np.float32)

            avg_reward = episode_reward / max(episode_steps, 1)
            logger.info(
                f'Episode {episode}: steps={episode_steps}, '
                f'reward={episode_reward:.1f} (avg={avg_reward:.2f}/step), '
                f'laps={real_laps}')

    except KeyboardInterrupt:
        logger.info('Stopped by user')
    finally:
        env.close()


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
    parser.add_argument('--eval', action='store_true',
                        help='Run inference only (no training)')
    parser.add_argument('--myconfig', type=str, default='myconfig.py')
    args = parser.parse_args()
    if args.eval:
        evaluate(args)
    else:
        train(args)