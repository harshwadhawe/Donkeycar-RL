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
    Reward shaping for Dreamer.

    Requires extendedTelemetry enabled in the sim UI.
    Uses: forward_vel, cte, pos (all real values with extended telemetry).
    """

    # Circling detection: if car drives N steps but stays within a small area
    CIRCLE_CHECK_INTERVAL = 50   # check every N steps
    CIRCLE_MIN_DISPLACEMENT = 3.0  # must move at least this far (meters) per window
    CIRCLE_MAX_CTE = 2.0        # avg |CTE| above this over window = off-track

    # Curriculum: CTE-focused early, blend in speed later
    CURRICULUM_CTE_PHASE = 80     # episodes where CTE dominates
    CURRICULUM_BLEND_PHASE = 120  # episodes to fully blend in speed

    def __init__(self, env, max_episode_steps=None):
        super().__init__(env)
        self.episode_step = 0
        self.max_episode_steps = max_episode_steps or cfg.DREAMER_MAX_EPISODE_STEPS
        self.last_steering = 0.0
        self.stuck_count = 0
        self.stuck_threshold = 10
        self._episode_num = 0  # set externally via set_episode()
        # Sustained steering detection
        self._same_dir_count = 0
        self._prev_steer_sign = 0
        # Circling detection state
        self._start_pos = None
        self._window_pos = None  # position at start of current check window
        self._window_ctes = []   # CTE values in current window

    def set_episode(self, episode_num):
        """Called by training loop to update curriculum phase."""
        self._episode_num = episode_num

    def reset(self, **kwargs):
        self.episode_step = 0
        self.last_steering = 0.0
        self.stuck_count = 0
        self._same_dir_count = 0
        self._prev_steer_sign = 0
        self._start_pos = None
        self._window_pos = None
        self._window_ctes = []
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, _sim_reward, terminated, truncated, info = self.env.step(action)
        self.episode_step += 1

        forward_vel = float(info.get("forward_vel", 0.0))
        cte = float(info.get("cte", 0.0))
        speed = float(info.get("speed", 0.0))
        steering = float(np.asarray(action)[0]) if hasattr(action, '__len__') else 0.0
        cur_pos = info.get("pos", None)

        # ── Curriculum: blend from CTE-only to CTE+speed ──
        # alpha=0 during CTE phase, ramps to 1 during blend phase
        ep = self._episode_num
        if ep < self.CURRICULUM_CTE_PHASE:
            alpha = 0.0
        elif ep < self.CURRICULUM_BLEND_PHASE:
            alpha = (ep - self.CURRICULUM_CTE_PHASE) / (
                self.CURRICULUM_BLEND_PHASE - self.CURRICULUM_CTE_PHASE)
        else:
            alpha = 1.0

        # CTE bell curve — primary signal for staying centered
        # sigma=1.0: at cte=0 → 5.0, at cte=0.5 → 3.9, at cte=1.0 → 1.8
        cte_reward = 5.0 * np.exp(-(cte / 1.0) ** 2)

        # Forward velocity — always present to distinguish driving from circling
        # Circles have forward_vel ≈ 0, real driving has forward_vel ≈ 1.0-2.0
        # Early: 1.0 * fwd_vel (secondary to CTE but enough to break ambiguity)
        # Late: ramps up so speed matters more
        fwd_vel_reward = (1.0 + alpha) * max(forward_vel, 0.0)

        # Light penalties — don't overwhelm the positive signals
        steer_change_penalty = 0.1 * (steering - self.last_steering) ** 2
        steer_mag_penalty = 0.15 * steering ** 2

        # Sustained steering penalty — catches smooth circling
        # Tracks consecutive steps with steering in the same direction.
        # Grace period allows legitimate corners; penalty ramps after.
        steer_sign = np.sign(steering)
        if steer_sign != 0 and steer_sign == self._prev_steer_sign:
            self._same_dir_count += 1
        else:
            self._same_dir_count = 0
        if steer_sign != 0:
            self._prev_steer_sign = steer_sign

        grace = 15       # ~0.75s at 20Hz — enough for any real corner
        excess = max(self._same_dir_count - grace, 0)
        sustained_penalty = 0.3 * (np.exp(0.04 * min(excess, 80)) - 1.0)
        # excess=0 → 0, excess=20 (~1s past grace) → 0.20
        # excess=40 (~2s) → 0.65, excess=60 (~3s) → 1.93

        reward = (cte_reward + fwd_vel_reward
                  - steer_change_penalty - steer_mag_penalty
                  - sustained_penalty)
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

        # ── Circling early termination ──
        # Every CIRCLE_CHECK_INTERVAL steps, verify the car actually moved
        # and stayed near the track. Circles have high |CTE| + low displacement.
        if cur_pos is not None:
            if self._start_pos is None:
                self._start_pos = cur_pos
            if self._window_pos is None:
                self._window_pos = cur_pos
            self._window_ctes.append(abs(cte))

            if self.episode_step % self.CIRCLE_CHECK_INTERVAL == 0 and self.episode_step > 0:
                dx = cur_pos[0] - self._window_pos[0]
                dz = cur_pos[2] - self._window_pos[2]
                window_disp = (dx**2 + dz**2) ** 0.5
                window_avg_cte = np.mean(self._window_ctes) if self._window_ctes else 0.0

                # Circling = not moving much OR way off-track for sustained period
                if (window_disp < self.CIRCLE_MIN_DISPLACEMENT and
                        window_avg_cte > self.CIRCLE_MAX_CTE):
                    terminated = True
                    info["circle_terminated"] = True

                # Reset window
                self._window_pos = cur_pos
                self._window_ctes = []

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
    best_test_score = -float('inf')  # displacement-based, ignores fake laps

    # Lap validation: multi-factor (not just time)
    real_lap_count = 0
    fake_lap_count = 0

    logger.info("=" * 60)
    logger.info(f"DREAMER v3 TRAINING")
    logger.info(f"  Episodes: {args.episodes}")
    logger.info(f"  Model: {model_path}")
    logger.info(f"  Curriculum: CTE-only eps 0-{RewardShapingWrapper.CURRICULUM_CTE_PHASE}, "
                f"blend to ep {RewardShapingWrapper.CURRICULUM_BLEND_PHASE}, "
                f"then full speed+CTE")
    logger.info("=" * 60)

    try:
        for episode in range(args.episodes):
            # Update curriculum phase
            reward_wrapper = env
            while hasattr(reward_wrapper, 'env'):
                if isinstance(reward_wrapper, RewardShapingWrapper):
                    break
                reward_wrapper = reward_wrapper.env
            if isinstance(reward_wrapper, RewardShapingWrapper):
                reward_wrapper.set_episode(episode)

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
            last_lap_time = None  # track sim-reported lap time changes
            # Per-lap-window metrics (reset each time a lap completes)
            lap_window_ctes = []
            lap_window_fwd_vels = []
            lap_window_max_disp = 0.0
            lap_window_start_pos = None
            t0 = time.time()

            is_seed = episode < cfg.DREAMER_SEED_EPISODES

            done = False
            while not done:
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                action_tensor = torch.from_numpy(last_action).unsqueeze(0)

                if is_seed:
                    # P-controller with noise: drives the track so world model
                    # sees actual turns, not just the first 3m before crashing
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
                episode_fwd_vels.append(fwd_vel)
                episode_ctes.append(step_cte)
                episode_steers.append(abs(float(np.asarray(action)[0])))

                # Track displacement from real pos
                cur_pos = info.get("pos", (0, 0, 0))
                if start_pos is None:
                    start_pos = cur_pos
                dx = cur_pos[0] - start_pos[0]
                dz = cur_pos[2] - start_pos[2]
                max_displacement = max(max_displacement, (dx**2 + dz**2)**0.5)

                # Accumulate per-lap-window metrics
                lap_window_ctes.append(abs(step_cte))
                lap_window_fwd_vels.append(fwd_vel)
                if lap_window_start_pos is None:
                    lap_window_start_pos = cur_pos
                ldx = cur_pos[0] - lap_window_start_pos[0]
                ldz = cur_pos[2] - lap_window_start_pos[2]
                lap_window_max_disp = max(lap_window_max_disp,
                                          (ldx**2 + ldz**2)**0.5)

                # Detect and classify laps — multi-factor validation
                cur_lap_time = info.get("last_lap_time", None)
                if cur_lap_time is not None:
                    cur_lap_time = float(cur_lap_time)
                    if last_lap_time is None or abs(cur_lap_time - last_lap_time) > 0.1:
                        # New lap reported by sim — validate it
                        avg_lap_cte = (np.mean(lap_window_ctes)
                                       if lap_window_ctes else 99.0)
                        avg_lap_fwd = (np.mean(lap_window_fwd_vels)
                                       if lap_window_fwd_vels else 0.0)
                        lap_disp = lap_window_max_disp

                        # Real lap: stayed on track + covered ground + moved forward
                        is_real = (avg_lap_cte < 2.0 and
                                   lap_disp > 5.0 and
                                   avg_lap_fwd > 0.3)

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
                            logger.debug(
                                f'  FAKE LAP {cur_lap_time:.1f}s '
                                f'(|cte|={avg_lap_cte:.2f}, '
                                f'disp={lap_disp:.1f}m, '
                                f'fwd={avg_lap_fwd:.2f})')

                        last_lap_time = cur_lap_time
                        # Reset lap window for next lap
                        lap_window_ctes = []
                        lap_window_fwd_vels = []
                        lap_window_max_disp = 0.0
                        lap_window_start_pos = cur_pos

                # Diagnostic: first episode
                if episode == 0 and episode_steps == 1:
                    logger.info(f'FIRST STEP INFO KEYS: {sorted(info.keys())}')
                    logger.info(
                        f'FIRST STEP INFO: cte={info.get("cte")}, '
                        f'forward_vel={info.get("forward_vel")}, '
                        f'pos={info.get("pos")}')

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
            avg_steer = np.mean(episode_steers) if episode_steers else 0.0
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

            # World model warmup: after seed episodes collected, train WM
            # for 100 steps before actor ever gets a gradient
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

                        # Accumulate per-lap-window metrics
                        t_cte = abs(float(test_info.get("cte", 0.0)))
                        t_fwd = float(test_info.get("forward_vel", 0.0))
                        test_lap_ctes.append(t_cte)
                        test_lap_fwds.append(t_fwd)
                        if test_lap_start is None:
                            test_lap_start = tp
                        tld = ((tp[0]-test_lap_start[0])**2 +
                               (tp[2]-test_lap_start[2])**2)**0.5
                        test_lap_max_disp_w = max(test_lap_max_disp_w, tld)

                        # Validate laps — same multi-factor check
                        tlt = test_info.get("last_lap_time", None)
                        if tlt is not None:
                            tlt = float(tlt)
                            if test_last_lap is None or abs(tlt - test_last_lap) > 0.1:
                                avg_c = np.mean(test_lap_ctes) if test_lap_ctes else 99.0
                                avg_f = np.mean(test_lap_fwds) if test_lap_fwds else 0.0
                                is_real = (avg_c < 2.0 and
                                           test_lap_max_disp_w > 5.0 and
                                           avg_f > 0.3)
                                if is_real:
                                    test_real_laps += 1
                                elif tlt > 0.5:
                                    test_fake_laps += 1
                                test_last_lap = tlt
                                # Reset window
                                test_lap_ctes = []
                                test_lap_fwds = []
                                test_lap_max_disp_w = 0.0
                                test_lap_start = tp

                test_avg = test_reward / max(test_steps, 1)
                test_lap_str = ''
                if test_real_laps or test_fake_laps:
                    test_lap_str = f', laps={test_real_laps}real/{test_fake_laps}fake'
                logger.info(
                    f'  TEST ep {episode}: steps={test_steps}, '
                    f'reward={test_reward:.1f} (avg={test_avg:.2f}/step), '
                    f'disp={test_max_disp:.1f}m'
                    f'{test_lap_str}')
                if writer:
                    writer.add_scalar('test/reward', test_reward, episode)
                    writer.add_scalar('test/steps', test_steps, episode)
                    writer.add_scalar('test/displacement', test_max_disp, episode)
                    writer.add_scalar('test/real_laps', test_real_laps, episode)

                # Save best model based on test performance
                # Score: displacement + big bonus per real lap
                test_score = test_max_disp + test_real_laps * 50.0
                if test_score > best_test_score:
                    best_test_score = test_score
                    best_path = model_path.replace('.pth', '_best_test.pth')
                    dreamer.save(best_path)
                    logger.info(
                        f'  New best test score {test_score:.1f} '
                        f'(disp={test_max_disp:.1f}, real_laps={test_real_laps}), '
                        f'saved to {best_path}')

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
