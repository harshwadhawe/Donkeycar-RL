#!/usr/bin/env python3
"""
train_sac.py — SB3 SAC + VAE training for Donkey Car simulator.

Uses a pre-trained VAE to encode camera images into 32-dim latent vectors,
then trains SAC with reward shaping (CTE bell curve, speed, survival, HOF).

Setup (one-time):
    conda activate donkey
    pip install -r requirements.txt

Step 1: Train VAE (only needed once):
    python -m vae.train_vae --tub=data/tub_sim --epochs=100

Step 2: Train SAC:
    python train_sac.py                        # new training
    python train_sac.py --resume               # resume from checkpoint
    python train_sac.py --timesteps=500000     # custom length

Step 3: Monitor:
    tensorboard --logdir ./logs/tb_logs/ --port 6006
"""

import os
import glob
import re
import argparse
from collections import deque

import numpy as np
import torch
import gymnasium as gym
import gym_donkeycar  # noqa: F401

# gym_donkeycar may register with old gym or gymnasium depending on version.
# If envs aren't in gymnasium registry, use shimmy to bridge.
try:
    gym.spec('donkey-generated-track-v0')
    _USE_SHIMMY = False
except gym.error.NameNotFound:
    import shimmy
    gym.register_envs(shimmy)
    _USE_SHIMMY = True

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

import donkeycar as dk

from vae.controller import VAEController


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
# VAE Observation Wrapper
# ==============================================================================
class DonkeyVAEWrapper(gym.ObservationWrapper):
    """Replace image observations with VAE latent vectors."""

    def __init__(self, env, vae_path, device):
        super().__init__(env)
        self.vae = VAEController(model_path=vae_path, z_dim=32, device=device)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.vae.z_dim,), dtype=np.float32
        )

    def observation(self, obs):
        return self.vae.encode_observation(obs)


# ==============================================================================
# Auto-Resume
# ==============================================================================
def find_latest_checkpoint(models_dir, prefix):
    """Find latest checkpoint: {prefix}_{STEPS}_steps.zip"""
    pattern = os.path.join(models_dir, f"{prefix}_*_steps.zip")
    files = glob.glob(pattern)
    if not files:
        return None, None, 0

    def get_steps(p):
        m = re.search(rf"{re.escape(prefix)}_(\d+)_steps\.zip$", os.path.basename(p))
        return int(m.group(1)) if m else 0

    latest_model = max(files, key=get_steps)
    latest_steps = get_steps(latest_model)

    buffer_path = latest_model.replace(".zip", "_replay_buffer.pkl")
    if not os.path.exists(buffer_path):
        buffer_path = None

    return latest_model, buffer_path, latest_steps


# ==============================================================================
# Wrappers
# ==============================================================================
class SmoothActionWrapper(gym.Wrapper):
    """Exponential moving average on actions to reduce jitter."""

    def __init__(self, env, alpha=0.7):
        super().__init__(env)
        self.alpha = float(alpha)
        self._prev_action = None

    def reset(self, **kwargs):
        self._prev_action = None
        return self.env.reset(**kwargs)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if self._prev_action is None:
            smoothed = action
        else:
            smoothed = self.alpha * action + (1.0 - self.alpha) * self._prev_action
        self._prev_action = smoothed
        return self.env.step(smoothed)


class RewardShapingWrapper(gym.Wrapper):
    """
    Reward shaping with CTE bell curve, speed reward, survival bonus,
    Hall-of-Fame lap bonus, and wobble penalties.
    """

    def __init__(self, env):
        super().__init__(env)

        self.episode_step = 0
        self.global_step = 0
        self.last_steering = 0.0
        self.last_steering_diff = 0.0

        # Wobble penalty curriculum
        self.wobble_penalty_start = 0.10
        self.wobble_penalty_final = 1.00
        self.wobble_ramp_steps = 50_000

        # CTE bell curve
        self.cte_amp = 5.0
        self.cte_sigma = 0.35

        # Speed
        self.speed_weight = 1.0
        self.speed_exponent = 1.15

        # Survival bonus
        self.survival_base = 0.03
        self.survival_max = 0.25
        self.survival_ramp = 10_000

        # Penalties
        self.min_speed = 0.25
        self.standstill_penalty = 1.0
        self.crash_penalty = 120.0

        # Steering smoothness
        self.steering_diff_weight = 0.20
        self.oscillation_extra = 1.50
        self.oscillation_threshold = 0.05

        # Hall of Fame
        self.hof_start_global_step = 50_000
        self.hof_window = 300
        self.hof_times = deque(maxlen=self.hof_window)
        self.hof_percentile = 10
        self.hof_max_bonus = 300.0
        self.hof_last_lap_recorded = 0.0

    def reset(self, **kwargs):
        self.episode_step = 0
        self.last_steering = 0.0
        self.last_steering_diff = 0.0
        return self.env.reset(**kwargs)

    @staticmethod
    def _gaussian(x, sigma):
        z = x / max(sigma, 1e-6)
        return float(np.exp(-(z * z)))

    def _wobble_scale(self):
        progress = min(self.global_step / max(self.wobble_ramp_steps, 1), 1.0)
        return self.wobble_penalty_start + \
            (self.wobble_penalty_final - self.wobble_penalty_start) * progress

    def step(self, action):
        obs, _raw_reward, terminated, truncated, info = self.env.step(action)

        self.episode_step += 1
        self.global_step += 1

        speed = float(info.get("speed", 0.0))
        cte = float(info.get("cte", 0.0))
        steering = float(np.asarray(action)[0])

        # Positive rewards
        speed_term = self.speed_weight * (max(speed, 0.0) ** self.speed_exponent)
        cte_term = self.cte_amp * self._gaussian(cte, self.cte_sigma)
        surv_progress = min(self.episode_step / max(self.survival_ramp, 1), 1.0)
        survival_term = self.survival_base + \
            (self.survival_max - self.survival_base) * surv_progress

        reward = speed_term + cte_term + survival_term

        # Steering smoothness penalties
        diff = steering - self.last_steering
        diff2 = diff * diff
        wobble_scale = self._wobble_scale()

        is_flipping = ((diff > 0 and self.last_steering_diff < 0) or
                       (diff < 0 and self.last_steering_diff > 0))
        wobble_mult = self.oscillation_extra \
            if (is_flipping and abs(diff) > self.oscillation_threshold) else 1.0

        speed_factor = 1.0 + min(abs(speed) * 0.25, 1.0)
        steering_penalty = (self.steering_diff_weight * diff2 *
                            speed_factor * wobble_scale * wobble_mult)
        reward -= steering_penalty

        if speed < self.min_speed:
            reward -= self.standstill_penalty

        # Hall-of-Fame bonus
        lap_time = float(info.get("last_lap_time", 0.0))
        if lap_time > 0.0 and lap_time != self.hof_last_lap_recorded:
            self.hof_last_lap_recorded = lap_time
            self.hof_times.append(lap_time)

            if (self.global_step >= self.hof_start_global_step and
                    len(self.hof_times) >= 50):
                p10 = float(np.percentile(
                    np.array(self.hof_times, dtype=np.float32),
                    self.hof_percentile
                ))
                if lap_time <= p10:
                    margin = (p10 - lap_time) / max(p10, 1e-6)
                    bonus = self.hof_max_bonus * (margin ** 2)
                    reward += bonus

        # Crash penalty
        if terminated or truncated:
            reward -= self.crash_penalty

        self.last_steering = steering
        self.last_steering_diff = diff

        if self.global_step % 10_000 == 0:
            print(f"[step={self.global_step}] wobble_scale={wobble_scale:.2f} "
                  f"speed={speed_term:.2f} cte={cte_term:.2f} surv={survival_term:.2f}")

        return obs, reward, terminated, truncated, info


# ==============================================================================
# Env Factory
# ==============================================================================
def make_env(env_id, conf, vae_path, device):
    """Create a single wrapped donkey env. Returned as a callable for DummyVecEnv."""
    def _init():
        if _USE_SHIMMY:
            env = gym.make(
                "GymV21Environment-v0",
                env_id=env_id,
                make_kwargs={"conf": conf},
            )
        else:
            env = gym.make(env_id, conf=conf)
        env = DonkeyVAEWrapper(env, vae_path=vae_path, device=device)
        env = SmoothActionWrapper(env, alpha=0.5)
        env = RewardShapingWrapper(env)
        env = Monitor(env)
        return env
    return _init


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description='SB3 SAC + VAE training')
    parser.add_argument('--timesteps', type=int, default=250_000,
                        help='Total training timesteps')
    parser.add_argument('--vae', type=str, default='logs/vae/vae_z32_best.pth',
                        help='Path to trained VAE model')
    parser.add_argument('--track', type=str, default=None,
                        help='Gym env name (default: from myconfig.py)')
    parser.add_argument('--resume', action='store_true',
                        help='Auto-resume from latest checkpoint')
    parser.add_argument('--myconfig', type=str, default='myconfig.py',
                        help='Donkeycar config override file')
    args = parser.parse_args()

    # Load donkeycar config for sim settings
    cfg = dk.load_config(myconfig=args.myconfig)

    device = get_device()
    print(f'Device: {device}')

    # Check VAE exists
    if not os.path.exists(args.vae):
        print(f'ERROR: VAE model not found at {args.vae}')
        print('Train one first:')
        print('  python -m vae.train_vae --tub=data/tub_sim --epochs=100')
        return

    track = args.track or getattr(cfg, 'DONKEY_GYM_ENV_NAME', 'donkey-generated-track-v0')
    sim_path = getattr(cfg, 'DONKEY_SIM_PATH', 'manual')

    conf = {
        "exe_path": sim_path,
        "host": getattr(cfg, 'SIM_HOST', '127.0.0.1'),
        "port": 9091,
        "frame_skip": 2,
        "cam_resolution": (
            getattr(cfg, 'IMAGE_H', 120),
            getattr(cfg, 'IMAGE_W', 160),
            getattr(cfg, 'IMAGE_DEPTH', 3),
        ),
        "log_level": 20,
        "throttle_max": 1.0,
    }

    print(f'Track: {track}')
    print(f'Sim: {sim_path}')

    models_dir = os.path.join(os.getcwd(), 'models')
    os.makedirs(models_dir, exist_ok=True)

    checkpoint_prefix = "sac_donkey"

    # DummyVecEnv needs a callable factory — avoids closure bugs
    env = DummyVecEnv([make_env(track, conf, vae_path=args.vae, device=device)])
    env = VecFrameStack(env, n_stack=4)

    # Auto-resume or start fresh
    if args.resume:
        latest_model, latest_buffer, start_steps = find_latest_checkpoint(
            models_dir, checkpoint_prefix
        )
    else:
        latest_model, latest_buffer, start_steps = None, None, 0

    if latest_model:
        print("=" * 60)
        print(f"RESUMING from step {start_steps}")
        print(f"Model : {latest_model}")
        print(f"Buffer: {latest_buffer or 'NONE'}")
        print("=" * 60)

        model = SAC.load(latest_model, env=env, device=device,
                         tensorboard_log="./logs/tb_logs/")

        if latest_buffer:
            try:
                model.load_replay_buffer(latest_buffer)
                print("Replay buffer loaded.")
            except Exception as e:
                print(f"Replay buffer load failed: {e}. Continuing with empty buffer.")

        model.num_timesteps = start_steps
        remaining_steps = max(args.timesteps - start_steps, 0)

        # Conservative LR when resuming
        low_lr = 1e-4
        optimizers = [model.actor.optimizer, model.critic.optimizer]
        if model.ent_coef_optimizer is not None:
            optimizers.append(model.ent_coef_optimizer)
        for opt in optimizers:
            for pg in opt.param_groups:
                pg["lr"] = low_lr
        print(f"Optimizer LR set to {low_lr}")
    else:
        print("=" * 60)
        print("STARTING NEW TRAINING")
        print(f"Total timesteps: {args.timesteps}")
        print("=" * 60)
        remaining_steps = args.timesteps

        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            device=device,
            learning_rate=3e-4,
            buffer_size=100_000,
            batch_size=512,
            ent_coef="auto",
            gamma=0.99,
            tau=0.005,
            train_freq=1,
            gradient_steps=1,
            use_sde=True,
            tensorboard_log="./logs/tb_logs/",
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=10_000,
        save_path=models_dir,
        name_prefix=checkpoint_prefix,
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    print(f"Training for {remaining_steps} steps...")

    try:
        if remaining_steps > 0:
            model.learn(
                total_timesteps=remaining_steps,
                callback=checkpoint_cb,
                reset_num_timesteps=False,
            )

        final_path = os.path.join(models_dir, f"{checkpoint_prefix}_final")
        model.save(final_path)
        model.save_replay_buffer(final_path + "_replay_buffer")
        print(f"Done. Final model: {final_path}")

    except KeyboardInterrupt:
        emergency_path = os.path.join(models_dir, f"{checkpoint_prefix}_interrupted")
        model.save(emergency_path)
        model.save_replay_buffer(emergency_path + "_replay_buffer")
        print(f"Interrupted. Emergency save: {emergency_path}")

    env.close()


if __name__ == "__main__":
    main()
