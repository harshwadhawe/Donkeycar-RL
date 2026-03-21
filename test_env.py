#!/usr/bin/env python3
"""Quick diagnostic: check sim connection, info dict, rewards, and Dreamer init."""

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

# ── 1. Device ────────────────────────────────────────────
print("=" * 60)
print("1. DEVICE CHECK")
print("=" * 60)
print(f"  PyTorch version: {torch.__version__}")
print(f"  CUDA available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU:             {torch.cuda.get_device_name(0)}")
    print(f"  VRAM:            {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
print(f"  MPS available:   {torch.backends.mps.is_available()}")
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"  Using device:    {device}")

# ── 2. Sim connection ───────────────────────────────────
print()
print("=" * 60)
print("2. SIM CONNECTION")
print("=" * 60)

cfg = dk.load_config(myconfig='myconfig.py')
track = getattr(cfg, 'DONKEY_GYM_ENV_NAME', 'donkey-generated-track-v0')
sim_path = getattr(cfg, 'DONKEY_SIM_PATH', 'manual')
print(f"  Track:    {track}")
print(f"  Sim path: {sim_path}")
print(f"  Shimmy:   {_USE_SHIMMY}")

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
    "max_cte": 3.0,
    "start_delay": 5.0,
}

if _USE_SHIMMY:
    env = gym.make("GymV21Environment-v0", env_id=track, make_kwargs={"conf": conf})
else:
    env = gym.make(track, conf=conf)

print("  Connected OK!")

# ── 3. Reset & info dict ────────────────────────────────
print()
print("=" * 60)
print("3. RESET & INFO DICT")
print("=" * 60)
obs, info = env.reset()
print(f"  Obs shape (reset): {np.array(obs).shape}, dtype={np.array(obs).dtype}")
print(f"  Info keys (reset): {sorted(info.keys()) if info else '** EMPTY **'}")
if info:
    for k, v in sorted(info.items()):
        print(f"    {k}: {v}")

# ── 4. Step & telemetry ─────────────────────────────────
print()
print("=" * 60)
print("4. STEP TELEMETRY (10 steps)")
print("=" * 60)
for i in range(10):
    action = np.array([0.0, 0.3], dtype=np.float32)  # straight, slow
    obs, reward, terminated, truncated, info = env.step(action)
    cte = info.get("cte", "MISSING")
    speed = info.get("speed", "MISSING")
    pos = info.get("pos", "MISSING")
    hit = info.get("hit", "MISSING")
    print(f"  step {i}: reward={reward:.4f}, cte={cte}, speed={speed}, "
          f"hit={hit}, terminated={terminated}")

print()
print(f"  Obs shape (step):  {np.array(obs).shape}")
print(f"  Info keys (step):  {sorted(info.keys())}")
print(f"  Full info dict:")
for k, v in sorted(info.items()):
    vstr = f"{v}" if not isinstance(v, np.ndarray) else f"ndarray{v.shape}"
    print(f"    {k}: {vstr}")

# ── 5. Steering test (does CTE change?) ─────────────────
print()
print("=" * 60)
print("5. STEERING TEST (hard left 20 steps)")
print("=" * 60)
for i in range(20):
    action = np.array([1.0, 0.3], dtype=np.float32)  # hard left
    obs, reward, terminated, truncated, info = env.step(action)
    cte = info.get("cte", 0.0)
    speed = info.get("speed", 0.0)
    if terminated or truncated:
        print(f"  step {i}: TERMINATED! reward={reward:.4f}, cte={cte}, speed={speed}")
        break
    if i % 5 == 0:
        print(f"  step {i}: reward={reward:.4f}, cte={cte}, speed={speed}")

env.close()

# ── 6. Dreamer model init on GPU ────────────────────────
print()
print("=" * 60)
print("6. DREAMER MODEL ON GPU")
print("=" * 60)
from rl.dreamer import Dreamer
from rl.buffer import EpisodeBuffer
from rl import config as rl_cfg

dreamer = Dreamer(device=device)
channels = 3 if rl_cfg.RGB else 1
print(f"  Image: {channels}x{rl_cfg.IMAGE_SIZE}x{rl_cfg.IMAGE_SIZE}")

x = torch.randn(1, channels, rl_cfg.IMAGE_SIZE, rl_cfg.IMAGE_SIZE, device=device)
emb = dreamer.encoder(x)
print(f"  Encoder: {x.shape} -> {emb.shape}")

bs = rl_cfg.DREAMER_BELIEF_SIZE
ss = rl_cfg.DREAMER_NUM_CLASSES * rl_cfg.DREAMER_NUM_CATEGORIES
feat = torch.randn(1, bs + ss, device=device)
dec = dreamer.decoder(feat)
print(f"  Decoder: {feat.shape} -> {dec.shape}")
assert dec.shape == (1, channels, rl_cfg.IMAGE_SIZE, rl_cfg.IMAGE_SIZE)

obs_t = torch.randn(1, channels, rl_cfg.IMAGE_SIZE, rl_cfg.IMAGE_SIZE)
act_t = torch.zeros(1, 2)
dreamer.reset_belief()
action = dreamer.select_action(obs_t, act_t, explore=True)
print(f"  Action: {action}")

# Quick training test
buf = EpisodeBuffer(max_steps=1000, obs_shape=(channels, rl_cfg.IMAGE_SIZE, rl_cfg.IMAGE_SIZE), action_dim=2)
for ep in range(3):
    for s in range(25):
        buf.add_step(
            np.random.randn(channels, rl_cfg.IMAGE_SIZE, rl_cfg.IMAGE_SIZE).astype(np.float32),
            np.random.randn(2).astype(np.float32), 1.0, s == 24,
        )
metrics = dreamer.update(buf, gradient_steps=2)
print(f"  Training (2 steps): {metrics}")

if device == 'cuda':
    mem = torch.cuda.memory_allocated() / 1e6
    print(f"  GPU memory used: {mem:.0f} MB")

print()
print("=" * 60)
print("ALL CHECKS PASSED")
print("=" * 60)
