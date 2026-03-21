#!/usr/bin/env python3
"""
Quick test: connect to sim, take a few steps, dump all telemetry.
Run on Mac and Linux to compare what's available.

Usage:
    python test_sim_telemetry.py
    python test_sim_telemetry.py --track=donkey-warehouse-v0
    python test_sim_telemetry.py --port=9091
"""

import argparse
import numpy as np
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


def main(args):
    dk_cfg = dk.load_config(myconfig=args.myconfig)
    sim_path = getattr(dk_cfg, 'DONKEY_SIM_PATH', 'manual')

    conf = {
        "exe_path": sim_path,
        "host": getattr(dk_cfg, 'SIM_HOST', '127.0.0.1'),
        "port": args.port,
        "frame_skip": 1,
        "cam_resolution": (120, 160, 3),
        "log_level": 20,
        "max_cte": 8.0,
        "start_delay": 5.0,
    }

    track = args.track or getattr(
        dk_cfg, 'DONKEY_GYM_ENV_NAME', 'donkey-generated-track-v0')

    print(f"Track: {track}")
    print(f"Sim:   {sim_path}")
    print(f"Port:  {args.port}")
    print()

    if _USE_SHIMMY:
        env = gym.make("GymV21Environment-v0", env_id=track,
                        make_kwargs={"conf": conf})
    else:
        env = gym.make(track, conf=conf)

    # === Reset ===
    obs, info = env.reset()
    print("=" * 70)
    print("RESET")
    print("=" * 70)
    print(f"Observation shape: {obs.shape}, dtype: {obs.dtype}")
    print(f"Observation range: [{obs.min()}, {obs.max()}]")
    print(f"\nInfo keys ({len(info)}): {sorted(info.keys())}")
    print("\nInfo values:")
    for k in sorted(info.keys()):
        v = info[k]
        if isinstance(v, np.ndarray):
            print(f"  {k}: ndarray shape={v.shape} dtype={v.dtype} "
                  f"range=[{v.min():.4f}, {v.max():.4f}]")
        elif isinstance(v, (list, tuple)) and len(v) > 10:
            print(f"  {k}: {type(v).__name__} len={len(v)}")
        else:
            print(f"  {k}: {v!r}")

    # === Take steps with different actions ===
    test_actions = [
        ("straight",    [0.0, 0.3]),
        ("straight",    [0.0, 0.3]),
        ("straight",    [0.0, 0.3]),
        ("left turn",   [-0.5, 0.3]),
        ("right turn",  [0.5, 0.3]),
        ("hard left",   [-1.0, 0.3]),
        ("full speed",  [0.0, 1.0]),
        ("reverse?",    [0.0, -0.3]),
        ("stopped",     [0.0, 0.0]),
        ("stopped",     [0.0, 0.0]),
    ]

    print("\n" + "=" * 70)
    print("STEPPING (10 steps with various actions)")
    print("=" * 70)

    for i, (label, action) in enumerate(test_actions):
        action = np.array(action, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"\n--- Step {i} ({label}): action={action} ---")
        print(f"  reward={reward:.4f}  terminated={terminated}  "
              f"truncated={truncated}")

        # Print key telemetry
        for k in sorted(info.keys()):
            v = info[k]
            if isinstance(v, np.ndarray):
                print(f"  {k}: ndarray shape={v.shape}")
            elif isinstance(v, (list, tuple)) and len(v) > 10:
                print(f"  {k}: {type(v).__name__} len={len(v)}")
            else:
                print(f"  {k}: {v!r}")

        if terminated or truncated:
            print("  *** EPISODE ENDED ***")
            break

    # === Summary of what's useful for RL ===
    print("\n" + "=" * 70)
    print("SUMMARY: Key fields for RL training")
    print("=" * 70)

    key_fields = ['cte', 'speed', 'forward_vel', 'pos', 'vel',
                  'car', 'hit', 'lap_count', 'last_lap_time',
                  'gyro', 'accel', 'lidar']

    for k in key_fields:
        v = info.get(k, "*** NOT AVAILABLE ***")
        if isinstance(v, np.ndarray):
            print(f"  {k}: ndarray shape={v.shape}")
        elif isinstance(v, (list, tuple)) and len(v) > 10:
            print(f"  {k}: {type(v).__name__} len={len(v)}")
        else:
            print(f"  {k}: {v!r}")

    env.close()
    print("\nDone.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test sim telemetry')
    parser.add_argument('--track', type=str, default=None)
    parser.add_argument('--port', type=int, default=9091)
    parser.add_argument('--myconfig', type=str, default='myconfig.py')
    args = parser.parse_args()
    main(args)
