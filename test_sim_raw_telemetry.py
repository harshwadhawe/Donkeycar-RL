#!/usr/bin/env python3
"""
Dump the RAW JSON keys the Unity sim sends in telemetry messages.
This tells us exactly what data the sim provides vs what gym_donkeycar expects.

Usage:
    python test_sim_raw_telemetry.py
"""

import argparse
import time
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
    print()

    if _USE_SHIMMY:
        env = gym.make("GymV21Environment-v0", env_id=track,
                        make_kwargs={"conf": conf})
    else:
        env = gym.make(track, conf=conf)

    # Monkey-patch the telemetry handler to capture raw message keys
    raw_keys_seen = set()
    raw_sample_msg = {}

    # Navigate to the actual sim handler
    actual_env = env.unwrapped if hasattr(env, 'unwrapped') else env
    # gym_donkeycar envs store the sim controller
    sim = None
    for attr in ['viewer', 'sim', 'simulator', 'env']:
        if hasattr(actual_env, attr):
            obj = getattr(actual_env, attr)
            if hasattr(obj, 'handler'):
                sim = obj
                break
    if sim is None and hasattr(actual_env, 'handler'):
        sim = actual_env

    if sim is not None and hasattr(sim, 'handler'):
        handler = sim.handler
        original_on_telemetry = handler.on_telemetry

        def patched_on_telemetry(message):
            nonlocal raw_sample_msg
            for k in message.keys():
                if k != 'image' and k != 'image_b':
                    raw_keys_seen.add(k)
                    if k not in raw_sample_msg:
                        raw_sample_msg[k] = message[k]
            original_on_telemetry(message)

        handler.on_telemetry = patched_on_telemetry
        print("Successfully patched telemetry handler!")
    else:
        print("WARNING: Could not find sim handler to patch.")
        print(f"  env type: {type(actual_env)}")
        print(f"  env attrs: {[a for a in dir(actual_env) if not a.startswith('_')]}")

    # Reset and take steps
    obs, info = env.reset()
    print(f"\nAfter reset - info keys: {sorted(info.keys())}")

    for i in range(20):
        action = np.array([0.0, 0.3], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

    print("\n" + "=" * 70)
    print("RAW TELEMETRY KEYS FROM UNITY SIM")
    print("=" * 70)
    print(f"\nKeys present in raw messages ({len(raw_keys_seen)}):")
    for k in sorted(raw_keys_seen):
        v = raw_sample_msg.get(k, "?")
        if isinstance(v, str) and len(v) > 50:
            v = v[:50] + "..."
        print(f"  {k}: {v!r}")

    # Check what's missing
    expected = ['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z',
                'cte', 'speed', 'forward_vel',
                'gyro_x', 'gyro_y', 'gyro_z',
                'accel_x', 'accel_y', 'accel_z',
                'roll', 'pitch', 'yaw', 'hit']

    print(f"\n{'Field':<15} {'In raw msg?':<12} {'Value'}")
    print("-" * 50)
    for field in expected:
        present = field in raw_keys_seen
        val = raw_sample_msg.get(field, "N/A")
        if isinstance(val, float):
            val = f"{val:.6f}"
        status = "YES" if present else "*** NO ***"
        print(f"  {field:<13} {status:<12} {val}")

    env.close()
    print("\nDone.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--track', type=str, default=None)
    parser.add_argument('--port', type=int, default=9091)
    parser.add_argument('--myconfig', type=str, default='myconfig.py')
    args = parser.parse_args()
    main(args)
