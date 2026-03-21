# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Donkey Car** autonomous RC car application — a Python-based self-driving car framework using TensorFlow/Keras and the `donkeycar` library.

## Environment Setup

Always activate the conda environment before running any commands:
```bash
conda activate donkey
pip install -r requirements.txt   # one-time: installs SB3, gymnasium, shimmy, etc.
```

**PyTorch** must be installed separately for the correct backend:
```bash
# macOS (MPS):
pip install torch torchvision
# Linux CUDA (e.g. RTX 4060):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Per-machine config (`myconfig.py`)

`myconfig.py` is gitignored — create one on each machine with the correct sim path:

```python
# macOS
DONKEY_GYM = True
DONKEY_SIM_PATH = "/path/to/DonkeySimMac/donkey_sim.app/Contents/MacOS/donkey_sim"
DONKEY_GYM_ENV_NAME = "donkey-generated-track-v0"

# Linux (CUDA)
DONKEY_GYM = True
DONKEY_SIM_PATH = "/home/pistar/harsh/DonkeySimLinux/donkey_sim.x86_64"
DONKEY_GYM_ENV_NAME = "donkey-generated-track-v0"
```

### Linux headless notes

On Linux the Unity sim needs a display. Always set `DISPLAY` and use `nohup` so training survives terminal close:
```bash
export DISPLAY=:0
```

## Common Commands

```bash
# Drive the car (manual or with a trained model)
python manage.py drive
python manage.py drive --model=models/mypilot.h5
python manage.py drive --myconfig=myconfig.py

# Train a model from recorded data
python train.py --tubs=data/tub_1 --model=models/mypilot.h5 --type=linear
# Model types: linear, inferred, tensorrt_linear, tflite_linear

# Calibrate hardware
python calibrate.py

# Drive with joystick
python manage.py drive --js

# RL driving via donkeycar vehicle loop
python drive_rl.py --type=sac --train                   # SAC+VAE: train from scratch
python drive_rl.py --type=dreamer --train               # Dreamer v3: train from scratch
python drive_rl.py --type=sac --model=models/sac.pth    # inference with trained SAC
python drive_rl.py --type=dreamer --model=models/dr.pth # inference with trained Dreamer

# SB3 SAC + VAE training (standalone, no vehicle loop)
python -m vae.train_vae --tub=data/tub_sim --epochs=100 # Step 1: train VAE (once)
python train_sac.py                                     # Step 2: train SAC (launches sim)
python train_sac.py --resume                            # auto-resume from checkpoint
python train_sac.py --timesteps=500000                  # custom timestep count

# Dreamer v3 training (standalone, no vehicle loop)
python train_dreamer.py                                  # train from scratch
python train_dreamer.py --resume                         # resume training
python train_dreamer.py --episodes=200                   # custom episode count

# Linux: run training in background (survives terminal close)
export DISPLAY=:0
nohup python train_sac.py > logs/train_sac.log 2>&1 &
nohup python train_dreamer.py > logs/train_dreamer.log 2>&1 &
tail -f logs/train_sac.log                               # follow progress

# Monitor training
tensorboard --logdir ./logs/tb_logs/ --port 6006
```

## Architecture

The app uses a **part-based vehicle architecture**. A `dk.vehicle.Vehicle()` instance is constructed by adding independent "parts" connected via named inputs/outputs:

```python
V = dk.vehicle.Vehicle()
V.add(SomePart(), inputs=['cam/image_array'], outputs=['pilot/angle', 'pilot/throttle'])
V.start(rate_hz=cfg.DRIVE_LOOP_HZ)  # runs the loop at 20Hz by default
```

Each part implements `run()` (or `run_threaded()`) — the vehicle loop calls all parts in sequence, passing named memory values between them.

### Key Files

- **`manage.py`** — Main entry point. `drive()` constructs the full vehicle by calling `add_camera()`, `add_user_controller()`, `add_drivetrain()`, etc.
- **`train.py`** — Thin wrapper around `donkeycar.pipeline.training.train()`.
- **`calibrate.py`** — Hardware calibration for servos and ESC.
- **`config.py`** — Primary configuration (770 lines). All hardware and behavior is configured here.
- **`myconfig.py`** — User overrides for `config.py`. Pass with `--myconfig=myconfig.py` to activate.

### Configuration System

`config.py` is loaded via `dk.load_config()` and controls everything: camera type/resolution, drivetrain type, PWM pin assignments and pulse ranges, drive loop frequency, web UI settings, and model training parameters. `myconfig.py` is a commented-out template for overrides — uncomment only what you need to change.

### Drivetrain Types (`DRIVE_TRAIN_TYPE` in config.py)

- `PWM_STEERING_THROTTLE` — default RC car (servo + ESC via PCA9685)
- `MM1`, `SERVO_HBRIDGE_2PIN`, `SERVO_HBRIDGE_3PIN`
- `DC_STEER_THROTTLE`, `DC_TWO_WHEEL`, `DC_TWO_WHEEL_L298N`
- `VESC` — VESC motor controller
- `MOCK` — testing without hardware

### Camera Types (`CAMERA_TYPE` in config.py)

`PICAM`, `WEBCAM`, `CVCAM`, `CSIC`, `V4L`, `D435` (RealSense), `OAKD`, `MOCK`, `IMAGE_LIST`

### Drive Modes

`user/mode` controls which source drives the car:
- `user` — human controls steering and throttle (web UI or joystick)
- `local_angle` — AI controls steering only; human controls throttle
- `local` — AI controls both steering and throttle

### Data Flow

Camera → (optional image processing) → Neural network inference → Steering/Throttle → Drivetrain

In manual mode, user input (web UI or joystick) bypasses the model. Training data is recorded to `data/` as "tubs" (directories of images + JSON metadata) and used to train models saved to `models/`.

### Simulator Support

Set `DONKEY_GYM = True` in config to use the Donkey Gym simulator instead of physical hardware. When active, CUDA is disabled to avoid resource conflicts.

### Hardware Abstraction

GPIO pin access is abstracted as `RPI_GPIO`, `PIGPIO`, or `PCA9685` — configured in `config.py`. The framework targets Raspberry Pi but supports Nano and custom boards.

### RL Module (`rl/`)

Reinforcement learning. Two algorithms: SAC+VAE (model-free) and Dreamer v3 (model-based, arXiv:2301.04104).

- **`rl/config.py`** — All hyperparameters for both algorithms
- **`rl/vae.py`** — VAE encoder/decoder (40x40 grayscale → 20D latent). Used by SAC only.
- **`rl/dreamer.py`** — Dreamer v3: categorical RSSM (256D belief + 16x16 categorical state), symlog predictions, KL balancing, LayerNorm+SiLU, percentile return normalization, slow EMA critic
- **`rl/sac.py`** — SAC+VAE: Gaussian actor (35→64→64→2), twin critics, auto-entropy tuning
- **`rl/buffer.py`** — `ReplayBuffer` (flat, for SAC) and `EpisodeBuffer` (sequential chunks, for Dreamer)
- **`rl/agent.py`** — `SACPilot` and `DreamerPilot` donkeycar parts with episode management, death detection, training
- **`drive_rl.py`** — Vehicle loop entry point: `--type=sac|dreamer`, `--train`, `--model=<path>`

### Standalone RL Training (`train_sac.py`, `train_dreamer.py`)

Bypass the donkeycar vehicle loop — train directly against the gym simulator. Includes reward shaping (CTE bell curve, speed, survival, HOF), action smoothing, checkpointing, and TensorBoard logging.

- **`vae/`** — VAE module for SAC observation encoding (32D latent from 120x160 images)
- **`train_sac.py`** — SB3 SAC + pre-trained VAE, frame stacking, auto-resume
- **`train_dreamer.py`** — Dreamer v3 standalone training with image preprocessing
