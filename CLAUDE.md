# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Donkey Car** autonomous RC car application — a Python-based self-driving car framework using TensorFlow/Keras and the `donkeycar` library.

## Environment Setup

Always activate the conda environment before running any commands:
```bash
conda activate donkey
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
