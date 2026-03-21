#!/usr/bin/env python3
"""
Drive the Donkey Car using Reinforcement Learning.

Supports two algorithms from the L2D paper (arXiv:2008.00715):
  - sac:     SAC+VAE (model-free, simpler, needs more episodes)
  - dreamer: Dreamer (model-based, learns in <5 min, heavier compute)

Usage:
    drive_rl.py [--model=<model>] [--train] [--type=<type>] [--myconfig=<filename>]

Options:
    -h --help               Show this screen.
    --model=<model>         Path to RL model weights
                            [default: models/rl_pilot.pth]
    --train                 Enable training mode (collect data + train)
    --type=<type>           RL algorithm: sac or dreamer [default: sac]
    --myconfig=<filename>   Specify myconfig file to use [default: myconfig.py]
"""

from docopt import docopt

try:
    import cv2
except ImportError:
    pass

import donkeycar as dk
from donkeycar.parts.tub_v2 import TubWriter
from donkeycar.parts.datastore import TubHandler
from donkeycar.parts.controller import LocalWebController, WebFpv
from donkeycar.parts.throttle_filter import ThrottleFilter
from donkeycar.parts.explode import ExplodeDict
from donkeycar.parts.transform import Lambda
from donkeycar.parts.pipe import Pipe
from donkeycar.utils import *

from manage import (
    add_simulator, add_odometry, add_camera,
    add_user_controller, add_drivetrain,
    DriveMode, UserPilotCondition,
)
from rl.agent import SACPilot, DreamerPilot, RLTrainToggle

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def drive_rl(cfg, model_path='models/rl_pilot.pth', train_mode=False, rl_type='sac'):
    """
    Construct vehicle with RL pilot.

    Uses the same camera, web controller, and drivetrain as manage.py,
    but replaces the Keras pilot with an RL agent (SAC+VAE or Dreamer).
    """
    logger.info(f'PID: {os.getpid()}')

    V = dk.vehicle.Vehicle()

    if cfg.HAVE_CONSOLE_LOGGING:
        logger.setLevel(logging.getLevelName(cfg.LOGGING_LEVEL))
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(cfg.LOGGING_FORMAT))
        logger.addHandler(ch)

    # Standard vehicle setup
    add_simulator(V, cfg)
    add_odometry(V, cfg)
    add_camera(V, cfg, 'single')

    # Web controller — auto-start in 'local' mode for RL training
    if train_mode:
        cfg.WEB_INIT_MODE = 'local'
    ctr = add_user_controller(V, cfg, use_joystick=False)
    V.add(Pipe(), inputs=['user/steering'], outputs=['user/angle'])
    V.add(ExplodeDict(V.mem, "web/"), inputs=['web/buttons'])

    th_filter = ThrottleFilter()
    V.add(th_filter, inputs=['user/throttle'], outputs=['user/throttle'])

    V.add(UserPilotCondition(show_pilot_image=getattr(cfg, 'SHOW_PILOT_IMAGE', False)),
          inputs=['user/mode', "cam/image_array", "cam/image_array_trans"],
          outputs=['run_user', "run_pilot", "ui/image_array"])

    # ── RL Pilot ────────────────────────────────────────
    if rl_type == 'dreamer':
        rl_pilot = DreamerPilot(model_path=model_path, train_mode=train_mode)
        V.add(rl_pilot,
              inputs=['cam/image_array'],
              outputs=['pilot/angle', 'pilot/throttle'],
              run_condition='run_pilot')
        algo_name = 'Dreamer'
    else:
        rl_pilot = SACPilot(model_path=model_path, train_mode=train_mode)
        V.add(rl_pilot,
              inputs=['cam/image_array'],
              outputs=['pilot/angle', 'pilot/throttle'],
              run_condition='run_pilot')
        algo_name = 'SAC+VAE'

    # Toggle training with web button w1 (rising-edge detection)
    V.add(RLTrainToggle(rl_pilot),
          inputs=['web/w1'])

    # Drive mode selection (user vs autopilot)
    V.add(DriveMode(cfg.AI_THROTTLE_MULT),
          inputs=['user/mode', 'user/angle', 'user/throttle',
                  'pilot/angle', 'pilot/throttle'],
          outputs=['steering', 'throttle'])

    # Drivetrain
    add_drivetrain(V, cfg)

    # Data recording (optional, for imitation learning data)
    inputs = ['cam/image_array', 'user/angle', 'user/throttle', 'user/mode']
    types = ['image_array', 'float', 'float', 'str']

    if cfg.RECORD_DURING_AI:
        inputs += ['pilot/angle', 'pilot/throttle']
        types += ['float', 'float']

    tub_path = TubHandler(path=cfg.DATA_PATH).create_tub_path() if \
        cfg.AUTO_CREATE_NEW_TUB else cfg.DATA_PATH
    tub_writer = TubWriter(tub_path, inputs=inputs, types=types)
    V.add(tub_writer, inputs=inputs, outputs=["tub/num_records"],
          run_condition='recording')

    print("=" * 60)
    print(f"RL DRIVING MODE ({algo_name})")
    print(f"  Model: {model_path}")
    print(f"  Training: {'ON' if train_mode else 'OFF'}")
    print(f"  Web UI: http://<hostname>:{cfg.WEB_CONTROL_PORT}")
    print(f"  W1 button in web UI toggles training on/off")
    print("=" * 60)

    V.start(rate_hz=cfg.DRIVE_LOOP_HZ, max_loop_count=cfg.MAX_LOOPS)


if __name__ == '__main__':
    args = docopt(__doc__)
    cfg = dk.load_config(myconfig=args['--myconfig'])

    drive_rl(
        cfg,
        model_path=args['--model'],
        train_mode=args['--train'],
        rl_type=args['--type'],
    )
