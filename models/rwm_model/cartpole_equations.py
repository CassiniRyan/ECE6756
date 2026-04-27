"""CartPole constants and small equations used by the reactive world model.

Keeping these equations outside the model class makes the neural wrapper easier
to read. These helpers are not learned; they encode known CartPole bounds and
terminal conditions used to keep imagined rollouts physically reasonable.
"""
import os

import numpy as np


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEBUG_DIR = os.path.join(ROOT_DIR, "debug")
LOG_PATH = os.path.join(DEBUG_DIR, "rwm_debug.log")

# Observation and delta clamps stop imagined states from wandering far outside
# the training distribution during planning.
SAMPLE_LOW = np.array([-4.8, -10.0, -0.418, -10.0], dtype=np.float32)
SAMPLE_HIGH = np.array([4.8, 10.0, 0.418, 10.0], dtype=np.float32)
DELTA_LOW = np.array([-0.30, -0.90, -0.12, -1.20], dtype=np.float32)
DELTA_HIGH = np.array([0.30, 0.90, 0.12, 1.20], dtype=np.float32)


def clip_sample_obs(s):
    """Clip imagined observations to a numerically stable CartPole range."""
    return np.clip(np.array(s, dtype=np.float32), SAMPLE_LOW, SAMPLE_HIGH)


def clip_sample_delta(delta):
    """Clamp one-step imagined deltas so planning stays local."""
    return np.clip(np.array(delta, dtype=np.float32), DELTA_LOW, DELTA_HIGH)


def cartpole_done(s):
    """Return whether an observation is terminal under CartPole-v1 thresholds."""
    x, _, theta, _ = np.array(s, dtype=np.float32)
    return bool(abs(x) > 2.4 or abs(theta) > 12.0 * np.pi / 180.0)
