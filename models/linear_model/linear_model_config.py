"""Shared paths and CSV schema for the linear world model diagnostics."""
import os

import numpy as np


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_DIR = os.path.join(ROOT_DIR, "logs")
CSV_PATH = os.path.join(LOG_DIR, "linear_model_metrics.csv")
OBS_LOW = np.array([-4.8, -5.0, -0.418, -5.0], dtype=np.float32)
OBS_HIGH = np.array([4.8, 5.0, 0.418, 5.0], dtype=np.float32)
OBS_SCALE = np.maximum(np.abs(OBS_HIGH), 1e-6)
CSV_FIELDS = [
    "step",
    "buffer",
    "mse",
    "ready",
    "delta_mae",
    "delta_mae_x",
    "delta_mae_v",
    "delta_mae_th",
    "delta_mae_om",
    "reward_mae",
    "plan_pred_err",
    "plan_count",
]
