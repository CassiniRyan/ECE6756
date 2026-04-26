"""Diagnostics and debug helpers for the reactive world model."""
import random

import numpy as np


def diagnostics(rwm, sample_size=128):
    """Measure one-step prediction error on real stored transitions."""
    if len(rwm.buffer) < max(32, sample_size):
        return None

    indices = random.sample(range(len(rwm.buffer)), min(sample_size, len(rwm.buffer)))
    delta_errors = []
    delta_errors_by_dim = []
    next_state_errors = []
    reward_errors = []
    disagreements = []
    reward_disagreements = []
    velocity_disagreements = []
    discrete_matches = []

    for idx in indices:
        s, a, delta_true, r_true, _ = rwm.buffer[idx]
        s_hist, a_hist = rwm._build_model_input(idx)
        delta_pred, r_pred, delta_disagreement, reward_disagreement, delta_std_by_dim = (
            rwm._planning_model().predict_with_uncertainty_by_dim(s_hist, a_hist)
        )
        delta_pred = np.array(delta_pred, dtype=np.float32)
        s_next_true = rwm._format_obs(s + delta_true)
        s_next_pred = rwm._format_obs(s + delta_pred)

        delta_errors.append(np.mean(np.abs(delta_pred - delta_true)))
        delta_errors_by_dim.append(np.abs(delta_pred - delta_true))
        next_state_errors.append(np.mean(np.abs(s_next_pred - s_next_true)))
        reward_errors.append(abs(float(r_pred) - float(r_true)))
        disagreements.append(delta_disagreement)
        reward_disagreements.append(reward_disagreement)
        velocity_disagreements.append(max(float(delta_std_by_dim[1]), float(delta_std_by_dim[3])))
        if rwm.discretizer is not None:
            discrete_matches.append(
                rwm.discretizer.discretize(s_next_pred)
                == rwm.discretizer.discretize(s_next_true)
            )

    rwm.last_diag = {
        "buffer_size": len(rwm.buffer),
        "training_steps": rwm.training_steps,
        "train_ratio": float(rwm._train_ratio()),
        "max_train_ratio": rwm.max_train_ratio,
        "loss_ema": None if rwm.loss_ema is None else float(rwm.loss_ema),
        "delta_mae": float(np.mean(delta_errors)),
        "delta_mae_by_dim": np.mean(np.array(delta_errors_by_dim), axis=0).tolist(),
        "next_state_mae": float(np.mean(next_state_errors)),
        "reward_mae": float(np.mean(reward_errors)),
        "delta_disagreement": float(np.mean(disagreements)),
        "velocity_disagreement": float(np.mean(velocity_disagreements)),
        "reward_disagreement": float(np.mean(reward_disagreements)),
        "discrete_bin_accuracy": (
            None if not discrete_matches else float(np.mean(discrete_matches))
        ),
        "training_frozen": rwm.training_frozen,
        "best_delta_mae": None if rwm.best_diag is None else rwm.best_diag["delta_mae"],
        "best_delta_disagreement": (
            None if rwm.best_diag is None else rwm.best_diag["delta_disagreement"]
        ),
    }

    _log_distribution_shift(rwm)
    _log_diagnostic_snapshot(rwm)

    rwm._update_best_model()
    rwm.last_diag["best_delta_mae"] = (
        None if rwm.best_diag is None else rwm.best_diag["delta_mae"]
    )
    rwm.last_diag["best_delta_disagreement"] = (
        None if rwm.best_diag is None else rwm.best_diag["delta_disagreement"]
    )

    is_ready = rwm._diagnostics_ready()
    if is_ready and not rwm._was_ready:
        rwm._log("MODEL_READY", train_steps=rwm.training_steps, buf=len(rwm.buffer))
    rwm._was_ready = is_ready

    return rwm.last_diag


def debug_prediction_snapshot(rwm, sample_size=256):
    """Collect real-vs-predicted next-state values for the debug scatter plot."""
    if len(rwm.buffer) < max(16, sample_size):
        return None

    indices = random.sample(range(len(rwm.buffer)), min(sample_size, len(rwm.buffer)))
    pred_next = []
    real_next = []

    for idx in indices:
        s, _, delta_true, _, _ = rwm.buffer[idx]
        s_hist, a_hist = rwm._build_model_input(idx)
        delta_pred, _, _, _ = rwm._planning_model().predict_with_uncertainty(s_hist, a_hist)
        pred_next.append(rwm._format_obs(s + np.array(delta_pred, dtype=np.float32)))
        real_next.append(rwm._format_obs(s + delta_true))

    return {
        "labels": ["x", "x_dot", "theta", "theta_dot"],
        "pred_next": np.array(pred_next, dtype=np.float32),
        "real_next": np.array(real_next, dtype=np.float32),
    }


def multistep_prediction_snapshot(rwm, sample_size=128, rollout_horizon=5):
    """Compare model rollouts against real future states using recorded actions."""
    if len(rwm.buffer) < max(16, sample_size + rollout_horizon):
        return None

    valid_starts = []
    last_start = len(rwm.buffer) - rollout_horizon - 1
    for idx in range(max(0, last_start + 1)):
        crosses_episode = any(rwm.buffer[j][4] for j in range(idx, idx + rollout_horizon))
        if not crosses_episode:
            valid_starts.append(idx)

    if not valid_starts:
        return None

    indices = random.sample(valid_starts, min(sample_size, len(valid_starts)))
    pred_by_step = [[] for _ in range(rollout_horizon)]
    real_by_step = [[] for _ in range(rollout_horizon)]
    mae_by_step = []
    bin_acc_by_step = []

    for idx in indices:
        state_history, action_history = rwm._build_model_input(idx)
        s_roll = np.array(rwm.buffer[idx][0], dtype=np.float32)

        for depth in range(rollout_horizon):
            action_history[-1] = int(rwm.buffer[idx + depth][1])
            delta_pred, _, _, _ = rwm._planning_model().predict_with_uncertainty(
                state_history,
                action_history,
            )
            s_roll = rwm._clip_sample_obs(
                s_roll + rwm._clip_sample_delta(np.array(delta_pred, dtype=np.float32))
            )
            s_real = rwm._format_obs(rwm.buffer[idx + depth][0] + rwm.buffer[idx + depth][2])

            pred_by_step[depth].append(s_roll)
            real_by_step[depth].append(s_real)

            if depth + 1 < rollout_horizon:
                next_action = int(rwm.buffer[idx + depth + 1][1])
                state_history = np.concatenate(
                    [state_history[1:], np.array([s_roll], dtype=np.float32)],
                    axis=0,
                )
                action_history = np.concatenate(
                    [action_history[1:], np.array([next_action], dtype=np.int64)],
                    axis=0,
                )

    for depth in range(rollout_horizon):
        pred = np.array(pred_by_step[depth], dtype=np.float32)
        real = np.array(real_by_step[depth], dtype=np.float32)
        mae_by_step.append(float(np.mean(np.abs(pred - real))))
        if rwm.discretizer is None:
            bin_acc_by_step.append(None)
        else:
            matches = [
                rwm.discretizer.discretize(pred[i]) == rwm.discretizer.discretize(real[i])
                for i in range(len(pred))
            ]
            bin_acc_by_step.append(float(np.mean(matches)))

    return {
        "labels": ["x", "x_dot", "theta", "theta_dot"],
        "pred_by_step": [np.array(v, dtype=np.float32) for v in pred_by_step],
        "real_by_step": [np.array(v, dtype=np.float32) for v in real_by_step],
        "mae_by_step": mae_by_step,
        "bin_acc_by_step": bin_acc_by_step,
    }


def _log_distribution_shift(rwm):
    """Log std-ratio shift between old and new replay-buffer states."""
    n = len(rwm.buffer)
    if n < 64:
        return

    q = n // 4
    old_states = np.array([rwm.buffer[i][0] for i in range(q)])
    new_states = np.array([rwm.buffer[i][0] for i in range(n - q, n)])
    old_std = old_states.std(axis=0)
    new_std = new_states.std(axis=0)
    std_ratio = new_std / (old_std + 1e-8)
    rwm._log(
        "DIST_SHIFT",
        x_ratio=f"{std_ratio[0]:.3f}",
        v_ratio=f"{std_ratio[1]:.3f}",
        th_ratio=f"{std_ratio[2]:.3f}",
        om_ratio=f"{std_ratio[3]:.3f}",
        old_th_std=f"{old_std[2]:.4f}",
        new_th_std=f"{new_std[2]:.4f}",
        buf=n,
    )


def _log_diagnostic_snapshot(rwm):
    """Write a compact one-line diagnostics summary to the RWM debug log."""
    d = rwm.last_diag
    dims = d["delta_mae_by_dim"]
    vel_errors = dims[1], dims[3]
    ready_vel_ok = max(vel_errors) <= rwm.ready_velocity_mae
    ready_mae_ok = d["delta_mae"] <= rwm.ready_delta_mae
    ready_dis_ok = d["delta_disagreement"] <= rwm.ready_disagreement

    rwm._log(
        "DIAG",
        buf=d["buffer_size"],
        train_steps=d["training_steps"],
        train_ratio=f"{d['train_ratio']:.3f}",
        frozen=d["training_frozen"],
        no_improve=rwm.no_improve_count,
        loss_ema=f"{d['loss_ema']:.8e}" if d["loss_ema"] else "None",
        delta_mae=f"{d['delta_mae']:.4f}",
        x_mae=f"{dims[0]:.4f}",
        v_mae=f"{dims[1]:.4f}",
        th_mae=f"{dims[2]:.4f}",
        om_mae=f"{dims[3]:.4f}",
        mean_dis=f"{d['delta_disagreement']:.5f}",
        vel_dis=f"{d['velocity_disagreement']:.5f}",
        bin_acc=(
            "None"
            if d["discrete_bin_accuracy"] is None
            else f"{d['discrete_bin_accuracy']:.3f}"
        ),
        best_mae=f"{d['best_delta_mae']:.4f}" if d["best_delta_mae"] else "None",
        ready_mae_ok=ready_mae_ok,
        ready_vel_ok=ready_vel_ok,
        ready_dis_ok=ready_dis_ok,
    )
