# Reactive world model wrapper that stores real transitions and generates synthetic samples.
import copy
import os
import random
import time
import numpy as np

_LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rwm_debug.log")

class RWMModel:
    """
    Reactive World Model - manages the buffer of real transitions
    and trains a neural network to predict state deltas and rewards.
    
    Key separation:
    - store(): saves REAL environment transitions
    - sample(): generates FAKE transitions using model predictions
    """
    
    def __init__(
        self,
        model,
        max_samples=20000,
        min_samples=800,
        retrain_batches=2,
        batch_size=64,
        ready_delta_mae=0.36,
        ready_velocity_mae=0.60,
        ready_disagreement=0.04,
        ready_velocity_disagreement=0.03,
        train_every=8,
        # diagnostics() is called every ~1000 steps; a smaller patience stops drift.
        freeze_patience=12,
        max_train_ratio=1.0,
    ):
        self.model = model
        self.best_model = copy.deepcopy(model)
        self.store_continuous = True
        self.returns_continuous = True
        self.sample_low = np.array([-4.8, -10.0, -0.418, -10.0], dtype=np.float32)
        self.sample_high = np.array([4.8, 10.0, 0.418, 10.0], dtype=np.float32)
        self.delta_low = np.array([-0.30, -0.90, -0.12, -1.20], dtype=np.float32)
        self.delta_high = np.array([0.30, 0.90, 0.12, 1.20], dtype=np.float32)
        self.buffer = []
        self.max_samples = max_samples
        self.min_samples = min_samples
        self.retrain_batches = retrain_batches
        self.batch_size = batch_size
        self.ready_delta_mae = float(ready_delta_mae)
        self.ready_velocity_mae = float(ready_velocity_mae)
        self.ready_disagreement = float(ready_disagreement)
        self.ready_velocity_disagreement = float(ready_velocity_disagreement)
        self.train_every = int(train_every)
        self.freeze_patience = int(freeze_patience)
        self.max_train_ratio = float(max_train_ratio)
        self.store_steps = 0
        self.training_steps = 0
        self.loss_ema = None
        self.last_diag = None
        self.best_diag = None
        self.best_score = float("inf")
        self.no_improve_count = 0
        self.training_frozen = False
        self._was_ready = False
        self._loss_500_ago = None
        self.history_horizon = int(getattr(self.model, "history_horizon", 1))
        self.rollback_count = 0

        # Sample-call counters — reset to zero for each log window.
        self._sc_total = 0
        self._sc_accepted = 0
        self._sc_rej_mean = 0    # rejected because mean disagreement too high
        self._sc_rej_vel = 0     # rejected because velocity disagreement too high
        self._sc_exhausted = 0   # all 10 retries failed

        # Wipe the log at construction so each run starts fresh.
        with open(_LOG_PATH, "w") as _f:
            _f.write(f"# rwm_debug.log — started {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            _f.write(
                "# Thresholds: ready_delta_mae={:.3f}  ready_velocity_mae={:.3f}"
                "  ready_disagreement={:.4f}  ready_velocity_disagreement={:.4f}"
                "  freeze_patience={}\n".format(
                    ready_delta_mae, ready_velocity_mae,
                    ready_disagreement, ready_velocity_disagreement,
                    freeze_patience,
                )
            )

    def _log(self, tag, **kwargs):
        """Append one structured line to rwm_debug.log."""
        ts = time.strftime("%H:%M:%S")
        pairs = " ".join(f"{k}={v}" for k, v in kwargs.items())
        with open(_LOG_PATH, "a") as f:
            f.write(f"[{ts}][store={self.store_steps}][{tag}] {pairs}\n")

    def _format_obs(self, s):
        """Keep the real observation values so the model can learn true velocities."""
        return np.array(s, dtype=np.float32)

    def _clip_sample_obs(self, s):
        """Only clip imagined rollouts to keep synthetic states numerically stable."""
        return np.clip(np.array(s, dtype=np.float32), self.sample_low, self.sample_high)

    def _clip_sample_delta(self, delta):
        """Clamp one-step fake dynamics so planning stays local and conservative."""
        return np.clip(np.array(delta, dtype=np.float32), self.delta_low, self.delta_high)

    def _sample_training_batch(self):
        """Sample a replay batch of short histories ending at real transitions."""
        n = len(self.buffer)
        if n < self.batch_size:
            raise ValueError("Not enough samples for a training batch")
        indices = random.choices(range(n), k=self.batch_size)
        state_seq = []
        action_seq = []
        delta = []
        reward = []
        for idx in indices:
            s_hist, a_hist = self._build_model_input(idx)
            state_seq.append(s_hist)
            action_seq.append(a_hist)
            delta.append(self.buffer[idx][2])
            reward.append(self.buffer[idx][3])
        return (
            np.array(state_seq, dtype=np.float32),
            np.array(action_seq, dtype=np.int64),
            np.array(delta, dtype=np.float32),
            np.array(reward, dtype=np.float32),
        )

    def _build_model_input(self, idx, action_override=None):
        """Build a short state-action history ending at transition idx."""
        states = []
        actions = []
        cursor = idx
        while len(states) < self.history_horizon and cursor >= 0:
            s, a, _, _, done = self.buffer[cursor]
            states.append(np.array(s, dtype=np.float32))
            actions.append(int(a))
            if cursor > 0 and self.buffer[cursor - 1][4]:
                break
            cursor -= 1

        states.reverse()
        actions.reverse()

        if action_override is not None and actions:
            actions[-1] = int(action_override)

        if len(states) < self.history_horizon:
            pad_state = np.array(states[0], dtype=np.float32) if states else np.zeros_like(self.sample_low)
            pad_action = 0
            pad_count = self.history_horizon - len(states)
            states = [pad_state.copy() for _ in range(pad_count)] + states
            actions = [pad_action for _ in range(pad_count)] + actions

        return np.array(states, dtype=np.float32), np.array(actions, dtype=np.int64)

    def store(self, s, a, s_next, r, done=False):
        """
        Store real transition from environment.
        Also trains model in background.
        """
        s = self._format_obs(s)
        s_next = self._format_obs(s_next)

        # Compute delta (change in state)
        delta = s_next - s

        # Store in buffer
        self.buffer.append((s, int(a), delta, float(r), float(done)))
        if len(self.buffer) > self.max_samples:
            self.buffer.pop(0)
        self.store_steps += 1

        train_ratio = self.training_steps / max(1, len(self.buffer))

        if (
            not self.training_frozen
            and self.store_steps >= self.min_samples
            and train_ratio < self.max_train_ratio
            and self.store_steps % self.train_every == 0
        ):
            # Update the neural model online from small replay batches.
            for _ in range(self.retrain_batches):
                batch = self._sample_training_batch()
                loss = self.model.train_batch(batch)
                if self.loss_ema is None:
                    self.loss_ema = float(loss)
                else:
                    self.loss_ema = 0.98 * self.loss_ema + 0.02 * float(loss)
                self.training_steps += 1

                if self.training_steps % 100 == 0:
                    train_ratio = self.training_steps / max(1, len(self.buffer))
                    # Flag if loss_ema is rising compared to 500 steps ago.
                    rising = (
                        self._loss_500_ago is not None
                        and self.loss_ema > self._loss_500_ago * 1.1
                    )
                    if self.training_steps % 500 == 0:
                        self._loss_500_ago = self.loss_ema
                    self._log(
                        "TRAIN",
                        train_step=self.training_steps,
                        loss=f"{float(loss):.5f}",
                        loss_ema=f"{self.loss_ema:.5f}",
                        loss_RISING=rising,
                        buffer=len(self.buffer),
                        train_ratio=f"{train_ratio:.3f}",
                        frozen=self.training_frozen,
                        no_improve=self.no_improve_count,
                    )

        prev_frozen = self.training_frozen
        if self.training_steps / max(1, len(self.buffer)) >= self.max_train_ratio:
            self.training_frozen = True
        if self.training_frozen and not prev_frozen:
            self._log(
                "FREEZE",
                reason="max_train_ratio",
                train_ratio=f"{self.training_steps / max(1, len(self.buffer)):.3f}",
                train_step=self.training_steps,
            )

    def sample(self):
        """
        Generate a FAKE transition using the model.
        This is different from reality and should only be used
        for model-based planning AFTER the model is trained.
        
        Returns:
            (s, a, s_next, r) - but s_next is PREDICTED, not real
        """
        if not self.ready():
            raise Exception("World model not ready")

        # MBPO/PETS-style branching: always start from a real replay state and
        # reject imagined samples when ensemble members disagree too much.
        self._sc_total += 1
        last_mean_dis = last_vel_dis = float("nan")
        for _ in range(10):
            idx = random.randrange(len(self.buffer))
            s, _, _, _, _ = self.buffer[idx]
            a = np.random.randint(0, self.model.num_actions)
            s_hist, a_hist = self._build_model_input(idx, action_override=a)
            delta_pred, r_pred, delta_disagreement, reward_disagreement, delta_std_by_dim = (
                self.best_model.predict_with_uncertainty_by_dim(s_hist, a_hist)
            )
            # Reject if mean disagreement OR either velocity dimension is too uncertain.
            velocity_disagreement = max(float(delta_std_by_dim[1]), float(delta_std_by_dim[3]))
            last_mean_dis = delta_disagreement
            last_vel_dis = velocity_disagreement

            mean_ok = delta_disagreement <= self.ready_disagreement
            vel_ok = velocity_disagreement <= self.ready_velocity_disagreement
            if mean_ok and vel_ok:
                self._sc_accepted += 1
                delta_pred = self._clip_sample_delta(delta_pred)
                s_next = self._clip_sample_obs(np.array(s) + np.array(delta_pred, dtype=np.float32))
                r_pred = float(np.clip(r_pred, 0.0, 1.0))

                # Log a sample-rate summary every 50 accepted+exhausted calls.
                if (self._sc_accepted + self._sc_exhausted) % 50 == 0:
                    self._log(
                        "SAMPLE_STATS",
                        total=self._sc_total,
                        accepted=self._sc_accepted,
                        rej_mean_dis=self._sc_rej_mean,
                        rej_vel_dis=self._sc_rej_vel,
                        exhausted=self._sc_exhausted,
                        accept_rate=f"{self._sc_accepted / max(1, self._sc_total):.3f}",
                        thr_mean={self.ready_disagreement},
                        thr_vel={self.ready_velocity_disagreement},
                    )

                return tuple(float(v) for v in s), a, tuple(float(v) for v in s_next), r_pred
            elif not mean_ok:
                self._sc_rej_mean += 1
            else:
                self._sc_rej_vel += 1

        self._sc_exhausted += 1
        # Log every exhausted event — these are the real signal that thresholds are too tight.
        self._log(
            "SAMPLE_EXHAUSTED",
            last_mean_dis=f"{last_mean_dis:.5f}",
            last_vel_dis=f"{last_vel_dis:.5f}",
            thr_mean=self.ready_disagreement,
            thr_vel=self.ready_velocity_disagreement,
            total_exhausted=self._sc_exhausted,
            total_calls=self._sc_total,
        )
        raise Exception("World model uncertainty too high for planning sample")

    def planning_ratio(self, step):
        """Ramp planning in linearly once the model is ready, capped at 1.0."""
        if not self._diagnostics_ready():
            return 0.0
        # Warm up over 20k steps after the model first becomes ready.
        if not hasattr(self, "_planning_start_step"):
            self._planning_start_step = step
        ramp = min(1.0, (step - self._planning_start_step) / 20000.0)
        return ramp

    def ready(self):
        """Model is ready when the best saved model has good prediction quality."""
        if len(self.buffer) < self.min_samples:
            return False
        if self.training_steps < 50:
            return False
        return self._diagnostics_ready()

    def diagnostics(self, sample_size=128):
        """Measure one-step prediction error on real stored transitions."""
        if len(self.buffer) < max(32, sample_size):
            return None

        indices = random.sample(range(len(self.buffer)), min(sample_size, len(self.buffer)))
        delta_errors = []
        delta_errors_by_dim = []
        next_state_errors = []
        reward_errors = []
        disagreements = []
        reward_disagreements = []

        velocity_disagreements = []
        for idx in indices:
            s, a, delta_true, r_true, _ = self.buffer[idx]
            s_hist, a_hist = self._build_model_input(idx)
            delta_pred, r_pred, delta_disagreement, reward_disagreement, delta_std_by_dim = (
                self.model.predict_with_uncertainty_by_dim(s_hist, a_hist)
            )
            delta_pred = np.array(delta_pred, dtype=np.float32)
            s_next_true = self._format_obs(s + delta_true)
            s_next_pred = self._format_obs(s + delta_pred)

            delta_errors.append(np.mean(np.abs(delta_pred - delta_true)))
            delta_errors_by_dim.append(np.abs(delta_pred - delta_true))
            next_state_errors.append(np.mean(np.abs(s_next_pred - s_next_true)))
            reward_errors.append(abs(float(r_pred) - float(r_true)))
            disagreements.append(delta_disagreement)
            reward_disagreements.append(reward_disagreement)
            velocity_disagreements.append(max(float(delta_std_by_dim[1]), float(delta_std_by_dim[3])))

        self.last_diag = {
            "buffer_size": len(self.buffer),
            "training_steps": self.training_steps,
            "train_ratio": float(self.training_steps / max(1, len(self.buffer))),
            "max_train_ratio": self.max_train_ratio,
            "loss_ema": None if self.loss_ema is None else float(self.loss_ema),
            "delta_mae": float(np.mean(delta_errors)),
            "delta_mae_by_dim": np.mean(np.array(delta_errors_by_dim), axis=0).tolist(),
            "next_state_mae": float(np.mean(next_state_errors)),
            "reward_mae": float(np.mean(reward_errors)),
            "delta_disagreement": float(np.mean(disagreements)),
            "velocity_disagreement": float(np.mean(velocity_disagreements)),
            "reward_disagreement": float(np.mean(reward_disagreements)),
            "training_frozen": self.training_frozen,
            "best_delta_mae": None if self.best_diag is None else self.best_diag["delta_mae"],
            "best_delta_disagreement": None if self.best_diag is None else self.best_diag["delta_disagreement"],
        }
        # Log distribution shift: compare std of oldest vs newest quarter.
        # CartPole is zero-mean by symmetry so mean shift is always ~0 — std shift
        # is the real signal (policy improvement narrows the variance of theta/omega).
        n = len(self.buffer)
        if n >= 64:
            q = n // 4
            old_states = np.array([self.buffer[i][0] for i in range(q)])
            new_states = np.array([self.buffer[i][0] for i in range(n - q, n)])
            old_std = old_states.std(axis=0)
            new_std = new_states.std(axis=0)
            std_ratio = new_std / (old_std + 1e-8)
            self._log(
                "DIST_SHIFT",
                x_ratio=f"{std_ratio[0]:.3f}",
                v_ratio=f"{std_ratio[1]:.3f}",
                th_ratio=f"{std_ratio[2]:.3f}",
                om_ratio=f"{std_ratio[3]:.3f}",
                old_th_std=f"{old_std[2]:.4f}",
                new_th_std=f"{new_std[2]:.4f}",
                buf=n,
            )

        # --- diagnostic snapshot log ---
        d = self.last_diag
        dims = d["delta_mae_by_dim"]
        vel_errors = dims[1], dims[3]
        ready_vel_ok = max(vel_errors) <= self.ready_velocity_mae
        ready_mae_ok = d["delta_mae"] <= self.ready_delta_mae
        ready_dis_ok = d["delta_disagreement"] <= self.ready_disagreement
        self._log(
            "DIAG",
            buf=d["buffer_size"],
            train_steps=d["training_steps"],
            train_ratio=f"{d['train_ratio']:.3f}",
            frozen=d["training_frozen"],
            no_improve=self.no_improve_count,
            loss_ema=f"{d['loss_ema']:.5f}" if d["loss_ema"] else "None",
            delta_mae=f"{d['delta_mae']:.4f}",
            x_mae=f"{dims[0]:.4f}",
            v_mae=f"{dims[1]:.4f}",
            th_mae=f"{dims[2]:.4f}",
            om_mae=f"{dims[3]:.4f}",
            mean_dis=f"{d['delta_disagreement']:.5f}",
            vel_dis=f"{d['velocity_disagreement']:.5f}",
            best_mae=f"{d['best_delta_mae']:.4f}" if d["best_delta_mae"] else "None",
            ready_mae_ok=ready_mae_ok,
            ready_vel_ok=ready_vel_ok,
            ready_dis_ok=ready_dis_ok,
        )

        self._update_best_model()

        # Log a readiness transition when the model first becomes usable.
        is_ready = self._diagnostics_ready()
        if is_ready and not self._was_ready:
            self._log("MODEL_READY", train_steps=self.training_steps, buf=len(self.buffer))
        self._was_ready = is_ready

        return self.last_diag

    def debug_prediction_snapshot(self, sample_size=256):
        """Collect real-vs-predicted next-state values for a quick debug plot.

        This helper is intentionally lightweight and plotting-specific so we can
        remove it later without touching the main training logic.
        """
        if len(self.buffer) < max(16, sample_size):
            return None

        indices = random.sample(range(len(self.buffer)), min(sample_size, len(self.buffer)))
        pred_next = []
        real_next = []

        for idx in indices:
            s, _, delta_true, _, _ = self.buffer[idx]
            s_hist, a_hist = self._build_model_input(idx)
            delta_pred, _, _, _ = self.model.predict_with_uncertainty(s_hist, a_hist)
            pred_next.append(self._format_obs(s + np.array(delta_pred, dtype=np.float32)))
            real_next.append(self._format_obs(s + delta_true))

        pred_next = np.array(pred_next, dtype=np.float32)
        real_next = np.array(real_next, dtype=np.float32)

        return {
            "labels": ["x", "x_dot", "theta", "theta_dot"],
            "pred_next": pred_next,
            "real_next": real_next,
        }

    def _diagnostics_ready(self):
        if self.best_diag is None:
            self.last_diag = self.diagnostics()
        if self.best_diag is None:
            return False

        velocity_errors = self.best_diag["delta_mae_by_dim"][1], self.best_diag["delta_mae_by_dim"][3]
        return (
            self.best_diag["delta_mae"] <= self.ready_delta_mae
            and max(velocity_errors) <= self.ready_velocity_mae
            and self.best_diag["delta_disagreement"] <= self.ready_disagreement
        )

    def _update_best_model(self):
        if self.last_diag is None:
            return
        # Once frozen, keep using the best snapshot — no more rollbacks or logging.
        if self.training_frozen:
            return

        score = (
            self.last_diag["delta_mae"]
            + 0.5 * self.last_diag["delta_disagreement"]
            + 0.1 * self.last_diag["reward_mae"]
        )

        if score + 1e-4 < self.best_score:
            prev_best = self.best_score
            self.best_score = score
            self.best_diag = dict(self.last_diag)
            self.best_model.load_exported_state(self.model.export_state())
            self.no_improve_count = 0
            self.training_frozen = self.training_steps / max(1, len(self.buffer)) >= self.max_train_ratio
            self._log(
                "BEST_MODEL",
                new_score=f"{score:.5f}",
                prev_score=f"{prev_best:.5f}",
                delta_mae=f"{self.last_diag['delta_mae']:.4f}",
                vel_dis=f"{self.last_diag['velocity_disagreement']:.5f}",
                train_steps=self.training_steps,
            )
        else:
            self.no_improve_count += 1
            self._log(
                "NO_IMPROVE",
                count=self.no_improve_count,
                best_score=f"{self.best_score:.5f}",
                current_score=f"{score:.5f}",
                train_steps=self.training_steps,
            )
            # If the live model drifts away from the best snapshot, roll it back.
            # If that keeps happening, stop live training and keep using the best model.
            if self.no_improve_count % 5 == 0:
                self.model.load_exported_state(self.best_model.export_state())
                # Reset Adam momentum/variance so stale optimiser state from the
                # degraded run doesn't immediately corrupt the restored weights.
                self.model.reset_optimizers()
                self.rollback_count += 1
                self._log(
                    "ROLLBACK",
                    reason="score_degraded",
                    no_improve_count=self.no_improve_count,
                    rollback_count=self.rollback_count,
                    best_score=f"{self.best_score:.5f}",
                    current_score=f"{score:.5f}",
                    train_steps=self.training_steps,
                )
            if self.no_improve_count >= self.freeze_patience:
                self.training_frozen = True
                self.model.load_exported_state(self.best_model.export_state())
                self.model.reset_optimizers()
                self._log(
                    "FREEZE",
                    reason="patience_exceeded",
                    no_improve_count=self.no_improve_count,
                    rollback_count=self.rollback_count,
                    best_score=f"{self.best_score:.5f}",
                    current_score=f"{score:.5f}",
                    train_steps=self.training_steps,
                )
