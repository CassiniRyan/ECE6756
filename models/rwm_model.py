# Reactive world model wrapper that stores real transitions and generates synthetic samples.
import copy
import random
import numpy as np

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
        max_samples=50000,
        min_samples=800,
        retrain_batches=2,
        batch_size=64,
        ready_delta_mae=0.36,
        ready_velocity_mae=0.60,
        ready_disagreement=0.04,
        ready_velocity_disagreement=0.03,
        train_every=8,
        # diagnostics() is called every ~1000 steps; 30 means ~30k steps before freeze.
        freeze_patience=30,
        max_train_ratio=0.90,
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
        """Sample a replay batch from the stored transitions."""
        if len(self.buffer) < self.batch_size:
            raise ValueError("Not enough samples for a training batch")

        return random.sample(self.buffer, self.batch_size)

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

                s_b = np.array([b[0] for b in batch])
                a_b = np.array([b[1] for b in batch])
                delta_b = np.array([b[2] for b in batch])
                r_b = np.array([b[3] for b in batch])

                loss = self.model.train_batch((s_b, a_b, delta_b, r_b))
                if self.loss_ema is None:
                    self.loss_ema = float(loss)
                else:
                    self.loss_ema = 0.98 * self.loss_ema + 0.02 * float(loss)
                self.training_steps += 1

        if self.training_steps / max(1, len(self.buffer)) >= self.max_train_ratio:
            self.training_frozen = True

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
        for _ in range(10):
            s, _, _, _, _ = random.choice(self.buffer)
            a = np.random.randint(0, self.model.num_actions)
            delta_pred, r_pred, delta_disagreement, reward_disagreement, delta_std_by_dim = (
                self.best_model.predict_with_uncertainty_by_dim(s, a)
            )
            # Reject if mean disagreement OR either velocity dimension is too uncertain.
            velocity_disagreement = max(float(delta_std_by_dim[1]), float(delta_std_by_dim[3]))
            if (
                delta_disagreement <= self.ready_disagreement
                and velocity_disagreement <= self.ready_velocity_disagreement
            ):
                delta_pred = self._clip_sample_delta(delta_pred)
                s_next = self._clip_sample_obs(np.array(s) + np.array(delta_pred, dtype=np.float32))
                r_pred = float(np.clip(r_pred, 0.0, 1.0))
                return tuple(float(v) for v in s), a, tuple(float(v) for v in s_next), r_pred

        raise Exception("World model uncertainty too high for planning sample")

    def planning_ratio(self, step):
        """Ramp in neural-model planning conservatively to avoid bad early rollouts."""
        if step < 3000:
            return 0.0
        if step < 6000:
            # Let the world model warm up before it contributes many synthetic updates.
            return 0.5 * (step - 3000) / 3000
        return 0.5

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

        batch = random.sample(self.buffer, min(sample_size, len(self.buffer)))
        delta_errors = []
        delta_errors_by_dim = []
        next_state_errors = []
        reward_errors = []
        disagreements = []
        reward_disagreements = []

        velocity_disagreements = []
        for s, a, delta_true, r_true, _ in batch:
            delta_pred, r_pred, delta_disagreement, reward_disagreement, delta_std_by_dim = (
                self.model.predict_with_uncertainty_by_dim(s, a)
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
        self._update_best_model()
        return self.last_diag

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

        score = (
            self.last_diag["delta_mae"]
            + 0.5 * self.last_diag["delta_disagreement"]
            + 0.1 * self.last_diag["reward_mae"]
        )

        if score + 1e-4 < self.best_score:
            self.best_score = score
            self.best_diag = dict(self.last_diag)
            self.best_model.load_exported_state(self.model.export_state())
            self.no_improve_count = 0
            self.training_frozen = self.training_steps / max(1, len(self.buffer)) >= self.max_train_ratio
        else:
            self.no_improve_count += 1
            if self.no_improve_count >= self.freeze_patience:
                self.training_frozen = True
