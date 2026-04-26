# Reactive world model wrapper that stores real transitions and generates synthetic samples.
import copy
import os
import random
import time
import numpy as np

from .cartpole_equations import (
    DEBUG_DIR,
    LOG_PATH,
    SAMPLE_LOW,
    cartpole_done,
    clip_sample_delta,
    clip_sample_obs,
)
from .debug import debug_prediction_snapshot as build_debug_prediction_snapshot
from .debug import multistep_prediction_snapshot as build_multistep_prediction_snapshot
from .debug import diagnostics as build_diagnostics

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
        min_samples=400,
        retrain_batches=2,
        batch_size=64,
        ready_delta_mae=0.12,
        ready_velocity_mae=0.30,
        ready_disagreement=0.04,
        ready_velocity_disagreement=0.03,
        train_every=8,
        max_planning_ratio=0.5,
        random_action_prob=0.5,
        planning_start_step=3000,
        planning_ramp_steps=3000,
        use_predicted_reward=False,
        reward_clip=(0.0, 1.0),
        use_shaped_reward=True,
        # Keep adapting by default; the policy changes the replay distribution
        # throughout training, so an early frozen snapshot quickly becomes stale.
        freeze_patience=None,
        max_train_ratio=1.0,
    ):
        self.model = model
        self.best_model = copy.deepcopy(model)
        self.store_continuous = True
        self.returns_continuous = True
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
        self.max_planning_ratio = float(max_planning_ratio)
        self.random_action_prob = float(random_action_prob)
        self.planning_start_step = int(planning_start_step)
        self.planning_ramp_steps = max(1, int(planning_ramp_steps))
        self.use_predicted_reward = bool(use_predicted_reward)
        self.reward_clip = reward_clip
        self.use_shaped_reward = bool(use_shaped_reward)
        self.freeze_patience = None if freeze_patience is None else int(freeze_patience)
        self.max_train_ratio = float(max_train_ratio)
        self.store_steps = 0
        self.training_steps = 0
        self.loss_ema = None
        self.last_diag = None
        self.best_diag = None
        self.discretizer = None
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
        self._sc_rej_bin = 0     # rejected because ensemble members predict different Q bins
        self._sc_exhausted = 0   # all 10 retries failed
        self._decision_total = 0
        self._decision_override = 0

        # Wipe the log at construction so each run starts fresh.
        os.makedirs(DEBUG_DIR, exist_ok=True)
        with open(LOG_PATH, "w") as _f:
            _f.write(f"# rwm_debug.log — started {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            _f.write(
                "# Thresholds: ready_delta_mae={:.3f}  ready_velocity_mae={:.3f}"
                "  ready_disagreement={:.4f}  ready_velocity_disagreement={:.4f}"
                "  use_predicted_reward={}  use_shaped_reward={}  history_horizon={}"
                "  freeze_patience={}\n".format(
                    ready_delta_mae, ready_velocity_mae,
                    ready_disagreement, ready_velocity_disagreement,
                    self.use_predicted_reward,
                    self.use_shaped_reward,
                    self.history_horizon,
                    freeze_patience,
                )
            )

    def _log(self, tag, **kwargs):
        """Append one structured line to rwm_debug.log."""
        ts = time.strftime("%H:%M:%S")
        pairs = " ".join(f"{k}={v}" for k, v in kwargs.items())
        with open(LOG_PATH, "a") as f:
            f.write(f"[{ts}][store={self.store_steps}][{tag}] {pairs}\n")

    def _format_obs(self, s):
        """Keep the real observation values so the model can learn true velocities."""
        return np.array(s, dtype=np.float32)

    def set_discretizer(self, discretizer):
        """Attach the Q-table discretizer so fake samples can be bin-consistent."""
        self.discretizer = discretizer

    def _clip_sample_obs(self, s):
        """Only clip imagined rollouts to keep synthetic states numerically stable."""
        return clip_sample_obs(s)

    def _clip_sample_delta(self, delta):
        """Clamp one-step fake dynamics so planning stays local and conservative."""
        return clip_sample_delta(delta)

    def _planning_model(self):
        """Use the live model unless training was explicitly frozen."""
        return self.best_model if self.training_frozen else self.model

    def _train_ratio(self):
        """Training updates per real environment transition."""
        return self.training_steps / max(1, self.store_steps)

    def _cartpole_done(self, s):
        """Predict CartPole termination from the imagined next observation."""
        return cartpole_done(s)

    def _format_planning_reward(self, r_pred):
        """Return the reward used for synthetic Q updates."""
        if not self.use_predicted_reward:
            return 1.0
        if self.reward_clip is None:
            return float(r_pred)
        lo, hi = self.reward_clip
        return float(np.clip(r_pred, lo, hi))

    def _planning_reward_for_next_state(self, r_pred, s_next, done):
        """Reward used by imagined planning transitions."""
        if self.use_shaped_reward:
            return self._stability_reward(s_next, done)
        return self._format_planning_reward(r_pred)

    def _stability_reward(self, s_next, done):
        """Dense CartPole stability reward used only by the world-model planner."""
        if done:
            return 0.0
        x, x_dot, theta, theta_dot = np.array(s_next, dtype=np.float32)
        x_score = 1.0 - min(abs(float(x)) / 2.4, 1.0)
        theta_score = 1.0 - min(abs(float(theta)) / 0.2095, 1.0)
        velocity_score = 1.0 - min(
            (abs(float(x_dot)) / 3.0 + abs(float(theta_dot)) / 3.5) / 2.0,
            1.0,
        )
        reward = 0.2 + 0.5 * theta_score + 0.2 * x_score + 0.1 * velocity_score
        return float(np.clip(reward, 0.0, 1.0))

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

    def _sample_nonterminal_index(self):
        """Choose a replay state that can be used as a valid planning start."""
        for _ in range(20):
            idx = random.randrange(len(self.buffer))
            if not self.buffer[idx][4]:
                return idx
        nonterminal = [i for i, sample in enumerate(self.buffer) if not sample[4]]
        if not nonterminal:
            raise Exception("No non-terminal replay states available for planning")
        return random.choice(nonterminal)

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
            pad_state = np.array(states[0], dtype=np.float32) if states else np.zeros_like(SAMPLE_LOW)
            pad_action = 0
            pad_count = self.history_horizon - len(states)
            states = [pad_state.copy() for _ in range(pad_count)] + states
            actions = [pad_action for _ in range(pad_count)] + actions

        return np.array(states, dtype=np.float32), np.array(actions, dtype=np.int64)

    def _ensemble_discrete_ok(self, s, ensemble_deltas):
        """Require ensemble members to agree on the Q-table bin of the fake next state."""
        if self.discretizer is None or ensemble_deltas is None:
            return True

        next_bins = []
        for delta in ensemble_deltas:
            member_delta = self._clip_sample_delta(delta)
            member_next = self._clip_sample_obs(np.array(s, dtype=np.float32) + member_delta)
            next_bins.append(self.discretizer.discretize(member_next))

        return len(set(next_bins)) == 1

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
        model_reward = self._stability_reward(s_next, done) if self.use_shaped_reward else float(r)
        self.buffer.append((s, int(a), delta, float(model_reward), float(done)))
        if len(self.buffer) > self.max_samples:
            self.buffer.pop(0)
        self.store_steps += 1

        train_ratio = self._train_ratio()

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
                    train_ratio = self._train_ratio()
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
                        loss=f"{float(loss):.8e}",
                        loss_ema=f"{self.loss_ema:.8e}",
                        loss_RISING=rising,
                        buffer=len(self.buffer),
                        train_ratio=f"{train_ratio:.3f}",
                        frozen=self.training_frozen,
                        no_improve=self.no_improve_count,
                    )

        prev_frozen = self.training_frozen
        if self._train_ratio() >= self.max_train_ratio:
            self.training_frozen = True
        if self.training_frozen and not prev_frozen:
            self._log(
                "FREEZE",
                reason="max_train_ratio",
                train_ratio=f"{self._train_ratio():.3f}",
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
        last_bin_ok = True
        for _ in range(10):
            idx = self._sample_nonterminal_index()
            s, replay_a, _, _, _ = self.buffer[idx]
            if random.random() < self.random_action_prob:
                a = np.random.randint(0, self.model.num_actions)
            else:
                a = int(replay_a)
            s_hist, a_hist = self._build_model_input(idx, action_override=a)
            if hasattr(self._planning_model(), "predict_all"):
                deltas, rewards = self._planning_model().predict_all(s_hist, a_hist)
                delta_std_by_dim = np.std(deltas, axis=0)
                delta_pred = np.mean(deltas, axis=0)
                r_pred = float(np.mean(rewards))
                delta_disagreement = float(np.mean(delta_std_by_dim))
                reward_disagreement = float(np.std(rewards))
            else:
                deltas = None
                delta_pred, r_pred, delta_disagreement, reward_disagreement, delta_std_by_dim = (
                    self._planning_model().predict_with_uncertainty_by_dim(s_hist, a_hist)
                )
            # Reject if mean disagreement OR either velocity dimension is too uncertain.
            velocity_disagreement = max(float(delta_std_by_dim[1]), float(delta_std_by_dim[3]))
            last_mean_dis = delta_disagreement
            last_vel_dis = velocity_disagreement

            mean_ok = delta_disagreement <= self.ready_disagreement
            vel_ok = velocity_disagreement <= self.ready_velocity_disagreement
            bin_ok = self._ensemble_discrete_ok(s, deltas)
            last_bin_ok = bin_ok
            if mean_ok and vel_ok and bin_ok:
                self._sc_accepted += 1
                delta_pred = self._clip_sample_delta(delta_pred)
                s_next = self._clip_sample_obs(np.array(s) + np.array(delta_pred, dtype=np.float32))
                done_pred = self._cartpole_done(s_next)
                r_pred = self._planning_reward_for_next_state(r_pred, s_next, done_pred)

                # Log a sample-rate summary every 50 accepted+exhausted calls.
                if (self._sc_accepted + self._sc_exhausted) % 50 == 0:
                    self._log(
                        "SAMPLE_STATS",
                        total=self._sc_total,
                        accepted=self._sc_accepted,
                        rej_mean_dis=self._sc_rej_mean,
                        rej_vel_dis=self._sc_rej_vel,
                        rej_bin=self._sc_rej_bin,
                        exhausted=self._sc_exhausted,
                        accept_rate=f"{self._sc_accepted / max(1, self._sc_total):.3f}",
                        thr_mean=f"{self.ready_disagreement:.4f}",
                        thr_vel=f"{self.ready_velocity_disagreement:.4f}",
                        pred_reward=self.use_predicted_reward,
                    )

                return tuple(float(v) for v in s), a, tuple(float(v) for v in s_next), r_pred, done_pred
            elif not mean_ok:
                self._sc_rej_mean += 1
            elif not vel_ok:
                self._sc_rej_vel += 1
            else:
                self._sc_rej_bin += 1

        self._sc_exhausted += 1
        # Log every exhausted event — these are the real signal that thresholds are too tight.
        self._log(
            "SAMPLE_EXHAUSTED",
            last_mean_dis=f"{last_mean_dis:.5f}",
            last_vel_dis=f"{last_vel_dis:.5f}",
            last_bin_ok=last_bin_ok,
            thr_mean=self.ready_disagreement,
            thr_vel=self.ready_velocity_disagreement,
            total_exhausted=self._sc_exhausted,
            total_calls=self._sc_total,
        )
        raise Exception("World model uncertainty too high for planning sample")

    def planning_ratio(self, step):
        """Ramp planning in linearly once ready, capped below full MBPO usage."""
        if not self._diagnostics_ready():
            return 0.0
        # Let real experience shape the early Q-table before synthetic planning joins in.
        if step < self.planning_start_step:
            return 0.0
        ramp = min(
            self.max_planning_ratio,
            self.max_planning_ratio * (step - self.planning_start_step) / self.planning_ramp_steps,
        )
        return ramp

    def ready(self):
        """Model is ready when recent diagnostics show good prediction quality."""
        if len(self.buffer) < self.min_samples:
            return False
        if self.training_steps < 20:
            return False
        return self._diagnostics_ready()

    def diagnostics(self, sample_size=128):
        """Measure one-step prediction error on real stored transitions."""
        return build_diagnostics(self, sample_size=sample_size)

    def debug_prediction_snapshot(self, sample_size=256):
        """Collect real-vs-predicted next-state values for a quick debug plot."""
        return build_debug_prediction_snapshot(self, sample_size=sample_size)

    def multistep_prediction_snapshot(self, sample_size=128, rollout_horizon=5):
        """Collect real-vs-predicted rollouts for multi-step debug plots."""
        return build_multistep_prediction_snapshot(
            self,
            sample_size=sample_size,
            rollout_horizon=rollout_horizon,
        )

    def _current_state_history(self, obs, previous_action=0):
        """Build a padded history for decision-time lookahead from the live env state."""
        state = self._format_obs(obs)
        states = [state.copy() for _ in range(self.history_horizon)]
        actions = [int(previous_action) for _ in range(self.history_horizon)]
        return np.array(states, dtype=np.float32), np.array(actions, dtype=np.int64)

    def _diagnostics_ready(self):
        if self.last_diag is None:
            self.last_diag = self.diagnostics()
        diag = self.last_diag
        if diag is None:
            return False

        velocity_errors = diag["delta_mae_by_dim"][1], diag["delta_mae_by_dim"][3]
        return (
            diag["delta_mae"] <= self.ready_delta_mae
            and max(velocity_errors) <= self.ready_velocity_mae
            and diag["delta_disagreement"] <= self.ready_disagreement
        )

    def _update_best_model(self):
        if self.last_diag is None:
            return
        if self.freeze_patience is None:
            self.best_score = (
                self.last_diag["delta_mae"]
                + 0.5 * self.last_diag["delta_disagreement"]
                + 0.1 * self.last_diag["reward_mae"]
            )
            self.best_diag = dict(self.last_diag)
            self.best_model.load_exported_state(self.model.export_state())
            self.no_improve_count = 0
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
            self.training_frozen = self._train_ratio() >= self.max_train_ratio
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


class RWMPredictModel(RWMModel):
    """Use the NN model to build multi-step bootstrapped targets and action vetoes."""

    def __init__(
        self,
        model,
        rollout_horizon=5,
        update_all_actions=True,
        decision_margin=0.35,
        decision_q_weight=0.02,
        decision_every=4,
        **kwargs,
    ):
        kwargs.setdefault("random_action_prob", 0.0)
        kwargs.setdefault("use_predicted_reward", False)
        super().__init__(model, **kwargs)
        self.rollout_horizon = int(rollout_horizon)
        self.update_all_actions = bool(update_all_actions)
        self.decision_margin = float(decision_margin)
        self.decision_q_weight = float(decision_q_weight)
        self.decision_every = max(1, int(decision_every))
        self._predict_update_count = 0
        self._decision_calls = 0

    def select_action(self, obs, q_action, q, discretizer):
        """Override a Q action when model lookahead predicts a clearly safer choice."""
        if not self.ready():
            return q_action

        self._decision_calls += 1
        if not self._should_check_decision(obs):
            return q_action

        state_history, action_history = self._current_state_history(obs, previous_action=q_action)
        obs = self._format_obs(obs)
        scores = []

        for first_action in range(self.model.num_actions):
            try:
                potential = self._rollout_potential(
                    state_history,
                    action_history,
                    obs,
                    first_action,
                    q,
                    discretizer,
                )
            except Exception:
                return q_action
            scores.append(potential)

        best_action = int(np.argmax(scores))
        self._decision_total += 1

        if best_action != int(q_action) and scores[best_action] > scores[int(q_action)] + self.decision_margin:
            self._decision_override += 1
            if self._decision_override % 50 == 0:
                self._log(
                    "DECISION_OVERRIDE",
                    total=self._decision_total,
                    overrides=self._decision_override,
                    q_action=q_action,
                    model_action=best_action,
                    q_score=f"{scores[int(q_action)]:.3f}",
                    model_score=f"{scores[best_action]:.3f}",
                )
            return best_action

        return q_action

    def _should_check_decision(self, obs):
        """Run expensive decision lookahead only periodically or near risky states."""
        x, x_dot, theta, theta_dot = np.array(obs, dtype=np.float32)
        risky = (
            abs(float(theta)) > 0.07
            or abs(float(x)) > 1.0
            or abs(float(theta_dot)) > 1.2
            or abs(float(x_dot)) > 1.5
        )
        return risky or self._decision_calls % self.decision_every == 0

    def apply_planning_update(self, q, discretizer):
        """Apply one imagined n-step return update directly to the Q-table."""
        if not self.ready():
            raise Exception("World model not ready")

        idx = self._sample_nonterminal_index()
        s0, a0, _, _, _ = self.buffer[idx]
        state_history, action_history = self._build_model_input(idx)
        actions = range(self.model.num_actions) if self.update_all_actions else [int(a0)]

        for first_action in actions:
            target = self._rollout_target(
                q,
                discretizer,
                state_history,
                action_history,
                s0,
                int(first_action),
            )
            s0_d = discretizer.discretize(s0)
            q.Q[s0_d][int(first_action)] += q.alpha * (
                target - q.Q[s0_d][int(first_action)]
            )

        self._predict_update_count += 1

        if self._predict_update_count % 50 == 0:
            self._log(
                "PREDICT_UPDATE",
                count=self._predict_update_count,
                horizon=self.rollout_horizon,
                updated_actions=self.model.num_actions if self.update_all_actions else 1,
            )

    def _rollout_target(self, q, discretizer, state_history, action_history, s0, first_action):
        """Estimate an n-step return for one candidate first action."""
        state_history = np.array(state_history, dtype=np.float32, copy=True)
        action_history = np.array(action_history, dtype=np.int64, copy=True)
        s_roll = np.array(s0, dtype=np.float32)
        a_roll = int(first_action)
        action_history[-1] = a_roll
        total_return = 0.0
        discount = 1.0
        done = False

        for _ in range(self.rollout_horizon):
            if hasattr(self._planning_model(), "predict_all"):
                deltas, rewards = self._planning_model().predict_all(state_history, action_history)
                delta_std_by_dim = np.std(deltas, axis=0)
                delta_pred = np.mean(deltas, axis=0)
                r_pred = float(np.mean(rewards))
                delta_disagreement = float(np.mean(delta_std_by_dim))
            else:
                deltas = None
                delta_pred, r_pred, delta_disagreement, _, delta_std_by_dim = (
                    self._planning_model().predict_with_uncertainty_by_dim(
                        state_history,
                        action_history,
                    )
                )

            velocity_disagreement = max(float(delta_std_by_dim[1]), float(delta_std_by_dim[3]))
            if (
                delta_disagreement > self.ready_disagreement
                or velocity_disagreement > self.ready_velocity_disagreement
                or not self._ensemble_discrete_ok(s_roll, deltas)
            ):
                raise Exception("World model uncertainty too high for predictive planning")

            delta_pred = self._clip_sample_delta(delta_pred)
            s_next = self._clip_sample_obs(s_roll + np.array(delta_pred, dtype=np.float32))
            done = self._cartpole_done(s_next)
            r_step = self._planning_reward_for_next_state(r_pred, s_next, done)
            total_return += discount * r_step
            discount *= q.gamma

            s_roll = s_next
            if done:
                break

            s_roll_d = discretizer.discretize(s_roll)
            a_roll = int(np.argmax(q.Q[s_roll_d]))
            state_history = np.concatenate(
                [state_history[1:], np.array([s_roll], dtype=np.float32)],
                axis=0,
            )
            action_history = np.concatenate(
                [action_history[1:], np.array([a_roll], dtype=np.int64)],
                axis=0,
            )

        if not done:
            s_bootstrap = discretizer.discretize(s_roll)
            total_return += discount * float(np.max(q.Q[s_bootstrap]))

        return total_return

    def _rollout_potential(self, state_history, action_history, s0, first_action, q, discretizer):
        """Score a candidate action by predicted stability over the rollout horizon."""
        state_history = np.array(state_history, dtype=np.float32, copy=True)
        action_history = np.array(action_history, dtype=np.int64, copy=True)
        s_roll = np.array(s0, dtype=np.float32)
        a_roll = int(first_action)
        action_history[-1] = a_roll
        score = 0.0
        discount = 1.0

        for _ in range(self.rollout_horizon):
            if hasattr(self._planning_model(), "predict_all"):
                deltas, _ = self._planning_model().predict_all(state_history, action_history)
                delta_std_by_dim = np.std(deltas, axis=0)
                delta_pred = np.mean(deltas, axis=0)
                delta_disagreement = float(np.mean(delta_std_by_dim))
            else:
                deltas = None
                delta_pred, _, delta_disagreement, _, delta_std_by_dim = (
                    self._planning_model().predict_with_uncertainty_by_dim(
                        state_history,
                        action_history,
                    )
                )

            velocity_disagreement = max(float(delta_std_by_dim[1]), float(delta_std_by_dim[3]))
            if (
                delta_disagreement > self.ready_disagreement
                or velocity_disagreement > self.ready_velocity_disagreement
                or not self._ensemble_discrete_ok(s_roll, deltas)
            ):
                raise Exception("World model uncertainty too high for decision lookahead")

            delta_pred = self._clip_sample_delta(delta_pred)
            s_next = self._clip_sample_obs(s_roll + np.array(delta_pred, dtype=np.float32))
            done = self._cartpole_done(s_next)
            score += discount * self._stability_reward(s_next, done)

            if done:
                return score

            discount *= q.gamma
            s_roll = s_next
            s_roll_d = discretizer.discretize(s_roll)
            q_values = q.Q[s_roll_d]
            score += self.decision_q_weight * discount * float(np.max(q_values))
            a_roll = int(np.argmax(q_values))
            state_history = np.concatenate(
                [state_history[1:], np.array([s_roll], dtype=np.float32)],
                axis=0,
            )
            action_history = np.concatenate(
                [action_history[1:], np.array([a_roll], dtype=np.int64)],
                axis=0,
            )

        return score
