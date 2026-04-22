
# Agent logic connecting the environment, the policy, and any planning model.
from rl_core.discretizer import Discretizer
from rl_core.q_learning import QLearning
import random

class Agent:
    """Agent wrapper for training, saving, loading, and policy evaluation."""

    def __init__(self, env, model=None, planning_steps=0, bins_per_dim=10):
        self.env = env
        self.disc = Discretizer(bins_per_dim=bins_per_dim)
        self.q = QLearning(env.action_space.n, bins_per_dim=bins_per_dim)
        self.model = model
        self.planning_steps = planning_steps
        self.planning_budget = 0.0
        self.global_step = 0  # Total number of environment steps taken.

    def planning_ratio(self, step):
        """Return how much model planning to apply for the current model."""
        if self.model is None:
            return 0.0
        if hasattr(self.model, "planning_ratio"):
            # Each model family can decide how cautiously it should be used.
            return float(self.model.planning_ratio(step))
        return 1.0

    def train(self, episodes):
        """Run training for a number of episodes and return episode rewards."""
        rewards = []

        for ep in range(episodes):
            obs, _ = self.env.reset()
            s = self.disc.discretize(obs)
            done = False
            total = 0

            while not done:
                # Select an action using the current Q-table.
                a = self.q.select_action(s, self.env.action_space.n)
                obs_next, r, term, trunc, _ = self.env.step(a)
                done = term or trunc
                s_next = self.disc.discretize(obs_next)

                # Always update the Q-table from the real transition.
                self.q.update(s, a, r, s_next, done=done)

                # Store the real transition in the model if available.
                if self.model is not None:
                    if getattr(self.model, "store_continuous", False):
                        # Learned dynamics models fit on raw continuous observations.
                        self.model.store(obs, a, obs_next, r, done)
                    else:
                        # Tabular models only need discretized state ids.
                        self.model.store(s, a, s_next, r)

                # Optionally perform additional model-based planning updates.
                ratio = self.planning_ratio(self.global_step)
                if self.model is not None and ratio > 0.0 and self.model.ready():
                    # Fractional planning budgets let gentle schedules still add up
                    # to occasional full planning updates.
                    self.planning_budget += self.planning_steps * ratio
                    model_updates = int(self.planning_budget)
                    if model_updates > 0:
                        self.planning_budget -= model_updates
                        for _ in range(model_updates):
                            try:
                                sample = self.model.sample()
                                if len(sample) == 5:
                                    s_m, a_m, s_next_m, r_m, done_m = sample
                                else:
                                    s_m, a_m, s_next_m, r_m = sample
                                    done_m = False

                                if getattr(self.model, "returns_continuous", False):
                                    # The Q-table lives in discretized state space even
                                    # when the planning model predicts continuous states.
                                    s_m_d = self.disc.discretize(s_m)
                                    s_next_m_d = self.disc.discretize(s_next_m)
                                else:
                                    s_m_d, s_next_m_d = s_m, s_next_m

                                self.q.update(s_m_d, a_m, r_m, s_next_m_d, done=done_m)
                            except Exception:
                                # Ignore sampling failures if the model is not ready.
                                pass

                s = s_next
                total += r
                self.global_step += 1

                if self.global_step % 1000 == 0:
                    model_ready = self.model is not None and self.model.ready()
                    print(
                        f"[Step {self.global_step}] planning_ratio={ratio:.3f} "
                        f"model_ready={model_ready}"
                    )
                    if self.model is not None and hasattr(self.model, "diagnostics"):
                        diag = self.model.diagnostics()
                        if diag is not None:
                            delta_dims = ", ".join(f"{v:.4f}" for v in diag["delta_mae_by_dim"])
                            print(
                                "[ModelDiag] "
                                f"buffer={diag['buffer_size']} "
                                f"train_steps={diag['training_steps']} "
                                f"train_ratio={diag['train_ratio']:.2f} "
                                f"training_frozen={diag['training_frozen']} "
                                f"loss_ema={diag['loss_ema']:.4f} "
                                f"delta_mae={diag['delta_mae']:.4f} "
                                f"delta_mae_by_dim=[{delta_dims}] "
                                f"next_state_mae={diag['next_state_mae']:.4f} "
                                f"reward_mae={diag['reward_mae']:.4f} "
                                f"delta_disagreement={diag['delta_disagreement']:.4f} "
                                f"reward_disagreement={diag['reward_disagreement']:.4f} "
                                f"best_delta_mae={diag['best_delta_mae'] if diag['best_delta_mae'] is not None else 'None'} "
                                f"best_delta_disagreement={diag['best_delta_disagreement'] if diag['best_delta_disagreement'] is not None else 'None'}"
                            )

            rewards.append(total)
            self.q.decay()

            if ep % 100 == 0:
                print(f"Ep {ep} Reward {total} Epsilon {self.q.epsilon:.3f}")

        return rewards

    def save(self, path):
        """Save the current Q-table to a NumPy .npy file."""
        import numpy as np
        np.save(path, self.q.Q)

    def load(self, path):
        """Load a saved Q-table from disk into the agent."""
        import numpy as np
        self.q.Q = np.load(path)

    def run(self, episodes=5):
        """Execute the greedy policy for a fixed number of episodes."""
        for ep in range(episodes):
            obs, _ = self.env.reset()
            s = self.disc.discretize(obs)
            done = False
            total = 0
            while not done:
                a = int(self.q.Q[s].argmax())
                obs, r, term, trunc, _ = self.env.step(a)
                done = term or trunc
                s = self.disc.discretize(obs)
                total += r
            print(f"[RUN] Ep {ep} Reward {total}")
