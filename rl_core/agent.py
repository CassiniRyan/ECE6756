
# Agent logic connecting the environment, the policy, and any planning model.
from rl_core.discretizer import Discretizer
from rl_core.q_learning import QLearning
import random

class Agent:
    """Agent wrapper for training, saving, loading, and policy evaluation."""

    def __init__(self, env, model=None, planning_steps=0):
        self.env = env
        self.disc = Discretizer()
        self.q = QLearning(env.action_space.n)
        self.model = model
        self.planning_steps = planning_steps
        self.global_step = 0  # Total number of environment steps taken.

    def model_ratio(self, step):
        """Return the fraction of planning updates to apply based on step count."""
        if step < 3000:
            # No model-based planning until enough real experience is gathered.
            return 0.0
        elif step < 6000:
            # Gradually ramp up model usage from 0 to 50%.
            return 0.5 * (step - 3000) / 3000
        else:
            # After initial burn-in, use the model at a fixed partial rate.
            return 0.5

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
                self.q.update(s, a, r, s_next)

                # Store the real transition in the model if available.
                if self.model is not None:
                    self.model.store(s, a, s_next, r)

                # Optionally perform additional model-based planning updates.
                ratio = self.model_ratio(self.global_step)
                if self.model is not None and ratio > 0.0 and self.model.ready():
                    model_updates = int(self.planning_steps * ratio)
                    if model_updates > 0:
                        for _ in range(model_updates):
                            try:
                                s_m, a_m, s_next_m, r_m = self.model.sample()
                                self.q.update(s_m, a_m, r_m, s_next_m)
                            except Exception:
                                # Ignore sampling failures if the model is not ready.
                                pass

                s = s_next
                total += r
                self.global_step += 1

                if self.global_step % 1000 == 0:
                    print(f"[Step {self.global_step}] model_ratio={ratio:.3f}")

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
