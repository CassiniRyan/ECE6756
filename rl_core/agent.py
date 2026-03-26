
from rl_core.discretizer import Discretizer
from rl_core.q_learning import QLearning
import random

class Agent:
    def __init__(self, env, model=None, planning_steps=0):
        self.env = env
        self.disc = Discretizer()
        self.q = QLearning(env.action_space.n)
        self.model = model
        self.planning_steps = planning_steps
        self.global_step = 0  # Track total steps (not episodes)

    def model_ratio(self, step):
        """
        Hard gating + gradual model usage.
        
        - Before step 3000: NO model (ratio = 0.0)
        - Step 3000-6000: Gradually increase from 0 to 0.5
        - After step 6000: Fixed at 0.5
        """
        if step < 3000:
            return 0.0
        elif step < 6000:
            return 0.5 * (step - 3000) / 3000
        else:
            return 0.5

    def train(self, episodes):
        rewards = []

        for ep in range(episodes):
            obs, _ = self.env.reset()
            s = self.disc.discretize(obs)
            done = False
            total = 0

            while not done:
                # --- STEP 1: Get action and real transition ---
                a = self.q.select_action(s, self.env.action_space.n)
                obs_next, r, term, trunc, _ = self.env.step(a)
                done = term or trunc
                s_next = self.disc.discretize(obs_next)

                # --- STEP 2: Always update Q with REAL transition ---
                self.q.update(s, a, r, s_next)

                # --- STEP 3: Store transition and train model (if enabled) ---
                if self.model is not None:
                    self.model.store(s, a, s_next, r)

                # --- STEP 4: Model-based training with hard gating ---
                ratio = self.model_ratio(self.global_step)
                
                if self.model is not None and ratio > 0.0 and self.model.ready():
                    # Perform model-based updates
                    model_updates = int(self.planning_steps * ratio)
                    
                    if model_updates > 0:
                        for _ in range(model_updates):
                            try:
                                s_m, a_m, s_next_m, r_m = self.model.sample()
                                self.q.update(s_m, a_m, r_m, s_next_m)
                            except Exception as e:
                                pass  # Silent fail if not enough data

                s = s_next
                total += r
                self.global_step += 1

                # Debug logging every 1000 steps
                if self.global_step % 1000 == 0:
                    print(f"[Step {self.global_step}] model_ratio={ratio:.3f}")

            # --- Episode cleanup ---
            rewards.append(total)
            self.q.decay()
            
            if ep % 100 == 0:
                print(f"Ep {ep} Reward {total} Epsilon {self.q.epsilon:.3f}")
        
        return rewards

    def save(self, path):
        import numpy as np
        np.save(path, self.q.Q)

    def load(self, path):
        import numpy as np
        self.q.Q = np.load(path)

    def run(self, episodes=5):
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
