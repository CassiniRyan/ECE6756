
from rl_core.discretizer import Discretizer
from rl_core.q_learning import QLearning

class Agent:
    def __init__(self, env, model=None, planning_steps=0):
        self.env = env
        self.disc = Discretizer()
        self.q = QLearning(env.action_space.n)
        self.model = model
        self.planning_steps = planning_steps

    def train(self, episodes):
        rewards = []

        for ep in range(episodes):
            obs, _ = self.env.reset()
            s = self.disc.discretize(obs)
            done = False
            total = 0

            while not done:
                a = self.q.select_action(s, self.env.action_space.n)
                obs_next, r, term, trunc, _ = self.env.step(a)
                done = term or trunc
                s_next = self.disc.discretize(obs_next)

                # --- always do real update ---
                self.q.update(s, a, r, s_next)

                if self.model is not None:
                    self.model.store(s, a, s_next, r)

                    # 🔥 WARM-UP: skip model usage early
                    if not self.model.ready():
                        s = s_next
                        total += r
                        continue

                    # --- model planning ---
                    print(f"[PLANNING] Ep {ep} running {self.planning_steps} planning steps")
                    for _ in range(self.planning_steps):
                        try:
                            s_m, a_m, s_next_m, r_m = self.model.sample()
                            self.q.update(s_m, a_m, r_m, s_next_m)
                        except Exception as e:
                            print("Model error:", e)
                            break

                s = s_next
                total += r

                
            rewards.append(total)

            self.q.decay()
            rewards.append(total)
            # log every episode for easier comparison
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
