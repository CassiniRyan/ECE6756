import random
import numpy as np

class RWMModel:
    def __init__(self, model):
        self.model = model
        self.buffer = []

    def store(self, s, a, s_next, r):
        s = np.array(s, dtype=np.float32)
        s_next = np.array(s_next, dtype=np.float32)

        delta = s_next - s
        self.buffer.append((s, a, delta, r))

        if len(self.buffer) > 500:
            for _ in range(5):
                batch = random.sample(self.buffer, 64)

                s_b = np.array([b[0] for b in batch])
                a_b = np.array([b[1] for b in batch])
                delta_b = np.array([b[2] for b in batch])
                r_b = np.array([b[3] for b in batch])

                self.model.train_batch((s_b, a_b, delta_b, r_b))

    def sample(self):
        if len(self.buffer) < 200:
            raise Exception("Not enough data")

        s, a, _, _ = random.choice(self.buffer)

        delta_pred, r = self.model.predict(s, a)

        s_next = np.array(s) + delta_pred
        s_next = tuple(np.clip(np.round(s_next), 0, 9).astype(int))

        return tuple(s.astype(int)), a, s_next, float(r)

    def ready(self):
        return len(self.buffer) > 3000