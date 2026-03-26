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
    
    def __init__(self, model):
        self.model = model
        self.buffer = []
        self.training_steps = 0

    def store(self, s, a, s_next, r):
        """
        Store real transition from environment.
        Also trains model in background.
        """
        s = np.array(s, dtype=np.float32)
        s_next = np.array(s_next, dtype=np.float32)

        # Compute delta (change in state)
        delta = s_next - s

        # Store in buffer
        self.buffer.append((s, a, delta, r))

        # Train model every 500 transitions
        if len(self.buffer) > 500:
            for _ in range(5):
                batch = random.sample(self.buffer, 64)

                s_b = np.array([b[0] for b in batch])
                a_b = np.array([b[1] for b in batch])
                delta_b = np.array([b[2] for b in batch])
                r_b = np.array([b[3] for b in batch])

                self.model.train_batch((s_b, a_b, delta_b, r_b))
                self.training_steps += 1

    def sample(self):
        """
        Generate a FAKE transition using the model.
        This is different from reality and should only be used
        for model-based planning AFTER the model is trained.
        
        Returns:
            (s, a, s_next, r) - but s_next is PREDICTED, not real
        """
        if len(self.buffer) < 200:
            raise Exception(f"Not ready: buffer has {len(self.buffer)}, need 200")

        # Pick a random real state
        s, a_sample, _, _ = random.choice(self.buffer)

        # Pick a random action (exploration)
        a = np.random.randint(0, 2)  # 0 or 1 for CartPole

        # Use model to predict next state and reward
        delta_pred, r_pred = self.model.predict(s, a)

        # Reconstruct state (add delta to original)
        s_next = np.array(s) + delta_pred
        s_next = tuple(np.clip(np.round(s_next), 0, 9).astype(int))

        return tuple(s.astype(int)), a, s_next, float(r_pred)

    def ready(self):
        """Model is ready after 200+ transitions stored"""
        return len(self.buffer) > 200