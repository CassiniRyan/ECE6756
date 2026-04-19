# Tabular model that stores recent transitions for Dyna-style planning.
import random
from collections import defaultdict

class TabularModel:
    """Simple memory-based model of environment transitions."""
    def __init__(self, max_per_state=8):
        # (s, a) -> list of (s_next, r)
        self.memory = defaultdict(list)

        # max transitions per (s,a)
        self.max_per_state = max_per_state

    def store(self, s, a, s_next, r):
        """Store a transition for a discrete state-action pair."""
        self.memory[(s, a)].append((s_next, r))

        # limit memory size per (s,a)
        if len(self.memory[(s, a)]) > self.max_per_state:
            self.memory[(s, a)].pop(0)

    def sample(self):
        """Sample an experience tuple from the stored transition memory."""
        # filter keys with enough data
        valid_keys = [k for k in self.memory.keys() if len(self.memory[k]) > 0]

        if not valid_keys:
            raise Exception("Model memory empty")

        s, a = random.choice(valid_keys)
        s_next, r = random.choice(self.memory[(s, a)])

        return s, a, s_next, r
    
    def ready(self):
        """Return True when the model has enough stored transitions to sample from."""
        return len(self.memory) > 50
