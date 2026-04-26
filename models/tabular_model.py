# Tabular model that stores recent transitions for Dyna-style planning.
import random
from collections import defaultdict

class TabularModel:
    """Simple memory-based model of environment transitions."""
    def __init__(self, max_per_state=8):
        # (s, a) -> list of (s_next, r, done)
        self.memory = defaultdict(list)

        # max transitions per (s,a)
        self.max_per_state = max_per_state

    def store(self, s, a, s_next, r, done=False):
        """Store a transition for a discrete state-action pair."""
        self.memory[(s, a)].append((s_next, r, bool(done)))

        # limit memory size per (s,a)
        if len(self.memory[(s, a)]) > self.max_per_state:
            self.memory[(s, a)].pop(0)

    def sample(self):
        """Sample an experience tuple from the stored transition memory."""
        # Sample from previously observed transitions rather than predicting.
        valid_keys = [k for k in self.memory.keys() if len(self.memory[k]) > 0]

        if not valid_keys:
            raise Exception("Model memory empty")

        s, a = random.choice(valid_keys)
        s_next, r, done = random.choice(self.memory[(s, a)])

        return s, a, s_next, r, done

    def planning_ratio(self, step):
        """Classic tabular Dyna can use planning immediately once samples exist."""
        return 1.0
    
    def ready(self):
        """Return True when the model has enough stored transitions to sample from."""
        return len(self.memory) > 50
