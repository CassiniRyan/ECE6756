import random
from collections import defaultdict

class TabularModel:
    def __init__(self):
        # (s, a) -> list of (s_next, r)
        self.memory = defaultdict(list)

        # max transitions per (s,a)
        self.max_per_state = 3

    def store(self, s, a, s_next, r):
        """
        Store transition (s, a) -> (s_next, r)
        Keep only recent transitions to reduce inconsistency
        """
        self.memory[(s, a)].append((s_next, r))

        # limit memory size per (s,a)
        if len(self.memory[(s, a)]) > self.max_per_state:
            self.memory[(s, a)].pop(0)

    def sample(self):
        """
        Sample a valid (s, a, s_next, r) transition
        Only sample from sufficiently populated entries
        """
        # filter keys with enough data
        valid_keys = [k for k in self.memory.keys() if len(self.memory[k]) > 0]

        if not valid_keys:
            raise Exception("Model memory empty")

        s, a = random.choice(valid_keys)
        s_next, r = random.choice(self.memory[(s, a)])

        return s, a, s_next, r
    
    def ready(self):
        return len(self.memory) > 50