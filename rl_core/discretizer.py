
import numpy as np

class Discretizer:
    """Convert continuous CartPole observations into discrete state indices."""

    def __init__(self):
        # Define fixed bins for each of the 4 observation dimensions.
        self.bins = [
            np.linspace(-4.8, 4.8, 9),
            np.linspace(-5, 5, 9),
            np.linspace(-0.418, 0.418, 9),
            np.linspace(-5, 5, 9),
        ]

    def discretize(self, obs):
        """Map each continuous observation value into a discrete bin index."""
        return tuple(np.digitize(o, b) for o, b in zip(obs, self.bins))
