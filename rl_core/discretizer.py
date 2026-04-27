
"""Observation discretization for tabular Q-learning.

CartPole observations are continuous, but the project uses a tabular Q-table.
This helper is the boundary between continuous simulators and discrete control.
"""
import numpy as np

class Discretizer:
    """Convert continuous CartPole observations into discrete state indices."""

    def __init__(self, bins_per_dim=10):
        # Define fixed bins for each of the 4 observation dimensions.
        # More bins give finer control but make the Q-table much larger.
        edge_count = bins_per_dim - 1
        self.bins = [
            np.linspace(-4.8, 4.8, edge_count),
            np.linspace(-5, 5, edge_count),
            np.linspace(-0.418, 0.418, edge_count),
            np.linspace(-5, 5, edge_count),
        ]

    def discretize(self, obs):
        """Map each continuous observation value into a discrete bin index."""
        # The tuple of bin indices becomes the state key used by tabular methods.
        return tuple(np.digitize(o, b) for o, b in zip(obs, self.bins))
