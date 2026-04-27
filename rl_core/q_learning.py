
"""Tabular Q-learning core shared by all algorithms.

Every experiment eventually updates this same Q-table. The difference between
algorithms is only where extra transitions or action guidance come from.
"""
import numpy as np

class QLearning:
    """Tabular Q-learning implementation for discretized CartPole states."""

    def __init__(self, action_space, bins_per_dim=10):
        # Q-table shape: [position, velocity, angle, angular_velocity, action]
        self.Q = np.zeros((bins_per_dim, bins_per_dim, bins_per_dim, bins_per_dim, action_space))
        self.alpha = 0.1
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01

    def select_action(self, s, action_space):
        """Choose an action using epsilon-greedy exploration from the Q-table.

        Exploration is shared by every algorithm so model-based methods are
        compared against the same basic behavior policy.
        """
        import numpy as np
        if np.random.rand() < self.epsilon:
            return np.random.randint(action_space)
        return int(np.argmax(self.Q[s]))

    def update(self, s, a, r, s_next, done=False):
        """Update a single Q-value using the TD(0) rule.

        Real and imagined transitions both arrive here, which is why model
        quality matters: a bad model writes directly into the same table as real
        experience.
        """
        # Terminal transitions should not bootstrap from future value estimates.
        target = r if done else r + self.gamma * np.max(self.Q[s_next])
        self.Q[s][a] += self.alpha * (
            target - self.Q[s][a]
        )

    def decay(self):
        """Decay the exploration rate after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
