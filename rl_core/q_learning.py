
import numpy as np
class QLearning:
    def __init__(self, action_space):
        self.Q = np.zeros((10,10,10,10,action_space))
        self.alpha = 0.1
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01

    def select_action(self, s, action_space):
        import numpy as np
        if np.random.rand() < self.epsilon:
            return np.random.randint(action_space)
        return int(np.argmax(self.Q[s]))

    def update(self, s, a, r, s_next):
        self.Q[s][a] += self.alpha * (
            r + self.gamma * np.max(self.Q[s_next]) - self.Q[s][a]
        )

    def decay(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
