import random
import numpy as np


class LinearModel:
    """Linear transition model fitted from stored Dyna-style samples.

    Uses ridge regression to improve numerical stability and computes a
    small validation MSE to gate when the model is used for planning.
    """

    def __init__(
        self,
        state_dim=4,
        action_space=2,
        max_state_index=15,
        max_samples=4000,
        retrain_every=50,
        reg=1e-3,
        min_samples=400,
        mse_threshold=0.5,
    ):
        self.state_dim = state_dim
        self.action_space = action_space
        self.max_state_index = max_state_index
        self.max_samples = max_samples
        self.retrain_every = retrain_every
        self.reg = float(reg)
        self.min_samples = min_samples
        self.mse_threshold = float(mse_threshold)

        self.buffer = []
        self.state_weights = None  # shape (feature_dim, state_dim)
        self.reward_weights = None  # shape (feature_dim,)
        self.last_mse = None

    def _features(self, s, a):
        s_arr = np.array(s, dtype=np.float32)
        action_one_hot = np.zeros(self.action_space, dtype=np.float32)
        action_one_hot[a] = 1.0
        return np.concatenate([s_arr, action_one_hot, np.array([1.0], dtype=np.float32)])

    def _fit(self):
        n = len(self.buffer)
        if n < self.min_samples:
            return

        X = np.array([self._features(s, a) for s, a, _, _ in self.buffer], dtype=np.float32)
        Y_state = np.array([s_next for _, _, s_next, _ in self.buffer], dtype=np.float32)
        Y_reward = np.array([r for _, _, _, r in self.buffer], dtype=np.float32)

        # Ridge solution: W = (X^T X + reg I)^{-1} X^T Y
        XtX = X.T @ X
        D = XtX.shape[0]
        reg_matrix = self.reg * np.eye(D, dtype=np.float32)

        try:
            inv = np.linalg.inv(XtX + reg_matrix)
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(XtX + reg_matrix)

        self.state_weights = inv @ (X.T @ Y_state)
        self.reward_weights = (inv @ (X.T @ Y_reward)).reshape(-1)

        # compute validation MSE on last 20% of buffer to estimate accuracy
        split = max(1, int(0.8 * n))
        X_val = X[split:]
        Y_val = Y_state[split:]
        if len(X_val) > 0:
            preds = X_val @ self.state_weights
            mse = float(np.mean((preds - Y_val) ** 2))
            self.last_mse = mse
        else:
            # if no validation split, compute MSE on training set
            preds = X @ self.state_weights
            self.last_mse = float(np.mean((preds - Y_state) ** 2))

        # reward mse (scalar) stored for debugging (not used for gating currently)
        # r_preds = X @ self.reward_weights
        # self.reward_mse = float(np.mean((r_preds - Y_reward) ** 2))

    def store(self, s, a, s_next, r):
        """Store a transition and periodically refit the linear predictors.

        We store continuous observations so the model learns continuous dynamics
        (the Agent discretizes model samples before updating the Q-table).
        """
        self.buffer.append((np.array(s, dtype=np.float32), int(a), np.array(s_next, dtype=np.float32), float(r)))
        if len(self.buffer) > self.max_samples:
            self.buffer.pop(0)

        if len(self.buffer) >= self.min_samples and len(self.buffer) % self.retrain_every == 0:
            self._fit()

    def sample(self):
        """Generate a synthetic transition from the fitted linear model."""
        if not self.ready():
            raise Exception("Linear model not ready")

        s, _, _, _ = random.choice(self.buffer)
        a = np.random.randint(self.action_space)
        x = self._features(s, a)

        # predict continuous next observation and reward
        s_next_pred = x @ self.state_weights
        r_pred = float(x @ self.reward_weights)

        # return continuous states (Agent will discretize when needed)
        return tuple(float(v) for v in s), a, tuple(float(v) for v in s_next_pred), r_pred

    def ready(self):
        """Return True when the model has enough data, fitted weights, and low validation error."""
        if len(self.buffer) < self.min_samples:
            return False
        if self.state_weights is None or self.reward_weights is None:
            return False
        if self.last_mse is None:
            return False
        return self.last_mse <= self.mse_threshold
