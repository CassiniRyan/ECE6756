import random
import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None


class LinearModel:
    """Linear transition model fitted from stored Dyna-style samples.

    Uses ridge regression to improve numerical stability and computes a
    small validation MSE to gate when the model is used for planning.
    """

    def __init__(
        self,
        state_dim=4,
        action_space=2,
        max_samples=4000,
        retrain_every=25,
        reg=1e-2,
        min_samples=300,
        mse_threshold=0.03,
    ):
        self.state_dim = state_dim
        self.action_space = action_space
        self.max_samples = max_samples
        self.retrain_every = retrain_every
        self.reg = float(reg)
        self.min_samples = min_samples
        self.mse_threshold = float(mse_threshold)
        self.store_continuous = True
        self.returns_continuous = True
        self.device = self._select_device()
        self.use_torch_backend = self.device != "cpu"

        self.obs_low = np.array([-4.8, -5.0, -0.418, -5.0], dtype=np.float32)
        self.obs_high = np.array([4.8, 5.0, 0.418, 5.0], dtype=np.float32)
        self.obs_scale = np.maximum(np.abs(self.obs_high), 1e-6)

        self.buffer = []
        self.delta_weights = None  # shape (feature_dim, state_dim)
        self.reward_weights = None  # shape (feature_dim,)
        self.last_mse = None

    def _select_device(self):
        """Use CUDA for the linear algebra backend when PyTorch can access it."""
        if torch is None:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _clip_obs(self, s):
        return np.clip(np.array(s, dtype=np.float32), self.obs_low, self.obs_high)

    def _features(self, s, a):
        # Use normalized observations plus a one-hot action so the linear model
        # can condition its prediction on both state and control input.
        s_arr = self._clip_obs(s) / self.obs_scale
        action_one_hot = np.zeros(self.action_space, dtype=np.float32)
        action_one_hot[a] = 1.0
        return np.concatenate([s_arr, action_one_hot, np.array([1.0], dtype=np.float32)])

    def _fit(self):
        n = len(self.buffer)
        if n < self.min_samples:
            return

        X_np = np.array([self._features(s, a) for s, a, _, _, _ in self.buffer], dtype=np.float32)
        S_np = np.array([s for s, _, _, _, _ in self.buffer], dtype=np.float32)
        Y_next_np = np.array([s_next for _, _, s_next, _, _ in self.buffer], dtype=np.float32)
        # Fit scaled state deltas instead of raw next states to keep the targets
        # in a similar numerical range across state dimensions.
        Y_delta_np = (Y_next_np - S_np) / self.obs_scale
        Y_reward_np = np.array([r for _, _, _, r, _ in self.buffer], dtype=np.float32)

        if self.use_torch_backend:
            self._fit_torch(X_np, Y_delta_np, Y_reward_np)
        else:
            self._fit_numpy(X_np, Y_delta_np, Y_reward_np)

    def _fit_numpy(self, X, Y_delta, Y_reward):
        """Fit the ridge-regression model with NumPy on CPU."""
        # Closed-form ridge regression: W = (X^T X + reg I)^-1 X^T Y
        XtX = X.T @ X
        D = XtX.shape[0]
        reg_matrix = self.reg * np.eye(D, dtype=np.float32)

        try:
            inv = np.linalg.inv(XtX + reg_matrix)
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(XtX + reg_matrix)

        self.delta_weights = inv @ (X.T @ Y_delta)
        self.reward_weights = (inv @ (X.T @ Y_reward)).reshape(-1)
        self.last_mse = self._validation_mse_numpy(X, Y_delta)

    def _fit_torch(self, X, Y_delta, Y_reward):
        """Fit the same ridge-regression model with Torch on GPU when available."""
        # This mirrors the NumPy solver; only the tensor backend changes.
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        Y_delta_t = torch.tensor(Y_delta, dtype=torch.float32, device=self.device)
        Y_reward_t = torch.tensor(Y_reward, dtype=torch.float32, device=self.device)

        XtX = X_t.T @ X_t
        D = XtX.shape[0]
        reg_matrix = self.reg * torch.eye(D, dtype=torch.float32, device=self.device)

        try:
            inv = torch.linalg.inv(XtX + reg_matrix)
        except RuntimeError:
            inv = torch.linalg.pinv(XtX + reg_matrix)

        self.delta_weights = inv @ (X_t.T @ Y_delta_t)
        self.reward_weights = (inv @ (X_t.T @ Y_reward_t)).reshape(-1)
        self.last_mse = self._validation_mse_torch(X_t, Y_delta_t)

    def _validation_mse_numpy(self, X, Y_delta):
        # Hold out the tail of the buffer as a lightweight readiness test.
        split = max(1, int(0.8 * len(X)))
        X_val = X[split:]
        Y_val = Y_delta[split:]
        if len(X_val) > 0:
            preds = X_val @ self.delta_weights
            return float(np.mean((preds - Y_val) ** 2))
        preds = X @ self.delta_weights
        return float(np.mean((preds - Y_delta) ** 2))

    def _validation_mse_torch(self, X, Y_delta):
        # Same readiness heuristic as the NumPy path.
        split = max(1, int(0.8 * X.shape[0]))
        X_val = X[split:]
        Y_val = Y_delta[split:]
        if X_val.shape[0] > 0:
            preds = X_val @ self.delta_weights
            return float(torch.mean((preds - Y_val) ** 2).item())
        preds = X @ self.delta_weights
        return float(torch.mean((preds - Y_delta) ** 2).item())

    def store(self, s, a, s_next, r, done=False):
        """Store a transition and periodically refit the linear predictors.

        We store continuous observations so the model learns continuous dynamics
        (the Agent discretizes model samples before updating the Q-table).
        """
        s_clip = self._clip_obs(s)
        s_next_clip = self._clip_obs(s_next)
        self.buffer.append((s_clip, int(a), s_next_clip, float(r), bool(done)))
        if len(self.buffer) > self.max_samples:
            self.buffer.pop(0)

        if len(self.buffer) >= self.min_samples and len(self.buffer) % self.retrain_every == 0:
            self._fit()

    def sample(self):
        """Generate a synthetic transition from the fitted linear model."""
        if not self.ready():
            raise Exception("Linear model not ready")

        # Start imagination from a real state-action pair so planning stays
        # close to states that were actually visited.
        s, a, _, _, done = random.choice(self.buffer)
        x = self._features(s, a)

        if self.use_torch_backend:
            x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
            delta_pred = (x_t @ self.delta_weights).detach().cpu().numpy() * self.obs_scale
            r_pred = float(torch.clamp(x_t @ self.reward_weights, 0.0, 1.0).item())
        else:
            delta_pred = (x @ self.delta_weights) * self.obs_scale
            r_pred = float(np.clip(x @ self.reward_weights, 0.0, 1.0))

        s_next_pred = self._clip_obs(s + delta_pred)

        return tuple(float(v) for v in s), a, tuple(float(v) for v in s_next_pred), r_pred, done

    def planning_ratio(self, step):
        """Use the linear model like a lightweight one-step Dyna planner."""
        return 1.0

    def ready(self):
        """Return True when the model has enough data, fitted weights, and low validation error."""
        if len(self.buffer) < self.min_samples:
            return False
        if self.delta_weights is None or self.reward_weights is None:
            return False
        if self.last_mse is None:
            return False
        return self.last_mse <= self.mse_threshold
