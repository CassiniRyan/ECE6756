# Torch-based structured ensemble world model for RWM-Q planning.
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class _DynamicsMember(nn.Module):
    """One ensemble member that predicts accelerations, then integrates dynamics."""

    def __init__(self, state_dim, num_actions, device="cpu"):
        super().__init__()
        self.device = device
        self.num_actions = num_actions
        self.state_dim = state_dim
        self.history_horizon = 1
        self.dt = 0.02

        self.state_scale = torch.tensor(
            [4.8, 10.0, 0.418, 10.0], dtype=torch.float32, device=device
        )
        self.accel_scale = torch.tensor([15.0, 20.0], dtype=torch.float32, device=device)
        self.next_state_loss_weights = torch.tensor(
            [2.0, 1.5, 2.5, 1.8], dtype=torch.float32, device=device
        )

        # Use hand-shaped features so the model sees pole geometry directly.
        in_dim = 5 + num_actions
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 3),
        ).to(device)

        self.lr = 3e-4
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def _encode_action(self, a):
        a_long = a.long().view(-1)
        one_hot = torch.zeros(
            (a_long.shape[0], self.num_actions), dtype=torch.float32, device=self.device
        )
        one_hot.scatter_(1, a_long.unsqueeze(1), 1.0)
        return one_hot

    def _features(self, state, action):
        x = state[:, 0:1] / self.state_scale[0]
        x_dot = state[:, 1:2] / self.state_scale[1]
        theta = state[:, 2:3]
        theta_dot = state[:, 3:4] / self.state_scale[3]
        geom = torch.cat([x, x_dot, torch.sin(theta), torch.cos(theta), theta_dot], dim=-1)
        return torch.cat([geom, self._encode_action(action)], dim=-1)

    def _integrate(self, state, accel_pred):
        x = state[:, 0:1]
        x_dot = state[:, 1:2]
        theta = state[:, 2:3]
        theta_dot = state[:, 3:4]

        x_acc = accel_pred[:, 0:1] * self.accel_scale[0]
        theta_acc = accel_pred[:, 1:2] * self.accel_scale[1]

        next_x = x + self.dt * x_dot
        next_x_dot = x_dot + self.dt * x_acc
        next_theta = theta + self.dt * theta_dot
        next_theta_dot = theta_dot + self.dt * theta_acc
        next_state = torch.cat([next_x, next_x_dot, next_theta, next_theta_dot], dim=-1)
        return next_state

    def forward(self, state, action):
        out = self.net(self._features(state, action))
        accel_pred = torch.tanh(out[:, :2])
        reward = out[:, 2:3]
        next_state = self._integrate(state, accel_pred)
        return next_state, reward

    def predict(self, state_seq, action_seq):
        self.eval()
        state = torch.tensor(state_seq[-1], dtype=torch.float32, device=self.device).unsqueeze(0)
        action = torch.tensor([action_seq[-1]], dtype=torch.long, device=self.device)
        with torch.no_grad():
            next_state, reward = self.forward(state, action)
        next_state = next_state.squeeze(0).detach().cpu().numpy()
        delta = next_state - np.array(state_seq[-1], dtype=np.float32)
        return delta, float(reward.squeeze(0).item())

    def train_batch(self, batch):
        self.train()
        state_seq, action_seq, delta, reward = batch
        state = torch.tensor(state_seq[:, -1, :], dtype=torch.float32, device=self.device)
        action = torch.tensor(action_seq[:, -1], dtype=torch.long, device=self.device)
        delta_t = torch.tensor(delta, dtype=torch.float32, device=self.device)
        reward_t = torch.tensor(reward, dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_state_true = state + delta_t

        next_state_pred, reward_pred = self.forward(state, action)

        state_error = (next_state_pred - next_state_true) / self.state_scale
        state_loss = torch.mean((state_error ** 2) * self.next_state_loss_weights)
        reward_loss = self.loss_fn(reward_pred, reward_t)
        loss = state_loss + 0.25 * reward_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        return float(loss.item())

    def reset_optimizer(self):
        """Reset optimizer state after loading a best checkpoint."""
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)


class WorldModel:
    """Structured PETS-style ensemble using acceleration prediction."""

    def __init__(
        self,
        state_dim,
        action_dim,
        num_actions=2,
        device="cpu",
        ensemble_size=5,
    ):
        self.device = device
        self.action_dim = action_dim
        self.num_actions = num_actions
        self.ensemble_size = ensemble_size
        self.history_horizon = 1
        self.members = [
            _DynamicsMember(state_dim=state_dim, num_actions=num_actions, device=device)
            for _ in range(ensemble_size)
        ]

    def train_batch(self, batch):
        """Train each member on a bootstrap sample of the same replay batch."""
        state_seq, action_seq, delta, reward = batch
        n = len(state_seq)
        losses = []
        for member in self.members:
            idx = np.random.randint(0, n, size=n)
            member_batch = (
                state_seq[idx],
                action_seq[idx],
                delta[idx],
                reward[idx],
            )
            losses.append(member.train_batch(member_batch))
        return float(np.mean(losses))

    def predict_all(self, state_seq, action_seq):
        preds = [member.predict(state_seq, action_seq) for member in self.members]
        deltas = np.array([p[0] for p in preds], dtype=np.float32)
        rewards = np.array([p[1] for p in preds], dtype=np.float32)
        return deltas, rewards

    def predict_with_uncertainty(self, state_seq, action_seq):
        deltas, rewards = self.predict_all(state_seq, action_seq)
        delta_mean = np.mean(deltas, axis=0)
        reward_mean = float(np.mean(rewards))
        delta_disagreement = float(np.mean(np.std(deltas, axis=0)))
        reward_disagreement = float(np.std(rewards))
        return delta_mean, reward_mean, delta_disagreement, reward_disagreement

    def predict_with_uncertainty_by_dim(self, state_seq, action_seq):
        deltas, rewards = self.predict_all(state_seq, action_seq)
        delta_mean = np.mean(deltas, axis=0)
        reward_mean = float(np.mean(rewards))
        delta_std_by_dim = np.std(deltas, axis=0)
        delta_disagreement = float(np.mean(delta_std_by_dim))
        reward_disagreement = float(np.std(rewards))
        return delta_mean, reward_mean, delta_disagreement, reward_disagreement, delta_std_by_dim

    def predict(self, state_seq, action_seq):
        delta_mean, reward_mean, _, _ = self.predict_with_uncertainty(state_seq, action_seq)
        return delta_mean, reward_mean

    def reset_optimizers(self):
        for member in self.members:
            member.reset_optimizer()

    def export_state(self):
        return [member.state_dict() for member in self.members]

    def load_exported_state(self, states):
        for member, state in zip(self.members, states):
            member.load_state_dict(state)
