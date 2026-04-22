# Torch-based ensemble world model for RWM-Q planning.
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class _DynamicsMember(nn.Module):
    """One ensemble member that predicts a one-step state delta and reward."""

    def __init__(self, state_dim, num_actions, device="cpu"):
        super().__init__()
        self.device = device
        self.num_actions = num_actions
        self.dt = 0.02
        # Wider velocity scales avoid crushing the two hard-to-model dimensions.
        self.state_scale = torch.tensor([4.8, 10.0, 0.418, 10.0], dtype=torch.float32, device=device)
        self.delta_scale = torch.tensor([0.25, 0.50, 0.10, 0.80], dtype=torch.float32, device=device)
        # Velocity dims (1, 3) get 3.5x weight: they have no kinematic prior and compound fastest.
        self.delta_loss_weights = torch.tensor([1.0, 3.5, 1.0, 3.5], dtype=torch.float32, device=device)

        self.net = nn.Sequential(
            nn.Linear(state_dim + num_actions, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, state_dim + 1),
        ).to(device)

        self.optimizer = optim.Adam(self.parameters(), lr=3e-4)
        self.loss_fn = nn.MSELoss()

    def _encode_action(self, a):
        a_long = a.long().view(-1)
        a_one_hot = torch.zeros((a_long.shape[0], self.num_actions), dtype=torch.float32, device=self.device)
        a_one_hot.scatter_(1, a_long.unsqueeze(1), 1.0)
        return a_one_hot

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))

    def _baseline_delta(self, s):
        """Cheap kinematic prior for the easy position/angle dimensions."""
        baseline = torch.zeros_like(s)
        baseline[:, 0] = self.dt * s[:, 1]
        baseline[:, 2] = self.dt * s[:, 3]
        return baseline

    def predict(self, s, a):
        self.eval()
        s_t = torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
        s_t = s_t / self.state_scale
        s_raw = torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
        a_t = self._encode_action(torch.tensor([a], dtype=torch.long, device=self.device))

        with torch.no_grad():
            out = self.forward(s_t, a_t).squeeze(0)

        out_np = out.detach().cpu().numpy()
        baseline = self._baseline_delta(s_raw).squeeze(0).detach().cpu().numpy()
        delta = baseline + out_np[:4] * self.delta_scale.detach().cpu().numpy()
        reward = float(out_np[4])
        return delta, reward

    def train_batch(self, batch):
        self.train()
        s, a, delta, r = batch

        s_raw = torch.tensor(s, dtype=torch.float32, device=self.device)
        s_t = s_raw / self.state_scale
        a_t = self._encode_action(torch.tensor(a, dtype=torch.long, device=self.device))
        baseline_delta = self._baseline_delta(s_raw)
        delta_t = (torch.tensor(delta, dtype=torch.float32, device=self.device) - baseline_delta) / self.delta_scale
        r_t = torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(-1)

        pred = self.forward(s_t, a_t)
        pred_delta = pred[:, :4]
        pred_reward = pred[:, 4:5]

        delta_sq_error = (pred_delta - delta_t) ** 2
        weighted_delta_loss = torch.mean(delta_sq_error * self.delta_loss_weights)
        reward_loss = self.loss_fn(pred_reward, r_t)
        loss = weighted_delta_loss + 0.5 * reward_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        return float(loss.item())


class WorldModel:
    """Small ensemble dynamics model in the style of PETS/MBPO."""

    def __init__(self, state_dim, action_dim, num_actions=2, device="cpu", ensemble_size=5):
        self.device = device
        self.action_dim = action_dim
        self.num_actions = num_actions
        self.ensemble_size = ensemble_size
        self.members = [
            _DynamicsMember(state_dim=state_dim, num_actions=num_actions, device=device)
            for _ in range(ensemble_size)
        ]

    def train_batch(self, batch):
        """Train each member on a different bootstrap sample of the same replay batch."""
        s, a, delta, r = batch
        n = len(s)
        losses = []

        for member in self.members:
            indices = np.random.randint(0, n, size=n)
            member_batch = (s[indices], a[indices], delta[indices], r[indices])
            losses.append(member.train_batch(member_batch))

        return float(np.mean(losses))

    def predict_all(self, s, a):
        """Return predictions from every ensemble member."""
        preds = [member.predict(s, a) for member in self.members]
        deltas = np.array([p[0] for p in preds], dtype=np.float32)
        rewards = np.array([p[1] for p in preds], dtype=np.float32)
        return deltas, rewards

    def predict_with_uncertainty(self, s, a):
        """Average the ensemble predictions and estimate epistemic disagreement."""
        deltas, rewards = self.predict_all(s, a)
        delta_mean = np.mean(deltas, axis=0)
        reward_mean = float(np.mean(rewards))
        delta_disagreement = float(np.mean(np.std(deltas, axis=0)))
        reward_disagreement = float(np.std(rewards))
        return delta_mean, reward_mean, delta_disagreement, reward_disagreement

    def predict_with_uncertainty_by_dim(self, s, a):
        """Like predict_with_uncertainty but also returns per-dimension std for targeted filtering."""
        deltas, rewards = self.predict_all(s, a)
        delta_mean = np.mean(deltas, axis=0)
        reward_mean = float(np.mean(rewards))
        delta_std_by_dim = np.std(deltas, axis=0)
        delta_disagreement = float(np.mean(delta_std_by_dim))
        reward_disagreement = float(np.std(rewards))
        return delta_mean, reward_mean, delta_disagreement, reward_disagreement, delta_std_by_dim

    def predict(self, s, a):
        """Compatibility helper returning the ensemble-mean prediction only."""
        delta_mean, reward_mean, _, _ = self.predict_with_uncertainty(s, a)
        return delta_mean, reward_mean

    def export_state(self):
        """Return a serializable copy of all ensemble weights."""
        return [member.state_dict() for member in self.members]

    def load_exported_state(self, exported_state):
        """Load a previously exported ensemble snapshot."""
        for member, state in zip(self.members, exported_state):
            member.load_state_dict(state)
