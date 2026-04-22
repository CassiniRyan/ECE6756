# Torch-based ensemble world model for RWM-Q planning.
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class _DynamicsMember(nn.Module):
    """
    One ensemble member.  Predicts (delta, reward) directly from (s, a).

    The NN learns all dynamics end-to-end — no kinematic baseline prior.
    A prior would force the network to learn near-zero residuals for the
    easy dimensions while the hard ones (velocity) stay unconstrained,
    creating an inconsistent loss landscape.  Plain MSE on the raw delta
    lets the network learn 'dx = v*dt' on its own in a few hundred steps.
    """

    def __init__(self, state_dim, num_actions, device="cpu"):
        super().__init__()
        self.device = device
        self.num_actions = num_actions
        # Normalise the input state only — output delta stays in original units.
        self.state_scale = torch.tensor(
            [4.8, 10.0, 0.418, 10.0], dtype=torch.float32, device=device
        )

        self.net = nn.Sequential(
            nn.Linear(state_dim + num_actions, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, state_dim + 1),
        ).to(device)

        self.lr = 1e-4
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def _encode_action(self, a):
        a_long = a.long().view(-1)
        one_hot = torch.zeros(
            (a_long.shape[0], self.num_actions), dtype=torch.float32, device=self.device
        )
        one_hot.scatter_(1, a_long.unsqueeze(1), 1.0)
        return one_hot

    def forward(self, s_norm, a_onehot):
        return self.net(torch.cat([s_norm, a_onehot], dim=-1))

    def predict(self, s, a):
        self.eval()
        s_t = (
            torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
            / self.state_scale
        )
        a_t = self._encode_action(
            torch.tensor([a], dtype=torch.long, device=self.device)
        )
        with torch.no_grad():
            out = self.forward(s_t, a_t).squeeze(0)
        delta = out[:4].detach().cpu().numpy()
        reward = float(out[4])
        return delta, reward

    def train_batch(self, batch):
        self.train()
        s, a, delta, r = batch
        s_t = (
            torch.tensor(s, dtype=torch.float32, device=self.device)
            / self.state_scale
        )
        a_t = self._encode_action(
            torch.tensor(a, dtype=torch.long, device=self.device)
        )
        delta_t = torch.tensor(delta, dtype=torch.float32, device=self.device)
        r_t = torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(-1)

        pred = self.forward(s_t, a_t)
        loss = self.loss_fn(pred[:, :4], delta_t) + 0.5 * self.loss_fn(pred[:, 4:5], r_t)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        return float(loss.item())

    def reset_optimizer(self):
        """Wipe Adam momentum/variance after a weight rollback."""
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)


class WorldModel:
    """Two-member ensemble dynamics model (PETS-style, minimal)."""

    def __init__(self, state_dim, action_dim, num_actions=2, device="cpu", ensemble_size=2):
        self.device = device
        self.num_actions = num_actions
        self.ensemble_size = ensemble_size
        self.members = [
            _DynamicsMember(state_dim=state_dim, num_actions=num_actions, device=device)
            for _ in range(ensemble_size)
        ]

    def train_batch(self, batch):
        """Each member trains on a bootstrap resample of the batch."""
        s, a, delta, r = batch
        n = len(s)
        losses = []
        for member in self.members:
            idx = np.random.randint(0, n, size=n)
            losses.append(member.train_batch((s[idx], a[idx], delta[idx], r[idx])))
        return float(np.mean(losses))

    def predict_all(self, s, a):
        preds = [m.predict(s, a) for m in self.members]
        deltas = np.array([p[0] for p in preds], dtype=np.float32)
        rewards = np.array([p[1] for p in preds], dtype=np.float32)
        return deltas, rewards

    def predict_with_uncertainty(self, s, a):
        deltas, rewards = self.predict_all(s, a)
        delta_mean = np.mean(deltas, axis=0)
        reward_mean = float(np.mean(rewards))
        delta_disagreement = float(np.mean(np.std(deltas, axis=0)))
        reward_disagreement = float(np.std(rewards))
        return delta_mean, reward_mean, delta_disagreement, reward_disagreement

    def predict_with_uncertainty_by_dim(self, s, a):
        deltas, rewards = self.predict_all(s, a)
        delta_mean = np.mean(deltas, axis=0)
        reward_mean = float(np.mean(rewards))
        delta_std_by_dim = np.std(deltas, axis=0)
        delta_disagreement = float(np.mean(delta_std_by_dim))
        reward_disagreement = float(np.std(rewards))
        return delta_mean, reward_mean, delta_disagreement, reward_disagreement, delta_std_by_dim

    def predict(self, s, a):
        delta_mean, reward_mean, _, _ = self.predict_with_uncertainty(s, a)
        return delta_mean, reward_mean

    def reset_optimizers(self):
        """Reset all member Adam states — must be called after a weight rollback."""
        for m in self.members:
            m.reset_optimizer()

    def export_state(self):
        return [m.state_dict() for m in self.members]

    def load_exported_state(self, states):
        for m, s in zip(self.members, states):
            m.load_state_dict(s)
