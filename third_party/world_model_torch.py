# Torch-based ensemble world model for RWM-Q planning.
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class _DynamicsMember(nn.Module):
    """
    One ensemble member. Predicts (delta, reward) from (s, a).

    Design choices:
    - No kinematic integration layer: integrating inside the model forces the
      NN to zero-out accelerations to minimise loss variance, causing mean
      collapse on x and x_dot (confirmed by the scatter plot).
    - sin/cos encoding for theta: circular geometry — the pole at +0.2 rad and
      -0.2 rad has symmetric dynamics; raw theta breaks that symmetry for the NN.
    - CartPole has exact kinematics for x/theta over one step, so the network
      learns only the harder velocity deltas. This avoids wasting capacity on
      an identity-like relationship and keeps the prediction scatter aligned
      with the y=x reference line.
    - lr=1e-4: 3e-4 caused divergence over long runs with continuous training.
    """

    def __init__(self, state_dim, num_actions, device="cpu"):
        super().__init__()
        self.device = device
        self.num_actions = num_actions
        self.state_scale = torch.tensor(
            [4.8, 10.0, 1.0, 10.0], dtype=torch.float32, device=device
        )
        self.delta_scale = torch.tensor(
            [1.0, 0.50, 1.0, 0.50], dtype=torch.float32, device=device
        )
        self.tau = 0.02

        # 5 geometric features + one-hot action
        # [x/4.8, x_dot/10, sin(theta), cos(theta), theta_dot/10] + [a0, a1]
        in_dim = 5 + num_actions
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, state_dim + 1),  # 4 delta dims + 1 reward
        ).to(device)

        self.lr = 1e-4
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.loss_fn = nn.SmoothL1Loss()

    def _encode_action(self, a):
        a_long = a.long().view(-1)
        one_hot = torch.zeros(
            (a_long.shape[0], self.num_actions), dtype=torch.float32, device=self.device
        )
        one_hot.scatter_(1, a_long.unsqueeze(1), 1.0)
        return one_hot

    def _features(self, state):
        """Geometric feature encoding — sin/cos for theta fixes circularity."""
        x      = state[:, 0:1] / self.state_scale[0]
        x_dot  = state[:, 1:2] / self.state_scale[1]
        theta  = state[:, 2:3]                         # raw radians for trig
        th_dot = state[:, 3:4] / self.state_scale[3]
        return torch.cat([x, x_dot, torch.sin(theta), torch.cos(theta), th_dot], dim=-1)

    def forward(self, state, action):
        feat = self._features(state)
        a_enc = self._encode_action(action)
        return self.net(torch.cat([feat, a_enc], dim=-1))

    def predict(self, state_seq, action_seq):
        """state_seq: [H, state_dim], action_seq: [H]  (H=history_horizon, currently 1)"""
        self.eval()
        s = torch.tensor(
            state_seq[-1], dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        a = torch.tensor([action_seq[-1]], dtype=torch.long, device=self.device)
        with torch.no_grad():
            out = self.forward(s, a).squeeze(0)
        delta = (out[:4] * self.delta_scale).detach().cpu().numpy()
        delta[0] = self.tau * float(state_seq[-1][1])
        delta[2] = self.tau * float(state_seq[-1][3])
        reward = float(out[4])
        return delta, reward

    def train_batch(self, batch):
        self.train()
        state_seq, action_seq, delta, reward = batch

        # Use the most recent state in each sequence (history_horizon may be >1 later)
        state = torch.tensor(
            state_seq[:, -1, :], dtype=torch.float32, device=self.device
        )
        action = torch.tensor(
            action_seq[:, -1], dtype=torch.long, device=self.device
        )
        delta_t = torch.tensor(delta, dtype=torch.float32, device=self.device)
        delta_t_scaled = delta_t / self.delta_scale
        reward_t = torch.tensor(reward, dtype=torch.float32, device=self.device).unsqueeze(-1)

        out = self.forward(state, action)
        pred_velocity_delta_scaled = out[:, [1, 3]]
        pred_reward = out[:, 4:5]

        loss = (
            self.loss_fn(pred_velocity_delta_scaled, delta_t_scaled[:, [1, 3]])
            + 0.05 * self.loss_fn(pred_reward, reward_t)
        )

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        return float(loss.item())

    def reset_optimizer(self):
        """Reset Adam state after a weight rollback."""
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)


class WorldModel:
    """Two-member ensemble — minimum needed for disagreement-based filtering."""

    def __init__(self, state_dim, action_dim, num_actions=2, device="cpu", ensemble_size=2):
        self.device = device
        self.num_actions = num_actions
        self.ensemble_size = ensemble_size
        self.history_horizon = 1
        self.members = [
            _DynamicsMember(state_dim=state_dim, num_actions=num_actions, device=device)
            for _ in range(ensemble_size)
        ]

    def train_batch(self, batch):
        state_seq, action_seq, delta, reward = batch
        n = len(state_seq)
        losses = []
        for m in self.members:
            idx = np.random.randint(0, n, size=n)
            losses.append(m.train_batch(
                (state_seq[idx], action_seq[idx], delta[idx], reward[idx])
            ))
        return float(np.mean(losses))

    def predict_all(self, state_seq, action_seq):
        preds   = [m.predict(state_seq, action_seq) for m in self.members]
        deltas  = np.array([p[0] for p in preds], dtype=np.float32)
        rewards = np.array([p[1] for p in preds], dtype=np.float32)
        return deltas, rewards

    def predict_with_uncertainty(self, state_seq, action_seq):
        deltas, rewards = self.predict_all(state_seq, action_seq)
        return (
            np.mean(deltas, axis=0),
            float(np.mean(rewards)),
            float(np.mean(np.std(deltas, axis=0))),
            float(np.std(rewards)),
        )

    def predict_with_uncertainty_by_dim(self, state_seq, action_seq):
        deltas, rewards = self.predict_all(state_seq, action_seq)
        delta_std = np.std(deltas, axis=0)
        return (
            np.mean(deltas, axis=0),
            float(np.mean(rewards)),
            float(np.mean(delta_std)),
            float(np.std(rewards)),
            delta_std,
        )

    def predict(self, state_seq, action_seq):
        d, r, _, _ = self.predict_with_uncertainty(state_seq, action_seq)
        return d, r

    def reset_optimizers(self):
        for m in self.members:
            m.reset_optimizer()

    def export_state(self):
        return [m.state_dict() for m in self.members]

    def load_exported_state(self, states):
        for m, s in zip(self.members, states):
            m.load_state_dict(s)
