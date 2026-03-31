# Simple neural world model for predicting state deltas and reward.
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class SimpleWorldModel(nn.Module):
    """Neural network that learns to predict environment transitions and reward."""
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, state_dim + 1)
        )

        # 🔥 lower LR = more stable
        self.optimizer = optim.Adam(self.parameters(), lr=5e-4)
        self.loss_fn = nn.MSELoss()

    def forward(self, s, a):
        """Concatenate state and action tensors, then run the network."""
        x = torch.cat([s, a], dim=-1)
        return self.net(x)

    def predict(self, s, a):
        """Run the model in evaluation mode and return predicted delta and reward."""
        self.eval()

        s = torch.tensor(s / 10.0, dtype=torch.float32).unsqueeze(0)
        a = torch.tensor([a], dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            out = self.forward(s, a)

        out = out.squeeze(0).numpy()

        delta = out[:-1]
        r = out[-1]

        return delta, r

    def train_step(self, s, a, delta, r):
        """Single step training"""
        self.train()

        s = torch.tensor(s / 10.0, dtype=torch.float32)
        a = torch.tensor([a], dtype=torch.float32)
        delta = torch.tensor(delta, dtype=torch.float32)
        r = torch.tensor([r], dtype=torch.float32)

        target = torch.cat([delta, r])

        pred = self.forward(s.unsqueeze(0), a.unsqueeze(0)).squeeze(0)

        loss = self.loss_fn(pred, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train_batch(self, batch_data):
        """Batch training"""
        self.train()
        
        s_b, a_b, delta_b, r_b = batch_data
        
        s_t = torch.tensor(s_b / 10.0, dtype=torch.float32)
        a_t = torch.tensor(a_b, dtype=torch.float32).unsqueeze(1) if len(a_b.shape) == 1 else torch.tensor(a_b, dtype=torch.float32)
        delta_t = torch.tensor(delta_b, dtype=torch.float32)
        r_t = torch.tensor(r_b, dtype=torch.float32).unsqueeze(1) if len(r_b.shape) == 1 else torch.tensor(r_b, dtype=torch.float32)

        target = torch.cat([delta_t, r_t], dim=1)

        pred = self.forward(s_t, a_t)

        loss = self.loss_fn(pred, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()