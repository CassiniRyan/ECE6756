import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class SimpleWorldModel(nn.Module):
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
        x = torch.cat([s, a], dim=-1)
        return self.net(x)

    def predict(self, s, a):
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