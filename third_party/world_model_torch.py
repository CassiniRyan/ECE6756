import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class WorldModel(nn.Module):
    def __init__(self, state_dim, action_dim, device="cpu"):
        super().__init__()

        self.device = device

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, state_dim + 1)
        ).to(device)

        self.optimizer = optim.Adam(self.parameters(), lr=3e-4)
        self.loss_fn = nn.MSELoss()

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return self.net(x)

    def predict(self, s, a):
        self.eval()

        s = torch.tensor(s / 10.0, dtype=torch.float32).to(self.device).unsqueeze(0)
        a = torch.tensor([[a]], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            out = self.forward(s, a)

        out = out.squeeze(0).cpu().numpy()

        delta = out[:-1]
        r = out[-1]

        return delta, r

    def train_batch(self, batch):
        self.train()

        s, a, delta, r = batch

        s = torch.tensor(s / 10.0, dtype=torch.float32).to(self.device)
        a = torch.tensor(a, dtype=torch.float32).unsqueeze(-1).to(self.device)
        delta = torch.tensor(delta, dtype=torch.float32).to(self.device)
        r = torch.tensor(r, dtype=torch.float32).unsqueeze(-1).to(self.device)

        target = torch.cat([delta, r], dim=-1)

        pred = self.forward(s, a)

        loss = self.loss_fn(pred, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()