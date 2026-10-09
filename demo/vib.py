"""Head VIB va baseline tuyen tinh tren dac trung CLIP (768-d) da cache.

VIBHead lap lai kien truc trong VIBAIGCDetect/models/clip_models.py (CLIPModel):
dropout -> fc_1 -> ReLU -> fc_2 -> ReLU -> fc_3 -> (mu, std=softplus(.-5)) -> z -> decode.
Phan KL + beta la phan bai bao co nhung repo khong cong bo (repo chi co code test).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

K = 256


class VIBHead(nn.Module):
    def __init__(self, in_dim=768, k=K):
        super().__init__()
        self.k = k
        self.dropout = nn.Dropout(0.5)
        self.fc_1 = nn.Linear(in_dim, 1024)
        self.fc_2 = nn.Linear(1024, 1024)
        self.fc_3 = nn.Linear(1024, 2 * k)
        self.decode = nn.Linear(k, 1)

    def encode(self, x):
        h = F.relu(self.fc_1(self.dropout(x)))
        h = F.relu(self.fc_2(h))
        s = self.fc_3(h)
        return s[:, :self.k], F.softplus(s[:, self.k:] - 5)

    def forward(self, x, sample=True):
        mu, std = self.encode(x)
        z = mu + torch.randn_like(std) * std if sample else mu
        return mu, std, self.decode(z).squeeze(1)


class LinearHead(nn.Module):
    """Baseline kieu Univfd: mot lop tuyen tinh tren dac trung CLIP."""

    def __init__(self, in_dim=768):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x, sample=True):
        return None, None, self.fc(x).squeeze(1)


def kl_to_std_normal(mu, std):
    # KL( N(mu, std^2) || N(0, I) ), tong theo chieu, trung binh theo batch
    return (0.5 * (mu ** 2 + std ** 2 - 1 - 2 * torch.log(std + 1e-8))).sum(1).mean()


def train_head(model, x, y, beta=0.0, epochs=60, lr=2e-4, bs=64, seed=0, history=None):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x, y = torch.as_tensor(x), torch.as_tensor(y, dtype=torch.float32)
    n = len(x)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            mu, std, logit = model(x[idx])
            loss = F.binary_cross_entropy_with_logits(logit, y[idx])
            if mu is not None and beta > 0:
                loss = loss + beta * kl_to_std_normal(mu, std)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        if history is not None:
            history.append(tot / n)
    return tot / n  # loss epoch cuoi


@torch.no_grad()
def predict(model, x):
    model.eval()
    _, _, logit = model(torch.as_tensor(x), sample=False)  # dung mu, khong lay mau -> on dinh
    return torch.sigmoid(logit).numpy()


def ap_acc(y, p, thr=0.5):
    return average_precision_score(y, p), float(((p > thr) == (y == 1)).mean())
