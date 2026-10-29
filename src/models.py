import torch, torch.nn as nn, numpy as np

SCALE = 10.0  # metres; keeps inputs/outputs near unit range

def make_batch(samples, device):
    hist = np.stack([s['hist'] for s in samples])              # N,11,2
    mask = np.stack([s['hist_mask'] for s in samples])         # N,11
    vel = np.zeros_like(hist); vel[:, 1:] = (hist[:, 1:] - hist[:, :-1]) / 0.1
    x = np.concatenate([hist / SCALE, vel / SCALE, mask[..., None]], -1)  # N,11,5
    x = x * mask[..., None]
    typ = np.zeros((len(samples), 3), np.float32)
    for i, s in enumerate(samples):
        typ[i, min(max(s['obj_type'] - 1, 0), 2)] = 1
    y = np.stack([s['fut'] for s in samples]) / SCALE          # N,80,2
    f = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
    return f(x), f(typ), f(y)

class GRUSingle(nn.Module):
    def __init__(self, hid=128):
        super().__init__()
        self.gru = nn.GRU(5, hid, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hid + 3, 256), nn.ReLU(), nn.Linear(256, 160))
    def forward(self, x, typ):
        _, h = self.gru(x)
        return self.head(torch.cat([h[-1], typ], -1)).view(-1, 80, 2)

class GRUMulti(nn.Module):
    def __init__(self, hid=128, K=6):
        super().__init__()
        self.K = K
        self.gru = nn.GRU(5, hid, batch_first=True)
        self.trunk = nn.Sequential(nn.Linear(hid + 3, 256), nn.ReLU())
        self.traj = nn.Linear(256, K * 160)
        self.prob = nn.Linear(256, K)
    def forward(self, x, typ):
        _, h = self.gru(x)
        z = self.trunk(torch.cat([h[-1], typ], -1))
        return self.traj(z).view(-1, self.K, 80, 2), self.prob(z)

def wta_loss(trajs, logits, y, relax=0.05):
    err = torch.linalg.norm(trajs - y[:, None], dim=-1).mean(-1)      # N,K ADE per mode
    best = err.argmin(1)
    n = torch.arange(len(y), device=y.device)
    reg = (1 - relax) * err[n, best] + relax * err.mean(1)
    cls = nn.functional.cross_entropy(logits, best)
    return reg.mean() + 0.1 * cls, best
