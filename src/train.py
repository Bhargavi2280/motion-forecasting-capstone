"""One-command reference run:  python -m src.train --out gru_multi.pt
Trains the multi-modal GRU (O2 reference) on train, evaluates on val only."""
import sys, pickle, argparse
sys.path.insert(0, '/content/wod_pb'); sys.path.insert(0, '.')
import numpy as np, torch
from src.models import GRUMulti, wta_loss, make_batch, SCALE
from src.metrics import evaluate

ap = argparse.ArgumentParser()
ap.add_argument('--train', default='/content/data/train_samples.pkl')
ap.add_argument('--val', default='/content/data/val_samples.pkl')
ap.add_argument('--out', default='gru_multi.pt')
ap.add_argument('--epochs', type=int, default=60)
ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args()

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
train = pickle.load(open(a.train, 'rb')); val = pickle.load(open(a.val, 'rb'))
torch.manual_seed(a.seed); np.random.seed(a.seed)
model = GRUMulti().to(dev)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for ep in range(1, a.epochs + 1):
    model.train()
    idx = np.random.permutation(len(train))
    for b in range(0, len(idx), 64):
        X, T, Y = make_batch([train[i] for i in idx[b:b+64]], dev)
        tr, lg = model(X, T)
        loss, _ = wta_loss(tr, lg, Y)
        opt.zero_grad(); loss.backward(); opt.step()
model.eval()
Xv, Tv, _ = make_batch(val, dev)
with torch.no_grad():
    tr, _ = model(Xv, Tv)
evaluate(tr.cpu().numpy() * SCALE, val)
torch.save(model.state_dict(), a.out)
