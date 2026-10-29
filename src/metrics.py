import numpy as np

MISS_THR = 4.0  # metres at the 8 s endpoint (frozen definition for this project)
TAGS = ["lane_change", "sudden_stop", "hard_brake", "extreme_turn", "vulnerable"]

def evaluate(pred, samples, miss_thr=MISS_THR):
    """pred: (N,K,80,2) in each sample's local frame."""
    gt = np.stack([s['fut'] for s in samples])
    err = np.linalg.norm(pred - gt[:, None], axis=-1)      # N,K,80
    minade = err.mean(-1).min(1)
    minfde = err[..., -1].min(1)
    miss = minfde > miss_thr
    rows = {"all": np.arange(len(samples))}
    for t in TAGS:
        rows[t] = np.array([i for i, s in enumerate(samples) if t in s['tags']])
    rare = [i for i, s in enumerate(samples) if any(t != 'vulnerable' for t in s['tags'])]
    rows["any_rare"] = np.array(rare)
    print(f"{'subset':13s}{'n':>6s}{'minADE':>9s}{'minFDE':>9s}{'miss%':>8s}{'coverage%':>11s}")
    out = {}
    for name, idx in rows.items():
        if len(idx) == 0: continue
        r = dict(n=len(idx), minADE=minade[idx].mean(), minFDE=minfde[idx].mean(),
                 miss=100 * miss[idx].mean(), coverage=100 * (1 - miss[idx].mean()))
        out[name] = r
        print(f"{name:13s}{r['n']:6d}{r['minADE']:9.2f}{r['minFDE']:9.2f}{r['miss']:8.1f}{r['coverage']:11.1f}")
    return out
