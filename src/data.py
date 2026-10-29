import sys, glob, pickle
sys.path.insert(0, '/content/wod_pb')
import numpy as np
import tensorflow as tf
from waymo_open_dataset.protos import scenario_pb2
from src.rare_tags import tag_trajectory

def track_array(t):
    return np.array([[s.center_x, s.center_y, s.heading, s.velocity_x, s.velocity_y, float(s.valid)]
                     for s in t.states], dtype=np.float32)

def to_local(xy, origin, heading):
    c, s = np.cos(-heading), np.sin(-heading)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    return (xy - origin) @ R.T

def extract_samples(sc, max_neighbors=8):
    cur = sc.current_time_index
    arrs = [track_array(t) for t in sc.tracks]
    out = []
    for ttp in sc.tracks_to_predict:
        i = ttp.track_index
        a = arrs[i]
        if not a[cur, 5] or a[cur + 1:, 5].sum() < len(a) - cur - 1:
            continue  # need valid current step and full future
        origin, h = a[cur, :2], a[cur, 2]
        hist = to_local(a[:cur + 1, :2], origin, h)
        fut = to_local(a[cur + 1:, :2], origin, h)
        speed = np.linalg.norm(a[cur:, 3:5], axis=1)
        tags = tag_trajectory(np.vstack([hist[-1:], fut]), speed, sc.tracks[i].object_type)
        others = [j for j in range(len(arrs)) if j != i and arrs[j][cur, 5]]
        others.sort(key=lambda j: np.linalg.norm(arrs[j][cur, :2] - origin))
        nb = np.zeros((max_neighbors, cur + 1, 2), np.float32)
        nb_mask = np.zeros((max_neighbors, cur + 1), np.float32)
        for k, j in enumerate(others[:max_neighbors]):
            nb[k] = to_local(arrs[j][:cur + 1, :2], origin, h)
            nb_mask[k] = arrs[j][:cur + 1, 5]
        out.append(dict(scenario_id=sc.scenario_id, track_index=i,
                        obj_type=int(sc.tracks[i].object_type),
                        hist=hist, hist_mask=a[:cur + 1, 5], fut=fut,
                        nb=nb, nb_mask=nb_mask, tags=tags))
    return out

def build_dataset(pattern, out_path, max_per_file=None):
    samples = []
    for f in sorted(glob.glob(pattern)):
        for n, rec in enumerate(tf.data.TFRecordDataset(f)):
            if max_per_file and n >= max_per_file:
                break
            sc = scenario_pb2.Scenario()
            sc.ParseFromString(rec.numpy())
            samples += extract_samples(sc)
    with open(out_path, 'wb') as fh:
        pickle.dump(samples, fh)
    return samples
