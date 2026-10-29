import sys; sys.path.insert(0, '.')
import numpy as np
from src.rare_tags import tag_trajectory
from src.metrics import evaluate

def straight(v=10.0):
    t = np.arange(81) * 0.1
    xy = np.stack([v * t, np.zeros_like(t)], 1)
    return xy, np.full(81, v)

def test_straight_driving_has_no_rare_tag():
    xy, sp = straight()
    assert tag_trajectory(xy, sp, 1) == []

def test_sudden_stop_tag():
    xy, sp = straight()
    sp = sp.copy(); sp[-30:] = 0.0
    assert "sudden_stop" in tag_trajectory(xy, sp, 1)

def test_hard_brake_tag():
    xy, sp = straight()
    sp = sp.copy(); sp[20:25] = np.linspace(10, 0, 5); sp[25:] = 0
    assert "hard_brake" in tag_trajectory(xy, sp, 1)

def test_vulnerable_tag_for_pedestrian():
    xy, sp = straight(1.0)
    assert "vulnerable" in tag_trajectory(xy, sp, 2)

def test_perfect_prediction_gives_zero_error_and_full_coverage():
    xy, sp = straight()
    fut = xy[1:].astype(np.float32)
    samples = [dict(fut=fut, tags=[])]
    out = evaluate(fut[None, None], samples)
    assert out["all"]["minADE"] < 1e-6
    assert out["all"]["coverage"] == 100.0

def test_far_prediction_is_a_miss():
    xy, sp = straight()
    fut = xy[1:].astype(np.float32)
    samples = [dict(fut=fut, tags=[])]
    out = evaluate((fut + 50)[None, None], samples)
    assert out["all"]["coverage"] == 0.0
