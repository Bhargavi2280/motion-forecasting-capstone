import numpy as np

# Frozen thresholds (v1). Changing these requires re-running extraction.
TH = dict(
    extreme_turn_deg=100,    # net heading change over 8 s (U-turns and very sharp turns)
    hard_brake_mps2=5.5,     # deceleration measured over 0.5 s
    stop_start_speed=5.0,    # was moving at least this fast ...
    stop_end_speed=0.5,      # ... and ends nearly stationary
    lane_shift_min_m=3.0,    # lateral shift range for a lane change
    lane_shift_max_m=5.0,
    lane_max_turn_deg=15,    # lane change = small net heading change
    lane_min_dist_m=25,
)

def tag_trajectory(xy, speed, obj_type):
    """xy: (T,2) future path in the agent's local frame (starts at current pos).
    speed: (T,) speed in m/s. Returns list of rare-behaviour tags."""
    tags = []
    d = np.diff(xy, axis=0)
    step = np.linalg.norm(d, axis=1)
    moving = step > 0.1
    turn = 0.0
    if moving.sum() >= 10:
        ang = np.unwrap(np.arctan2(d[moving, 1], d[moving, 0]))
        turn = abs(np.degrees(ang[-1] - ang[0]))
        if turn > TH['extreme_turn_deg']:
            tags.append("extreme_turn")
    if len(speed) > 5:
        acc5 = (speed[5:] - speed[:-5]) / 0.5
        if acc5.min() < -TH['hard_brake_mps2']:
            tags.append("hard_brake")
    if speed[0] > TH['stop_start_speed'] and speed[-10:].max() < TH['stop_end_speed']:
        tags.append("sudden_stop")
    if (turn < TH['lane_max_turn_deg'] and step.sum() > TH['lane_min_dist_m']
            and TH['lane_shift_min_m'] < abs(xy[-1, 1]) < TH['lane_shift_max_m']):
        tags.append("lane_change")
    if obj_type in (2, 3):
        tags.append("vulnerable")
    return tags
