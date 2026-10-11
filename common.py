import os

BASE = os.path.expanduser("~/Documents/data")  # folder name is ("data");
MODEL_PATH = os.path.join(HERE, "model.joblib")
CAMERA_CSV = os.path.join(BASE, "camera.csv")   # samples recorded from your own camera

# This is the label order (0..5). no_gesture = no gesture being made
NAMES = ["little_finger", "one", "peace", "grabbing", "two_up", "no_gesture"]
CLASSES = {n: i for i, n in enumerate(NAMES)}

def normalize(pts):
    """pts: list of 21 (x, y) points. Wrist is the origin, then divide by the max abs value. Returns 42 numbers.
    This is the single normalization function in the whole project, used for preparation, recording and runtime."""
    pts = [(p[0], p[1]) for p in pts]
    wx, wy = pts[0]
    pts = [(x - wx, y - wy) for x, y in pts]
    m = max(max(abs(x), abs(y)) for x, y in pts) or 1
    return [v / m for p in pts for v in p]

def mirror(row):
    """Left/right flip (flips x only)."""
    return [(-v if i % 2 == 0 else v) for i, v in enumerate(row)]