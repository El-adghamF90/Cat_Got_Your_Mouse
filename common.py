import os

BASE = os.path.expanduser("C:/Users/HP/Desktop/Mouse/tom-jerry/Data/annotations")
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model.joblib")
CAMERA_CSV = os.path.join(BASE, "camera.csv")   # samples recorded from your own camera

# The order defines the numeric labels (0..N-1). Keep no_gesture last.
# Names must match the annotation file names exactly (e.g. two_up.json).
NAMES = ["little_finger", "one", "peace", "grabbing", "two_up",
         "like", "dislike", "fist", "no_gesture"]
CLASSES = {n: i for i, n in enumerate(NAMES)}


def normalize(pts):
    """pts: list of 21 (x, y) points. Wrist becomes the origin, then divide by the
    largest absolute value. Returns 42 numbers. Single source of truth for
    preprocessing, used by data conversion, recording and live inference."""
    pts = [(p[0], p[1]) for p in pts]
    wx, wy = pts[0]
    pts = [(x - wx, y - wy) for x, y in pts]
    m = max(max(abs(x), abs(y)) for x, y in pts) or 1
    return [v / m for p in pts for v in p]


def mirror(row):
    """Flip left/right (negates x only)."""
    return [(-v if i % 2 == 0 else v) for i, v in enumerate(row)]
