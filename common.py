import os

BASE = os.path.expanduser("~/Documents/داتا")
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model.joblib")
CAMERA_CSV = os.path.join(BASE, "camera.csv")   # عينات من الكاميرا بتاعتك

# الترتيب ده هو ترتيب الـ labels (0..5). no_gesture = مفيش إيماءة
NAMES = ["fist", "one", "ok", "call", "peace", "no_gesture"]
CLASSES = {n: i for i, n in enumerate(NAMES)}


def normalize(pts):
    """pts: قايمة 21 نقطة (x, y). المعصم أصل، وقسمة على أكبر قيمة. بترجع 42 رقم.
    دي الدالة الوحيدة في المشروع كله، بتتستخدم في التجهيز والتسجيل والتشغيل."""
    pts = [(p[0], p[1]) for p in pts]
    wx, wy = pts[0]
    pts = [(x - wx, y - wy) for x, y in pts]
    m = max(max(abs(x), abs(y)) for x, y in pts) or 1
    return [v / m for p in pts for v in p]


def mirror(row):
    """قلب يمين/شمال (بيقلب الـ x بس)."""
    return [(-v if i % 2 == 0 else v) for i, v in enumerate(row)]