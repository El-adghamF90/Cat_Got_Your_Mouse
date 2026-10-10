"""يرسم شكل كل إيماءة موجودة في annotations بتاعة HaGRID (عينتين لكل إيماءة).
التشغيل:  python view_all.py
بيحفظ all_gestures.png جنب الكود. اقرأ أسماء الإيماءات من عناوين الرسم،
وهي نفس الأسماء اللي تحطها في NAMES جوه common.py.
"""
import glob
import json
import math
import os
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import BASE, HERE, normalize

SPLIT = "val"            # val أصغر وأسرع في التحميل من train
SAMPLES = 2              # عدد العينات لكل إيماءة
GESTURES_PER_ROW = 4
random.seed(1)

CONNECTIONS = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),
               (10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),
               (18,19),(19,20),(0,17)]


def find_files(split):
    files = []
    for p in glob.glob(f"{BASE}/**/{split}/*.json", recursive=True):
        if ".ipynb_checkpoints" not in p:
            files.append(p)
    return sorted(files)


files = find_files(SPLIT) or find_files("train")
if not files:
    raise SystemExit(f"مش لاقي ملفات json جوه {BASE}")

print(f"لقيت {len(files)} ملف:")
gestures = {}
for path in files:
    name = os.path.splitext(os.path.basename(path))[0]
    with open(path) as f:
        data = json.load(f)
    keys = list(data.keys())
    random.shuffle(keys)
    found = []
    for k in keys:
        item = data[k]
        for lab, pts in zip(item.get("labels", []), item.get("hand_landmarks", [])):
            if lab == name and pts and len(pts) == 21:
                found.append(normalize(pts))
                break
        if len(found) >= SAMPLES:
            break
    gestures[name] = found
    print(f"  {name}: {len(found)} عينة")

names = sorted(gestures)
cols = GESTURES_PER_ROW * SAMPLES
rows = math.ceil(len(names) / GESTURES_PER_ROW)
fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.6 * rows), squeeze=False)

for ax in axes.flat:
    ax.axis("off")

for i, name in enumerate(names):
    r, c0 = divmod(i, GESTURES_PER_ROW)
    for s in range(SAMPLES):
        ax = axes[r][c0 * SAMPLES + s]
        if s >= len(gestures[name]):
            continue
        ax.axis("on")
        row = gestures[name][s]
        xs, ys = row[0::2], row[1::2]
        for a, b in CONNECTIONS:
            ax.plot([xs[a], xs[b]], [ys[a], ys[b]], "b-", linewidth=1)
        ax.scatter(xs, ys, c="r", s=10)
        ax.scatter(xs[0], ys[0], c="g", s=30)      # المعصم
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        if s == 0:
            ax.set_title(name, loc="left", fontsize=11, fontweight="bold")

plt.tight_layout()
out = os.path.join(HERE, "all_gestures.png")
plt.savefig(out, dpi=100)
print("اتحفظت:", out)