import csv
import glob
import json
import random

from common import BASE, CLASSES, mirror, normalize

LIMITS = {"train": 3000, "val": 500, "test": 500}  # أقصى عدد صور لكل إيماءة
random.seed(42)


def find_json(split, name):
    for p in glob.glob(f"{BASE}/**/{split}/{name}.json", recursive=True):
        if ".ipynb_checkpoints" not in p:
            return p
    raise FileNotFoundError(f"مش لاقي {split}/{name}.json جوه {BASE}")


for split, limit in LIMITS.items():
    rows = []
    for name, label in CLASSES.items():
        try:
            path = find_json(split, name)
        except FileNotFoundError as e:
            print("تحذير:", e)
            continue
        with open(path) as f:
            data = json.load(f)
        keys = list(data.keys())
        random.shuffle(keys)
        count = 0
        for k in keys:
            if count >= limit:
                break
            item = data[k]
            labels = item.get("labels", [])
            hands = item.get("hand_landmarks", [])
            for lab, pts in zip(labels, hands):
                if lab != name or not pts or len(pts) != 21:
                    continue
                row = normalize(pts)
                rows.append(row + [label])
                count += 1
                if split == "train":
                    rows.append(mirror(row) + [label])
                break                      # إيد واحدة من كل صورة
        print(split, name, count)
        if count == 0:
            print(f"  تحذير: مفيش عينات للكلاس {name} (لو no_gesture، "
                  "ممكن الـ landmarks فاضية في الملف ده، سجّله من الكاميرا بـ record.py)")
    random.shuffle(rows)
    out = f"{BASE}/{split}.csv"
    with open(out, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"{out}: {len(rows)} سطر\n")