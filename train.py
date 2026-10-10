import os
import time

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import BASE, CAMERA_CSV, MODEL_PATH, NAMES


def load(path):
    d = np.loadtxt(path, delimiter=",", ndmin=2)
    return d[:, :-1], d[:, -1].astype(int)


X_train, y_train = load(f"{BASE}/train.csv")
X_val, y_val = load(f"{BASE}/val.csv")
X_test, y_test = load(f"{BASE}/test.csv")

# عينات الكاميرا بتاعتك: تتضاف للـ train، ونص منها (بشكل عشوائي) للـ val
if os.path.exists(CAMERA_CSV):
    Xc, yc = load(CAMERA_CSV)
    rng = np.random.default_rng(42)
    mask = rng.random(len(yc)) < 0.2
    X_train = np.vstack([X_train, Xc[~mask]])
    y_train = np.concatenate([y_train, yc[~mask]])
    X_val = np.vstack([X_val, Xc[mask]])
    y_val = np.concatenate([y_val, yc[mask]])
    print(f"ضفنا {len(yc)} عينة من الكاميرا")
    print("(ملحوظة: عينات الكاميرا المتتالية شبه بعض، فالـ val هنا متفائل. "
          "الاختبار الحقيقي هو التجربة بالإيد.)")

print("train:", X_train.shape, "| val:", X_val.shape, "| test:", X_test.shape)
print("توزيع الكلاسات في train:", np.bincount(y_train, minlength=len(NAMES)))

models = {
    "RandomForest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
    "MLP": make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300,
                      early_stopping=True, random_state=42),
    ),
}

best_name, best_acc, best_model = None, 0, None
for name, model in models.items():
    model.fit(X_train, y_train)
    acc = accuracy_score(y_val, model.predict(X_val))

    t = time.perf_counter()
    for i in range(200):                         # latency لفريم واحد
        model.predict_proba(X_val[i % len(X_val)].reshape(1, -1))
    ms = (time.perf_counter() - t) / 200 * 1000
    print(f"{name}: val accuracy = {acc:.4f} | {ms:.2f} ms/frame")

    if acc > best_acc:
        best_name, best_acc, best_model = name, acc, model

print(f"\nالأفضل: {best_name}")
pred = best_model.predict(X_test)
print(f"test accuracy = {accuracy_score(y_test, pred):.4f}\n")
labels = sorted(set(y_test) | set(pred))
print("confusion matrix (الصفوف = الحقيقي، الأعمدة = المتوقع):")
print(confusion_matrix(y_test, pred, labels=labels))
print()
print(classification_report(y_test, pred, labels=labels,
                            target_names=[NAMES[i] for i in labels], zero_division=0))

joblib.dump(best_model, MODEL_PATH)
print("اتحفظ:", MODEL_PATH)