"""تسجيل عينات من الكاميرا بتاعتك.
اضغط رقم الكلاس (0-5) علشان يبدأ يسجل، واضغط نفس الرقم تاني أو space علشان يقف.
0 fist | 1 one | 2 ok | 3 call | 4 peace | 5 no_gesture (إيد عادية/بتتحرك/مرتاحة)
ESC للخروج والحفظ. سجّل 200-300 عينة لكل إيماءة، بزوايا وأوضاع مختلفة.
"""
import csv
import os

import cv2
import mediapipe as mp

from common import CAMERA_CSV, NAMES, normalize

mp_hands = mp.solutions.hands
draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7,
                       min_tracking_confidence=0.6)
cap = cv2.VideoCapture(0)

rows, counts = [], {n: 0 for n in NAMES}
recording = None

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)              # نفس الـ flip بتاع mouse.py
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0]
            draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
            if recording is not None:
                pts = [(p.x, p.y) for p in lm.landmark]
                rows.append(normalize(pts) + [recording])
                counts[NAMES[recording]] += 1

        status = f"REC {NAMES[recording]}" if recording is not None else "idle"
        cv2.putText(frame, status, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 0, 255) if recording is not None else (200, 200, 200), 2)
        y = 80
        for n in NAMES:
            cv2.putText(frame, f"{NAMES.index(n)} {n}: {counts[n]}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y += 25
        cv2.imshow("record", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord(" "):
            recording = None
        elif ord("0") <= key <= ord("0") + len(NAMES) - 1:
            c = key - ord("0")
            recording = None if recording == c else c
finally:
    cap.release()
    cv2.destroyAllWindows()
    if rows:
        new = not os.path.exists(CAMERA_CSV)
        with open(CAMERA_CSV, "a", newline="") as f:
            csv.writer(f).writerows(rows)
        print(f"اتضاف {len(rows)} سطر في {CAMERA_CSV}", counts)