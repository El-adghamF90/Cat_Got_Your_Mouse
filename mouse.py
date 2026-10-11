"""Hand Mouse
SPACE = toggle control on/off (starts off) | ESC = quit

Gestures (change them from the G_* constants below if you want):
one           = move the cursor
peace         = left click
little_finger = right click
two_up        = scroll (move your hand up and down)
grabbing      = select/drag (holds the button down + move your hand, releases when you drop the gesture)
"""
import platform
import time
from collections import Counter, deque

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pyautogui

from common import MODEL_PATH, NAMES, normalize

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = True   # quickly move the mouse to the top-left corner of the screen for an emergency stop

# ---- Map gestures to actions (names must match NAMES in common.py exactly) ----
G_MOVE = "one"
G_CLICK = "peace"
G_RIGHT = "little_finger"
G_SCROLL = "two_up"
G_SELECT = "grabbing"
# -------------------------------------------------------------------------------

model = joblib.load(MODEL_PATH)
CLASS_NAMES = [NAMES[c] for c in model.classes_]   # order guaranteed by the model
for g in (G_MOVE, G_CLICK, G_RIGHT, G_SCROLL, G_SELECT):
    if g not in CLASS_NAMES:
        raise SystemExit(f"Gesture '{g}' is not in the model. Available names: {CLASS_NAMES}")

SCREEN_W, SCREEN_H = pyautogui.size()
MARGIN = 0.15
SMOOTH = 0.35
CLICK_COOLDOWN = 0.8
POINTER_LM = 5           # index finger base: relatively stable no matter how the fingers move
MIN_CONF = 0.7

# On Windows scrolling uses units of 120 per notch, on Mac/Linux it is per line. Tune the number by trial
# (if scrolling is reversed, make it negative)
SCROLL_GAIN = 3000 if platform.system() == "Windows" else 300
SCROLL_DEADZONE = 0.004

mp_hands = mp.solutions.hands
draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7,
                       min_tracking_confidence=0.6)
cap = cv2.VideoCapture(0)

history = deque(maxlen=5)
stable = "none"
enabled = False
dragging = False
last_click = 0.0
cur_x, cur_y = pyautogui.position()
prev_y = None
conf = 0.0


def release_drag():
    global dragging
    if dragging:
        pyautogui.mouseUp()
        dragging = False


try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        lm = None
        conf = 0.0
        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0]
            draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
            pts = [(p.x, p.y) for p in lm.landmark]
            proba = model.predict_proba(np.array([normalize(pts)]))[0]
            idx = int(proba.argmax())
            conf = float(proba[idx])
            name = CLASS_NAMES[idx]
            history.append(name if conf >= MIN_CONF and name != "no_gesture" else "none")
        else:
            history.append("none")

        gesture = Counter(history).most_common(1)[0][0]
        now = time.time()

        if enabled:
            # actions that fire once when the gesture changes
            if gesture != stable:
                if gesture == G_CLICK and now - last_click > CLICK_COOLDOWN:
                    pyautogui.click()
                    last_click = now
                elif gesture == G_RIGHT and now - last_click > CLICK_COOLDOWN:
                    pyautogui.rightClick()
                    last_click = now
                elif gesture == G_SELECT and not dragging:
                    pyautogui.mouseDown()
                    dragging = True
                if stable == G_SELECT and gesture != G_SELECT:
                    release_drag()

            # move the cursor (in move mode or while selecting)
            if gesture in (G_MOVE, G_SELECT) and lm:
                p = lm.landmark[POINTER_LM]
                tx = np.clip((p.x - MARGIN) / (1 - 2 * MARGIN), 0, 1) * SCREEN_W
                ty = np.clip((p.y - MARGIN) / (1 - 2 * MARGIN), 0, 1) * SCREEN_H
                cur_x += (tx - cur_x) * SMOOTH
                cur_y += (ty - cur_y) * SMOOTH
                pyautogui.moveTo(np.clip(cur_x, 2, SCREEN_W - 2),
                                 np.clip(cur_y, 2, SCREEN_H - 2))
            else:
                cur_x, cur_y = pyautogui.position()

            # scroll
            if gesture == G_SCROLL and lm:
                y = lm.landmark[12].y
                if prev_y is not None and abs(y - prev_y) > SCROLL_DEADZONE:
                    pyautogui.scroll(int(-(y - prev_y) * SCROLL_GAIN))
                prev_y = y
            else:
                prev_y = None
        else:
            release_drag()
            prev_y = None

        stable = gesture

        color = (0, 255, 0) if enabled else (0, 0, 255)
        cv2.putText(frame, f"{'ON' if enabled else 'OFF (SPACE)'} | {gesture} {conf:.2f}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.imshow("Hand Mouse (ESC = quit)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord(" "):
            enabled = not enabled
            if enabled:
                cur_x, cur_y = pyautogui.position()
finally:
    release_drag()
    cap.release()
    cv2.destroyAllWindows()