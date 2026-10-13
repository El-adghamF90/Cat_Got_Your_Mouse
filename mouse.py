"""Hand Mouse
SPACE = toggle control on/off (starts OFF) | ESC = quit

Gestures (change them via the G_* constants below):
one           = move the cursor
peace         = left click
little_finger = right click
two_up        = scroll (move your hand up/down)
grabbing      = select/drag (holds the button; move your hand)
like          = volume up: move your hand up (macOS)
dislike       = volume down: move your hand down (macOS)
fist          = play/pause video: hold still for about a second (presses space)
"""
import platform
import subprocess
import time
from collections import Counter, deque

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pyautogui

from common import MODEL_PATH, NAMES, normalize

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = True   # emergency stop: move the mouse quickly to the top-left corner

# ---- Gesture to action mapping (names must match NAMES in common.py exactly) ----
G_MOVE = "one"
G_CLICK = "peace"
G_RIGHT = "little_finger"
G_SCROLL = "two_up"
G_SELECT = "grabbing"
G_VOL_UP = "like"
G_VOL_DOWN = "dislike"
G_PLAYPAUSE = "fist"
# ---------------------------------------------------------------------------------

model = joblib.load(MODEL_PATH)
CLASS_NAMES = [NAMES[c] for c in model.classes_]   # label order taken from the model
for g in (G_MOVE, G_CLICK, G_RIGHT, G_SCROLL, G_SELECT, G_VOL_UP, G_VOL_DOWN, G_PLAYPAUSE):
    if g not in CLASS_NAMES:
        raise SystemExit(f"Gesture '{g}' is not in the model. Available: {CLASS_NAMES}")

SCREEN_W, SCREEN_H = pyautogui.size()
MARGIN = 0.15            # ignore camera edges so the screen corners are reachable
SMOOTH = 0.35            # lower = smoother but slower
CLICK_COOLDOWN = 0.8     # seconds between clicks
POINTER_LM = 5           # index finger base: stays steady when the fingers change shape
MIN_CONF = 0.7           # minimum model confidence

# Windows scrolls in units of 120 per notch; macOS/Linux scroll per line.
# Tune by testing (use a negative value if scrolling is reversed).
SCROLL_GAIN = 3000 if platform.system() == "Windows" else 300
SCROLL_DEADZONE = 0.004

VOLUME_GAIN = 150        # higher = volume changes faster with hand movement
VOLUME_DEADZONE = 0.003
PLAY_HOLD = 1.0          # seconds to hold the gesture before play/pause fires
PLAY_COOLDOWN = 2.0


SOUNDS_ENABLED = True
SOUND_DIR = "/System/Library/Sounds"       # built-in macOS sounds (.aiff)
SOUNDS = {                                  # action -> sound name; pick any file in SOUND_DIR
    "click": "Tink",
    "right": "Pop",
    "select": "Morse",
    "release": "Bottle",
    "play": "Glass",
}


def play_sound(action):
    """Play a short sound without blocking the loop (macOS only)."""
    if not SOUNDS_ENABLED or platform.system() != "Darwin":
        return
    subprocess.Popen(["afplay", f"{SOUND_DIR}/{SOUNDS[action]}.aiff"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def get_volume():
    """macOS output volume (0-100), or None if unsupported (not macOS, external audio, ...)."""
    if platform.system() != "Darwin":
        return None
    try:
        out = subprocess.run(
            ["osascript", "-e", "output volume of (get volume settings)"],
            capture_output=True, text=True, timeout=2).stdout.strip()
        return float(out)
    except Exception:
        return None


def set_volume(v):
    subprocess.Popen(["osascript", "-e", f"set volume output volume {int(v)}"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


mp_hands = mp.solutions.hands
draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7,
                       min_tracking_confidence=0.6)
cap = cv2.VideoCapture(0)

history = deque(maxlen=5)   # majority vote over the last 5 frames
stable = "none"
enabled = False
dragging = False
last_click = 0.0
cur_x, cur_y = pyautogui.position()
prev_y = None
conf = 0.0

vol = get_volume()
if vol is None:
    print("Note: volume control is unavailable here (not macOS, or audio is on an external device).")
vol_prev_y = None
vol_prev_g = None
last_vol_sent = -1
last_vol_time = 0.0
hold_start = None
play_fired = False
last_play = 0.0


def release_drag():
    global dragging
    if dragging:
        pyautogui.mouseUp()
        dragging = False
        play_sound("release")


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
            # One-shot actions fire when the gesture changes
            if gesture != stable:
                if gesture == G_CLICK and now - last_click > CLICK_COOLDOWN:
                    pyautogui.click()
                    play_sound("click")
                    last_click = now
                elif gesture == G_RIGHT and now - last_click > CLICK_COOLDOWN:
                    pyautogui.rightClick()
                    play_sound("right")
                    last_click = now
                elif gesture == G_SELECT and not dragging:
                    pyautogui.mouseDown()
                    play_sound("select")
                    dragging = True
                if stable == G_SELECT and gesture != G_SELECT:
                    release_drag()

            # Cursor movement (in move mode or while selecting)
            if gesture in (G_MOVE, G_SELECT) and lm:
                p = lm.landmark[POINTER_LM]
                tx = np.clip((p.x - MARGIN) / (1 - 2 * MARGIN), 0, 1) * SCREEN_W
                ty = np.clip((p.y - MARGIN) / (1 - 2 * MARGIN), 0, 1) * SCREEN_H
                cur_x += (tx - cur_x) * SMOOTH
                cur_y += (ty - cur_y) * SMOOTH
                pyautogui.moveTo(np.clip(cur_x, 2, SCREEN_W - 2),
                                 np.clip(cur_y, 2, SCREEN_H - 2))
            else:
                cur_x, cur_y = pyautogui.position()   # avoids a jump if you moved the mouse by hand

            # Scroll
            if gesture == G_SCROLL and lm:
                y = lm.landmark[12].y
                if prev_y is not None and abs(y - prev_y) > SCROLL_DEADZONE:
                    pyautogui.scroll(int(-(y - prev_y) * SCROLL_GAIN))
                prev_y = y
            else:
                prev_y = None

            # Volume: like + hand up = louder | dislike + hand down = quieter
            if gesture in (G_VOL_UP, G_VOL_DOWN) and lm and vol is not None:
                y = lm.landmark[9].y
                if vol_prev_y is not None and vol_prev_g == gesture:
                    dy = y - vol_prev_y            # negative = hand moving up
                    if gesture == G_VOL_UP and dy < -VOLUME_DEADZONE:
                        vol = float(np.clip(vol - dy * VOLUME_GAIN, 0, 100))
                    elif gesture == G_VOL_DOWN and dy > VOLUME_DEADZONE:
                        vol = float(np.clip(vol - dy * VOLUME_GAIN, 0, 100))
                    if int(vol) != last_vol_sent and now - last_vol_time > 0.1:
                        set_volume(vol)
                        last_vol_sent = int(vol)
                        last_vol_time = now
                vol_prev_y = y
                vol_prev_g = gesture
            else:
                vol_prev_y = None
                vol_prev_g = None

            # Play/pause toggle: hold the gesture for PLAY_HOLD seconds.
            # It fires once per hold; drop the gesture and show it again to toggle back.
            if gesture == G_PLAYPAUSE:
                if hold_start is None:
                    hold_start = now
                    play_fired = False
                elif (not play_fired and now - hold_start >= PLAY_HOLD
                      and now - last_play > PLAY_COOLDOWN):
                    pyautogui.press("space")
                    play_sound("play")
                    last_play = now
                    play_fired = True
            else:
                hold_start = None
                play_fired = False
        else:
            release_drag()
            prev_y = None
            vol_prev_y = None
            vol_prev_g = None
            hold_start = None
            play_fired = False

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
                v = get_volume()
                if v is not None:
                    vol = v
finally:
    release_drag()
    cap.release()
    cv2.destroyAllWindows()