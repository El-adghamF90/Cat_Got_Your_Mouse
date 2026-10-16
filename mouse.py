"""
Hand Mouse - Windows Version

SPACE = toggle control on/off (starts OFF)
ESC   = quit

Gestures:
one           = move cursor
peace         = left click
little_finger = right click
two_up        = scroll
grabbing      = select/drag
like          = volume up: move hand up
dislike       = volume down: move hand down
fist          = play/pause video: hold still for 1 second
"""

import time
import winsound
from collections import Counter, deque

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pyautogui

from pycaw.pycaw import AudioUtilities
from common import MODEL_PATH, NAMES, normalize


# ---------------- SAFETY & SETTINGS ----------------

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = True

G_MOVE = "one"
G_CLICK = "peace"
G_RIGHT = "little_finger"
G_SCROLL = "two_up"
G_SELECT = "grabbing"
G_VOL_UP = "like"
G_VOL_DOWN = "dislike"
G_PLAYPAUSE = "fist"

SCREEN_W, SCREEN_H = pyautogui.size()

MARGIN = 0.15
SMOOTH = 0.35
CLICK_COOLDOWN = 0.8
POINTER_LM = 5
MIN_CONF = 0.7

SCROLL_GAIN = 3000
SCROLL_DEADZONE = 0.004

VOLUME_GAIN = 150
VOLUME_DEADZONE = 0.003

PLAY_HOLD = 1.0
PLAY_COOLDOWN = 2.0

SOUNDS_ENABLED = True

SOUNDS = {
    "click": winsound.MB_OK,
    "right": winsound.MB_ICONEXCLAMATION,
    "select": winsound.MB_ICONASTERISK,
    "release": winsound.MB_ICONQUESTION,
    "play": winsound.MB_ICONHAND,
}


# ---------------- WINDOWS SOUND EFFECTS ----------------

def play_sound(action):
    """Play a built-in Windows system sound."""
    if not SOUNDS_ENABLED:
        return

    try:
        winsound.MessageBeep(SOUNDS.get(action, winsound.MB_OK))
    except (RuntimeError, OSError):
        pass


# ---------------- WINDOWS MASTER VOLUME ----------------

def get_volume():
    """Return Windows master volume as a percentage, 0-100."""
    try:
        device = AudioUtilities.GetSpeakers()
        volume = device.EndpointVolume
        return volume.GetMasterVolumeLevelScalar() * 100
    except Exception as e:
        print(f"Volume control unavailable: {e}")
        return None


def set_volume(value):
    """Set Windows master volume, 0-100."""
    try:
        device = AudioUtilities.GetSpeakers()
        volume = device.EndpointVolume
        volume.SetMasterVolumeLevelScalar(
            float(np.clip(value, 0, 100)) / 100,
            None
        )
    except Exception as e:
        print(f"Could not change volume: {e}")


# ---------------- LOAD TRAINED MODEL ----------------

model = joblib.load(MODEL_PATH)
CLASS_NAMES = [NAMES[c] for c in model.classes_]

required_gestures = (
    G_MOVE, G_CLICK, G_RIGHT, G_SCROLL,
    G_SELECT, G_VOL_UP, G_VOL_DOWN, G_PLAYPAUSE
)

for gesture_name in required_gestures:
    if gesture_name not in CLASS_NAMES:
        raise SystemExit(
            f"Gesture '{gesture_name}' is not in the model. "
            f"Available gestures: {CLASS_NAMES}"
        )


# ---------------- MEDIAPIPE & WEBCAM ----------------

mp_hands = mp.solutions.hands
draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    hands.close()
    raise SystemExit(
        "Could not open webcam. Check that your camera is connected "
        "and not being used by another application."
    )


# ---------------- STATE VARIABLES ----------------

history = deque(maxlen=5)

stable = "none"
enabled = False
dragging = False

last_click = 0.0

cur_x, cur_y = pyautogui.position()

prev_y = None
conf = 0.0

vol = get_volume()
if vol is None:
    print("Windows volume control is unavailable. Volume gestures are disabled.")

vol_prev_y = None
vol_prev_g = None
last_vol_sent = -1
last_vol_time = 0.0

hold_start = None
play_fired = False
last_play = 0.0


def release_drag():
    """Release the mouse button if a drag is active."""
    global dragging

    if dragging:
        pyautogui.mouseUp()
        dragging = False
        play_sound("release")


# ---------------- MAIN LOOP ----------------

try:
    print("Hand Mouse started.")
    print("SPACE = enable/disable | ESC = quit")

    while True:
        ok, frame = cap.read()

        if not ok:
            print("Could not read webcam frame.")
            break

        frame = cv2.flip(frame, 1)

        res = hands.process(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )

        lm = None
        conf = 0.0

        # -------- DETECT HAND & PREDICT GESTURE --------

        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0]

            draw.draw_landmarks(
                frame, lm, mp_hands.HAND_CONNECTIONS
            )

            pts = [(p.x, p.y) for p in lm.landmark]

            features = np.array([normalize(pts)])
            proba = model.predict_proba(features)[0]

            idx = int(proba.argmax())
            conf = float(proba[idx])
            name = CLASS_NAMES[idx]

            if conf >= MIN_CONF and name != "no_gesture":
                history.append(name)
            else:
                history.append("none")

        else:
            history.append("none")

        gesture = Counter(history).most_common(1)[0][0]
        now = time.time()

        # -------- MOUSE CONTROL ENABLED --------

        if enabled:

            # Click, right-click, and drag
            if gesture != stable:

                if (
                    gesture == G_CLICK
                    and now - last_click > CLICK_COOLDOWN
                ):
                    pyautogui.click()
                    play_sound("click")
                    last_click = now

                elif (
                    gesture == G_RIGHT
                    and now - last_click > CLICK_COOLDOWN
                ):
                    pyautogui.rightClick()
                    play_sound("right")
                    last_click = now

                elif gesture == G_SELECT and not dragging:
                    pyautogui.mouseDown()
                    play_sound("select")
                    dragging = True

                if stable == G_SELECT and gesture != G_SELECT:
                    release_drag()

            # Cursor movement, including while dragging
            if gesture in (G_MOVE, G_SELECT) and lm:
                p = lm.landmark[POINTER_LM]

                tx = np.clip(
                    (p.x - MARGIN) / (1 - 2 * MARGIN),
                    0, 1
                ) * SCREEN_W

                ty = np.clip(
                    (p.y - MARGIN) / (1 - 2 * MARGIN),
                    0, 1
                ) * SCREEN_H

                cur_x += (tx - cur_x) * SMOOTH
                cur_y += (ty - cur_y) * SMOOTH

                pyautogui.moveTo(
                    np.clip(cur_x, 2, SCREEN_W - 2),
                    np.clip(cur_y, 2, SCREEN_H - 2)
                )

            else:
                cur_x, cur_y = pyautogui.position()

            # Scroll
            if gesture == G_SCROLL and lm:
                y = lm.landmark[12].y

                if (
                    prev_y is not None
                    and abs(y - prev_y) > SCROLL_DEADZONE
                ):
                    pyautogui.scroll(
                        int(-(y - prev_y) * SCROLL_GAIN)
                    )

                prev_y = y
            else:
                prev_y = None

            # Windows volume: like = up, dislike = down
            if (
                gesture in (G_VOL_UP, G_VOL_DOWN)
                and lm
                and vol is not None
            ):
                y = lm.landmark[9].y

                if vol_prev_y is not None and vol_prev_g == gesture:
                    dy = y - vol_prev_y

                    if (
                        gesture == G_VOL_UP
                        and dy < -VOLUME_DEADZONE
                    ):
                        vol = float(np.clip(
                            vol - dy * VOLUME_GAIN, 0, 100
                        ))

                    elif (
                        gesture == G_VOL_DOWN
                        and dy > VOLUME_DEADZONE
                    ):
                        vol = float(np.clip(
                            vol - dy * VOLUME_GAIN, 0, 100
                        ))

                    if (
                        int(vol) != last_vol_sent
                        and now - last_vol_time > 0.1
                    ):
                        set_volume(vol)
                        last_vol_sent = int(vol)
                        last_vol_time = now

                vol_prev_y = y
                vol_prev_g = gesture

            else:
                vol_prev_y = None
                vol_prev_g = None

            # Play/pause: hold fist for one second
            if gesture == G_PLAYPAUSE:

                if hold_start is None:
                    hold_start = now
                    play_fired = False

                elif (
                    not play_fired
                    and now - hold_start >= PLAY_HOLD
                    and now - last_play > PLAY_COOLDOWN
                ):
                    pyautogui.press("space")
                    play_sound("play")

                    last_play = now
                    play_fired = True

            else:
                hold_start = None
                play_fired = False

        # -------- CONTROL DISABLED --------

        else:
            release_drag()
            prev_y = None
            vol_prev_y = None
            vol_prev_g = None
            hold_start = None
            play_fired = False

        stable = gesture

        # -------- DISPLAY STATUS --------

        color = (0, 255, 0) if enabled else (0, 0, 255)

        cv2.putText(
            frame,
            f"{'ON' if enabled else 'OFF (SPACE)'} | "
            f"{gesture} {conf:.2f}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

        cv2.imshow("Hand Mouse (ESC = quit)", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break

        elif key == ord(" "):  # SPACE
            enabled = not enabled

            if enabled:
                cur_x, cur_y = pyautogui.position()

                current_volume = get_volume()
                if current_volume is not None:
                    vol = current_volume

                print("Mouse control ON")
            else:
                print("Mouse control OFF")

finally:
    release_drag()
    cap.release()
    hands.close()
    cv2.destroyAllWindows()
    print("Hand Mouse closed.")
