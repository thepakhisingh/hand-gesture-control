import cv2
import mediapipe as mp
import pyautogui
import math
import time

# Disable pyautogui fail-safe (mouse to corner)
pyautogui.FAILSAFE = False

# Screen size
screen_w, screen_h = pyautogui.size()

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Camera
cap = cv2.VideoCapture(0)

# States
dragging = False
last_action_time = 0
ACTION_DELAY = 0.8

def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])
 
def fingers_up(lm):
    fingers = []

    # Thumb (left/right independent)
    fingers.append(lm[4].x < lm[3].x)

    # Other fingers
    tips = [8, 12, 16, 20]
    for tip in tips:
        fingers.append(lm[tip].y < lm[tip - 2].y)

    return fingers

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    h, w, _ = img.shape

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_lms in result.multi_hand_landmarks:
            lm = hand_lms.landmark
            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)

            fingers = fingers_up(lm)
            count = fingers.count(True)

            # Index finger controls mouse
            ix, iy = int(lm[8].x * w), int(lm[8].y * h)
            screen_x = int(lm[8].x * screen_w)
            screen_y = int(lm[8].y * screen_h)
            pyautogui.moveTo(screen_x, screen_y, duration=0.05)

            # Pinch = drag
            thumb_tip = (int(lm[4].x * w), int(lm[4].y * h))
            index_tip = (ix, iy)
            pinch_dist = distance(thumb_tip, index_tip)

            if pinch_dist < 35:
                if not dragging:
                    pyautogui.mouseDown()
                    dragging = True
            else:
                if dragging:
                    pyautogui.mouseUp()
                    dragging = False

            # Timed gestures
            now = time.time()
            if now - last_action_time > ACTION_DELAY:

                # Open palm → Play / Pause
                if count == 5:
                    pyautogui.press("playpause")
                    last_action_time = now

                # Fist → Pause
                elif count == 0:
                    pyautogui.press("playpause")
                    last_action_time = now

                # Two fingers → Next tab
                elif count == 2:
                    pyautogui.hotkey("ctrl", "tab")
                    last_action_time = now

                # Thumbs up → Volume up
                elif fingers[0] and not any(fingers[1:]):
                    pyautogui.press("volumeup")
                    last_action_time = now

                # One finger down → Volume down
                elif not fingers[0] and count == 1:
                    pyautogui.press("volumedown")
                    last_action_time = now

                # Four fingers → Minimize all
                elif count == 4:
                    pyautogui.hotkey("win", "d")
                    last_action_time = now

    cv2.imshow("Gesture Control", img)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
