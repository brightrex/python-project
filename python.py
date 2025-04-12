'''This is hand gesture scanning code to run this code please install cv ,mediapipe, pyautogui and numpy using this commands
pip install opencv-python
pip install mediapipe
pip install pyautogui
pip install numpy '''
import cv2
import mediapipe as mp
import pyautogui
import math
import numpy as np
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

screen_width, screen_height = pyautogui.size()
pyautogui.FAILSAFE = False

cap = cv2.VideoCapture(0)
cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

smoothing = 5
prev_x, prev_y = 0, 0

is_dragging = False
prev_hand_closed = False
prev_finger_closed = None
last_click_time = 0
click_cooldown = 0.5
move_sensitivity = 2.0

scroll_sensitivity = 80
scroll_smoothing = 3
fist_state_duration = 0
fist_detection_threshold = 3

y_positions = []
max_positions = 5

def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x)*2 + (point1.y - point2.y)*2)

def is_hand_closed(landmarks):
    palm_base = landmarks[0]
    fingertips = [landmarks[4], landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
    palm_center = landmarks[9]
    distances = [calculate_distance(tip, palm_center) for tip in fingertips]
    avg_distance = sum(distances) / len(distances)
    hand_size = calculate_distance(landmarks[0], landmarks[9])
    return avg_distance < hand_size * 0.5

def is_fist(landmarks):
    fingertips = [landmarks[4], landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
    mid_joints = [landmarks[3], landmarks[6], landmarks[10], landmarks[14], landmarks[18]]
    palm_center = landmarks[9]
    all_fingers_bent = all(tip.y > mid.y for tip, mid in zip(fingertips[1:], mid_joints[1:]))
    thumb_tucked = calculate_distance(fingertips[0], palm_center) < calculate_distance(landmarks[0], palm_center) * 0.4
    return all_fingers_bent and thumb_tucked

def is_finger_closed(landmarks, finger_id):
    fingertips = [4, 8, 12, 16, 20]
    finger_bases = [2, 5, 9, 13, 17]
    middle_joints = [3, 6, 10, 14, 18]
    tip = landmarks[fingertips[finger_id]]
    base = landmarks[finger_bases[finger_id]]
    mid = landmarks[middle_joints[finger_id]]
    palm = landmarks[0]
    if finger_id == 0:
        v1 = np.array([base.x - palm.x, base.y - palm.y])
        v1 = v1 / np.linalg.norm(v1)
        v2 = np.array([tip.x - base.x, tip.y - base.y])
        v2 = v2 / np.linalg.norm(v2)
        angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
        return angle > 0.6
    else:
        return tip.y > mid.y

def hand_gesture(landmarks):
    hand_closed = is_hand_closed(landmarks)
    fingers_closed = [is_finger_closed(landmarks, i) for i in range(5)]
    fist = is_fist(landmarks)
    return {
        'hand_closed': hand_closed,
        'fist': fist,
        'fingers_closed': fingers_closed,
        'palm_position': (landmarks[9].x, landmarks[9].y)
    }

fist_detected = False
start_y = None
prev_scroll_amount = 0
current_fist_frames = 0

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    status_text = "No hand detected"
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark
            gesture = hand_gesture(landmarks)
            current_time = time.time()
            palm_x, palm_y = gesture['palm_position']
            if gesture['fist']:
                current_fist_frames += 1
                status_text = "Fist detected - Scrolling mode"
                if current_fist_frames >= fist_detection_threshold:
                    if not fist_detected:
                        fist_detected = True
                        start_y = palm_y
                        y_positions = [palm_y]
                    else:
                        y_positions.append(palm_y)
                        if len(y_positions) > max_positions:
                            y_positions.pop(0)
                        current_y = sum(y_positions) / len(y_positions)
                        y_diff = current_y - start_y
                        raw_scroll_amount = int(y_diff * scroll_sensitivity)
                        scroll_amount = prev_scroll_amount + (raw_scroll_amount - prev_scroll_amount) / scroll_smoothing
                        prev_scroll_amount = scroll_amount
                        pyautogui.scroll(-int(scroll_amount))
                        cv2.putText(image, f"Scroll: {-int(scroll_amount)}", (10, 60), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                fist_detected = False
                current_fist_frames = 0
                start_y = None
                prev_scroll_amount = 0
                if not gesture['hand_closed']:
                    status_text = "Open palm - Moving cursor"
                    target_x = int((palm_x * screen_width * move_sensitivity) % screen_width)
                    target_y = int((palm_y * screen_height * move_sensitivity) % screen_height)
                    cursor_x = prev_x + (target_x - prev_x) / smoothing
                    cursor_y = prev_y + (target_y - prev_y) / smoothing
                    prev_x, prev_y = cursor_x, cursor_y
                    pyautogui.moveTo(cursor_x, cursor_y)
                    if not prev_hand_closed:
                        for i, is_closed in enumerate(gesture['fingers_closed']):
                            if is_closed and prev_finger_closed and prev_finger_closed[i] == False:
                                if current_time - last_click_time > click_cooldown:
                                    if i == 0:
                                        pyautogui.rightClick()
                                        status_text = "Right click!"
                                    else:
                                        pyautogui.click()
                                        status_text = f"Click with finger {i+1}!"
                                    last_click_time = current_time
                prev_finger_closed = gesture['fingers_closed']
            prev_hand_closed = gesture['hand_closed']
    cv2.putText(image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow('Hand Gesture Mouse Control', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
