import cv2
import grad
import os
import csv
import time
import random
from datetime import datetime
import numpy as np
import mediapipe as mp

# Paths
bg_path = r"Personal Projects\data\Nur M-Background.png"
cap_path = r"\Personal Projects\data\faces\gradcap_nobg.png"
attendance_dir = r"\Personal Projects\data"

message_timer = 0
MESSAGE_DURATION = 90  # ~3 seconds at 30 FPS

# Load background image
imgBackground = cv2.imread(bg_path)

# Load grad cap image
cap_img = cv2.imread(cap_path, cv2.IMREAD_UNCHANGED)
cap_w, cap_h = 100, 100
cap_img = cv2.resize(cap_img, (cap_w, cap_h))
cap_visible = True

# Camera feed dimensions
cam_w, cam_h = 420, 340
cam_x = (imgBackground.shape[1] - cam_w) // 2
cam_y = 90


bg_h, bg_w = imgBackground.shape[:2]
required_height = cam_y + cam_h
required_width = cam_x + cam_w
if bg_h < required_height or bg_w < required_width:
    imgBackground = cv2.resize(imgBackground, (max(required_width, bg_w), max(required_height, bg_h)))

# Initialize webcam
video = cv2.VideoCapture(0)


COL_NAMES = ['NAME', 'TIME']

# Confetti state
confetti_active = False
confetti_timer = 0
CONFETTI_DURATION = 30

# MediaPipe hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

def draw_confetti(frame):
    for _ in range(100):
        x = random.randint(0, frame.shape[1])
        y = random.randint(0, frame.shape[0])
        color = random.choice([
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255)
        ])
        radius = random.randint(2, 5)
        cv2.circle(frame, (x, y), radius, color, -1)

def draw_static_message(frame, x, y, w, h):
    message1 = "Name: "
    message2 = "This user will graduate tomorrow."
    message3 = "Congratulations!"

    font = cv2.FONT_HERSHEY_COMPLEX
    font_scale = 0.7
    thickness = 2

    (w1, h1), _ = cv2.getTextSize(message1, font, font_scale, thickness)
    (w2, h2), _ = cv2.getTextSize(message2, font, font_scale, thickness)
    (w3, h3), _ = cv2.getTextSize(message3, font, font_scale, thickness)

    text_x1 = x + (w - w1) // 2
    text_x2 = x + (w - w2) // 2
    text_x3 = x + (w - w3) // 2

    text_y1 = y + h + 40
    text_y2 = text_y1 + h1 + 15
    text_y3 = text_y2 + h2 + 15
    text_y3 = min(text_y3, frame.shape[0] - 20)

    cv2.rectangle(frame, (text_x1 - 15, text_y1 - 30), (text_x1 + w1 + 15, text_y1 + 10), (0, 165, 255), -1)
    cv2.rectangle(frame, (text_x2 - 15, text_y2 - 30), (text_x2 + w2 + 15, text_y2 + 10), (0, 165, 255), -1)
    cv2.rectangle(frame, (text_x3 - 15, text_y3 - 30), (text_x3 + w3 + 15, text_y3 + 10), (0, 165, 255), -1)

    cv2.putText(frame, message1, (text_x1, text_y1), font, font_scale, (0, 0, 0), 3)
    cv2.putText(frame, message1, (text_x1, text_y1), font, font_scale, (255, 255, 255), 1)

    cv2.putText(frame, message2, (text_x2, text_y2), font, font_scale, (0, 0, 0), 3)
    cv2.putText(frame, message2, (text_x2, text_y2), font, font_scale, (255, 255, 255), 1)

    cv2.putText(frame, message3, (text_x3, text_y3), font, font_scale, (0, 0, 0), 3)
    cv2.putText(frame, message3, (text_x3, text_y3), font, font_scale, (255, 255, 255), 1)

def overlay_image_alpha(background, overlay, x, y):
    b, g, r, a = cv2.split(overlay)
    overlay_rgb = cv2.merge((b, g, r))
    mask = cv2.merge((a, a, a))

    h, w = overlay_rgb.shape[:2]
    roi = background[y:y+h, x:x+w]

    img1_bg = cv2.bitwise_and(roi, cv2.bitwise_not(mask))
    img2_fg = cv2.bitwise_and(overlay_rgb, mask)
    dst = cv2.add(img1_bg, img2_fg)
    background[y:y+h, x:x+w] = dst

while True:
    ret, frame = video.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, (cam_w, cam_h))
    rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(rgb_frame)
    detections = grad.detect_faces(frame_resized)
    imgBackground_copy = imgBackground.copy()

    # Place camera feed onto background
    imgBackground_copy[cam_y:cam_y+cam_h, cam_x:cam_x+cam_w] = frame_resized

    # Grad cap position relative to full background
    cap_x = cam_x + (cam_w - cap_w) // 2
    cap_y = cam_y + (cam_h - cap_h) // 2

    # Show grad cap directly on background copy
    if cap_visible:
        overlay_image_alpha(imgBackground_copy, cap_img, cap_x, cap_y)

    # Check for hand grab
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x_px = int(index_tip.x * cam_w)
            y_px = int(index_tip.y * cam_h)

            # Convert to full background coordinates
            x_px += cam_x
            y_px += cam_y

        if cap_visible and cap_x < x_px < cap_x + cap_w and cap_y < y_px < cap_y + cap_h:
            cap_visible = False
            confetti_active = True
            confetti_timer = CONFETTI_DURATION
            draw_static_message(imgBackground_copy, cap_x, cap_y, cap_w, cap_h)
            message_timer = MESSAGE_DURATION
        
        if message_timer > 0:
            draw_static_message(imgBackground_copy, cap_x, cap_y, cap_w, cap_h)
            message_timer -= 1

    # Face detection and attendance
    attendance = None
    for result in detections:
        x, y, w, h = result["coords"]
        label = result["label"]

        # Draw on frame_resized
        cv2.rectangle(frame_resized, (x, y), (x+w, y+h), (0, 165, 255), 3)
        cv2.rectangle(frame_resized, (x, y-40), (x+w, y), (0, 165, 255), -1)
        cv2.putText(frame_resized, label, (x+5, y-15), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)

        if label == "Name":
            confetti_active = True
            confetti_timer = CONFETTI_DURATION
            draw_static_message(imgBackground_copy, cam_x + x, cam_y + y, w, h)

        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")
        attendance = [label, timestamp]
        os.makedirs(attendance_dir, exist_ok=True)
        attendance_file = os.path.join(attendance_dir, f"Attendance_{date}.csv")
        attendance_exists = os.path.isfile(attendance_file)

    # Draw confetti
    if confetti_active:
        draw_confetti(imgBackground_copy)
        confetti_timer -= 1
        if confetti_timer <= 0:
            confetti_active = False

    # Show final frame
    cv2.imshow("Graduation Face Recognition", imgBackground_copy)

    key = cv2.waitKey(1)
    if key == ord('o') and attendance:
        time.sleep(1)
        with open(attendance_file, "a", newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not attendance_exists:
                writer.writerow(COL_NAMES)
            writer.writerow(attendance)

    elif key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()