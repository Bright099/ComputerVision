# Importing our Libraries
import cv2
import mediapipe as mp
import time
import math

# Mediapipe Aliases
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Shared variable for async detection results
latest_result = None

def result_callback(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

# configure HandLandmarker
options = HandLandmarkerOptions(
    base_options = BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode = VisionRunningMode.LIVE_STREAM,
    num_hands = 2,
    min_hand_detection_confidence = 0.5,
    min_tracking_confidence = 0.5,
    result_callback = result_callback
)

# Explicit Skeletal connections for all 5 fingers and palm
FINGER_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (0,9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky finger
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm knuckles bridge
    (5, 9), (9, 13), (13, 17)
]

cap = cv2.VideoCapture(0)

def euclidean_distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def classify_gestures(landmarks):
    wrist = landmarks[0]

    finger_pairs = [
        (8, 5),
        (12, 9),
        (16, 13),
        (20, 17)
    ]

    # Check whether each finger is extended i.e. tip is longer than pip
    fingers_open = []
    for tip_idx, pip_idx in finger_pairs:
        d_tip = euclidean_distance(wrist, landmarks[tip_idx])
        d_mcp = euclidean_distance(wrist, landmarks[pip_idx])
        fingers_open.append(d_tip > d_mcp)

    if fingers_open == [False, False, False, False]:
        return "Rock"
    elif fingers_open == [True, True, True, True]:
        return "Paper"
    elif fingers_open == [True, True, False, False]:
        return "Scissors"
    else:
        return "Unknown"


with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        h, w, _ = frame.shape

        # Convert to RGB for Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        #send frame to async landmarker
        timestamp = int(time.time() * 1000)
        landmarker.detect_async(mp_image, timestamp)

        # divide the screen view into two
        mid_x = w // 2
        cv2.line(frame, (mid_x, 0), (mid_x, h), (255, 255, 255), 2)

        # Render hand landmarks
        if latest_result and latest_result.hand_landmarks:
            for hand_landmarks in latest_result.hand_landmarks:
                # Convert all 21 landmarks to pixel coordinates
                coords = []
                for pt in hand_landmarks:
                    px, py = int(pt.x * w), int(pt.y * h)
                    coords.append((px, py))

                # Draw finger bones
                for start_idx, end_idx in FINGER_CONNECTIONS:
                    cv2.line(frame, coords[start_idx], coords[end_idx], (0, 255, 255), 2)

                # Draw joints: Highlight fingertips in red, other in yellow
                for idx, (px, py) in enumerate(coords):
                    color = (0, 0, 255) if idx in [4, 8, 12, 16, 20] else (0, 255, 0)
                    radius = 5 if idx in [4, 8, 12, 16, 20] else 3
                    cv2.circle(frame, (px, py), radius, color, -1)

                # classify gesture for current hand
                current_gesture = classify_gestures(hand_landmarks)

                # dividing the hands into left and right sides
                wrist_x = int(hand_landmarks[0].x * w)

                if wrist_x < mid_x:
                    side = "Left Side"
                    # Render left side measurement
                    cv2.putText(
                        frame,
                        f"Left: {current_gesture}",
                        (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (255, 255, 255),
                        3
                    )
                else:
                    side = "Right Side"
                    # Render right side measurement
                    cv2.putText(
                        frame,
                        f"Right: {current_gesture}",
                        (mid_x + 30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (255, 255, 255),
                        3
                    )

        cv2.imshow("Rock, Paper, Scissors detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()