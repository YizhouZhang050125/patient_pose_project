import csv
import os

import cv2
import mediapipe as mp

from utils import calculate_angle

from collections import deque

INPUT_VIDEO = "input/78466397363__C4D3D6CF-0C02-4B4A-846E-A6635882E15D.MOV"
OUTPUT_VIDEO = "output/annotated_output1.mp4"
OUTPUT_CSV = "output/angles1.csv"

os.makedirs("output", exist_ok=True)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def to_pixel_coords(landmark, width, height):
    return int(landmark.x * width), int(landmark.y * height)


def is_visible(landmark, threshold=0.5):
    # MediaPipe Pose includes visibility for landmarks
    return getattr(landmark, "visibility", 1.0) >= threshold


cap = cv2.VideoCapture(INPUT_VIDEO)

if not cap.isOpened():
    raise RuntimeError(f"Could not open input video: {INPUT_VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30.0

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "time_sec", "joint", "angle_deg"])

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:

        frame_index = 0

        while True:
            success, frame = cap.read()
            if not success:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

                L_SHOULDER = mp_pose.PoseLandmark.LEFT_SHOULDER.value
                L_ELBOW = mp_pose.PoseLandmark.LEFT_ELBOW.value
                L_WRIST = mp_pose.PoseLandmark.LEFT_WRIST.value

                R_SHOULDER = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
                R_ELBOW = mp_pose.PoseLandmark.RIGHT_ELBOW.value
                R_WRIST = mp_pose.PoseLandmark.RIGHT_WRIST.value

                L_HIP = mp_pose.PoseLandmark.LEFT_HIP.value
                L_KNEE = mp_pose.PoseLandmark.LEFT_KNEE.value
                L_ANKLE = mp_pose.PoseLandmark.LEFT_ANKLE.value

                R_HIP = mp_pose.PoseLandmark.RIGHT_HIP.value
                R_KNEE = mp_pose.PoseLandmark.RIGHT_KNEE.value
                R_ANKLE = mp_pose.PoseLandmark.RIGHT_ANKLE.value

                joint_definitions = [
                    ("left_elbow", L_SHOULDER, L_ELBOW, L_WRIST),
                    ("right_elbow", R_SHOULDER, R_ELBOW, R_WRIST),
                    ("left_knee", L_HIP, L_KNEE, L_ANKLE),
                    ("right_knee", R_HIP, R_KNEE, R_ANKLE),
                ]

                for joint_name, a_idx, b_idx, c_idx in joint_definitions:
                    a_lm = landmarks[a_idx]
                    b_lm = landmarks[b_idx]
                    c_lm = landmarks[c_idx]

                    if not (is_visible(a_lm) and is_visible(b_lm) and is_visible(c_lm)):
                        continue

                    a = to_pixel_coords(a_lm, width, height)
                    b = to_pixel_coords(b_lm, width, height)
                    c = to_pixel_coords(c_lm, width, height)

                    angle = calculate_angle(a, b, c)
                    if angle is None:
                        continue

                    label = f"{joint_name}: {int(round(angle))}°"

                    cv2.putText(
                        frame,
                        label,
                        (b[0] + 10, b[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA
                    )

                    csv_writer.writerow([
                        frame_index,
                        round(frame_index / fps, 3),
                        joint_name,
                        round(angle, 2)
                    ])

            video_writer.write(frame)
            frame_index += 1

cap.release()
video_writer.release()

print(f"Saved annotated video to: {OUTPUT_VIDEO}")
print(f"Saved angle CSV to: {OUTPUT_CSV}")