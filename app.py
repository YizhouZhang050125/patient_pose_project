import csv
import os
from collections import defaultdict, deque

import cv2
import mediapipe as mp

from utils import calculate_angle

INPUT_VIDEO = "input/test1.MOV"
OUTPUT_VIDEO = "output/annotated_output2.mp4"
OUTPUT_CSV = "output/angles2.csv"

os.makedirs("output", exist_ok=True)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def to_pixel_coords(landmark, width, height):
    return int(landmark.x * width), int(landmark.y * height)


def is_visible(landmark, threshold=0.5):
    return getattr(landmark, "visibility", 1.0) >= threshold


def clamp_point(x, y, width, height, pad=10):
    x = max(pad, min(x, width - pad))
    y = max(pad, min(y, height - pad))
    return x, y


def draw_label(frame, text, anchor_point, width, height, used_boxes):
    """
    Draw label near anchor_point while trying to avoid overlap with other labels.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 2
    padding = 4

    text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
    tw, th = text_size

    # Candidate offsets around the joint
    offsets = [
        (15, -15),
        (15, 20),
        (-tw - 15, -15),
        (-tw - 15, 20),
        (20, -35),
        (20, 40),
        (-tw - 20, -35),
        (-tw - 20, 40),
        (0, -45),
        (0, 55),
    ]

    ax, ay = anchor_point
    chosen_box = None
    chosen_text_origin = None

    def overlaps(box1, box2):
        x1, y1, x2, y2 = box1
        a1, b1, a2, b2 = box2
        return not (x2 < a1 or a2 < x1 or y2 < b1 or b2 < y1)

    for dx, dy in offsets:
        tx = ax + dx
        ty = ay + dy

        tx, ty = clamp_point(tx, ty, width, height)

        box = (
            tx - padding,
            ty - th - padding,
            tx + tw + padding,
            ty + baseline + padding,
        )

        if not any(overlaps(box, existing) for existing in used_boxes):
            chosen_box = box
            chosen_text_origin = (tx, ty)
            break

    if chosen_box is None:
        tx, ty = clamp_point(ax + 15, ay - 15, width, height)
        chosen_box = (
            tx - padding,
            ty - th - padding,
            tx + tw + padding,
            ty + baseline + padding,
        )
        chosen_text_origin = (tx, ty)

    used_boxes.append(chosen_box)

    x1, y1, x2, y2 = chosen_box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.putText(
        frame,
        text,
        chosen_text_origin,
        font,
        font_scale,
        (0, 255, 0),
        thickness,
        cv2.LINE_AA,
    )


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

# Smooth angles a bit so labels do not jump too much
angle_history = defaultdict(lambda: deque(maxlen=5))

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

                # Landmark indices
                L_SHOULDER = mp_pose.PoseLandmark.LEFT_SHOULDER.value
                L_ELBOW = mp_pose.PoseLandmark.LEFT_ELBOW.value
                L_WRIST = mp_pose.PoseLandmark.LEFT_WRIST.value
                L_INDEX = mp_pose.PoseLandmark.LEFT_INDEX.value
                L_HIP = mp_pose.PoseLandmark.LEFT_HIP.value
                L_KNEE = mp_pose.PoseLandmark.LEFT_KNEE.value
                L_ANKLE = mp_pose.PoseLandmark.LEFT_ANKLE.value
                L_HEEL = mp_pose.PoseLandmark.LEFT_HEEL.value

                R_SHOULDER = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
                R_ELBOW = mp_pose.PoseLandmark.RIGHT_ELBOW.value
                R_WRIST = mp_pose.PoseLandmark.RIGHT_WRIST.value
                R_INDEX = mp_pose.PoseLandmark.RIGHT_INDEX.value
                R_HIP = mp_pose.PoseLandmark.RIGHT_HIP.value
                R_KNEE = mp_pose.PoseLandmark.RIGHT_KNEE.value
                R_ANKLE = mp_pose.PoseLandmark.RIGHT_ANKLE.value
                R_HEEL = mp_pose.PoseLandmark.RIGHT_HEEL.value

                # (joint_name, pointA, jointPointB, pointC)
                joint_definitions = [
                    ("left_shoulder", L_HIP, L_SHOULDER, L_ELBOW),
                    ("right_shoulder", R_HIP, R_SHOULDER, R_ELBOW),
                    ("left_elbow", L_SHOULDER, L_ELBOW, L_WRIST),
                    ("right_elbow", R_SHOULDER, R_ELBOW, R_WRIST),
                    ("left_wrist", L_ELBOW, L_WRIST, L_INDEX),
                    ("right_wrist", R_ELBOW, R_WRIST, R_INDEX),
                    ("left_hip", L_SHOULDER, L_HIP, L_KNEE),
                    ("right_hip", R_SHOULDER, R_HIP, R_KNEE),
                    ("left_knee", L_HIP, L_KNEE, L_ANKLE),
                    ("right_knee", R_HIP, R_KNEE, R_ANKLE),
                    ("left_ankle", L_KNEE, L_ANKLE, L_HEEL),
                    ("right_ankle", R_KNEE, R_ANKLE, R_HEEL),
                ]

                used_boxes = []

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

                    angle_history[joint_name].append(angle)
                    smooth_angle = sum(angle_history[joint_name]) / len(angle_history[joint_name])

                    label = f"{joint_name}: {smooth_angle:.2f}°"
                    draw_label(frame, label, b, width, height, used_boxes)

                    csv_writer.writerow([
                        frame_index,
                        round(frame_index / fps, 3),
                        joint_name,
                        round(smooth_angle, 2)
                    ])

            video_writer.write(frame)
            frame_index += 1

cap.release()
video_writer.release()

print(f"Saved annotated video to: {OUTPUT_VIDEO}")
print(f"Saved angle CSV to: {OUTPUT_CSV}")