import cv2
import numpy as np
import os

file_path = './data/15-zack-squat-1/diag.mp4'
output_dir = './sampled_frames'
N = 10  # 저장할 프레임 개수

os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(file_path)
if not cap.isOpened():
    raise IOError(f"Cannot open video file {file_path}")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if total_frames == 0:
    raise ValueError("Video contains no frames")

indices = np.linspace(1, total_frames, N, endpoint=True, dtype=int)

for i, frame_no in enumerate(indices, start=1):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no - 1)
    ret, frame = cap.read()
    if not ret:
        print(f"Warning: Frame {frame_no} could not be read.")
        continue

    # 보정: 시계 방향으로 90도 회전
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    out_path = os.path.join(output_dir, f"frame_{i+20}.png")
    cv2.imwrite(out_path, frame)
    print(f"Saved frame {frame_no} -> {out_path}")

cap.release()
