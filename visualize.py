import cv2
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 화면 출력 없이 백엔드 Agg 사용
import matplotlib.pyplot as plt
import time

# --- 설정 ---
exercises = ['1-frontcut', '1-leftcut', '1-rightcut']
video_paths = [f'./output/{excercise}/{excercise}.mp4' for excercise in exercises]

# 1. 비디오 프레임 미리 로드 (preload)
video_frames = []
for vp in video_paths:
    cap = cv2.VideoCapture(vp)
    frames_list = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 매 프레임마다 BGR->RGB 한 번만 수행하여 저장
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_list.append(frame_rgb)
    cap.release()
    video_frames.append(frames_list)

# 2. 각 비디오의 스켈레톤 데이터와 confidence data 로드
# pickle 파일에는 data['data']와 data['conf']가 있으며, 각 리스트의 0번째 요소를 사용 (관절 index 0)
joints_data_list = []
conf_data_list = []
for excercise in exercises:
    with open(f'./output/{excercise}/{excercise}.pkl', 'rb') as f:
        data = pickle.load(f)
        joints_data = np.array(data['data'][15])   # shape: (n_frames, 3)
        conf_data = np.array(data['conf'][15])       # shape: (n_frames,) (confidence score)
        joints_data_list.append(joints_data)
        conf_data_list.append(conf_data)

# 3. 모든 비디오, 스켈레톤, confidence 데이터의 최소 프레임 수 사용 (동기화)
min_frames_video = min(len(frames) for frames in video_frames)
min_frames_skel = min(joints.shape[0] for joints in joints_data_list)
min_frames_conf = min(conf.shape[0] for conf in conf_data_list)
frames = min(min_frames_video, min_frames_skel, min_frames_conf)

# --- 프레임 스킵 설정 (속도 가속용) ---
frame_skip = 2  # 예: 2프레임마다 한 프레임 처리 (전체 프레임의 절반 사용)
new_frames = list(range(0, frames, frame_skip))
new_total_frames = len(new_frames)

# 4. FPS 정보 읽기 (첫 번째 비디오 기준; 실패 시 30fps)
cap_temp = cv2.VideoCapture(video_paths[0])
fps = cap_temp.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30
cap_temp.release()
# 효과적인 출력 FPS (frame skip에 따라 낮춤)
new_fps = fps / frame_skip

# --- Matplotlib Figure 구성 ---
N_VIDEOS = len(video_paths)
# 3행: 상단 비디오, 중간 관절 y 좌표, 하단 confidence score
fig, axs = plt.subplots(3, N_VIDEOS, figsize=(15, 12))

# 상단: 비디오 프레임 표시
img_plots = []
for i in range(N_VIDEOS):
    first_frame = video_frames[i][0]
    img_plot = axs[0, i].imshow(first_frame)
    axs[0, i].axis('off')
    axs[0, i].set_title(f'Camera {i+1}')
    img_plots.append(img_plot)

# 중간: 관절 y 좌표 변화 (x축: 누적 frame index, y축: 관절의 y 좌표)
line_plots_joint = []
for i in range(N_VIDEOS):
    line_plot, = axs[1, i].plot([], [], 'o-', lw=2)
    axs[1, i].set_xlim(0, new_total_frames)
    y_vals = joints_data_list[i][:, 1]  # 관절의 y 좌표 (index 1)
    margin = 10
    axs[1, i].set_ylim(np.min(y_vals) - margin, np.max(y_vals) + margin)
    axs[1, i].set_xlabel('Frame')
    axs[1, i].set_ylabel('Joint Y Coordinate')
    axs[1, i].set_title(f'Joint Y over Frames (Camera {i+1})')
    line_plots_joint.append(line_plot)

# 하단: confidence score 변화 (x축: 누적 frame index, y축: confidence score)
line_plots_conf = []
for i in range(N_VIDEOS):
    line_plot, = axs[2, i].plot([], [], 'o-', lw=2)
    axs[2, i].set_xlim(0, new_total_frames)
    conf_vals = conf_data_list[i]
    margin_conf = 0.1  # 보통 confidence는 0~1 범위로 가정 (필요시 조정)
    axs[2, i].set_ylim(0, 1)
    axs[2, i].set_xlabel('Frame')
    axs[2, i].set_ylabel('Confidence Score')
    axs[2, i].set_title(f'Confidence over Frames (Camera {i+1})')
    line_plots_conf.append(line_plot)

# 누적 데이터를 저장할 리스트 초기화 (joint y와 confidence 각각)
accumulated_frames = [[] for _ in range(N_VIDEOS)]
accumulated_y = [[] for _ in range(N_VIDEOS)]
accumulated_conf = [[] for _ in range(N_VIDEOS)]

# --- cv2.VideoWriter 설정 ---
fig.canvas.draw()
s, (width, height) = fig.canvas.print_to_buffer()
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter("output_video-wconf.mp4", fourcc, new_fps, (width, height))

start_time = time.time()
for idx, frame_idx in enumerate(new_frames):
    for i in range(N_VIDEOS):
        # 상단: 미리 로드한 비디오 프레임 업데이트
        img_plots[i].set_data(video_frames[i][frame_idx])
        # 중간: 관절 y 좌표 누적 업데이트
        joint_y = joints_data_list[i][frame_idx][1]
        accumulated_frames[i].append(idx)
        accumulated_y[i].append(joint_y)
        line_plots_joint[i].set_data(accumulated_frames[i], accumulated_y[i])
        # 하단: confidence score 누적 업데이트
        conf_score = conf_data_list[i][frame_idx]
        accumulated_conf[i].append(conf_score)
        line_plots_conf[i].set_data(accumulated_frames[i], accumulated_conf[i])
    
    # Figure를 canvas에 그린 후 버퍼에서 이미지 배열로 변환
    fig.canvas.draw()
    s, (width, height) = fig.canvas.print_to_buffer()
    frame_image = np.frombuffer(s, dtype=np.uint8).reshape(height, width, 4)
    # RGBA -> BGR (OpenCV 포맷)
    frame_image = cv2.cvtColor(frame_image, cv2.COLOR_RGBA2BGR)
    
    video_writer.write(frame_image)
    
    if idx % 50 == 0:
        elapsed = time.time() - start_time
        print(f"Processed frame {idx}/{new_total_frames} in {elapsed:.2f} seconds")

video_writer.release()
plt.close(fig)
print("Video saved as output_video.mp4")
