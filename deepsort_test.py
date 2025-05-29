import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# IoU 계산 함수 (두 박스 형식: [x1, y1, x2, y2])
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    if area1 + area2 - inter_area == 0:
        return 0
    return inter_area / float(area1 + area2 - inter_area)

# COCO 기준 skeleton 연결 정보 (0-indexed)
skeleton_connections = [
    (15, 13), (13, 11), (16, 14), (14, 12),
    (11, 12), (5, 11), (6, 12), (5, 6),
    (5, 7), (6, 8), (7, 9), (8, 10),
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4)
]

# YOLO 모델 로드
model = YOLO("yolo11x-pose.pt")

# Deep SORT 트래커 초기화
tracker = DeepSort(max_age=30, n_init=3)

# 비디오 파일 열기
cap = cv2.VideoCapture("./output.mp4")

# 출력 비디오 저장 설정 (입력 비디오와 동일한 크기 및 FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("tracked_output.mp4", fourcc, fps, (width, height))

# detection_data: YOLO 검출 시 bbox와 함께 keypoints 저장
# 각 항목: {"bbox": [x1, y1, x2, y2], "keypoints": keypoints_array}
detection_data = []
for_save = {'keypoints': [], 'conf':[]}  # keypoints 저장용
while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = []
    detection_data.clear()  # 매 프레임마다 초기화

    # YOLO를 통한 객체 및 pose 검출
    results = model(frame)
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            # bounding box: [x1, y1, x2, y2]
            xyxy = box.xyxy[0].tolist()
            confidence = box.conf.item()
            if confidence < 0.5:
                continue

            # Deep SORT용 bbox: [x, y, w, h]
            x1, y1, x2, y2 = xyxy
            w = x2 - x1
            h = y2 - y1
            detections.append(([x1, y1, w, h], confidence, "person"))

            # keypoints 처리: keypoints는 보통 (N, 3) 배열 (x, y, conf)
            kp = None
            if hasattr(result, "keypoints") and result.keypoints is not None:
                # keypoints를 numpy array로 변환 (shape: [num_keypoints, 3])
                kp = result.keypoints.cpu().numpy() if hasattr(result.keypoints, 'cpu') else np.array(box.keypoints)
            # kp.data 대신, kp 자체가 numpy array라고 가정 (필요한 경우 np.array(kp)로 변환)
            detection_data.append({
                "bbox": [x1, y1, x2, y2],
                "keypoints": kp  # np.array 형태로 저장됨
            })

    # Deep SORT 트래커 갱신 (appearance embedding 추출용으로 frame 전달)
    tracks = tracker.update_tracks(detections, frame=frame)

    # 각 track에 대해 bounding box와 track id 표시 및 skeleton overlay (여기선 ID '5'만 예시)
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        if str(track_id) == '5':
            ltrb = track.to_ltrb()  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = map(int, ltrb)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            # track bbox와 가장 잘 매칭되는 detection의 keypoints 찾기 (IoU 기준)
            best_iou = 0
            best_kp = None
            for idx, det in enumerate(detection_data):
                iou = compute_iou(det["bbox"], [x1, y1, x2, y2])
                # print(f"IoU: {iou:.2f}")
                if iou > best_iou:
                    best_iou = iou
                    best_kp = det["keypoints"][idx]
                
            # IoU가 일정 threshold 이상이면 skeleton을 그리기 (예: 0.3 이상)
         
            if best_kp is not None and best_iou > 0.3:
                # 각 keypoint에 대해 원 그리기 (신뢰도 0.3 이상)
                for point in best_kp:
                    # print(point.data[0])
                    kp_x, kp_y, kp_conf = point.data[0].transpose(1,0) # (17,3) -> (3,17)
                    for_save['keypoints'].append([kp_x, kp_y])
                    for_save['conf'].append(kp_conf)
                    for x_, y_, conf_ in zip(kp_x, kp_y, kp_conf):
                        if float(conf_) < 0.3:
                            continue
                        cv2.circle(frame, (int(x_), int(y_)), 3, (0, 0, 255), -1)
                    for p1, p2 in skeleton_connections:
                        if kp_conf[p1] > 0.5 and kp_conf[p2] > 0.5:  # 신뢰도 체크
                            cv2.line(frame, (int(kp_x[p1]), int(kp_y[p1])), (int(kp_x[p2]), int(kp_y[p2])), (255, 0, 0), 2)

    # 처리된 프레임 출력 또는 저장
    out.write(frame)

cap.release()
out.release()
print("처리 완료. 'tracked_output.mp4' 파일을 확인하세요.")
import pickle
print(np.shape(for_save["keypoints"]))
print(np.shape(for_save["conf"]))
for_save['keypoints'] = np.array(for_save['keypoints']).transpose(0,2,1)
for_save['conf'] = np.array(for_save['conf'])
with open("tracked_output.pkl", "wb") as f:
    pickle.dump(for_save, f)