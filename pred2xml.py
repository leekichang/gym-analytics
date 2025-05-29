import os
import cv2
import pickle
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, List

import numpy as np
from ultralytics import YOLO
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

import videoProcess as vp

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
YOLO_MODEL_PATH = "yolo11x-pose.pt"
FRAME_STEP = 2
KEYPOINT_LABELS: List[str] = [
    "nose", "leye", "reye", "lear", "rear",
    "lshoulder", "rshoulder", "lelbow", "relbow",
    "lwrist", "rwrist", "lhip", "rhip",
    "lknee", "rknee", "lankle", "rankle",
]

# ---------------------------------------------------------------------------
# XML helpers
# ---------------------------------------------------------------------------

def prettify_xml(elem: ET.Element) -> str:
    """Return a pretty-printed XML string for *elem*."""
    rough = ET.tostring(elem, "utf-8")
    return minidom.parseString(rough).toprettyxml(indent="  ")


def save_xml(elem: ET.Element, path: Path) -> None:
    """Save *elem* to *path* (UTF-8, pretty-printed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(prettify_xml(elem), encoding="utf-8")

# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------

def ensure_vertical(video_path: str) -> str:
    """Return a path to a **portrait-oriented** video.

    If the input is already portrait/square (H ≥ W) the original path is
    returned. Otherwise frames are rotated 90° clockwise and written to a
    temporary file.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if h >= w:
        return video_path  # already portrait

    tmp_path = tempfile.mktemp(suffix="_rot.mp4")
    cmd = [
        "ffmpeg", "-y", "-noautorotate", "-i", video_path,
        "-vf", "transpose=1",  # 90° CW
        "-c:v", "libx264", "-crf", "18",
        "-c:a", "copy",
        "-metadata:s:v:0", "rotate=0",
        tmp_path,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return tmp_path

# ---------------------------------------------------------------------------
# YOLO helpers
# ---------------------------------------------------------------------------

def load_yolo(model_path: str = YOLO_MODEL_PATH) -> YOLO:
    print("[YOLO] Loading model …")
    return YOLO(model_path)

# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------

def cache_path_for(video_name: str) -> Path:
    return Path("./output") / video_name / f"{video_name}.pkl"


def load_cached_results(video_name: str) -> Optional[dict]:
    cache = cache_path_for(video_name)
    if cache.is_file():
        print(f"[cache] Found {cache}. Loading …")
        return pickle.loads(cache.read_bytes())
    return None


def post_process_cached(results: dict) -> dict:
    kp = np.array(results["keypoints"])
    kp = vp.interpolate_keypoints(kp)
    kp = vp.smooth_all_keypoints(kp, window_size=21)
    results["keypoints"] = kp
    return results

# ---------------------------------------------------------------------------
# XML construction
# ---------------------------------------------------------------------------

def build_cvat_xml(results, live: bool, frame_step: int = FRAME_STEP) -> ET.Element:
    """Convert YOLO-Pose output → CVAT XML tree."""
    ann = ET.Element("annotations")
    ET.SubElement(ann, "version").text = "1.1"

    tracks = [
        ET.SubElement(ann, "track", id=str(i), label=lbl, source="manual")
        for i, lbl in enumerate(KEYPOINT_LABELS)
    ]

    total_frames = len(results) if live else len(results["keypoints"])
    frame_idx = 0

    for ridx in range(total_frames):
        if ridx % frame_step:
            continue

        # ------------------------------------------------------------------
        # Extract keypoints for this frame
        # ------------------------------------------------------------------
        if live:
            kp_tensor = results[ridx].keypoints.xy  # torch.Tensor [N, 17, 2]
            if kp_tensor is None or kp_tensor.numel() == 0:
                frame_idx += frame_step
                continue
            keypoints = kp_tensor[0].cpu().tolist()  # first person
        else:
            keypoints = results["keypoints"][ridx]

        # ------------------------------------------------------------------
        # Add <points> entries to the corresponding <track>
        # ------------------------------------------------------------------
        for k, (x, y) in enumerate(keypoints):
            pt = ET.SubElement(
                tracks[k], "points",
                frame=str(frame_idx), keyframe="1", outside="0",
                occluded="0", points=f"{x:.2f},{y:.2f}", z_order="0",
            )
            ET.SubElement(pt, "attribute", name="visibility").text = "visible"

        frame_idx += frame_step

    return ann

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def process_video(video_name: str, camera: str, model: Optional[YOLO] = None) -> None:
    raw_path = Path("./data") / camera / f"{video_name}.mp4"

    cached = load_cached_results(video_name)
    live = cached is None

    if live:
        if model is None:
            model = load_yolo()
        vid_path = ensure_vertical(str(raw_path))
        results = model(source=vid_path)
    else:
        results = post_process_cached(cached)

    xml_root = build_cvat_xml(results, live)
    xml_out = Path("./xmls") / f"{video_name}.xml"
    save_xml(xml_root, xml_out)
    print(f"[✓] Saved annotations → {xml_out.resolve()}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    prs = argparse.ArgumentParser(
        description="Run YOLO-Pose on a video and export CVAT-style XML annotations."
    )
    prs.add_argument("--camera", "-c", required=True,
                     help="Camera/run directory name (e.g. Diag-views)")
    prs.add_argument("--video", "-v", required=True,
                     help="Video file base name without extension (e.g. D12)")
    prs.add_argument("--frame-step", "-s", type=int, default=2,
                     help=f"Process every N-th frame (default={2})")
    args = prs.parse_args()

    # allow dynamic override of FRAME_STEP if user supplies --frame-step
    FRAME_STEP = args.frame_step

    process_video(args.video, args.camera)





# import cv2
# import pickle
# import tempfile
# import subprocess
# import numpy as np
# from ultralytics import YOLO
# import xml.dom.minidom as minidom
# import xml.etree.ElementTree as ET

# import videoProcess as vp

# def prettify_and_save_xml(root_element, filename):
#     rough_string = ET.tostring(root_element, 'utf-8')
#     reparsed = minidom.parseString(rough_string)
#     with open(filename, "w", encoding="utf-8") as f:
#         f.write(reparsed.toprettyxml(indent="  "))  # 2-space indentation

# def ensure_vertical(path: str) -> str:
#     """가로 영상이면 시계방향 90° 회전한 임시 MP4 경로를 반환.
#        이미 세로면 원본 경로 그대로 반환."""
#     cap = cv2.VideoCapture(path)
#     if not cap.isOpened():
#         raise FileNotFoundError(path)
#     w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     cap.release()

#     if h >= w:                       # 이미 세로(또는 정사각형)
#         return path

#     # ---- 가로 => 세로로 물리적 회전 ----
#     tmp = tempfile.mktemp(suffix="_rot.mp4")
#     ffmpeg_cmd = [
#         "ffmpeg", "-y", "-noautorotate",
#         "-i", path,
#         "-vf", "transpose=1",        # 1 = 90° CW, 2 = 90° CCW
#         "-c:v", "libx264", "-crf", "18",
#         "-c:a", "copy",
#         "-metadata:s:v:0", "rotate=0",
#         tmp
#     ]
#     subprocess.run(ffmpeg_cmd, check=True)
#     return tmp

# # ----------------- 사용 예 -----------------
# def run_inference(run_idx: str, video_name: str):
#     print(f"File not found for {video_name}. Loading YOLO model...")
#     model = YOLO("yolo11x-pose.pt")
#     print("Model loaded. Processing video...")

#     raw_path  = f"./data/{run_idx}/{video_name}.mp4"
#     fixed_vid = ensure_vertical(raw_path)   # ← 핵심 추가

#     results = model(source=fixed_vid)       # path 또는 제너레이터 모두 OK
#     return results

# if __name__ == '__main__':
#     video_name = 'D12'
#     run_idx = video_name.split('-')[0]
#     run_flag = False
#     try:
#         with open(f'./output/{video_name}/{video_name}.pkl', 'rb') as f:
#             results = pickle.load(f)
#         N_FRAMES = len(results['keypoints'])
#         keypoints = np.array(results['keypoints'])
#         interpolated_keypoints = vp.interpolate_keypoints(keypoints)
#         smoothed_keypoints = vp.smooth_all_keypoints(interpolated_keypoints, window_size=21)
#         results['keypoints'] = smoothed_keypoints
        
#     except FileNotFoundError:
#         # print(f"File not found for {video_name}. Loading YOLO model...")
#         # model = YOLO('yolo11x-pose.pt')
#         # print("Model loaded. Processing video...")
#         # video_path = f'./data/{run_idx}/{video_name}.mp4'
#         # results = model(source=video_path)
#         results = run_inference('Diag-views', video_name=video_name)
        
#         N_FRAMES = len(results)
#         run_flag = True
    
#     keypoint_labels = ["nose", "leye", "reye", "lear", "rear",
#                     "lshoulder", "rshoulder", "lelbow", "relbow",
#                     "lwrist", "rwrist", "lhip", "rhip",
#                     "lknee", "rknee", "lankle", "rankle"]
#     num_keypoints = len(keypoint_labels)

#     annotations = ET.Element("annotations")
#     ET.SubElement(annotations, "version").text = "1.1"

#     # keypoint별 track 생성
#     tracks = []
#     for idx, label in enumerate(keypoint_labels):
#         track = ET.SubElement(annotations, "track", id=str(idx), label=label, source="manual")
#         tracks.append(track)

#     frame_num = 0
#     frame_step = 2
#     for ridx in range(N_FRAMES):
#         if ridx % frame_step == 0:
#             if run_flag:
#                 keypoints_list = results[ridx].keypoints.xy
#                 if len(keypoints_list) == 0:
#                     frame_num += frame_step
#                     continue
#                 keypoints = keypoints_list[0].tolist() # (17, 2)
#             else:
#                 keypoints = results['keypoints'][ridx]  # (17, 2)
                
            
#             # keypoints_list = results[ridx].keypoints.xy  if run_flag else results['keypoints'][ridx]
#             # if len(keypoints_list) == 0:
#             #     frame_num += frame_step
#             #     continue
#             # keypoints = keypoints_list[0].tolist()  # 첫 번째 사람만 사용 (원하면 여러명 지원 가능)

#             for kp_idx, (x, y) in enumerate(keypoints):
#                 point = ET.SubElement(tracks[kp_idx], "points",
#                                     frame=str(frame_num),
#                                     keyframe="1",
#                                     outside="0",
#                                     occluded="0",
#                                     points=f"{x:.2f},{y:.2f}",
#                                     z_order="0")
#                 attr = ET.SubElement(point, "attribute", name="visibility")
#                 attr.text = "visible"  # YOLO는 visibility score 안 줌

#             frame_num += frame_step

#     # XML 저장
#     tree = ET.ElementTree(annotations)
#     prettify_and_save_xml(annotations, f"./xmls/{video_name}.xml")
