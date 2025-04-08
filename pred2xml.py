import pickle
import numpy as np
from ultralytics import YOLO
import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET

import videoProcess as vp

def prettify_and_save_xml(root_element, filename):
    rough_string = ET.tostring(root_element, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(reparsed.toprettyxml(indent="  "))  # 2-space indentation

if __name__ == '__main__':
    video_name = '1-leftcut'
    run_idx = video_name.split('-')[0]
    run_flag = False
    try:
        with open(f'./output/{video_name}/{video_name}.pkl', 'rb') as f:
            results = pickle.load(f)
        N_FRAMES = len(results['keypoints'])
        keypoints = np.array(results['keypoints'])
        interpolated_keypoints = vp.interpolate_keypoints(keypoints)
        smoothed_keypoints = vp.smooth_all_keypoints(interpolated_keypoints, window_size=21)
        results['keypoints'] = smoothed_keypoints
        
    except FileNotFoundError:
        model = YOLO('yolo11x-pose.pt')
        video_path = f'./data/{run_idx}/{video_name}.mp4'
        results = model(source=video_path)
        N_FRAMES = len(results)
        run_flag = True
    
    keypoint_labels = ["nose", "leye", "reye", "lear", "rear",
                    "lshoulder", "rshoulder", "lelbow", "relbow",
                    "lwrist", "rwrist", "lhip", "rhip",
                    "lknee", "rknee", "lankle", "rankle"]
    num_keypoints = len(keypoint_labels)

    annotations = ET.Element("annotations")
    ET.SubElement(annotations, "version").text = "1.1"

    # keypoint별 track 생성
    tracks = []
    for idx, label in enumerate(keypoint_labels):
        track = ET.SubElement(annotations, "track", id=str(idx), label=label, source="manual")
        tracks.append(track)

    frame_num = 0
    frame_step = 2
    for ridx in range(N_FRAMES):
        if ridx % frame_step == 0:
            if run_flag:
                keypoints_list = results[ridx].keypoints.xy
                if len(keypoints_list) == 0:
                    frame_num += frame_step
                    continue
                keypoints = keypoints_list[0].tolist() # (17, 2)
            else:
                keypoints = results['keypoints'][ridx]  # (17, 2)
                
            
            # keypoints_list = results[ridx].keypoints.xy  if run_flag else results['keypoints'][ridx]
            # if len(keypoints_list) == 0:
            #     frame_num += frame_step
            #     continue
            # keypoints = keypoints_list[0].tolist()  # 첫 번째 사람만 사용 (원하면 여러명 지원 가능)

            for kp_idx, (x, y) in enumerate(keypoints):
                point = ET.SubElement(tracks[kp_idx], "points",
                                    frame=str(frame_num),
                                    keyframe="1",
                                    outside="0",
                                    occluded="0",
                                    points=f"{x:.2f},{y:.2f}",
                                    z_order="0")
                attr = ET.SubElement(point, "attribute", name="visibility")
                attr.text = "visible"  # YOLO는 visibility score 안 줌

            frame_num += frame_step

    # XML 저장
    tree = ET.ElementTree(annotations)
    prettify_and_save_xml(annotations, f"{video_name}.xml")
