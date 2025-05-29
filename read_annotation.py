import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

def parse_cvat_annotations_to_arrays(xml_path):
    """
    CVAT annotations.xml을 읽어서,
    각 관절(label)별로 (N, 2) NumPy 배열을 반환합니다.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    joint_arrays = {}
    # <track> 요소 반복: 각 track은 하나의 관절(joint)
    for track in root.findall('track'):
        label = track.get('label')  # e.g. "nose", "leye", ...
        pts = []
        # <points> 태그마다 "points" 속성(x,y)을 추출
        for pt_elem in track.findall('points'):
            x_str, y_str = pt_elem.get('points').split(',')
            x, y = float(x_str), float(y_str)
            pts.append([x, y])
        # (N,2) 배열로 변환
        arr = np.array(pts, dtype=float)
        joint_arrays[label] = arr

    return joint_arrays

# 사용 예
if __name__ == "__main__":
    xml_file = "./data/L1/annotations.xml"
    joints = parse_cvat_annotations_to_arrays(xml_file)

    # 예: 코(“nose”) 관절 좌표 시퀀스
    print(joints.keys())
    nose_arr = joints["rhip"]
    plt.plot(nose_arr[:, 1], 'ro-')
    plt.show()
    import pickle
    with open('joints.pkl', 'wb') as f:
        pickle.dump(joints, f)
    
    