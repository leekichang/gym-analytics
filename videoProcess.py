import numpy as np

def is_missing(point):
    """
    좌표 point가 (0,0)인지 판별.
    """
    return np.all(point == 0)

def find_missing_intervals(joint_data):
    """
    joint_data: (N, 2) shape의 한 관절 좌표 배열
    반환: missing 구간의 리스트 [(start_idx, end_idx), ...] 
          start_idx는 missing 구간의 시작 프레임, end_idx는 missing이 끝난 첫 프레임의 인덱스
    """
    intervals = []
    N = joint_data.shape[0]
    t = 0
    while t < N:
        if is_missing(joint_data[t]):
            start_idx = t
            while t < N and is_missing(joint_data[t]):
                t += 1
            end_idx = t  # missing이 아닌 첫 프레임 (또는 N)
            intervals.append((start_idx, end_idx))
        else:
            t += 1
    return intervals

def interpolate_interval(joint_data, prev_idx, next_idx, start_idx, end_idx):
    """
    joint_data: (N, 2) 배열, 한 관절에 대한 데이터
    prev_idx: missing 구간 이전에 valid한 프레임 인덱스
    next_idx: missing 구간 이후에 valid한 프레임 인덱스
    start_idx, end_idx: missing 구간 (start_idx 포함, end_idx 미포함)
    
    선형 보간을 수행하여 missing 구간을 채움.
    """
    prev_point = joint_data[prev_idx].astype(np.float32)
    next_point = joint_data[next_idx].astype(np.float32)
    gap = next_idx - prev_idx
    for t in range(start_idx, end_idx):
        ratio = (t - prev_idx) / gap
        joint_data[t] = prev_point + ratio * (next_point - prev_point)
    return joint_data

def interpolate_keypoints(keypoints):
    """
    keypoints: (N, 17, 2) numpy array, missing 좌표는 (0,0)으로 표시됨.
    반환: missing 좌표가 선형 보간 혹은 경계 처리로 채워진 배열.
    
    - 만약 가장 초기 프레임들이 missing인 경우, 첫 valid한 값으로 채움.
    - 만약 마지막 프레임들이 missing인 경우, 마지막 valid한 값으로 채움.
    - 중간에 missing 구간이 있으면 이전과 이후 valid 값으로 선형 보간.
    """
    keypoints_interp = keypoints.copy()
    num_frames, num_joints, _ = keypoints.shape

    # 각 관절별 처리
    for joint in range(num_joints):
        joint_data = keypoints_interp[:, joint].copy()  # (N,2)
        
        # 경계 처리: 처음 프레임이 missing인 경우
        if is_missing(joint_data[0]):
            # 첫번째 valid한 값 찾기
            first_valid = None
            for i in range(num_frames):
                if not is_missing(joint_data[i]):
                    first_valid = joint_data[i]
                    break
            # 전체가 missing인 경우는 건너뜀
            if first_valid is not None:
                joint_data[:i] = first_valid

        # 경계 처리: 마지막 프레임이 missing인 경우
        if is_missing(joint_data[-1]):
            last_valid = None
            for i in range(num_frames-1, -1, -1):
                if not is_missing(joint_data[i]):
                    last_valid = joint_data[i]
                    break
            if last_valid is not None:
                joint_data[i+1:] = last_valid

        # 중간 missing 구간에 대해 선형 보간 적용
        missing_intervals = find_missing_intervals(joint_data)
        for (start_idx, end_idx) in missing_intervals:
            # 이미 처리된 경계 구간은 건너뛰기
            if start_idx == 0 or end_idx == num_frames:
                continue

            prev_idx = start_idx - 1
            next_idx = end_idx
            # 이전과 이후 valid 값이 있는 경우 선형 보간
            if not is_missing(joint_data[prev_idx]) and not is_missing(joint_data[next_idx]):
                joint_data = interpolate_interval(joint_data, prev_idx, next_idx, start_idx, end_idx)
                
        keypoints_interp[:, joint] = joint_data

    return keypoints_interp

def smooth_joint_keypoints(joint_data, window_size=3):
    """
    joint_data: (N, 2) 배열, 한 관절의 x,y 좌표 시계열 데이터
    window_size: 이동 평균 윈도우 사이즈
    반환: smoothing 처리된 (N, 2) 배열
    
    경계 문제를 피하기 위해 'edge' 패딩을 사용.
    """
    smoothed_data = np.zeros_like(joint_data)
    for coord in range(2):
        x = joint_data[:, coord]
        pad_width = window_size // 2
        # 경계값 확장을 위해 'edge' 모드 사용
        x_padded = np.pad(x, pad_width, mode='reflect')
        # padding 후 valid convolution
        smoothed_data[:, coord] = np.convolve(x_padded, np.ones(window_size)/window_size, mode='valid')
    return smoothed_data

def smooth_all_keypoints(keypoints, window_size=3):
    """
    keypoints: (N, 17, 2) numpy array
    window_size: 이동 평균 윈도우 사이즈
    반환: 각 관절별로 moving average smoothing이 적용된 keypoints 배열.
    """
    keypoints_smoothed = keypoints.copy()
    num_frames, num_joints, _ = keypoints.shape

    for joint in range(num_joints):
        joint_data = keypoints_smoothed[:, joint].copy()  # (N,2)
        keypoints_smoothed[:, joint] = smooth_joint_keypoints(joint_data, window_size)
    return keypoints_smoothed