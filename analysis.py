import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt

import config as cfg

def smoothing(data):
    if len(data.shape) == 1:
        return moving_average(medfilt(data, kernel_size=7))
    elif len(data.shape) == 2:
        new_temp = []
        for i in range(data.shape[1]):
            data[:, i] = medfilt(data[:, i], kernel_size=7)
            new_temp.append(moving_average(data[:, i]))
        new_temp = np.array(new_temp)
        return new_temp.T

def moving_average(data, window_size=5):
    # add multi channel support
    if len(data.shape) == 1:
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    elif len(data.shape) == 2:
        new_temp = []
        for i in range(data.shape[1]):
            new_temp.append(np.convolve(data[:, i], np.ones(window_size)/window_size, mode='valid'))
        new_temp = np.array(new_temp)
        return new_temp.T
    else:
        raise ValueError("Data shape is not supported")

def get_velocity(data):
    return np.diff(data, axis=0)

def angles(start_point, e1, e2):
    a = e1 - start_point
    b = e2 - start_point
    dot_product = torch.sum(a * b, dim=1)  # 각 행별로 내적 계산

    # 벡터 크기 계산
    a_norm = torch.norm(a, dim=1)  # 각 벡터의 크기
    b_norm = torch.norm(b, dim=1)  # 각 벡터의 크기

    # 코사인 값 계산 (클리핑 추가로 안정성 확보)
    cos_theta = dot_product / (a_norm * b_norm + 1e-8)
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)  # -1 ~ 1 사이로 제한

    # 각도 변환 (라디안 → 도)
    angles = torch.acos(cos_theta)  # 라디안 단위
    angles_deg = torch.rad2deg(angles)  # 도(degree) 단위 변환
    return angles_deg

data_path = './output'
# exercises = ['1-frontcut', '1-leftcut', '1-rightcut']
exercises = ['D1', 'B1', 'L1']
datas = [f'{data_path}/{p}' for p in exercises]

# exercises = ['pullup-zackery-1']
# exercises = ['pullup-zackery-1', 'pullup-zackery-2', 'pullup-zackery-3']
datas = [f'{data_path}/{p}' for p in exercises]
for idx, p in enumerate(datas):
    with open(p+f'/{exercises[idx]}.pkl', 'rb') as f:
        data = pickle.load(f)
    fig, ax1 = plt.subplots(2, 1, figsize=(12, 8))
    dict_kp = {}
    for i in range(17):
        dict_kp[i] = np.array(data['keypoints'][i])
    data['keypoints'] = dict_kp
    for k,v in data['keypoints'].items():
        # plt.subplot(2,1,1)
        # plt.plot(moving_average(medfilt(torch.stack(v)[:,0].numpy(), kernel_size=7)), label=cfg.label2key[k])
        if k in cfg.upper_body:
            plt.subplot(2,1,1)
            # plt.plot(moving_average(medfilt(torch.stack(v)[:,1].numpy(), kernel_size=7)), label=cfg.label2key[k])
            plt.plot(moving_average(medfilt(v[:,1], kernel_size=7)), label=cfg.label2key[k])
            plt.ylabel('Pixel Value')
            plt.ylim([1000,0])
            plt.legend(ncol=9, loc='upper center', bbox_to_anchor=(0.5, 1.15))
            plt.subplot(2,1,2)
            plt.plot(get_velocity(smoothing(v[:,1])), label=cfg.label2key[k])
            plt.ylabel('Velocity (pixel/frame)')
            plt.ylim([15,-15])
            plt.xlabel('Frame')
    plt.show()
    # plt.figure(figsize=(12,4))
    # left_elbow     = torch.tensor(smoothing(torch.stack(data['keypoints'][cfg.key2label['Left Elbow']]).numpy())    )
    # right_elbow    = torch.tensor(smoothing(torch.stack(data['keypoints'][cfg.key2label['Right Elbow']]).numpy())   )
    # left_shoulder  = torch.tensor(smoothing(torch.stack(data['keypoints'][cfg.key2label['Left Shoulder']]).numpy()) )
    # right_shoulder = torch.tensor(smoothing(torch.stack(data['keypoints'][cfg.key2label['Right Shoulder']]).numpy()))
    # left_wrist     = torch.tensor(smoothing(torch.stack(data['keypoints'][cfg.key2label['Left Wrist']]).numpy())    )
    # right_wrist    = torch.tensor(smoothing(torch.stack(data['keypoints'][cfg.key2label['Right Wrist']]).numpy())   )
    # left_hip       = torch.tensor(smoothing(torch.stack(data['keypoints'][cfg.key2label['Left Hip']]).numpy())      )
    # right_hip      = torch.tensor(smoothing(torch.stack(data['keypoints'][cfg.key2label['Right Hip']]).numpy())     )
    left_elbow     = smoothing(data['keypoints'][cfg.key2label['Left Elbow']])    
    right_elbow    = smoothing(data['keypoints'][cfg.key2label['Right Elbow']])   
    left_shoulder  = smoothing(data['keypoints'][cfg.key2label['Left Shoulder']]) 
    right_shoulder = smoothing(data['keypoints'][cfg.key2label['Right Shoulder']])
    left_wrist     = smoothing(data['keypoints'][cfg.key2label['Left Wrist']])    
    right_wrist    = smoothing(data['keypoints'][cfg.key2label['Right Wrist']])   
    left_hip       = smoothing(data['keypoints'][cfg.key2label['Left Hip']])      
    right_hip      = smoothing(data['keypoints'][cfg.key2label['Right Hip']])     

plt.figure(figsize=(12,4))
plt.subplot(2,1,1)
plt.plot(left_elbow[:,1], label='Left Elbow')
plt.plot(right_elbow[:,1], label='Right Elbow')
plt.plot(left_shoulder[:,1], label='Left Shoulder')
plt.plot(right_shoulder[:,1], label='Right Shoulder')
plt.plot(left_wrist[:,1], label='Left Wrist')
plt.plot(right_wrist[:,1], label='Right Wrist')
plt.plot(left_hip[:,1], label='Left Hip')
plt.plot(right_hip[:,1], label='Right Hip')
plt.ylabel('Pixel Value')
plt.ylim([1000,0])
plt.xlabel('Frame')
plt.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.15))
plt.subplot(2,1,2)
plt.plot(get_velocity(left_elbow[:,1]), label='Left Elbow')
plt.plot(get_velocity(right_elbow[:,1]), label='Right Elbow')
plt.plot(get_velocity(left_shoulder[:,1]), label='Left Shoulder')
plt.plot(get_velocity(right_shoulder[:,1]), label='Right Shoulder')
plt.plot(get_velocity(left_wrist[:,1]), label='Left Wrist')
plt.plot(get_velocity(right_wrist[:,1]), label='Right Wrist')
plt.plot(get_velocity(left_hip[:,1]), label='Left Hip')
plt.plot(get_velocity(right_hip[:,1]), label='Right Hip')
plt.ylabel('Velocity (pixel/frame)')
plt.ylim([15,-15])
plt.xlabel('Frame')
plt.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.15))
plt.show()
    
    
    
    # right_angle = angles(right_elbow[:,:2], right_shoulder[:,:2], right_wrist[:,:2])
    # left_angle = angles(left_elbow[:,:2], left_shoulder[:,:2], left_wrist[:,:2])
    # right_elbow_hip = angles(right_shoulder[:,:2], right_elbow[:,:2], right_hip[:,:2])
    # left_elbow_hip = angles(left_shoulder[:,:2], left_elbow[:,:2], left_hip[:,:2])
    # plt.figure(figsize=(12,4))
    # plt.plot(right_angle, label='Right Elbow Angle')
    # plt.plot(left_angle, label='Left Elbow Angle')
    # plt.plot(right_elbow_hip, label='Right Elbow-Hip Angle')
    # plt.plot(left_elbow_hip, label='Left Elbow-Hip Angle')
    # plt.ylabel('Degree')
    # plt.xlabel('Frame')
    # plt.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.15))
    # plt.show()