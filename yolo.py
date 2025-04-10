import os
import sys
import cv2
import torch
import pickle
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

import utils
import config as cfg # include the label-joint pairs

import subprocess
def fix_video_rotation(input_path, output_path):
            subprocess.run([
                'ffmpeg', '-y', '-i', input_path,
                '-c', 'copy', '-metadata:s:v', 'rotate=90', output_path
            ])
            
if __name__=='__main__':
    model = YOLO('yolo11x-pose.pt')
    data_path = f'./data/L1'
    save_path = './output'
    # files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
    files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path,f))]
    for video_name in files:
        video_name = video_name.split('/')[-1].split('.')[0]
        os.makedirs(f'{save_path}/{video_name}', exist_ok=True)
        video_format = utils.get_format(files, video_name)
        
        # if os.path.exists(f'{save_path}/{video_name}/{video_name}.pkl'):
        #     print(f"Video {video_name} already processed.")
        #     continue
        # elif video_format is None:
        #     print(f"Video {video_name} not found.")
        #     continue
            
        src = f'{data_path}/{video_name}.{video_format}'
        
        # corrected_src = f'{save_path}/{video_name}/{video_name}_fixed.mp4'
        corrected_src = './output.mp4'
        results = model.predict(source=corrected_src, save=True, project=save_path, name=video_name)
        # results = model.predict(source=src, save=True, project=save_path, name=video_name)
        data = {'keypoints': [], 'conf': []}
        for frame_idx, result in enumerate(results):
            if result.keypoints.conf is None:
                data['keypoints'].append(np.zeros((17,2)))
                data['conf'].append(np.zeros(17))
            else:
                data['keypoints'].append(result.keypoints.xy.tolist()[0])
                data['conf'].append(result.keypoints.conf.tolist()[0])
        with open(f'{save_path}/{video_name}/{video_name}.pkl', 'wb') as f:
            pickle.dump(data, f)
            
        utils.convert_avi_to_mp4(f'{save_path}/{video_name}/{video_name}.avi', f'{save_path}/{video_name}/{video_name}.mp4')
        
        if os.path.exists(f'{save_path}/{video_name}/{video_name}.avi'):
            os.remove(f'{save_path}/{video_name}/{video_name}.avi')
            
    # for i in range(14):
    #     data_path = f'./data/{i+1}'
    #     save_path = './output'
    #     # files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
    #     files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path,f))]
    #     for video_name in files:
    #         video_name = video_name.split('/')[-1].split('.')[0]
    #         os.makedirs(f'{save_path}/{video_name}', exist_ok=True)
    #         video_format = utils.get_format(files, video_name)
            
    #         if os.path.exists(f'{save_path}/{video_name}/{video_name}.pkl'):
    #             print(f"Video {video_name} already processed.")
    #             continue
    #         elif video_format is None:
    #             print(f"Video {video_name} not found.")
    #             continue
                
    #         src = f'{data_path}/{video_name}.{video_format}'
    #         results = model.predict(source=src) # , save=True, project=save_path, name=video_name)
    #         data = {'keypoints': [], 'conf': []}
    #         for frame_idx, result in enumerate(results):
    #             if result.keypoints.conf is None:
    #                 data['keypoints'].append(np.zeros((17,2)))
    #                 data['conf'].append(np.zeros(17))
    #             else:
    #                 data['keypoints'].append(result.keypoints.xy.tolist()[0])
    #                 data['conf'].append(result.keypoints.conf.tolist()[0])
    #         with open(f'{save_path}/{video_name}/{video_name}.pkl', 'wb') as f:
    #             pickle.dump(data, f)
                
    #         utils.convert_avi_to_mp4(f'{save_path}/{video_name}/{video_name}.avi', f'{save_path}/{video_name}/{video_name}.mp4')
            
    #         if os.path.exists(f'{save_path}/{video_name}/{video_name}.avi'):
    #             os.remove(f'{save_path}/{video_name}/{video_name}.avi')