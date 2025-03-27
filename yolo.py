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

if __name__=='__main__':
    model = YOLO('yolo11x-pose.pt')
    data_path = './data/1'
    save_path = './output'
    # files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
    files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path,f))]
    for video_name in files:
        video_name = video_name.split('/')[-1].split('.')[0]
        video_format = utils.get_format(files, video_name)
        
        if os.path.exists(f'{save_path}/{video_name}/{video_name}.pkl'):
            print(f"Video {video_name} already processed.")
            continue
        elif video_format is None:
            print(f"Video {video_name} not found.")
            continue
            
        src = f'{data_path}/{video_name}.{video_format}'
        result = model.predict(source=src, save=True, project=save_path, name=video_name)
        
        with open(f'{save_path}/{video_name}/{video_name}.pkl', 'wb') as f:
            c,d = utils.get_data(result)
            data = {'conf':c, 'data':d}
            pickle.dump(data, f)
            
        utils.convert_avi_to_mp4(f'{save_path}/{video_name}/{video_name}.avi', f'{save_path}/{video_name}/{video_name}.mp4')
        
        if os.path.exists(f'{save_path}/{video_name}/{video_name}.avi'):
            os.remove(f'{save_path}/{video_name}/{video_name}.avi')