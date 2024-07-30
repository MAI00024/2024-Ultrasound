#!/usr/bin/env python
# coding: utf-8

import os
import shutil
import subprocess
import wandb

# Clone YOLOv5 repository
def clone_yolov5():
    subprocess.run(['git', 'clone', 'https://github.com/ultralytics/yolov5.git'])
    subprocess.run(['pip', 'install', '-qr', 'yolov5/requirements.txt'])

# Move "split_for_region_crop" folder to "yolov5" folder
def move_split_folder():
    src_path = './split_for_region_crop'
    dst_path = './yolov5/split_for_region_crop'
    if os.path.exists(src_path):
        shutil.move(src_path, dst_path)
    else:
        print(f"Source path {src_path} does not exist")

# Login to Weights and Biases
def login_wandb(api_key):
    wandb.login(key=api_key)

# Train YOLOv5 model
def train_yolov5(fold, data_path, weights='yolov5x.pt', img_size=1024, batch_size=8, epochs=100, device=0, project='Yolov5_10kfold_region_crop'):
    cmd = [
        'python', 'train.py',
        '--img', str(img_size),
        '--batch', str(batch_size),
        '--epochs', str(epochs),
        '--data', os.path.join(data_path, f'fold_{fold}/custom.yaml'),
        '--device', str(device),
        '--weights', weights,
        '--name', f'test_fold_{fold}',
        '--project', project
    ]
    subprocess.run(cmd)

# Perform inference with YOLOv5 model
def infer_yolov5(fold, data_path, weights_path, img_size=1024, conf_thres=0.5, device=0):
    cmd = [
        'python', 'val.py',
        '--img', str(img_size),
        '--conf-thres', str(conf_thres),
        '--task', 'test',
        '--data', os.path.join(data_path, f'fold_{fold}/custom.yaml'),
        '--weights', os.path.join(weights_path, f'test_fold_{fold}/weights/best.pt'),
        '--device', str(device),
        '--save-txt',
        '--name', f'test_fold_{fold}'
    ]
    subprocess.run(cmd)

# Perform detection with YOLOv5 model
def detect_yolov5(weights_path, source='./test_image/'):
    cmd = [
        'python', 'detect.py',
        '--weights', weights_path,
        '--source', source
    ]
    subprocess.run(cmd)

def main():
    # Clone YOLOv5 repository
    clone_yolov5()

    # Move "split_for_region_crop" folder to "yolov5" folder
    move_split_folder()

    # Change directory to yolov5
    os.chdir('yolov5')

    # Login to Weights and Biases
    wandb_api_key = "Your Key"
    login_wandb(wandb_api_key)

    # Path configuration
    data_path = './split_for_region_crop'
    project = 'Yolov5_10kfold_region_crop'

    # Train and inference for each fold
    for i in range(10):
        # Train YOLOv5 model
        train_yolov5(i, data_path, project=project)
        
        # Perform inference with YOLOv5 model
        infer_yolov5(i, data_path, f'./{project}')

if __name__ == "__main__":
    main()
