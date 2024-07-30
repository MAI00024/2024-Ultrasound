#!/usr/bin/env python
# coding: utf-8

# Describe

# * 1. Retrieve prediction results from yolov5/runs/val and save them in Result_circle_detect/images and Result_circle_detect/preds.
# * 2. Compare with the original images to find regions that were not detected.
# * 3. Crop the detected bounding boxes and save them in the Result_circle_detect/crop_results folder.

# ### Save labels from runs/val/test_fold_{i} to "./Result_circle_detect/preds" folder
# * Retrieve labels from all 10 folds using GroupKFold

import os
import shutil

# Destination directory path
dest_dir = "./Result_circle_detect/preds"

# Create destination directory if it doesn't exist
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

# Loop through each fold
for i in range(10):
    # Source directory path
    src_dir = f"./yolov5/runs/val/test_fold_{i}/labels"
    
    # Get all txt files from the source directory
    for filename in os.listdir(src_dir):
        if filename.endswith('.txt'):
            src_file_path = os.path.join(src_dir, filename)
            dest_file_path = os.path.join(dest_dir, filename)
            
            # Copy file
            shutil.copy(src_file_path, dest_file_path)


# ### Save all images from "./Result_region_detect/Result_crop_images" to "./Result_circle_detect/images"

# Source directory containing the original images
src_dir = "./Result_region_detect/Result_crop_images"
# Directory to save the copied images
dest_dir = "./Result_circle_detect/images"

# Create destination directory if it doesn't exist
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

# Get all image files from the source directory
for filename in os.listdir(src_dir):
    # Common image file extensions
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
        src_file_path = os.path.join(src_dir, filename)
        dest_file_path = os.path.join(dest_dir, filename)
        
        # Copy file
        shutil.copy(src_file_path, dest_file_path)


# Check Missing Files

# Get list of filenames (without extension) from each folder
images_filenames = {os.path.splitext(f)[0] for f in os.listdir("../2_circle_data/Result_circle_detect/images") if f.endswith(".jpg")}
preds_filenames = {os.path.splitext(f)[0] for f in os.listdir("../2_circle_data/Result_circle_detect/preds") if f.endswith(".txt")}

# Print filenames that are in images but not in preds
for filename in images_filenames - preds_filenames:
    print(f"'{filename}.jpg' is in 'images' but not in 'preds'")

# Print filenames that are in preds but not in images
for filename in preds_filenames - images_filenames:
    print(f"'{filename}.txt' is in 'preds' but not in 'images'")
