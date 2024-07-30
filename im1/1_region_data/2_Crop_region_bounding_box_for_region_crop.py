#!/usr/bin/env python
# coding: utf-8

# # Describe

# * 1. Retrieve prediction results from yolov5/runs/val and save them in ../2_echo_data/Result_region_detect/images and ../2_echo_data/Result_region_detect/preds.
# * 2. Compare with the original images to find regions that were not detected.
# * 3. Crop the detected bounding boxes and save them in the 2_echo_data/Result_region_detect/Result_crop_images folder.

# ### Save labels from runs/val/test_fold_{i} to "../2_echo_data/Result_region_detect/preds" folder
# * Retrieve labels from all 10 folds using GroupKFold

import os
import shutil
import cv2

# Destination directory path
dest_dir = "../2_echo_data/Result_region_detect/preds"

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


# ### Save all images from "./preprocessed_data/images" folder to "../2_echo_data/Result_region_detect/images"

# Source directory containing the original images
src_dir = "./preprocessed_data/images"
# Directory to save the copied images
dest_dir = "../2_echo_data/Result_region_detect/images"

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


# ## Check Missing Files

# Get list of filenames (without extension) from each folder
images_filenames = {os.path.splitext(f)[0] for f in os.listdir("../2_echo_data/Result_region_detect/images") if f.endswith(".jpg")}
preds_filenames = {os.path.splitext(f)[0] for f in os.listdir("../2_echo_data/Result_region_detect/preds") if f.endswith(".txt")}

# Print filenames that are in images but not in preds
for filename in images_filenames - preds_filenames:
    print(f"'{filename}.jpg' is in 'images' but not in 'preds'")

# Print filenames that are in preds but not in images
for filename in preds_filenames - images_filenames:
    print(f"'{filename}.txt' is in 'preds' but not in 'images'")


# # Crop region bounding box
# * Save cropped images in "2_echo_data/Result_region_detect/Result_crop_images" folder

def crop_bounding_boxes_from_folder(image_folder, txt_folder, output_folder, expansion_factor=1.2):
    for image_filename in os.listdir(image_folder):
        if image_filename.endswith(".jpg"):
            image_path = os.path.join(image_folder, image_filename)
            txt_filename = os.path.splitext(image_filename)[0] + ".txt"
            txt_path = os.path.join(txt_folder, txt_filename)
            
            if os.path.exists(txt_path):
                image = cv2.imread(image_path)
                
                original_height, original_width, _ = image.shape
                
                with open(txt_path, 'r') as f:
                    lines = f.readlines()

                for idx, line in enumerate(lines):
                    line = line.strip().split()  # 공백으로 분리된 각 열을 리스트로 변환
                    x, y, w, h = map(float, line[1:5])  # 두 번째부터 다섯 번째 열을 추출하여 float 형태로 변환

                    # 원본 이미지의 크기에 맞게 바운딩 박스 좌표 조정
                    new_x = int(x * original_width)
                    new_y = int(y * original_height)
                    new_w = int(w * original_width)
                    new_h = int(h * original_height)

                    # 바운딩 박스의 좌표를 이용하여 주변 영역 확장
                    x_min = max(0, int(new_x - new_w / 2 * expansion_factor))
                    y_min = max(0, int(new_y - new_h / 2 * expansion_factor))
                    x_max = min(original_width, int(new_x + new_w / 2 * expansion_factor))
                    y_max = min(original_height, int(new_y + new_h / 2 * expansion_factor))

                    # 바운딩 박스 영역과 주변 영역 crop
                    cropped_image = image[y_min:y_max, x_min:x_max]

                    # Crop한 이미지 저장
                    output_filename = f"{image_filename}"
                    output_path = os.path.join(output_folder, output_filename)
                    cv2.imwrite(output_path, cropped_image)

# Path to image files
image_folder = "../2_echo_data/Result_region_detect/images"

# Path to bounding box information in text files
txt_folder = "../2_echo_data/Result_region_detect/preds"

# Path to save cropped images
output_folder = "../2_echo_data/Result_region_detect/Result_crop_images"

# Create output directory if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Expansion factor for bounding box area (e.g., 1.5 expands to 1.5 times the original size)
expansion_factor = 1.2

crop_bounding_boxes_from_folder(image_folder, txt_folder, output_folder, expansion_factor)


# * Create "./Result_region_detect/Result_crop_images" Folder

# # Adjust coordinates for class_id 1 based on cropped images
# * Updated epoch folder: "2_echo_data/Result_region_detect/Result_echo_updated"

# Set folder paths
images_origin_folder = "preprocessed_data/images"
labels_echo_folder = 'preprocessed_data/only_echo_labels'
region_detect_folder = '../2_echo_data/Result_region_detect/preds'
result_crop_images_folder = '../2_echo_data/Result_region_detect/Result_crop_images'
Result_echo_updated_folder = '../2_echo_data/Result_region_detect/Result_echo_updated'

# Create result folder if it doesn't exist
if not os.path.exists(Result_echo_updated_folder):
    os.makedirs(Result_echo_updated_folder)

# Get list of original image files
image_files = os.listdir(images_origin_folder)

for image_file in image_files:
    image_path = os.path.join(images_origin_folder, image_file)
    
    # Read image
    img = cv2.imread(image_path)
    
    # Create label file paths
    label_name = os.path.splitext(image_file)[0] + '.txt'
    label_path = os.path.join(labels_echo_folder, label_name)
    region_label_path = os.path.join(region_detect_folder, label_name)
    
    # Check if label files exist
    if os.path.exists(label_path) and os.path.exists(region_label_path):
        with open(label_path, 'r') as f:
            labels = f.readlines()
        
        with open(region_label_path, 'r') as f:
            region_labels = f.readlines()
        
        region_count = 0
        for region_label in region_labels:
            _, x_center, y_center, width, height = map(float, region_label.strip().split())
            
            x_min = int((x_center - width / 2) * img.shape[1])
            y_min = int((y_center - height / 2) * img.shape[0])
            x_max = int((x_center + width / 2) * img.shape[1])
            y_max = int((y_center + height / 2) * img.shape[0])
            
            # Path to save cropped image
            cropped_img_name = f"{os.path.splitext(image_file)[0]}.jpg"
            cropped_img_path = os.path.join(result_crop_images_folder, cropped_img_name)
            # Path to save updated label
            updated_label_name = f"{os.path.splitext(label_name)[0]}.txt"
            updated_label_path = os.path.join(Result_echo_updated_folder, updated_label_name)
            
            # Create cropped image
            crop_img = img[y_min:y_max, x_min:x_max]
            cv2.imwrite(cropped_img_path, crop_img)
            
            with open(updated_label_path, 'w') as f:
                for label in labels:
                    class_id, echo_x_center, echo_y_center, echo_width, echo_height = map(float, label.strip().split())
                    
                    # Convert relative coordinates to absolute
                    abs_x_center = echo_x_center * img.shape[1]
                    abs_y_center = echo_y_center * img.shape[0]
                    abs_width = echo_width * img.shape[1]
                    abs_height = echo_height * img.shape[0]
                    
                    # Update relative coordinates and dimensions
                    new_x_center = (abs_x_center - x_min) / (x_max - x_min)
                    new_y_center = (abs_y_center - y_min) / (y_max - y_min)
                    new_width = abs_width / (x_max - x_min)
                    new_height = abs_height / (y_max - y_min)
                    
                    # Write updated values to file
                    f.write(f"{class_id} {new_x_center} {new_y_center} {new_width} {new_height}\n")
            
            region_count += 1

print("Done.")
# # Preparation for echo Detection completed.

