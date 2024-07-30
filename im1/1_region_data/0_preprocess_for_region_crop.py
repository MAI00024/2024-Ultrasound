#!/usr/bin/env python
# coding: utf-8

import json
import cv2
import os
import glob
import shutil
from sklearn.model_selection import KFold, GroupKFold
import random
import yaml
import numpy as np

# Function to convert annotations from JSON format to YOLO format
def convert_to_yolo_format(image_path, json_path, output_dir):
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    shapes = data["shapes"]
    yolo_annotations = []

    for shape in shapes:
        points = shape["points"]
        label = shape["label"]
        min_x = min(point[0] for point in points)
        max_x = max(point[0] for point in points)
        min_y = min(point[1] for point in points)
        max_y = max(point[1] for point in points)

        x_center = (min_x + max_x) / 2 / width
        y_center = (min_y + max_y) / 2 / height
        bbox_width = (max_x - min_x) / width
        bbox_height = (max_y - min_y) / height

        class_id = 0 if label == "region" else 1
        yolo_annotations.append(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + ".txt")
    
    with open(output_path, "w") as file:
        file.write("\n".join(yolo_annotations))


# Convert JSON annotations to YOLO format
root_folder_path = "./data"
output_dir = "./preprocessed_data/labels"

# Process all JPG files in the specified folder and its subfolders
for folder_path, _, _ in os.walk(root_folder_path):
    jpg_files = glob.glob(os.path.join(folder_path, "*.jpg"))
    
    for image_file_path in jpg_files:
        json_file_name = os.path.splitext(os.path.basename(image_file_path))[0] + ".json"
        json_file_path = os.path.join(folder_path, json_file_name)
        
        if os.path.exists(json_file_path):
            convert_to_yolo_format(image_file_path, json_file_path, output_dir)
        else:
            print(f"No JSON file found for image: {image_file_path}")

print("All image and JSON files have been converted.")


# Copy all image files to a designated folder

source_root_folder = "./data"
destination_folder = "./preprocessed_data/images"

# Create the destination folder if it does not exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Find and copy all JPG files to the destination folder
for folder_path, _, _ in os.walk(source_root_folder):
    jpg_files = glob.glob(os.path.join(folder_path, "*.jpg"))

    for jpg_file in jpg_files:
        file_name = os.path.basename(jpg_file)
        destination_file_path = os.path.join(destination_folder, file_name)

        # Copy the original file to the destination folder
        shutil.copy(jpg_file, destination_file_path)

print("All images have been copied to the destination folder.")


# Filter annotations to keep only those with class_id 0 (Region)

input_dir = 'preprocessed_data/labels'
output_dir = 'preprocessed_data/only_region_labels'

# Create output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Process each label file and filter lines based on class_id
for filename in os.listdir(input_dir):
    if filename.endswith('.txt'):
        input_file_path = os.path.join(input_dir, filename)
        output_file_path = os.path.join(output_dir, filename)

        # Write lines with class_id 0 to the new file
        with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
            for line in infile:
                class_id = int(line.split()[0])
                if class_id == 0:
                    outfile.write(line)


# Filter annotations to keep only those with class_id 1 (Echo)

input_dir = 'preprocessed_data/labels'
output_dir = 'preprocessed_data/only_echo_labels'

# Create output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Process each label file and filter lines based on class_id
for filename in os.listdir(input_dir):
    if filename.endswith('.txt'):
        input_file_path = os.path.join(input_dir, filename)
        output_file_path = os.path.join(output_dir, filename)

        # Write lines with class_id 1 to the new file
        with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
            for line in infile:
                class_id = int(line.split()[0])
                if class_id == 1:
                    outfile.write(line)


# Split dataset into train, validation, and test sets

def create_file_list_txt(folder_path, output_file):
    # Get a list of files in the folder
    file_list = os.listdir(folder_path)
    
    # Create or overwrite the output text file
    with open(output_file, 'w') as f:
        # Write each file path to a new line in the text file
        for file_name in file_list:
            file_path = os.path.join(folder_path, file_name)
            
            # Remove './split' part from the path
            file_path = file_path.replace('split/', '')
            
            f.write(file_path + '\n')


def split_dataset(images_path, labels_path, output_path, n_splits=10, random_state=42):
    random.seed(random_state)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    all_images = glob.glob(os.path.join(images_path, "*.jpg"))
    random.shuffle(all_images)
    
    # Extract device IDs and group images accordingly
    device_ids = [os.path.basename(img).split('_')[0] for img in all_images]
    print(device_ids)

    # Set up K-Fold cross-validation based on device IDs
    kf = GroupKFold(n_splits=n_splits)

    for fold, (train_valid_idx, test_idx) in enumerate(kf.split(all_images, groups=device_ids)):
        train_valid_images = [all_images[i] for i in train_valid_idx]
        test_images = [all_images[i] for i in test_idx]

        train_count = int(len(train_valid_images) * 0.9)
        train_images = train_valid_images[:train_count]
        valid_images = train_valid_images[train_count:]

        sets = {"train": train_images, "valid": valid_images, "test": test_images}

        fold_output_path = os.path.join(output_path, f"fold_{fold}")
        if not os.path.exists(fold_output_path):
            os.makedirs(fold_output_path)

        for set_name, set_images in sets.items():
            set_images_path = os.path.join(fold_output_path, set_name, "images")
            set_labels_path = os.path.join(fold_output_path, set_name, "labels")

            if not os.path.exists(set_images_path):
                os.makedirs(set_images_path)
            if not os.path.exists(set_labels_path):
                os.makedirs(set_labels_path)

            with open(os.path.join(fold_output_path, f"{set_name}.txt"), "w") as f:
                f.write("\n".join(set_images) + "\n")

            for image_path in set_images:
                image_filename = os.path.basename(image_path)
                label_filename = os.path.splitext(image_filename)[0] + ".txt"

                shutil.copy(image_path, os.path.join(set_images_path, image_filename))
                shutil.copy(os.path.join(labels_path, label_filename), os.path.join(set_labels_path, label_filename))

        for set_name in sets.keys():
            create_file_list_txt(os.path.join(fold_output_path, set_name, "images"), os.path.join(fold_output_path, f"{set_name}.txt"))

        # Creating yaml file for each fold
        yaml_data = {
            "names": ['region', 'echo'],  # Class names
            "nc": 2,  # Number of classes
            "path": fold_output_path,  # Root path
            "train": "train.txt",  # Train.txt path
            "val": "valid.txt",  # Valid.txt path
            "test": "test.txt",  # Test.txt path
        }

        with open(os.path.join(fold_output_path, "custom.yaml"), "w") as f:
            yaml.dump(yaml_data, f)

        print(f"Created fold {fold} with train: {len(train_images)}, valid: {len(valid_images)}, test: {len(test_images)} images.")

split_dataset("preprocessed_data/images", "preprocessed_data/only_region_labels", "split_for_region_crop", random_state=42)
