#!/usr/bin/env python
# coding: utf-8

import json
import cv2
import os
import glob
import shutil
import random
from sklearn.model_selection import GroupKFold
import yaml


# Function to convert annotations to YOLO format
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


# Convert all images and annotations in the dataset to YOLO format
def convert_all_to_yolo_format(root_folder_path, output_dir):
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


# Copy all images to a single directory
def copy_all_images(source_root_folder, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for folder_path, _, _ in os.walk(source_root_folder):
        jpg_files = glob.glob(os.path.join(folder_path, "*.jpg"))

        for jpg_file in jpg_files:
            file_name = os.path.basename(jpg_file)
            destination_file_path = os.path.join(destination_folder, file_name)

            shutil.copy(jpg_file, destination_file_path)

    print("All images have been copied to the destination folder.")


# Filter labels to keep only those with class_id 0 (Region)
def filter_labels(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            input_file_path = os.path.join(input_dir, filename)
            output_file_path = os.path.join(output_dir, filename)

            with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
                for line in infile:
                    class_id = int(line.split()[0])
                    if class_id == 0:
                        outfile.write(line)


# Create a list of file paths for a given folder
def create_file_list_txt(folder_path, output_file):
    file_list = os.listdir(folder_path)
    
    with open(output_file, 'w') as f:
        for file_name in file_list:
            file_path = os.path.join(folder_path, file_name)
            file_path = file_path.replace('split/', '')
            f.write(file_path + '\n')


# Split the dataset into train, validation, and test sets using GroupKFold
def split_dataset(images_path, labels_path, output_path, n_splits=10, random_state=42):
    random.seed(random_state)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    all_images = glob.glob(os.path.join(images_path, "*.jpg"))
    random.shuffle(all_images)
    
    device_ids = [os.path.basename(img).split('_')[0] for img in all_images]

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

        yaml_data = {
            "names": ['region'],
            "nc": 1,
            "path": fold_output_path,
            "train": "train.txt",
            "val": "valid.txt",
            "test": "test.txt",
        }

        with open(os.path.join(fold_output_path, "custom.yaml"), "w") as f:
            yaml.dump(yaml_data, f)

        print(f"Created fold {fold} with train: {len(train_images)}, valid: {len(valid_images)}, test: {len(test_images)} images.")


if __name__ == "__main__":
    # Paths
    root_folder_path = "./data"
    output_dir_labels = "./preprocessed_data/labels"
    output_dir_images = "./preprocessed_data/images"
    output_dir_only_region_labels = "preprocessed_data/only_region_labels"
    split_output_path = "split_for_region_crop"

    # Step 1: Convert all annotations to YOLO format
    convert_all_to_yolo_format(root_folder_path, output_dir_labels)

    # Step 2: Copy all images to a single directory
    copy_all_images(root_folder_path, output_dir_images)

    # Step 3: Filter labels to keep only those with class_id 0 (Region)
    filter_labels(output_dir_labels, output_dir_only_region_labels)

    # Step 4: Split the dataset into train, validation, and test sets
    split_dataset(output_dir_images, output_dir_only_region_labels, split_output_path, random_state=42)
