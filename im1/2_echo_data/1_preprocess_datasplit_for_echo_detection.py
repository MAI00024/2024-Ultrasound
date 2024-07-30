#!/usr/bin/env python
# coding: utf-8

# GroupKFold

from sklearn.model_selection import KFold
import random
import os
import shutil
import glob
import yaml
from sklearn.model_selection import GroupKFold
import numpy as np

# Function to create a text file listing all files in a folder
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

# Function to split dataset into train, validation, and test sets using GroupKFold
def split_dataset(images_path, labels_path, output_path, n_splits=10, random_state=42):
    random.seed(random_state)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Get all image file paths
    all_images = glob.glob(os.path.join(images_path, "*.jpg"))
    random.shuffle(all_images)
    
    # Extract device IDs to group images
    device_ids = [os.path.basename(img).split('_')[0] for img in all_images]
    print(device_ids)

    # Set up GroupKFold cross-validation based on device IDs
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
            "train": "train.txt",  # Path to train.txt
            "val": "valid.txt",  # Path to valid.txt
            "test": "test.txt",  # Path to test.txt
        }

        # Save yaml configuration file
        with open(os.path.join(fold_output_path, "custom.yaml"), "w") as f:
            yaml.dump(yaml_data, f)

        print(f"Created fold {fold} with train: {len(train_images)}, valid: {len(valid_images)}, test: {len(test_images)} images.")


try:
    # Split the dataset and create cross-validation folds
    split_dataset("Result_region_detect/Result_crop_images", "Result_region_detect/Result_echo_updated", "split_for_echo_detection", random_state=42)
except Exception as e:
    print("The following code requires the datasets 'Result_region_detect/Result_crop_images' and 'Result_region_detect/Result_echo_updated' to be present in the specified directories. Please ensure these datasets are available before running the script.")