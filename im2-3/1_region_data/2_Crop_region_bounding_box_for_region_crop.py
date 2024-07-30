#!/usr/bin/env python
# coding: utf-8

import os
import shutil
import cv2

# Step 1: Copy prediction label files to the destination directory
def copy_prediction_labels(src_root_dir, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for i in range(10):
        src_dir = f"./yolov5/runs/val/test_fold_{i}/labels"
        for filename in os.listdir(src_dir):
            if filename.endswith('.txt'):
                src_file_path = os.path.join(src_dir, filename)
                dest_file_path = os.path.join(dest_dir, filename)
                shutil.copy(src_file_path, dest_file_path)

# Step 2: Copy original images to the destination directory
def copy_original_images(src_dir, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for filename in os.listdir(src_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            src_file_path = os.path.join(src_dir, filename)
            dest_file_path = os.path.join(dest_dir, filename)
            shutil.copy(src_file_path, dest_file_path)

# Step 3: Check for missing files
def check_missing_files(images_dir, preds_dir):
    images_filenames = {os.path.splitext(f)[0] for f in os.listdir(images_dir) if f.endswith(".jpg")}
    preds_filenames = {os.path.splitext(f)[0] for f in os.listdir(preds_dir) if f.endswith(".txt")}

    for filename in images_filenames - preds_filenames:
        print(f"'{filename}.jpg' is in 'images' but not in 'preds'")
    for filename in preds_filenames - images_filenames:
        print(f"'{filename}.txt' is in 'preds' but not in 'images'")

# Step 4: Crop bounding boxes from images based on prediction labels
def crop_bounding_boxes_from_folder(image_folder, txt_folder, output_folder, expansion_factor=1.2):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

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
                    line = line.strip().split()
                    x, y, w, h = map(float, line[1:5])
                    
                    new_x = int(x * original_width)
                    new_y = int(y * original_height)
                    new_w = int(w * original_width)
                    new_h = int(h * original_height)

                    x_min = max(0, int(new_x - new_w / 2 * expansion_factor))
                    y_min = max(0, int(new_y - new_h / 2 * expansion_factor))
                    x_max = min(original_width, int(new_x + new_w / 2 * expansion_factor))
                    y_max = min(original_height, int(new_y + new_h / 2 * expansion_factor))

                    cropped_image = image[y_min:y_max, x_min:x_max]
                    output_filename = f"{image_filename}"
                    output_path = os.path.join(output_folder, output_filename)
                    cv2.imwrite(output_path, cropped_image)

def main():
    # Path configurations
    preds_dest_dir = "../2_classification/Result_region_detect/preds"
    images_src_dir = "./preprocessed_data/images"
    images_dest_dir = "../2_classification/Result_region_detect/images"
    output_crop_dir = "../2_classification/Result_region_detect/Result_crop_images"
    
    # Step 1: Copy prediction label files    
    copy_prediction_labels('./yolov5/runs/val', preds_dest_dir)

    # Step 2: Copy original images
    copy_original_images(images_src_dir, images_dest_dir)

    # Step 3: Check for missing files
    check_missing_files(images_dest_dir, preds_dest_dir)

    # Step 4: Crop bounding boxes from images    
    expansion_factor = 1.2
    crop_bounding_boxes_from_folder(images_dest_dir, preds_dest_dir, output_crop_dir, expansion_factor)
    print("Done.")

if __name__ == "__main__":
    main()
