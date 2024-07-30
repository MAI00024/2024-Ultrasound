#!/bin/bash

# Navigate to im2-3/1_region_data and run the scripts
cd im2-3/1_region_data
python 0_preprocess_for_region_crop.py
python 1_yolov5_models-imgsize1024-for_region_crop.py
python 2_Crop_region_bounding_box_for_region_crop.py

# Navigate to im2-3/2_classification and run the scripts
cd ../2_classification
python 0_preprocess_for_image_classification.py
python 1_1-train-vgg16-classification-v2-imgsize384.py
python 1_2-infer-vgg16-classification-v2-imgsize384.py
