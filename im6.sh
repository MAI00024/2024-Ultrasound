#!/bin/bash

# Navigate to im6/1_region_data and run the scripts
cd im6/1_region_data
python 0_preprocess_for_region_crop.py
python 1_yolov5_models-imgsize1024-for_region_crop.py
python 2_Crop_region_bounding_box_for_region_crop.py

# Navigate to im6/2_circle_data and run the scripts
cd ../2_circle_data
python 1_preprocess_datasplit_for_circle_detection.py
python 2_yolov5_models-imgsize1024-for_circle_detection.py
python 3_Check_circle_prediction_results.py
python 4_circle_count_and_eval.py