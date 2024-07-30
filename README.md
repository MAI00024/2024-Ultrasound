<<<<<<< HEAD
<<<<<<< HEAD
# 2024-USQual
## How to run

1. Provide execution permissions to the shell script files
```
chmod +x im1.sh
chmod +x im2-3.sh
chmod +x im6.sh
chmod +x Quality_Score_Determination.sh
```
2. Execute the shell scripts
```
./im1.sh
./im2-3.sh
./im6.sh
./Quality_Score_Determination.sh
```

## File Description
```
├── im1
│   ├── 1_region_data
│   │   ├── data
│   │   ├── 0_preprocess_for_region_crop.py
│   │   ├── 1_yolov5_models-imgsize1024-for_region_crop.py
│   │   └── 2_Crop_region_bounding_box_for_region_crop.py
│   └── 2_echo_data
│       ├── data
│       ├── 0_visualization_echo.ipynb
│       ├── 1_preprocess_datasplit_for_echo_detection.py
│       ├── 2_yolov5_models-imgsize1024-for_echo_detection.py
│       ├── 3_Check_Echo_prediction_results.py
│       └── 4_echo_count_and_eval.py
│
├── im2-3
│   ├── 1_region_data
│   │   ├── data
│   │   ├── 0_preprocess_for_region_crop.py
│   │   ├── 1_yolov5_models-imgsize1024-for_region_crop.py
│   │   └── 2_Crop_region_bounding_box_for_region_crop.py
│   └── 2_classification
│       ├── data
│       ├── 0_preprocess_for_image_classification.py
│       ├── 1_1-train-vgg16-classification-v2-imgsize384.py
│       └── 1_2-infer-vgg16-classification-v2-imgsize384.py
│
├── im6
│   ├── 1_region_data
│   │    ├── data
│   │    ├── 0_preprocess_for_region_crop.py
│   │    ├── 1_yolov5_models-imgsize1024-for_region_crop.py
│   │    └── 2_Crop_region_bounding_box_for_region_crop.py
│   └── 2_circle_data
│       ├── data
│       ├── 0_visualization_circle.ipynb
│       ├── 1_preprocess_datasplit_for_circle_detection.py
│       ├── 2_yolov5_models-imgsize1024-for_circle_detection.py
│       ├── 3_Check_circle_prediction_results.py
│       └── 4_circle_count_and_eval.py
│
├── Quality_Score_Determination
│   ├── data
│   ├── True_manual_image
│   │   ├── im1
│   │   ├── im2-3
│   │   └── im6
│   ├── 0_making_the_final_decision-multiple-case-for-INH_and_GMC.py
│   └── 1_logistic_regression_5KFold_using_true_value-De-identification.py
├── im1.sh
├── im2-3.sh
├── im6.sh
└── Quality_Score_Determination.sh
```

* im1: 'Dead zone'
* Im2-3: 'Axial/lateral resolution'
* im6: 'Gray scale and dynamic range'
=======
# 2024-USQual
>>>>>>> 8dc317d31f70bbbd5cc0baa4f5a396435c6ba07a
=======
# 2024-Ultrasound
>>>>>>> 0450c735985626926f3ad87b06b6c812d524c844
