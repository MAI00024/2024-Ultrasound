#!/bin/bash

# Navigate to Quality_Score_Determination and run the scripts
cd Quality_Score_Determination
python 0_making_the_final_decision-multiple-case-for-negative_device.py
python 1_logistic_regression_5KFold_using_true_value-De-identification.py