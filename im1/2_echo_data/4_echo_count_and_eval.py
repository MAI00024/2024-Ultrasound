#!/usr/bin/env python
# coding: utf-8

# echo_count.py

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc, confusion_matrix
import seaborn as sns
import cv2

# Part 1: Count occurrences of class_id = 1 in text files

# Set the folder path
folder_path = './Result_echo_detect/preds'

# Dictionary to store the count of class_id = 1
count_dict = {}

# Loop through all text files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        # File path
        file_path = os.path.join(folder_path, filename)
        
        # Read the file
        with open(file_path, 'r') as file:
            lines = file.readlines()
            
            # Count occurrences of class_id = 1
            count = sum(1 for line in lines if line.strip().split()[0] == '1')
            
            # Remove extension and store the result
            name_without_extension, _ = os.path.splitext(filename)
            count_dict[name_without_extension] = count

# Convert the dictionary to a DataFrame
df = pd.DataFrame(list(count_dict.items()), columns=['name', 'im1_preds_count'])

# Create a new column based on the count
df['im1_preds'] = df['im1_preds_count'].apply(lambda x: 1 if x >= 9 else 0)

# Part 2: Load labels and compare performance

# Load label data
df_labels = pd.read_csv('./data/im1_label.csv')
df_labels.drop(columns=['Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5'], inplace=True)

# Remove '.jpg' from the 'name' column
df_labels['name'] = df_labels['name'].str.replace('.jpg', '')

# Part 3: Merge and fill missing predictions with 0

# Merge df_labels and df on 'name' column
result_df = pd.merge(df_labels, df, on='name', how='left')
print(result_df[result_df.isnull().any(axis=1)])  # Display rows with missing values

# Fill NaN values with 0
result_df = result_df.fillna(0)

# Convert 'im1_preds_count' and 'im1_preds' columns to int
result_df['im1_preds_count'] = result_df['im1_preds_count'].astype(int)
result_df['im1_preds'] = result_df['im1_preds'].astype(int)

# Part 4: Check cases where predictions are incorrect

print(result_df[result_df['im1_label'] != result_df['im1_preds']])

# Check if the 'result' directory exists, if not, create it
result_folder = 'result'
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

# Save the result to a CSV file
result_df.to_csv(os.path.join(result_folder, "Im1_result.csv"), index=False)

# Part 5: Evaluate and visualize performance

# Actual labels and predictions
y_true = result_df['im1_label']
y_preds = result_df['im1_preds']

# Calculate accuracy
accuracy = accuracy_score(y_true, y_preds)
print(f'Accuracy: {accuracy}')

# Calculate precision
precision = precision_score(y_true, y_preds)
print(f'Precision: {precision}')

# Calculate recall (sensitivity)
sensitivity = recall_score(y_true, y_preds)
print(f'Sensitivity: {sensitivity}')

# Calculate F1 score
f1 = f1_score(y_true, y_preds)
print(f'F1 Score: {f1}')

# Calculate AUC
fpr, tpr, thresholds = roc_curve(y_true, y_preds)
auc_value = auc(fpr, tpr)
print(f'AUC: {auc_value}')

# Visualize metrics
metrics_names = ['Accuracy', 'Precision', 'Sensitivity', 'F1 Score', 'AUC']
metrics_values = [accuracy, precision, sensitivity, f1, auc_value]

plt.figure(figsize=(10, 6))
bars = plt.bar(metrics_names, metrics_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
plt.ylim(0, 1.1)
plt.ylabel('Score')
plt.title('Classification Metrics')

# Add values on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.3f}', ha='center', va='bottom', fontsize=12)

plt.show()

# Part 6: Visualize confusion matrix

# Calculate confusion matrix
cm = confusion_matrix(y_true, y_preds)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Part 7: Visualize predictions

# Set paths for images and labels
image_dir = './Result_echo_detect/images'
label_dir = './Result_echo_detect/preds'

# Get all image and label files
image_files = os.listdir(image_dir)
label_files = os.listdir(label_dir)

# Process each image and label file
for index, image_file in enumerate(image_files):
    # Process only jpg or png files
    if image_file.endswith(('.jpg', '.png')):
        # Generate label file name
        label_file = os.path.splitext(image_file)[0] + '.txt'
        print(label_file)
        
        # Check if label file exists
        if label_file in label_files:
            # Get paths for image and label
            image_path = os.path.join(image_dir, image_file)
            label_path = os.path.join(label_dir, label_file)
            
            # Read the image
            image = cv2.imread(image_path)
            height, width = image.shape[:2]
            
            # Read label file for bounding box info
            with open(label_path, 'r') as file:
                for line in file.readlines():
                    class_id, x_center, y_center, w, h = map(float, line.strip().split())
                    # Convert YOLO format to pixel coordinates
                    x_min = int((x_center - w / 2) * width)
                    y_min = int((y_center - h / 2) * height)
                    x_max = int((x_center + w / 2) * width)
                    y_max = int((y_center + h / 2) * height)
                    
                    # Draw bounding box on image
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            
            # Display the image
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.show()
