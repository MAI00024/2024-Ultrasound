#!/usr/bin/env python
# coding: utf-8

import os
import torch
import torch.nn as nn
from torchsummary import summary
import torchvision
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import cv2
from skimage import io, color
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold

# Load dataset
data = pd.read_csv("data/im2-3_label_updated240102.csv")
print(data.shape)
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Add image path
data['path'] = data['name'].apply(lambda x: "./Result_region_detect/Result_crop_images/" + x)

# # Display the first image
# def display_first_image(image_path):
#     image = Image.open(image_path)
#     plt.imshow(image)
#     plt.axis('off')
#     plt.show()

# display_first_image(data['path'][0])

# Data split
df = data.copy()

# Display label distribution
print(df['im2_3_label'].value_counts())

# Stratified Group K-Fold Split
df['device_id'] = df['name'].apply(lambda x: x.split('_')[0])
n_splits = 5
sgkf = StratifiedGroupKFold(n_splits=n_splits, random_state=2, shuffle=True)
fold_results = pd.DataFrame()

for fold, (train_valid_idx, test_idx) in enumerate(sgkf.split(df, df['im2_3_label'], groups=df['device_id'])):
    train_valid_df = df.iloc[train_valid_idx].copy()
    test_df = df.iloc[test_idx].copy()

    valid_groups = train_valid_df['device_id'].drop_duplicates().sample(frac=0.2, random_state=42)
    valid_df = train_valid_df[train_valid_df['device_id'].isin(valid_groups)].copy()
    train_df = train_valid_df[~train_valid_df['device_id'].isin(valid_groups)].copy()

    train_df.loc[:, 'set'], valid_df.loc[:, 'set'], test_df.loc[:, 'set'] = 'train', 'valid', 'test'
    train_df.loc[:, 'fold'], valid_df.loc[:, 'fold'], test_df.loc[:, 'fold'] = fold, fold, fold

    fold_results = pd.concat([fold_results, train_df, valid_df, test_df])

# Display the number of samples in each set for each fold
for i in range(5):
    train_shape = fold_results[(fold_results['fold'] == i) & (fold_results['set'] == 'train')].shape[0]
    valid_shape = fold_results[(fold_results['fold'] == i) & (fold_results['set'] == 'valid')].shape[0]
    test_shape = fold_results[(fold_results['fold'] == i) & (fold_results['set'] == 'test')].shape[0]
    print(f'Fold {i}: Train={train_shape}, Valid={valid_shape}, Test={test_shape}')

# Display label distribution in each set for each fold
for i in range(5):
    print(f"Fold {i} Train Label Distribution:")
    print(fold_results[(fold_results['fold'] == i) & (fold_results['set'] == 'train')]['im2_3_label'].value_counts())
    print(f"Fold {i} Valid Label Distribution:")
    print(fold_results[(fold_results['fold'] == i) & (fold_results['set'] == 'valid')]['im2_3_label'].value_counts())
    print(f"Fold {i} Test Label Distribution:")
    print(fold_results[(fold_results['fold'] == i) & (fold_results['set'] == 'test')]['im2_3_label'].value_counts())
    print()

# Save the fold results to a CSV file
fold_results.to_csv('data/im2_3_5fold_Results_dataset_v2.csv', index=False)

if __name__ == "__main__":
    print("Data processing completed successfully.")
