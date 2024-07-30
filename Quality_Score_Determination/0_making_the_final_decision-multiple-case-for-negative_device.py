#!/usr/bin/env python
# coding: utf-8

# Libraries
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import gc
import pprint
from itertools import product
import glob

pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')

# Load Data

# Load Im1 result data
df_Im1 = pd.read_csv("data/Im1_result.csv")
print("Load Im1 dataset...")

# Load Im2-3 result data
print("Load Im2-3 dataset...")
dfs = []
for i in range(5):
    df = pd.read_csv(f"data/Im2_3_result_fold{i}.csv")
    dfs.append(df)
    
df_Im2_3 = pd.concat(dfs).reset_index(drop=True)
df_Im2_3['name'] = df_Im2_3['name'].apply(lambda x: x.split('.jpg')[0])


# Load Im6 result data
df_Im6 = pd.read_csv("data/Im6_result.csv")
print("Load Im6 dataset...")


# Combine device results

# Load image paths
Im1_path_pattern = "True_manual_image/im1/*"
Im2_3_path_pattern = "True_manual_image/im2-3/*"
Im6_path_pattern = "True_manual_image/im6/*"

# Find all file names matching the pattern
Im1_file_list = glob.glob(Im1_path_pattern)
Im2_3_file_list = glob.glob(Im2_3_path_pattern)
Im6_file_list = glob.glob(Im6_path_pattern)

Im1_file_name = [path.split('\\')[-1].split('.jpg')[0] for path in Im1_file_list]
Im2_3_file_name = [path.split('\\')[-1].split('.jpg')[0] for path in Im2_3_file_list]
Im6_file_name = [path.split('\\')[-1].split('.jpg')[0] for path in Im6_file_list]

# print(f"Im1_file_name len: {len(Im1_file_name)}")
# print(f"Im2_3_file_name len: {len(Im2_3_file_name)}")
# print(f"Im6_file_name len: {len(Im6_file_name)}")

# Extract rows matching file names
representative_df_Im1 = df_Im1[df_Im1['name'].isin(Im1_file_name)].reset_index(drop=True)
representative_df_Im2_3 = df_Im2_3[df_Im2_3['name'].isin(Im2_3_file_name)].reset_index(drop=True)
representative_df_Im6 = df_Im6[df_Im6['name'].isin(Im6_file_name)].reset_index(drop=True)

print(f"representative_df_Im1 len: {len(representative_df_Im1)}")
print(f"representative_df_Im2_3 len: {len(representative_df_Im2_3)}")
print(f"representative_df_Im6 len: {len(representative_df_Im6)}")

# Increase abnormal cases for INH and GMC devices

# Create all possible combinations for INH
representative_df_Im1_names = representative_df_Im1[representative_df_Im1['name'].str.startswith('INH')]['name'].tolist()
representative_df_Im2_3_names = representative_df_Im2_3[representative_df_Im2_3['name'].str.startswith('INH')]['name'].tolist()
representative_df_Im6_names = representative_df_Im6[representative_df_Im6['name'].str.startswith('INH')]['name'].tolist()

all_combinations = list(product(representative_df_Im1_names, representative_df_Im2_3_names, representative_df_Im6_names))
named_combinations = {f"INH{i+1}": comb for i, comb in enumerate(all_combinations)}

# Generate combined dataframe for INH
final_rows = []
for name, (im1_name, im2_3_name, im6_name) in named_combinations.items():
    row_im1 = representative_df_Im1[representative_df_Im1['name'] == im1_name].iloc[0].to_dict()
    row_im2_3 = representative_df_Im2_3[representative_df_Im2_3['name'] == im2_3_name].iloc[0].to_dict()
    row_im6 = representative_df_Im6[representative_df_Im6['name'] == im6_name].iloc[0].to_dict()
    merged_row = {**row_im1, **row_im2_3, **row_im6, 'name': name}
    final_rows.append(merged_row)

final_INH_df = pd.DataFrame(final_rows)
final_INH_df['name'] = [f'INH{i+1}' for i in range(len(final_INH_df))]

# Create all possible combinations for GMC
representative_df_Im1_names = representative_df_Im1[representative_df_Im1['name'].str.startswith('GMC')]['name'].tolist()
representative_df_Im2_3_names = representative_df_Im2_3[representative_df_Im2_3['name'].str.startswith('GMC')]['name'].tolist()
representative_df_Im6_names = representative_df_Im6[representative_df_Im6['name'].str.startswith('GMC')]['name'].tolist()

all_combinations = list(product(representative_df_Im1_names, representative_df_Im2_3_names, representative_df_Im6_names))
named_combinations = {f"GMC{i+1}": comb for i, comb in enumerate(all_combinations)}

# Generate combined dataframe for GMC
final_rows = []
for name, (im1_name, im2_3_name, im6_name) in named_combinations.items():
    row_im1 = representative_df_Im1[representative_df_Im1['name'] == im1_name].iloc[0].to_dict()
    row_im2_3 = representative_df_Im2_3[representative_df_Im2_3['name'] == im2_3_name].iloc[0].to_dict()
    row_im6 = representative_df_Im6[representative_df_Im6['name'] == im6_name].iloc[0].to_dict()
    merged_row = {**row_im1, **row_im2_3, **row_im6, 'name': name}
    final_rows.append(merged_row)

final_GMC_df = pd.DataFrame(final_rows)
final_GMC_df['name'] = [f'GMC{i+1}' for i in range(len(final_GMC_df))]

# Merge all DataFrames

# Exclude rows starting with "GMC" and "INH" in 'name' column
filtered_df_Im1 = representative_df_Im1[~(representative_df_Im1['name'].str.startswith('GMC') | representative_df_Im1['name'].str.startswith('INH'))]
filtered_df_Im2_3 = representative_df_Im2_3[~(representative_df_Im2_3['name'].str.startswith('GMC') | representative_df_Im2_3['name'].str.startswith('INH'))]
filtered_df_Im6 = representative_df_Im6[~(representative_df_Im6['name'].str.startswith('GMC') | representative_df_Im6['name'].str.startswith('INH'))]

filtered_df_Im1['name'] = filtered_df_Im1['name'].apply(lambda x: x.split('_')[0])
filtered_df_Im2_3['name'] = filtered_df_Im2_3['name'].apply(lambda x: x.split('_')[0])
filtered_df_Im6['name'] = filtered_df_Im6['name'].apply(lambda x: x.split('_')[0])

merged_df = pd.merge(pd.merge(filtered_df_Im1, filtered_df_Im2_3, on='name', how='outer'), filtered_df_Im6, on='name', how='outer').drop_duplicates('name').reset_index(drop=True)
# print(merged_df.shape)
# print(merged_df.head(5))

merged_df['target'] = 1
final_GMC_df['target'] = 0
final_INH_df['target'] = 0

final_df = pd.concat([merged_df, final_GMC_df, final_INH_df], axis=0).reset_index(drop=True)

print()
print("final_df shape: ",final_df.shape)
# print(final_df.head(1))

final_df['target'].value_counts()

# Drop 'name' column for anonymization
final_df = final_df.drop(columns='name')

# Save final DataFrame to CSV
final_df.to_csv("final_df.csv", index=False)
print("Save as 'final_df.csv'")
print("Done.")

# Summary
# We have combined the data from different sources and increased the abnormal cases by creating combinations of images for INH and GMC devices.
# We now have 32 positive devices and 35 negative devices.

# End of script
