#!/usr/bin/env python
# coding: utf-8

# Import Libraries
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Explanation:
# * true label value: ["im1_count", "im1_label", "im2_3_label", "im6_count", "im6_label"]
# * pred value: ["im1_preds_count", "im1_preds", "im2_3_pred_prob_class_1", "im6_preds_count", "im6_preds"]
# * target value: "target"

# Load Data
merged_df = pd.read_csv("final_df.csv")

# Select True Label Columns
Label_true_column = [
    "im1_count", "im1_label", 
    "im2_3_label",
    "im6_count", "im6_label", 
    "target"
]

merged_df = merged_df[Label_true_column]


from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
import gc

df = merged_df.copy()
df['final_device_score1'] = -1
gc.collect()

# KFold Configuration
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Lists to store model performance metrics
scores = []
weights_list = []
bias_list = []
y_pred_list = []
model_list = []


print("Training...")
# KFold Split
for train_index, test_index in kf.split(df):
    # Split Train and Test Data
    X_train, X_test = df.drop(['target'], axis=1).iloc[train_index], df.drop(['target'], axis=1).iloc[test_index]
    y_train, y_test = df['target'].iloc[train_index], df['target'].iloc[test_index]
    
    # Train Regression Model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    model_list.append(model)
    
    # Extract Weights and Bias
    weights = model.coef_
    bias = model.intercept_
    
    weights_list.append(weights)
    bias_list.append(bias)

    # Predict and Evaluate on Test Data
    y_pred = model.predict_proba(X_test)[:, 1]  # Extract probabilities for class 1
    
    df.loc[test_index, 'final_device_score1'] = y_pred    
    y_pred_list.append(y_pred)
    score = mean_squared_error(y_test, y_pred, squared=False)  # Calculate RMSE
    scores.append(score)
print("Training Done.")


# Inference Part
# Train the model with true labels, then use it to predict on the test set (ultrasound case results)

# Load Data Again for Inference
merged_df = pd.read_csv("final_df.csv")

# Select Prediction Columns
Label_pred_column = [
    "im1_preds_count", "im1_preds", 
    "im2_3_pred_prob_class_1",
    "im6_preds_count", "im6_preds", 
    "target"
]

merged_df = merged_df[Label_pred_column]

# Match columns with training features
merged_df.columns = [
    "im1_count", "im1_label", 
    "im2_3_label",
    "im6_count", "im6_label", 
    "target"
]

merged_df['final_device_score1'] = -1
gc.collect()

# KFold Configuration
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Lists to store predictions and model parameters
scores = []
weights_list = []
bias_list = []
y_pred_list = []

print("Test...")
# KFold Split for Inference
for i, (train_index, test_index) in enumerate(kf.split(merged_df)):
    _, X_test = merged_df.drop(['target'], axis=1).iloc[train_index], merged_df.drop(['target'], axis=1).iloc[test_index]
    _, y_test = merged_df['target'].iloc[train_index], merged_df['target'].iloc[test_index]
        
    # Extract Weights and Bias from Trained Model
    weights = model_list[i].coef_
    bias = model_list[i].intercept_
    
    weights_list.append(weights)
    bias_list.append(bias)

    # Predict and Evaluate on Test Data
    y_pred = model_list[i].predict_proba(X_test)[:, 1]  # Extract probabilities for class 1
    
    merged_df.loc[test_index, 'final_device_score1'] = y_pred    
    y_pred_list.append(y_pred)
    score = mean_squared_error(y_test, y_pred, squared=False)  # Calculate RMSE
    scores.append(score)
print("Test result:")

# Print RMSE Score, Weights, and Bias
for i in range(len(scores)):
    print('#'*25)
    print(f'### {i} Fold')
    print(f'### Score: {scores[i]}')
    print(f'### weights: {weights_list[i]}')
    print(f'### bias: {bias_list[i]}')          
    print(f'### y_pred: {y_pred_list[i]}')
    print('#'*25)

mean_score = np.mean(scores)
print(f'\nMean Scores: {mean_score}')

# De-identification
merged_df['de-identification'] = merged_df.index.astype(str)

# Plotting Results
import matplotlib.pyplot as plt
import seaborn as sns

# Update target labels for visualization
merged_df['target'] = merged_df['target'].map({0: 'Non-pass', 1: 'Pass'})

# Color Palette for target labels
palette = {'Non-pass': '#ff6666', 'Pass': '#66cc66'}  # Red for 'Non-pass', Green for 'Pass'

# Sort DataFrame by 'final_device_score1'
sorted_df = merged_df.sort_values('final_device_score1', ascending=True)

# Create Horizontal Bar Plot
plt.figure(figsize=(15, 12))  # Adjust figure size as needed
barplot = sns.barplot(x=sorted_df['final_device_score1']*100, y='de-identification', hue='target', data=sorted_df, palette=palette, dodge=False)

# Display 'final_device_score1' in percentage at the end of each bar
for index, value in enumerate(sorted_df['final_device_score1']):
    barplot.text(value*100, index, f'{value*100:.1f}%', color='black', va='center', fontsize=11)

# Set plot title and labels
plt.xlabel('Device Quality Score (%)', fontsize=25)
plt.ylabel('De-identified devices', fontsize=25)

# Adjust font size for x and y ticks
plt.xticks(fontsize=20)
plt.yticks(fontsize=11)

# Set legend location and size
plt.legend(title='Target', loc='upper right', fontsize=25, title_fontsize='25')

# Show plot
plt.show()
