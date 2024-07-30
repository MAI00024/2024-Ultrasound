#!/usr/bin/env python
# coding: utf-8

# **Versions:**
# * v2: Changed optimizer to SGD, it converged better and faster. Accuracy **%94**

import os
import time
import torch
import torch.nn as nn
from torchsummary import summary
import torchvision
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
from skimage import io, color
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import argparse

# Configuration class
class CFG:
    N_LABEL = 2
    N_CLASS_2 = True if N_LABEL == 2 else False
    IMG_SIZE = 384
    N_EPOCHS = 200
    BATCH_SIZE = 8
    PATIENCE = 30  # early stopping
    NUM_WORKERS = 0
    OPTIMIZER = 'Adam'  # 'SGD', 'Adam'
    SCHEDULER = 'CosineAnnealingLR'  # CosineAnnealingLR
    LR_value = 1e-5


# Argument parser for fold selection
parser = argparse.ArgumentParser(description='Train a classification model')
parser.add_argument('--fold', type=int, required=True, help='Select fold for training/testing')
args = parser.parse_args()
CFG.select_Fold_case = args.fold


# Check device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Load data
data = pd.read_csv("./data/im2_3_5fold_Results_dataset_v2.csv")
print(data.shape)
if CFG.N_CLASS_2:
    data['im2_3_label'] = data['im2_3_label'].apply(lambda x: 1 if x == 2 else x)
data.head()

print(data['im2_3_label'].unique())
print(data['set'].unique())
print(data.info())

# Function to split data into train, validation, and test sets based on fold
def split_data(data, i):
    train_df = data[(data['fold'] == i) & (data['set'] == 'train')]
    val_df = data[(data['fold'] == i) & (data['set'] == 'valid')]
    test_df = data[(data['fold'] == i) & (data['set'] == 'test')]
    return train_df, val_df, test_df

# Split data
train_df, val_df, test_df = split_data(data, CFG.select_Fold_case)
print(f'********* {CFG.select_Fold_case} Fold *********')
print("Length of train dataset: ", len(train_df))
print("Length of validation dataset: ", len(val_df))
print("Length of test dataset: ", len(test_df))

print(train_df['im2_3_label'].value_counts())
print(val_df['im2_3_label'].value_counts())
print(test_df['im2_3_label'].value_counts())

# Data transforms
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(CFG.IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Dataset class
class USQualDataset(Dataset):
    def __init__(self, df: pd.DataFrame, train: bool, transforms=None):
        self.df = df
        self.train = train
        self.transforms = transforms
    
    def __getitem__(self, index):
        im_path = self.df.iloc[index]["path"]
        x = cv2.imread(im_path)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = cv2.resize(x, (CFG.IMG_SIZE, CFG.IMG_SIZE))

        if self.transforms:
            x = self.transforms(x)
        
        if self.train:
            y = self.df.iloc[index]["im2_3_label"]
            return x, y
        else:
            return x
    
    def __len__(self):
        return len(self.df)

# Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = models.vgg16(pretrained=True)
        self.model.fc = nn.Linear(512, CFG.N_LABEL) 
    
    def forward(self, x):        
        output = self.model(x)
        return output
    
# Prepare dataset and dataloaders
train_data = USQualDataset(df=train_df, train=True, transforms=train_transform)
val_data = USQualDataset(df=val_df, train=True, transforms=test_transform)
test_data = USQualDataset(df=test_df, train=True, transforms=test_transform)

train_loader = DataLoader(dataset=train_data, shuffle=True, batch_size=CFG.BATCH_SIZE, num_workers=CFG.NUM_WORKERS)
val_loader = DataLoader(dataset=val_data, shuffle=False, batch_size=CFG.BATCH_SIZE, num_workers=CFG.NUM_WORKERS)
test_loader = DataLoader(dataset=test_data, shuffle=False, batch_size=CFG.BATCH_SIZE, num_workers=CFG.NUM_WORKERS)

# Set model parameters
criterion = nn.CrossEntropyLoss()
model = Net()
model.to(device)

if CFG.OPTIMIZER == 'SGD':
    optim = torch.optim.SGD(model.parameters(), lr=CFG.LR_value, momentum=0.9)
elif CFG.OPTIMIZER == 'Adam':
    optim = torch.optim.Adam(model.parameters(), lr=CFG.LR_value, weight_decay=1e-5)

if CFG.SCHEDULER == 'CosineAnnealingLR':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=CFG.N_EPOCHS)

# Summary of the model
summary(model, input_size=(3, CFG.IMG_SIZE, CFG.IMG_SIZE))

# Training function
def train_model(model, optimizer, n_epochs, criterion, patience=5, scheduler=None):
    early_stopping_counter = 0
    best_accuracy = -float('inf')
    best_model = None
    best_epoch = 0
    lrs = []
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
                
    start_time = time.time()
    for epoch in range(1, n_epochs + 1):
        epoch_time = time.time()
        epoch_loss = 0
        correct = 0
        total = 0
        print(f"Epoch {epoch} / {n_epochs}")
        model.train()
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            _, pred = torch.max(outputs, dim=1)
            correct += (pred.cpu() == labels.cpu()).sum().item()
            total += labels.shape[0]
        acc = correct / total
        
        model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inp_val, lab_val in val_loader:
                inp_val, lab_val = inp_val.to(device), lab_val.to(device)
                out_val = model(inp_val)
                loss_val = criterion(out_val, lab_val)
                val_loss += loss_val.item()
                _, pred_val = torch.max(out_val, dim=1)
                correct_val += (pred_val.cpu() == lab_val.cpu()).sum().item()
                total_val += lab_val.shape[0]
            acc_val = correct_val / total_val
        
        if scheduler:
            scheduler.step()
            lrs.append(optimizer.param_groups[0]["lr"])

        epoch_duration = time.time() - epoch_time
        print(f"Duration: {epoch_duration:.0f}s, Train Loss: {epoch_loss / total:.4f}, Train Acc: {acc:.4f}, Val Loss: {val_loss / total_val:.4f}, Val Acc: {acc_val:.4f}")
        
        train_losses.append(epoch_loss / len(train_loader))
        train_accuracies.append(acc)
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(acc_val)
        
        if acc_val > best_accuracy:
            best_accuracy = acc_val
            best_model = model.state_dict()
            best_epoch = epoch
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter > patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    if best_model is not None:
        model.load_state_dict(best_model)
        save_path = f'./img_cls_fold{CFG.select_Fold_case}.pth'
        torch.save(best_model, save_path)
        print(f"Best model loaded from epoch {best_epoch}")
                                
    end_time = time.time()
    print(f"Total Time: {end_time - start_time:.0f}s")
    
    return train_losses, train_accuracies, val_losses, val_accuracies, epoch, lrs

# Evaluation function
from sklearn.metrics import f1_score, precision_score, recall_score

def eval_model(model, data_loader, data_df, data_type):
    correct = 0
    total = 0
    model.eval()
    misclassified_info = {}
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, pred = torch.max(outputs, dim=1)
            correct_preds = (pred == labels)
            correct += correct_preds.sum().item()
            total += labels.shape[0]
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())

            misclassified_indices = ~correct_preds
            if misclassified_indices.any():
                misclassified_indices = misclassified_indices.cpu()
                misclassified_batch_indices = np.where(misclassified_indices.numpy())[0]
                misclassified_batch_indices += i * data_loader.batch_size

                for idx in misclassified_batch_indices:
                    image_name = data_df.iloc[idx]['name']
                    image_pred = pred[idx - i * data_loader.batch_size].item()
                    image_label = labels[idx - i * data_loader.batch_size].item()
                    misclassified_info[image_name] = {'pred': image_pred, 'label': image_label}

    accuracy = 100 * correct / total
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    print(f'Accuracy of the network on the {data_type} images: {accuracy:.1f} %')
    print(f'Precision of the network on the {data_type} images: {precision:.2f}')
    print(f'Recall of the network on the {data_type} images: {recall:.2f}')
    print(f'F1 Score of the network on the {data_type} images: {f1:.2f}')

    return misclassified_info

# Load pretrained model
model = Net()
load_path = f'./model_weight/img_cls_fold{CFG.select_Fold_case}.pth'
model.load_state_dict(torch.load(load_path))
model.to(device)

# Function to evaluate and return probabilities
def eval_model_return(model, data_loader, data_df, data_type):
    correct = 0
    total = 0
    model.eval()
    misclassified_info = {}
    all_labels = []
    all_preds = []
    all_prob_class_1 = []
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, pred = torch.max(outputs, dim=1)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            prob_class_1 = probabilities[:, 1]
                    
            correct_preds = (pred == labels)
            correct += correct_preds.sum().item()
            total += labels.shape[0]
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())
            all_prob_class_1.extend(prob_class_1.cpu().numpy())

            misclassified_indices = ~correct_preds
            if misclassified_indices.any():
                misclassified_indices = misclassified_indices.cpu()
                misclassified_batch_indices = np.where(misclassified_indices.numpy())[0]
                misclassified_batch_indices += i * data_loader.batch_size

                for idx in misclassified_batch_indices:
                    image_name = data_df.iloc[idx]['name']
                    image_pred = pred[idx - i * data_loader.batch_size].item()
                    image_label = labels[idx - i * data_loader.batch_size].item()
                    misclassified_info[image_name] = {'pred': image_pred, 'label': image_label}

    accuracy = 100 * correct / total
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    print(f'Accuracy of the network on the {data_type} images: {accuracy:.1f} %')
    print(f'Precision of the network on the {data_type} images: {precision:.2f}')
    print(f'Recall of the network on the {data_type} images: {recall:.2f}')
    print(f'F1 Score of the network on the {data_type} images: {f1:.2f}')

    return all_prob_class_1, all_labels, all_preds, misclassified_info

# Extract test set results
all_prob_class_1, all_labels, all_preds, misclassified_examples_info = eval_model_return(model, test_loader, test_df, 'test')
for name, info in misclassified_examples_info.items():
    print(f'Image Name: {name}, Prediction: {info["pred"]}, Label: {info["label"]}')
print()

# Save results to CSV
save_df = test_df[['name', 'im2_3_count', 'im2_3_label']].copy()
save_df['im2_3_pred'] = all_preds
save_df['im2_3_pred_prob_class_1'] = all_prob_class_1
print(save_df.shape)
print(save_df.head())

save_df.to_csv(f"Im2_3_result_fold{CFG.select_Fold_case}.csv", index=False)
