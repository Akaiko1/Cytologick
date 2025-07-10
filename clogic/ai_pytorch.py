"""
PyTorch implementation of AI module for Cytologick - neural network training and data processing.

This module provides PyTorch equivalents to the TensorFlow implementation,
maintaining identical functionality for U-Net model training and inference.
"""

import os
import random
from typing import Tuple, Optional, Union
import albumentations as A
from albumentations.pytorch import ToTensorV2

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from PIL import Image

import config

# Global configurations
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CytologyDataset(Dataset):
    """
    PyTorch Dataset for cytology image segmentation.
    
    Loads image-mask pairs and applies transformations.
    """
    
    def __init__(self, images_path: str, masks_path: str, transform=None):
        """
        Initialize the dataset.
        
        Args:
            images_path: Path to directory containing training images
            masks_path: Path to directory containing corresponding masks
            transform: Albumentations transform pipeline
        """
        self.images_path = images_path
        self.masks_path = masks_path
        self.transform = transform
        
        self.images = [f for f in os.listdir(images_path) if f.endswith('.bmp')]
        self.masks = [f for f in os.listdir(masks_path) if f.endswith('.bmp')]
        
        # Ensure images and masks are aligned
        self.images.sort()
        self.masks.sort()
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        image_path = os.path.join(self.images_path, self.images[idx])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_path = os.path.join(self.masks_path, self.masks[idx])
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Ensure mask is long type for loss computation
        if isinstance(mask, torch.Tensor):
            mask = mask.long()
        
        return image, mask


def get_train_transforms():
    """
    Get training transforms that match TensorFlow augmentations.
    
    Returns:
        Albumentations transform pipeline for training
    """
    return A.Compose([
        A.Resize(128, 128),
        A.OneOf([
            A.Rotate(limit=360, p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.35, 
                scale_limit=0.2, 
                rotate_limit=360, 
                p=0.5
            ),
        ], p=0.8),
        A.OneOf([
            A.GaussNoise(var_limit=50.0, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.25, p=0.5),
        ], p=0.5),
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
        ToTensorV2(),
    ])


def get_val_transforms():
    """
    Get validation transforms (no augmentation).
    
    Returns:
        Albumentations transform pipeline for validation
    """
    return A.Compose([
        A.Resize(128, 128),
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
        ToTensorV2(),
    ])


class JaccardLoss(nn.Module):
    """
    Jaccard (IoU) Loss for segmentation.
    """
    
    def __init__(self, smooth=1e-7):
        super(JaccardLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, y_pred, y_true):
        y_pred = F.softmax(y_pred, dim=1)
        y_true_one_hot = F.one_hot(y_true.long(), num_classes=y_pred.shape[1]).permute(0, 3, 1, 2).float()
        
        intersection = (y_pred * y_true_one_hot).sum(dim=(2, 3))
        union = y_pred.sum(dim=(2, 3)) + y_true_one_hot.sum(dim=(2, 3)) - intersection
        
        jaccard = (intersection + self.smooth) / (union + self.smooth)
        return 1 - jaccard.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    """
    
    def __init__(self, alpha=1, gamma=2, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduce:
            return torch.mean(focal_loss)
        else:
            return focal_loss


class CombinedLoss(nn.Module):
    """
    Combined Jaccard and Focal Loss to match TensorFlow implementation.
    """
    
    def __init__(self, jaccard_weight=1.0, focal_weight=1.0):
        super(CombinedLoss, self).__init__()
        self.jaccard_loss = JaccardLoss()
        self.focal_loss = FocalLoss()
        self.jaccard_weight = jaccard_weight
        self.focal_weight = focal_weight
    
    def forward(self, y_pred, y_true):
        jaccard = self.jaccard_loss(y_pred, y_true)
        focal = self.focal_loss(y_pred, y_true)
        return self.jaccard_weight * jaccard + self.focal_weight * focal


def iou_score(y_pred, y_true, num_classes=3):
    """
    Calculate IoU score for segmentation.
    
    Args:
        y_pred: Model predictions
        y_true: Ground truth masks
        num_classes: Number of segmentation classes
        
    Returns:
        Mean IoU score
    """
    y_pred = torch.argmax(y_pred, dim=1)
    
    ious = []
    for cls in range(num_classes):
        pred_cls = (y_pred == cls)
        true_cls = (y_true == cls)
        
        intersection = (pred_cls & true_cls).float().sum()
        union = (pred_cls | true_cls).float().sum()
        
        if union == 0:
            ious.append(1.0)  # If no pixels for this class, perfect score
        else:
            ious.append((intersection / union).item())
    
    return np.mean(ious)


def f1_score(y_pred, y_true, num_classes=3):
    """
    Calculate F1 score for segmentation.
    
    Args:
        y_pred: Model predictions
        y_true: Ground truth masks
        num_classes: Number of segmentation classes
        
    Returns:
        Mean F1 score
    """
    y_pred = torch.argmax(y_pred, dim=1)
    
    f1_scores = []
    for cls in range(num_classes):
        pred_cls = (y_pred == cls)
        true_cls = (y_true == cls)
        
        tp = (pred_cls & true_cls).float().sum()
        fp = (pred_cls & ~true_cls).float().sum()
        fn = (~pred_cls & true_cls).float().sum()
        
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
        
        f1_scores.append(f1.item())
    
    return np.mean(f1_scores)


def get_datasets(images_path: str, masks_path: str, train_split: float = 0.8):
    """
    Create PyTorch datasets from image and mask directories.
    
    Args:
        images_path: Path to directory containing training images
        masks_path: Path to directory containing corresponding masks
        train_split: Fraction of data to use for training
        
    Returns:
        Tuple containing (train_dataset, val_dataset, total_samples)
    """
    # Create full dataset
    full_dataset = CytologyDataset(images_path, masks_path, transform=None)
    total_samples = len(full_dataset)
    
    # Split dataset
    train_size = int(train_split * total_samples)
    val_size = total_samples - train_size
    
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, total_samples))
    
    # Create datasets with transforms
    train_dataset = CytologyDataset(images_path, masks_path, transform=get_train_transforms())
    val_dataset = CytologyDataset(images_path, masks_path, transform=get_val_transforms())
    
    # Create subset datasets
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)
    
    return train_subset, val_subset, total_samples


def train_new_model_pytorch(model_path: str, output_classes: int, epochs: int, batch_size: int = 64):
    """
    Train a new U-Net model with EfficientNetB3 encoder using PyTorch.
    
    This function mirrors the TensorFlow implementation exactly.
    
    Args:
        model_path: Path where the trained model will be saved
        output_classes: Number of segmentation classes (e.g., 3 for background, LSIL, HSIL)
        epochs: Number of training epochs
        batch_size: Training batch size (default: 64)
    """
    # Construct dataset paths from configuration
    images_path = os.path.join(config.DATASET_FOLDER, config.IMAGES_FOLDER)
    masks_path = os.path.join(config.DATASET_FOLDER, config.MASKS_FOLDER)
    
    # Load and prepare datasets
    train_dataset, val_dataset, total_samples = get_datasets(images_path, masks_path)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Total samples: {total_samples}")
    
    # Create U-Net model with EfficientNetB3 backbone
    model = smp.Unet(
        encoder_name='efficientnet-b3',
        encoder_weights='imagenet',
        classes=output_classes,
        activation='softmax2d'
    )
    model = model.to(DEVICE)
    
    # Define loss function and optimizer
    criterion = CombinedLoss()
    optimizer = torch.optim.NAdam(model.parameters(), lr=0.001)
    
    # Training loop
    best_iou = 0.0
    train_losses = []
    val_ious = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(DEVICE).float()
            masks = masks.to(DEVICE).long()
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_iou = 0.0
        val_f1 = 0.0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(DEVICE).float()
                masks = masks.to(DEVICE).long()
                
                outputs = model(images)
                
                # Calculate metrics
                iou = iou_score(outputs, masks, output_classes)
                f1 = f1_score(outputs, masks, output_classes)
                
                val_iou += iou
                val_f1 += f1
        
        avg_val_iou = val_iou / len(val_loader)
        avg_val_f1 = val_f1 / len(val_loader)
        val_ious.append(avg_val_iou)
        
        print(f'Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val IoU: {avg_val_iou:.4f}, Val F1: {avg_val_f1:.4f}')
        
        # Save best model
        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            torch.save(model.state_dict(), f"{model_path}_best.pth")
            print(f'New best model saved with IoU: {best_iou:.4f}')
    
    # Save final model
    torch.save(model.state_dict(), f"{model_path}_final.pth")
    print(f'Training completed. Best IoU: {best_iou:.4f}')


def train_current_model_pytorch(model_path: str, epochs: int, batch_size: int = 64, lr: float = 0.0001):
    """
    Continue training an existing PyTorch model.
    
    Args:
        model_path: Path to the existing trained model to load and continue training
        epochs: Number of additional epochs to train
        batch_size: Training batch size (default: 64)
        lr: Learning rate for the optimizer (default: 0.0001)
    """
    # Construct dataset paths from configuration
    images_path = os.path.join(config.DATASET_FOLDER, config.IMAGES_FOLDER)
    masks_path = os.path.join(config.DATASET_FOLDER, config.MASKS_FOLDER)
    
    # Load and prepare datasets
    train_dataset, val_dataset, total_samples = get_datasets(images_path, masks_path)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    # Load existing model
    model = smp.Unet(
        encoder_name='efficientnet-b3',
        encoder_weights='imagenet',
        classes=config.CLASSES,
        activation='softmax2d'
    )
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    
    # Define loss function and optimizer with specified learning rate
    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Continue training
    best_iou = 0.0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(DEVICE).float()
            masks = masks.to(DEVICE).long()
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_iou = 0.0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(DEVICE).float()
                masks = masks.to(DEVICE).long()
                
                outputs = model(images)
                iou = iou_score(outputs, masks, config.CLASSES)
                val_iou += iou
        
        avg_val_iou = val_iou / len(val_loader)
        
        print(f'Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val IoU: {avg_val_iou:.4f}')
        
        # Save best model
        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            torch.save(model.state_dict(), model_path)
            print(f'Model updated with IoU: {best_iou:.4f}')
    
    print(f'Training completed. Best IoU: {best_iou:.4f}')


def create_mask_pytorch(pred_mask):
    """
    Create a mask from PyTorch model predictions by taking the argmax.
    
    Args:
        pred_mask: Model prediction tensor
        
    Returns:
        Predicted mask tensor
    """
    pred_mask = torch.argmax(pred_mask, dim=1)
    return pred_mask[0].cpu().numpy()


def display_pytorch(display_list, titles=['Input Image', 'True Mask', 'Predicted Mask']):
    """
    Display a list of images in a single row for comparison.
    
    Args:
        display_list: List of images to display
        titles: List of titles for each image
    """
    plt.figure(figsize=(15, 5))
    
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(titles[i])
        
        if isinstance(display_list[i], torch.Tensor):
            img = display_list[i].cpu().numpy()
            if len(img.shape) == 3 and img.shape[0] == 3:  # CHW format
                img = np.transpose(img, (1, 2, 0))
        else:
            img = display_list[i]
            
        plt.imshow(img)
        plt.axis('off')
    plt.show()