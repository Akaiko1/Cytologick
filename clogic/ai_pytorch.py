"""
PyTorch implementation of AI module for Cytologick - neural network training and data processing.

This module provides PyTorch equivalents to the TensorFlow implementation,
maintaining identical functionality for U-Net model training and inference.
"""

import os
import random
from contextlib import nullcontext
from typing import Tuple, Optional, Union
import albumentations as A
import inspect
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

# Progress bar (tqdm) - optional
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, *args, **kwargs):
        return x


def set_seed(seed: int = 42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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


def _build_gauss_noise():
    """Return a Gauss/Gaussian noise transform compatible with installed Albumentations.

    Handles API differences between versions by inspecting constructor signatures.
    """
    try:
        params = inspect.signature(A.GaussNoise.__init__).parameters
        if 'var_limit' in params:
            return A.GaussNoise(var_limit=(10.0, 30.0), p=0.5)
        if 'std_range' in params:
            return A.GaussNoise(std_range=(0.1, 0.3), p=0.5)
    except Exception:
        pass

    # Fallback to GaussianNoise if available
    try:
        params = inspect.signature(A.GaussianNoise.__init__).parameters
        if 'var_limit' in params:
            return A.GaussianNoise(var_limit=(10.0, 30.0), p=0.5)
    except Exception:
        pass

    # As a last resort, no-op to keep pipeline functional
    return A.NoOp(p=0.0)


def get_train_transforms():
    """
    Get training transforms that match TensorFlow augmentations.
    
    Returns:
        Albumentations transform pipeline for training
    """
    noise_tf = _build_gauss_noise()

    return A.Compose([
        A.Resize(128, 128),
        A.OneOf([
            A.Rotate(limit=360, p=0.5),
            A.Affine(
                translate_percent=0.35,
                scale=(0.8, 1.2),
                rotate=(-360, 360),
                p=0.5
            ),
        ], p=0.8),
        A.OneOf([
            noise_tf,
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



def mixup_data(x, y, alpha=0.4, device=DEVICE):
    """
    Returns mixed inputs, pairs of targets, and lambda.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


class CombinedLoss(nn.Module):
    """
    Combined Lovasz Softmax and Cross Entropy Loss (with Label Smoothing).
    """
    
    def __init__(self, lovasz_weight=1.0, ce_weight=1.0):
        super(CombinedLoss, self).__init__()
        # Lovasz Softmax Loss (typically for multiclass, using logit inputs)
        self.lovasz_loss = smp.losses.LovaszLoss(mode='multiclass', per_image=True)
        # Cross Entropy with Label Smoothing
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.lovasz_weight = lovasz_weight
        self.ce_weight = ce_weight
    
    def forward(self, y_pred, y_true):
        # y_pred: (N, C, H, W) logits
        # y_true: (N, H, W) long indices
        
        # Ensure y_true is long
        if y_true.dtype != torch.long:
            y_true = y_true.long()

        lovasz = self.lovasz_loss(y_pred, y_true)
        ce = self.ce_loss(y_pred, y_true)
        
        return self.lovasz_weight * lovasz + self.ce_weight * ce


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


def _train_model_loop(model, train_loader, val_loader, criterion, optimizer, scheduler, scaler, amp_ctx, 
                      epochs, model_path, output_classes_metrics):
    """
    Common training loop for PyTorch models.
    """
    best_iou = 0.0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False, dynamic_ncols=True)
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(DEVICE).float()
            masks = masks.to(DEVICE).long()
            
            optimizer.zero_grad()
            
            # Apply Mixup
            inputs, targets_a, targets_b, lam = mixup_data(images, masks, alpha=0.4, device=DEVICE)

            with amp_ctx():
                outputs = model(inputs)
                # Compute loss for both targets and mix
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()
            if hasattr(pbar, 'set_postfix'):
                # Handle both scalar and tensor loss for display
                loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss
                avg_loss = running_loss / (batch_idx + 1)
                lr_now = optimizer.param_groups[0]['lr']
                pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr_now:.3e}")
            
        avg_train_loss = running_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_iou = 0.0
        val_f1 = 0.0
        
        with torch.no_grad():
            vbar = tqdm(val_loader, desc=f"Val   {epoch+1}/{epochs}", leave=False, dynamic_ncols=True)
            for images, masks in vbar:
                images = images.to(DEVICE).float()
                masks = masks.to(DEVICE).long()
                
                outputs = model(images)
                
                # Calculate metrics
                iou = iou_score(outputs, masks, output_classes_metrics)
                f1 = f1_score(outputs, masks, output_classes_metrics)
                
                val_iou += iou
                val_f1 += f1
                if hasattr(vbar, 'set_postfix'):
                    seen = max(1, vbar.n)
                    vbar.set_postfix(iou=f"{val_iou/seen:.4f}", f1=f"{val_f1/seen:.4f}")
        
        avg_val_iou = val_iou / len(val_loader)
        avg_val_f1 = val_f1 / len(val_loader)
        
        print(f'Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val IoU: {avg_val_iou:.4f}, Val F1: {avg_val_f1:.4f}')

        # Step LR scheduler
        try:
            scheduler.step(epoch + 1)
        except Exception:
            pass

        # Save weights every epoch
        base_path = os.path.splitext(model_path)[0]
        epoch_weights = f"{base_path}_epoch{epoch+1:03d}.pth"
        torch.save(model.state_dict(), epoch_weights)
        torch.save(model.state_dict(), f"{base_path}_last.pth")

        # Save best model
        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            torch.save(model.state_dict(), f"{base_path}_best.pth")
            print(f'New best model saved with IoU: {best_iou:.4f}')
    
    # Save final model
    torch.save(model.state_dict(), f"{base_path}_final.pth")
    print(f'Training completed. Best IoU: {best_iou:.4f}')


def _setup_training_components(model, lr, use_amp):
    """
    Setup optimizer, scheduler, criterion, and mixed precision components.
    """
    criterion = CombinedLoss()
    optimizer = torch.optim.NAdam(model.parameters(), lr=lr)
    # Note: T_mult=2 matches original train_new_model implementation
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    
    if torch.cuda.is_available():
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
        amp_ctx = lambda: torch.amp.autocast('cuda', enabled=use_amp)
    else:
        scaler = None
        amp_ctx = nullcontext
        
    return criterion, optimizer, scheduler, scaler, amp_ctx


def _prepare_data_loaders(batch_size):
    """
    Create data loaders from config paths.
    """
    images_path = os.path.join(config.DATASET_FOLDER, config.IMAGES_FOLDER)
    masks_path = os.path.join(config.DATASET_FOLDER, config.MASKS_FOLDER)
    
    train_dataset, val_dataset, total_samples = get_datasets(images_path, masks_path)
    
    num_workers = min(4, os.cpu_count() or 4)
    pin_mem = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=pin_mem,
        persistent_workers=bool(num_workers)
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=pin_mem,
        persistent_workers=bool(num_workers)
    )
    
    return train_loader, val_loader, total_samples


def train_new_model_pytorch(model_path: str, output_classes: int, epochs: int, batch_size: int = 64,
                            lr: float = 0.001, use_amp: bool = True):
    """
    Train a new U-Net model using PyTorch.
    """
    set_seed(42)
    
    train_loader, val_loader, total_samples = _prepare_data_loaders(batch_size)
    print(f"Total samples: {total_samples}")
    
    model = smp.Unet(
        encoder_name='efficientnet-b3',
        encoder_weights='imagenet',
        classes=output_classes,
        activation=None
    )
    model = model.to(DEVICE)
    
    criterion, optimizer, scheduler, scaler, amp_ctx = _setup_training_components(model, lr, use_amp)
    
    _train_model_loop(
        model, train_loader, val_loader, criterion, optimizer, scheduler, scaler, amp_ctx,
        epochs, model_path, output_classes
    )


def train_current_model_pytorch(model_path: str, epochs: int, batch_size: int = 64, lr: float = 0.0001, use_amp: bool = True):
    """
    Continue training an existing PyTorch model.
    """
    set_seed(42)
    
    train_loader, val_loader, _ = _prepare_data_loaders(batch_size)
    
    # Load existing architecture
    model = smp.Unet(
        encoder_name='efficientnet-b3',
        encoder_weights='imagenet',
        classes=config.CLASSES,
        activation=None
    )
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    
    criterion, optimizer, scheduler, scaler, amp_ctx = _setup_training_components(model, lr, use_amp)
    
    _train_model_loop(
        model, train_loader, val_loader, criterion, optimizer, scheduler, scaler, amp_ctx,
        epochs, model_path, config.CLASSES
    )


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
