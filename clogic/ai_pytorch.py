"""
PyTorch implementation of AI module for Cytologick - neural network training and data processing.

This module provides PyTorch equivalents to the TensorFlow implementation,
maintaining identical functionality for U-Net model training and inference.
"""

import os
import random
import sys
from contextlib import nullcontext
from functools import partial
from typing import Tuple, Optional, Union
import albumentations as A
import inspect
from albumentations.pytorch import ToTensorV2

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from PIL import Image

from config import Config
from clogic.preprocessing_pytorch import preprocess_rgb_image

# Global configurations
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _log_device_selection(prefix: str = ""):
    prefix = (prefix + " ") if prefix else ""
    if torch.cuda.is_available():
        try:
            device_count = torch.cuda.device_count()
            current_index = torch.cuda.current_device()
            current_name = torch.cuda.get_device_name(current_index)
            all_names = [torch.cuda.get_device_name(i) for i in range(device_count)]
            print(f"{prefix}CUDA available: True | torch device: {DEVICE} | current GPU: {current_index} ({current_name}) | GPUs: {all_names}")
        except Exception:
            print(f"{prefix}CUDA available: True | torch device: {DEVICE}")
    else:
        print(f"{prefix}CUDA available: False | torch device: {DEVICE}")

# Progress bar (tqdm) - optional
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, *args, **kwargs):
        return x


def _identity_mask(x, **kwargs):
    return x


# Mask class indices (after conversion from visible values)
_MASK_CLASS_BACKGROUND = 0
_MASK_CLASS_NORMAL = 1
_MASK_CLASS_PATHOLOGY = 2


class SafeGeometricTransform(A.DualTransform):
    """
    Wrapper for geometric transforms that protects pathology from being truncated.

    If a geometric transform would reduce pathology area below a threshold,
    the transform is rejected and original image/mask returned.
    """

    def __init__(
        self,
        transform: A.BasicTransform,
        min_pathology_ratio: float = 0.4,
        pathology_class: int = _MASK_CLASS_PATHOLOGY,
        always_apply: bool = False,
        p: float = 1.0,
    ):
        """
        Args:
            transform: The geometric transform to wrap
            min_pathology_ratio: Minimum ratio of pathology area to preserve (0.4 = 40%)
            pathology_class: Class index for pathology in mask
        """
        super().__init__(always_apply=always_apply, p=p)
        self.transform = transform
        self.min_pathology_ratio = min_pathology_ratio
        self.pathology_class = pathology_class
        self._original_image = None
        self._original_mask = None
        self._original_pathology_area = 0

    def apply(self, img, **params):
        # Store original for potential rollback
        self._original_image = img.copy()
        return self.transform.apply(img, **params)

    def apply_to_mask(self, mask, **params):
        self._original_mask = mask.copy()
        self._original_pathology_area = np.sum(mask == self.pathology_class)
        return self.transform.apply_to_mask(mask, **params)

    def get_params(self):
        return self.transform.get_params() if hasattr(self.transform, 'get_params') else {}

    def get_params_dependent_on_data(self, params, data):
        if hasattr(self.transform, 'get_params_dependent_on_data'):
            return self.transform.get_params_dependent_on_data(params, data)
        return {}

    def get_transform_init_args_names(self):
        return ('min_pathology_ratio', 'pathology_class')

    @property
    def targets_as_params(self):
        if hasattr(self.transform, 'targets_as_params'):
            return self.transform.targets_as_params
        return []


def _check_pathology_preserved(
    original_mask: np.ndarray,
    transformed_mask: np.ndarray,
    min_ratio: float = 0.4,
    pathology_class: int = _MASK_CLASS_PATHOLOGY,
) -> bool:
    """
    Check if pathology area is preserved after transformation.

    Args:
        original_mask: Mask before transformation
        transformed_mask: Mask after transformation
        min_ratio: Minimum ratio of pathology to preserve
        pathology_class: Class index for pathology

    Returns:
        True if pathology is sufficiently preserved
    """
    original_area = np.sum(original_mask == pathology_class)
    if original_area == 0:
        return True  # No pathology to protect

    transformed_area = np.sum(transformed_mask == pathology_class)
    ratio = transformed_area / original_area

    return ratio >= min_ratio


def _safe_geometric_augmentation(
    image: np.ndarray,
    mask: np.ndarray,
    transform: A.Compose,
    max_attempts: int = 3,
    min_pathology_ratio: float = 0.4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply geometric augmentation with pathology protection.

    Retries transform if pathology is truncated too much.
    Falls back to original if all attempts fail.
    """
    original_mask = mask.copy()
    has_pathology = np.any(mask == _MASK_CLASS_PATHOLOGY)

    if not has_pathology:
        # No pathology — apply transform freely
        result = transform(image=image, mask=mask)
        return result['image'], result['mask']

    for attempt in range(max_attempts):
        result = transform(image=image, mask=mask)
        if _check_pathology_preserved(original_mask, result['mask'], min_pathology_ratio):
            return result['image'], result['mask']

    # All attempts failed — return original (only resize applied)
    return image, mask


def _preprocess_image(
    img,
    *,
    use_encoder_preprocessing: bool,
    encoder_name: str,
    encoder_weights,
    **kwargs,
):
    return preprocess_rgb_image(
        img,
        use_encoder_preprocessing=use_encoder_preprocessing,
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
    )


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
    Automatically selects conservative or aggressive augmentation based on
    whether the tile contains pathology.
    """

    def __init__(
        self,
        images_path: str,
        masks_path: str,
        transform=None,
        transform_aggressive=None,
    ):
        """
        Initialize the dataset.

        Args:
            images_path: Path to directory containing training images
            masks_path: Path to directory containing corresponding masks
            transform: Albumentations transform pipeline (conservative, for pathology)
            transform_aggressive: Albumentations transform pipeline (aggressive, for normal tiles)
                                  If None, uses same transform for all tiles.
        """
        self.images_path = images_path
        self.masks_path = masks_path
        self.transform = transform
        self.transform_aggressive = transform_aggressive

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

        # Convert visible mask values (0, 127, 255) back to class indices (0, 1, 2)
        # This allows masks to be viewed in image viewers while still training correctly
        mask = self._convert_mask_to_labels(mask)

        # Apply transforms with pathology protection
        if self.transform:
            original_pathology_area = np.sum(mask == _MASK_CLASS_PATHOLOGY)
            has_pathology = original_pathology_area > 0

            if has_pathology:
                # Use conservative transform with retry for pathology tiles
                max_attempts = 3
                min_ratio = 0.4  # Keep at least 40% of pathology

                # Store original for fallback (before any transform)
                original_image = image.copy()
                original_mask = mask.copy()

                for attempt in range(max_attempts):
                    # Always start from original to get fresh random params
                    transformed = self.transform(image=original_image, mask=original_mask)
                    new_mask = transformed['mask']

                    # Check pathology preservation
                    if isinstance(new_mask, torch.Tensor):
                        new_pathology_area = (new_mask == _MASK_CLASS_PATHOLOGY).sum().item()
                    else:
                        new_pathology_area = np.sum(new_mask == _MASK_CLASS_PATHOLOGY)

                    ratio = new_pathology_area / original_pathology_area if original_pathology_area > 0 else 1.0

                    if ratio >= min_ratio:
                        image = transformed['image']
                        mask = new_mask
                        break
                else:
                    # All attempts failed - use original without augmentation
                    # Apply only resize and preprocessing (no geometric transforms)
                    # This matches validation transforms behavior
                    h, w = self.transform.transforms[0].height, self.transform.transforms[0].width
                    image = cv2.resize(original_image, (w, h))
                    mask = cv2.resize(original_mask, (w, h), interpolation=cv2.INTER_NEAREST)

                    # Apply encoder preprocessing (same as in transform pipeline)
                    # Find the Lambda transform in the pipeline and apply its image function
                    for t in self.transform.transforms:
                        if isinstance(t, A.Lambda) and hasattr(t, 'image'):
                            image = t.image(image)
                            break

                    # Convert to tensor format
                    image = torch.from_numpy(image.transpose(2, 0, 1)).float()
                    mask = torch.from_numpy(mask)
            else:
                # No pathology - use aggressive transform if available
                transform_to_use = self.transform_aggressive or self.transform
                transformed = transform_to_use(image=image, mask=mask)
                image = transformed['image']
                mask = transformed['mask']

        # Ensure mask is long type for loss computation
        if isinstance(mask, torch.Tensor):
            mask = mask.long()

        return image, mask

    @staticmethod
    def _convert_mask_to_labels(mask: np.ndarray) -> np.ndarray:
        """
        Convert visible mask values to class indices.

        Mask values saved for visibility:
            0   -> class 0 (background)
            127 -> class 1 (normal cells)
            255 -> class 2 (abnormal cells)

        Returns:
            Mask with values 0, 1, 2
        """
        labels = np.zeros_like(mask, dtype=np.uint8)
        labels[mask == 127] = 1
        labels[mask == 255] = 2
        return labels


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


def _get_color_augmentations(noise_tf):
    """Common color/stain augmentations for both pathology and normal tiles."""
    return [
        # Color/Stain augmentation (critical for pathology - stain variability between labs)
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        ], p=0.7),

        # Brightness/Contrast + CLAHE (lighting and contrast variation)
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.25, p=0.5),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        ], p=0.5),

        # Blur/Sharpness (focus varies across slides)
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.MotionBlur(blur_limit=5, p=0.5),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),
        ], p=0.3),

        # Noise
        noise_tf,
    ]


def get_train_transforms_aggressive(cfg: Config):
    """
    Get aggressive training transforms for normal/background tiles (no pathology).

    These tiles don't need protection - can use full augmentation range.
    """
    noise_tf = _build_gauss_noise()
    h, w = int(cfg.IMAGE_SHAPE[0]), int(cfg.IMAGE_SHAPE[1])

    return A.Compose([
        A.Resize(h, w),

        # Aggressive geometric augmentations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.Rotate(limit=360, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
            A.Affine(
                translate_percent=0.35,  # Full range - OK for normal tiles
                scale=(0.8, 1.2),
                rotate=(-360, 360),
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.5
            ),
        ], p=0.7),

        # Elastic/grid distortion
        A.OneOf([
            A.ElasticTransform(
                alpha=120,
                sigma=120 * 0.05,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.5
            ),
            A.GridDistortion(num_steps=5, distort_limit=0.3, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        ], p=0.3),

        # Color augmentations (same as conservative)
        *_get_color_augmentations(noise_tf),

        # Aggressive coarse dropout (OK for normal tiles)
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(int(h * 0.05), int(h * 0.15)),
            hole_width_range=(int(w * 0.05), int(w * 0.15)),
            fill=0,
            p=0.4
        ),

        # Preprocessing
        A.Lambda(
            image=partial(
                _preprocess_image,
                use_encoder_preprocessing=bool(cfg.PT_USE_ENCODER_PREPROCESSING),
                encoder_name=str(cfg.PT_ENCODER_NAME),
                encoder_weights=cfg.PT_ENCODER_WEIGHTS,
            ),
            mask=_identity_mask,
        ),
        ToTensorV2(),
    ])


def get_train_transforms(cfg: Config):
    """
    Get conservative training transforms for pathology tiles.

    Uses conservative geometric params to avoid truncating pathology.
    For normal tiles, use get_train_transforms_aggressive() instead.

    Returns:
        Albumentations transform pipeline for training
    """
    noise_tf = _build_gauss_noise()
    h, w = int(cfg.IMAGE_SHAPE[0]), int(cfg.IMAGE_SHAPE[1])

    # Geometric transforms - split into safe (rotation/flip) and risky (translate/scale)
    # Rotation and flips are safe - they don't remove content
    safe_geometric = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=360, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
    ])

    # Affine with conservative translate (max 15% instead of 35%)
    # Small translations are OK - they simulate slight positioning variation
    risky_geometric = A.OneOf([
        A.Affine(
            translate_percent=(-0.15, 0.15),  # Conservative: max 15% shift
            scale=(0.9, 1.1),  # Conservative: max 10% scale change
            rotate=(-15, 15),  # Small additional rotation
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.5
        ),
        # Elastic/grid distortion (cells deform naturally) - generally safe
        A.ElasticTransform(
            alpha=80,  # Reduced from 120
            sigma=80 * 0.05,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.5
        ),
        A.GridDistortion(
            num_steps=5,
            distort_limit=0.2,  # Reduced from 0.3
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.5
        ),
    ], p=0.5)

    return A.Compose([
        A.Resize(h, w),

        # Safe geometric augmentations (don't remove content)
        safe_geometric,

        # Risky geometric (conservative params to avoid truncating pathology)
        risky_geometric,

        # Color augmentations
        *_get_color_augmentations(noise_tf),

        # Coarse dropout / cutout (occlusion robustness)
        # Small holes only - won't remove significant pathology
        A.CoarseDropout(
            num_holes_range=(1, 4),  # Reduced from (1, 8)
            hole_height_range=(int(h * 0.03), int(h * 0.08)),  # Smaller holes
            hole_width_range=(int(w * 0.03), int(w * 0.08)),
            fill=0,
            p=0.2  # Reduced probability
        ),

        # Preprocessing (encoder-specific normalization)
        A.Lambda(
            image=partial(
                _preprocess_image,
                use_encoder_preprocessing=bool(cfg.PT_USE_ENCODER_PREPROCESSING),
                encoder_name=str(cfg.PT_ENCODER_NAME),
                encoder_weights=cfg.PT_ENCODER_WEIGHTS,
            ),
            mask=_identity_mask,
        ),
        ToTensorV2(),
    ])


def get_val_transforms(cfg: Config):
    """
    Get validation transforms (no augmentation).
    
    Returns:
        Albumentations transform pipeline for validation
    """
    h, w = int(cfg.IMAGE_SHAPE[0]), int(cfg.IMAGE_SHAPE[1])

    return A.Compose([
        A.Resize(h, w),
        A.Lambda(
            image=partial(
                _preprocess_image,
                use_encoder_preprocessing=bool(cfg.PT_USE_ENCODER_PREPROCESSING),
                encoder_name=str(cfg.PT_ENCODER_NAME),
                encoder_weights=cfg.PT_ENCODER_WEIGHTS,
            ),
            mask=_identity_mask,
        ),
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
    
    def __init__(self, lovasz_weight=1.0, ce_weight=1.0, label_smoothing: float = 0.0):
        super(CombinedLoss, self).__init__()
        # Lovasz Softmax Loss (typically for multiclass, using logit inputs)
        self.lovasz_loss = smp.losses.LovaszLoss(mode='multiclass', per_image=True)
        # Cross Entropy with Label Smoothing
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=float(label_smoothing))
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
    # Ensure class-index masks
    if y_true.dtype != torch.long:
        y_true = y_true.long()

    y_pred = torch.argmax(y_pred, dim=1)
    
    ious = []
    for cls in range(num_classes):
        pred_cls = (y_pred == cls)
        true_cls = (y_true == cls)
        
        intersection = (pred_cls & true_cls).float().sum()
        union = (pred_cls | true_cls).float().sum()
        
        if union.item() == 0:
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
    # Ensure class-index masks
    if y_true.dtype != torch.long:
        y_true = y_true.long()

    y_pred = torch.argmax(y_pred, dim=1)
    
    f1_scores = []
    for cls in range(num_classes):
        pred_cls = (y_pred == cls)
        true_cls = (y_true == cls)
        
        tp = (pred_cls & true_cls).float().sum()
        fp = (pred_cls & ~true_cls).float().sum()
        fn = (~pred_cls & true_cls).float().sum()

        # Dice/F1 for segmentation: 2TP / (2TP + FP + FN)
        denom = (2 * tp + fp + fn)
        if denom.item() == 0:
            f1_scores.append(1.0)  # No pixels in both pred and gt for this class
        else:
            f1 = (2 * tp) / (denom + 1e-7)
            f1_scores.append(f1.item())
    
    return np.mean(f1_scores)


def get_datasets(
    cfg: Config,
    images_path: str,
    masks_path: str,
    train_split: float | None = None,
    *,
    epoch: int = 0,
):
    """
    Create PyTorch datasets from image and mask directories.
    
    Args:
        images_path: Path to directory containing training images
        masks_path: Path to directory containing corresponding masks
        train_split: Fraction of data to use for training
        
    Returns:
        Tuple containing (train_dataset, val_dataset, total_samples)
    """
    # NOTE:
    # - 'static' split: one random holdout subset (val) fixed across epochs.
    # - 'cyclic' split: rotating/rolling holdout subset (val) changes each epoch.
    #   This effectively swaps the current validation subset back into training
    #   and brings in a new validation subset of the same size.

    # Allow cfg-driven split ratio by default.
    if train_split is None:
        val_frac = float(getattr(cfg, 'PT_VAL_FRACTION', 0.2) or 0.2)
        if not (0.0 < val_frac < 1.0):
            raise ValueError(f'PT_VAL_FRACTION must be in (0, 1), got {val_frac}')
        train_split = 1.0 - val_frac

    # Create full dataset just to compute length and establish deterministic file ordering.
    full_dataset = CytologyDataset(images_path, masks_path, transform=None)
    total_samples = len(full_dataset)

    # Split dataset sizes
    train_size = int(train_split * total_samples)
    val_size = total_samples - train_size

    # Edge cases: tiny datasets
    if total_samples == 0:
        raise ValueError('Dataset is empty: no images found')
    if val_size <= 0:
        # Keep at least one validation sample when possible
        val_size = 1 if total_samples > 1 else 0
        train_size = total_samples - val_size

    # Determine split strategy
    strategy = str(getattr(cfg, 'PT_VAL_STRATEGY', 'cyclic') or 'cyclic').strip().lower()
    if strategy in {'cycle', 'cycling', 'rotate', 'rotating', 'rolling'}:
        strategy = 'cyclic'
    if strategy in {'holdout', 'random', 'fixed'}:
        strategy = 'static'
    if strategy not in {'cyclic', 'static'}:
        raise ValueError(f"PT_VAL_STRATEGY must be 'cyclic' or 'static', got {strategy!r}")

    seed = int(getattr(cfg, 'PT_VAL_SEED', 42) or 42)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(total_samples)

    # Compute indices
    if val_size == 0:
        val_indices: list[int] = []
        train_indices = perm.tolist()
    elif strategy == 'static':
        val_indices = perm[:val_size].tolist()
        train_indices = perm[val_size:].tolist()
    else:
        # Rotating/rolling holdout: shift a validation window each epoch.
        if epoch < 0:
            raise ValueError(f'epoch must be >= 0, got {epoch}')
        start = (epoch * val_size) % total_samples
        window = perm.tolist()
        if start + val_size <= total_samples:
            val_indices = window[start:start + val_size]
        else:
            tail = window[start:]
            head = window[: (start + val_size) - total_samples]
            val_indices = tail + head

        # Complement for train indices
        val_set = set(val_indices)
        train_indices = [i for i in window if i not in val_set]

    # Create datasets with transforms
    # Train dataset gets both conservative (for pathology) and aggressive (for normal) transforms
    train_dataset = CytologyDataset(
        images_path,
        masks_path,
        transform=get_train_transforms(cfg),
        transform_aggressive=get_train_transforms_aggressive(cfg),
    )
    val_dataset = CytologyDataset(images_path, masks_path, transform=get_val_transforms(cfg))

    # Create subset datasets
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)

    return train_subset, val_subset, total_samples


def _train_model_loop(
    cfg: Config,
    model,
    criterion,
    optimizer,
    scheduler,
    scaler,
    amp_ctx,
    epochs,
    model_path,
    output_classes_metrics,
    batch_size: int,
    save_base_path: str | None = None,
):
    """
    Common training loop for PyTorch models.
    """
    best_iou = 0.0
    base_path = os.path.splitext(save_base_path or model_path)[0]
    
    # Prepare loaders. For cyclic validation we rebuild loaders each epoch.
    val_strategy = str(getattr(cfg, 'PT_VAL_STRATEGY', 'cyclic') or 'cyclic').strip().lower()
    if val_strategy in {'cycle', 'cycling', 'rotate', 'rotating', 'rolling'}:
        val_strategy = 'cyclic'
    if val_strategy in {'holdout', 'random', 'fixed'}:
        val_strategy = 'static'

    cached_loaders = None
    cached_total_samples = None

    for epoch in range(epochs):
        if val_strategy == 'cyclic':
            train_loader, val_loader, total_samples = _prepare_data_loaders(cfg, batch_size, epoch=epoch)
            cached_total_samples = total_samples
        else:
            if cached_loaders is None:
                train_loader, val_loader, total_samples = _prepare_data_loaders(cfg, batch_size, epoch=0)
                cached_loaders = (train_loader, val_loader)
                cached_total_samples = total_samples
            else:
                train_loader, val_loader = cached_loaders
        # Training phase
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False, dynamic_ncols=True)
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(DEVICE).float()
            masks = masks.to(DEVICE).long()
            
            optimizer.zero_grad(set_to_none=True)

            mixup_alpha = float(getattr(cfg, 'PT_MIXUP_ALPHA', 0.2) or 0.2)
            if mixup_alpha < 0:
                raise ValueError(f'PT_MIXUP_ALPHA must be >= 0, got {mixup_alpha}')

            with amp_ctx():
                if mixup_alpha > 0:
                    # Apply Mixup
                    inputs, targets_a, targets_b, lam = mixup_data(images, masks, alpha=mixup_alpha, device=DEVICE)
                    outputs = model(inputs)
                    # Compute loss for both targets and mix
                    loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                
            clip_norm = float(getattr(cfg, 'PT_GRAD_CLIP_NORM', 0.0) or 0.0)
            if scaler is not None:
                scaler.scale(loss).backward()
                if clip_norm > 0:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), max_norm=clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if clip_norm > 0:
                    clip_grad_norm_(model.parameters(), max_norm=clip_norm)
                optimizer.step()
            
            running_loss += loss.item()
            if hasattr(pbar, 'set_postfix'):
                # Handle both scalar and tensor loss for display
                loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss
                avg_loss = running_loss / (batch_idx + 1)
                lr_now = max((pg.get('lr', 0.0) for pg in optimizer.param_groups), default=0.0)
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


def _setup_training_components(cfg: Config, model, lr, use_amp):
    """
    Setup optimizer, scheduler, criterion, and mixed precision components.
    """
    label_smoothing = float(getattr(cfg, 'PT_LABEL_SMOOTHING', 0.0) or 0.0)
    if not (0.0 <= label_smoothing <= 1.0):
        raise ValueError(f'PT_LABEL_SMOOTHING must be in [0, 1], got {label_smoothing}')

    criterion = CombinedLoss(label_smoothing=label_smoothing)

    pt_optimizer = str(getattr(cfg, 'PT_OPTIMIZER', 'adamw')).lower()
    weight_decay = float(getattr(cfg, 'PT_WEIGHT_DECAY', 1e-4))
    encoder_lr_mult = float(getattr(cfg, 'PT_ENCODER_LR_MULT', 0.1))

    def _is_no_decay_param(param_name: str, param: torch.nn.Parameter) -> bool:
        if param_name.endswith('.bias'):
            return True
        if param.ndim <= 1:
            return True
        lname = param_name.lower()
        if 'bn' in lname or 'norm' in lname:
            return True
        return False

    def _build_param_groups(base_lr: float):
        enc_decay, enc_no_decay, other_decay, other_no_decay = [], [], [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            is_encoder = name.startswith('encoder.')
            no_decay = _is_no_decay_param(name, param)

            if is_encoder and no_decay:
                enc_no_decay.append(param)
            elif is_encoder:
                enc_decay.append(param)
            elif no_decay:
                other_no_decay.append(param)
            else:
                other_decay.append(param)

        enc_lr = base_lr * encoder_lr_mult
        groups = []
        if enc_decay:
            groups.append({'params': enc_decay, 'lr': enc_lr, 'weight_decay': weight_decay})
        if enc_no_decay:
            groups.append({'params': enc_no_decay, 'lr': enc_lr, 'weight_decay': 0.0})
        if other_decay:
            groups.append({'params': other_decay, 'lr': base_lr, 'weight_decay': weight_decay})
        if other_no_decay:
            groups.append({'params': other_no_decay, 'lr': base_lr, 'weight_decay': 0.0})
        return groups

    if pt_optimizer == 'nadam':
        optimizer = torch.optim.NAdam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.AdamW(_build_param_groups(lr), lr=lr)
    # Note: T_mult=2 matches original train_new_model implementation
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    
    if torch.cuda.is_available():
        # PyTorch AMP APIs differ slightly across versions.
        # - `torch.cuda.amp.GradScaler` exists broadly.
        # - `torch.amp.GradScaler` exists in newer versions.
        try:
            scaler = torch.amp.GradScaler('cuda', enabled=use_amp)  # type: ignore[attr-defined]
        except AttributeError:
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        def amp_ctx():
            try:
                return torch.amp.autocast(device_type='cuda', enabled=use_amp)  # type: ignore[attr-defined]
            except AttributeError:
                return torch.cuda.amp.autocast(enabled=use_amp)
    else:
        scaler = None

        def amp_ctx():
            return nullcontext()
        
    return criterion, optimizer, scheduler, scaler, amp_ctx


def _prepare_data_loaders(cfg: Config, batch_size, *, epoch: int = 0):
    """
    Create data loaders from config paths.
    """
    images_path = os.path.join(cfg.DATASET_FOLDER, cfg.IMAGES_FOLDER)
    masks_path = os.path.join(cfg.DATASET_FOLDER, cfg.MASKS_FOLDER)
    
    # If caller didn't provide an explicit train_split, default to cfg.PT_VAL_FRACTION (80/20).
    val_frac = float(getattr(cfg, 'PT_VAL_FRACTION', 0.2) or 0.2)
    if not (0.0 < val_frac < 1.0):
        raise ValueError(f'PT_VAL_FRACTION must be in (0, 1), got {val_frac}')
    train_split = 1.0 - val_frac

    train_dataset, val_dataset, total_samples = get_datasets(
        cfg,
        images_path,
        masks_path,
        train_split=train_split,
        epoch=epoch,
    )
    
    if int(getattr(cfg, 'PT_NUM_WORKERS', -1)) >= 0:
        num_workers = int(cfg.PT_NUM_WORKERS)
    else:
        # On macOS the multiprocessing start method is spawn, which often adds
        # overhead for cv2/albumentations-heavy pipelines. Default to 0 unless
        # explicitly configured.
        if sys.platform == 'darwin':
            num_workers = 0
        else:
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


def train_new_model_pytorch(
    cfg: Config,
    model_path: str,
    output_classes: int,
    epochs: int,
    batch_size: int = 64,
    lr: float | None = None,
    use_amp: bool = True,
):
    """
    Train a new U-Net model using PyTorch.
    """
    set_seed(42)
    _log_device_selection(prefix="[train_new_model_pytorch]")
    
    # Print total samples once (epoch 0 split).
    _, _, total_samples = _prepare_data_loaders(cfg, batch_size, epoch=0)
    print(f"Total samples: {total_samples}")
    
    model = smp.Unet(
        encoder_name=str(cfg.PT_ENCODER_NAME),
        encoder_weights=cfg.PT_ENCODER_WEIGHTS,
        classes=output_classes,
        activation=None
    )
    model = model.to(DEVICE)
    
    if lr is None:
        lr = float(getattr(cfg, 'PT_LR', 1e-3))
    criterion, optimizer, scheduler, scaler, amp_ctx = _setup_training_components(cfg, model, lr, use_amp)
    
    _train_model_loop(
        cfg, model, criterion, optimizer, scheduler, scaler, amp_ctx,
        epochs, model_path, output_classes, batch_size=batch_size, save_base_path=model_path
    )


def train_current_model_pytorch(
    cfg: Config,
    model_path: str,
    epochs: int,
    batch_size: int = 64,
    lr: float | None = None,
    use_amp: bool = True,
    save_base_path: str | None = None,
):
    """
    Continue training an existing PyTorch model.
    """
    set_seed(42)
    _log_device_selection(prefix="[train_current_model_pytorch]")
    
    # Warm up loaders (epoch 0 split) to validate dataset and print device info early.
    _prepare_data_loaders(cfg, batch_size, epoch=0)
    
    # Load existing architecture
    model = smp.Unet(
        encoder_name=str(cfg.PT_ENCODER_NAME),
        encoder_weights=cfg.PT_ENCODER_WEIGHTS,
        classes=cfg.CLASSES,
        activation=None
    )
    try:
        state = torch.load(model_path, map_location=DEVICE, weights_only=True)
    except TypeError:  # older torch
        state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model = model.to(DEVICE)
    
    if lr is None:
        lr = float(getattr(cfg, 'PT_LR', 1e-3))
    criterion, optimizer, scheduler, scaler, amp_ctx = _setup_training_components(cfg, model, lr, use_amp)
    
    if save_base_path is None:
        save_base_path = model_path
        for suffix in ("_last.pth", "_best.pth", "_final.pth"):
            if save_base_path.endswith(suffix):
                save_base_path = save_base_path[: -len(suffix)]
                break
        else:
            if save_base_path.endswith('.pth') and '_epoch' in save_base_path:
                save_base_path = save_base_path.rsplit('_epoch', 1)[0]

    _train_model_loop(
        cfg, model, criterion, optimizer, scheduler, scaler, amp_ctx,
        epochs, model_path, cfg.CLASSES, batch_size=batch_size, save_base_path=save_base_path
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
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "matplotlib is unavailable (often due to a NumPy/Matplotlib binary mismatch). "
            "Fix by installing compatible wheels (commonly: 'pip install \"numpy<2\" matplotlib') "
            "or by running without display functions."
        ) from e
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
