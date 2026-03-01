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

# Avoid network calls/version-check noise in restricted environments.
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')

import albumentations as A
import inspect
from albumentations.pytorch import ToTensorV2

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import segmentation_models_pytorch as smp
from PIL import Image

from config import Config
from clogic.preprocessing_pytorch import preprocess_rgb_image

try:
    from skimage import color as skcolor
    _HAS_SKIMAGE_COLOR = True
except Exception:
    skcolor = None
    _HAS_SKIMAGE_COLOR = False
_PRINTED_HED_UNAVAILABLE_WARNING = False

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


def _resolve_hw_tuple(value, fallback=(256, 256)) -> tuple[int, int]:
    try:
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return int(value[0]), int(value[1])
    except Exception:
        pass
    return int(fallback[0]), int(fallback[1])

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
        pathology_min_keep_ratio: float = 0.9,
        pathology_min_pixels_after_aug: int = 10,
        aug_skip_prob: float = 0.2,
        sanity_check: bool = True,
        sanity_check_max_masks: int = 200,
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
        self.pathology_min_keep_ratio = float(pathology_min_keep_ratio)
        self.pathology_min_pixels_after_aug = max(1, int(pathology_min_pixels_after_aug))
        self.aug_skip_prob = float(min(max(float(aug_skip_prob), 0.0), 1.0))
        self._pathology_pixels_cache: dict[int, int] = {}

        self.images = [f for f in os.listdir(images_path) if f.endswith('.bmp')]
        self.masks = [f for f in os.listdir(masks_path) if f.endswith('.bmp')]

        # Ensure images and masks are aligned
        self.images.sort()
        self.masks.sort()

        if sanity_check:
            _sanity_check_dataset_pairs(
                images_path,
                masks_path,
                self.images,
                self.masks,
                max_masks=sanity_check_max_masks,
            )

    @staticmethod
    def _apply_resize_preprocess_only(image: np.ndarray, mask: np.ndarray, transform_to_use) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply deterministic train path: resize + encoder preprocess + tensor conversion."""
        h, w = image.shape[:2]
        if transform_to_use is not None and hasattr(transform_to_use, 'transforms'):
            for t in transform_to_use.transforms:
                if isinstance(t, A.Resize):
                    h = int(t.height)
                    w = int(t.width)
                    break
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        if transform_to_use is not None and hasattr(transform_to_use, 'transforms'):
            for t in transform_to_use.transforms:
                if isinstance(t, A.Lambda) and hasattr(t, 'image'):
                    image = t.image(image)
                    break

        image_t = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask_t = torch.from_numpy(mask)
        return image_t, mask_t
        
    def __len__(self):
        return len(self.images)

    def pathology_pixels(self, idx: int) -> int:
        """Return pathology pixel count for a sample (cached)."""
        cached = self._pathology_pixels_cache.get(int(idx))
        if cached is not None:
            return int(cached)

        mask_path = os.path.join(self.masks_path, self.masks[idx])
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f'Failed to read mask: {mask_path}')
        labels = self._convert_mask_to_labels(mask)
        count = int(np.count_nonzero(labels == _MASK_CLASS_PATHOLOGY))
        self._pathology_pixels_cache[int(idx)] = count
        return count
    
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
            transform_to_use = self.transform if has_pathology else (self.transform_aggressive or self.transform)
            skip_augs = False

            if self.aug_skip_prob > 0.0 and random.random() < self.aug_skip_prob:
                image, mask = self._apply_resize_preprocess_only(image, mask, transform_to_use)
                skip_augs = True

            if not skip_augs and has_pathology:
                # Use conservative transform with retry for pathology tiles
                max_attempts = 3
                min_ratio = self.pathology_min_keep_ratio

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

                    required_by_ratio = int(np.ceil(float(min_ratio) * float(original_pathology_area)))
                    required_by_abs = min(self.pathology_min_pixels_after_aug, int(original_pathology_area))
                    required_area = max(required_by_ratio, required_by_abs)
                    ratio = new_pathology_area / original_pathology_area if original_pathology_area > 0 else 1.0

                    if ratio >= min_ratio and int(new_pathology_area) >= int(required_area):
                        image = transformed['image']
                        mask = new_mask
                        break
                else:
                    # All attempts failed - use original without augmentation
                    image, mask = self._apply_resize_preprocess_only(original_image, original_mask, self.transform)
            elif not skip_augs:
                # No pathology - use aggressive transform if available
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


def _sanity_check_dataset_pairs(
    images_path: str,
    masks_path: str,
    images: list[str],
    masks: list[str],
    *,
    max_masks: int = 200,
) -> None:
    """Validate that image/mask pairs align and masks have expected values."""
    if not images:
        raise ValueError(f'No .bmp images found in: {images_path}')
    if not masks:
        raise ValueError(f'No .bmp masks found in: {masks_path}')

    image_stems = [os.path.splitext(name)[0] for name in images]
    mask_stems = [os.path.splitext(name)[0] for name in masks]

    if len(image_stems) != len(mask_stems):
        raise ValueError(
            f'Mismatched dataset sizes: images={len(image_stems)}, masks={len(mask_stems)}. '
            f'Check {images_path} and {masks_path} for extra/missing files.'
        )

    if image_stems != mask_stems:
        image_set = set(image_stems)
        mask_set = set(mask_stems)
        missing_masks = sorted(image_set - mask_set)[:10]
        missing_images = sorted(mask_set - image_set)[:10]
        raise ValueError(
            'Image/mask filename mismatch after sorting. '
            f'Missing masks (examples): {missing_masks}. '
            f'Missing images (examples): {missing_images}.'
        )

    # Spot-check mask values to ensure they are visible-coded (0, 127, 255).
    allowed_values = {0, 127, 255}
    to_check = masks if len(masks) <= max_masks else random.sample(masks, max_masks)
    for name in to_check:
        mask_path = os.path.join(masks_path, name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f'Failed to read mask: {mask_path}')
        unique_vals = set(np.unique(mask).tolist())
        unexpected = sorted(v for v in unique_vals if v not in allowed_values)
        if unexpected:
            raise ValueError(
                f'Unexpected mask values in {mask_path}: {unexpected}. '
                'Expected only {0, 127, 255}. '
                'This usually means masks are RGB/debug or already label-indexed.'
            )


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


def _clip_u8(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0, 255).astype(np.uint8, copy=False)


def _hed_stain_jitter(
    image: np.ndarray,
    sigma: float = 0.05,
    bias_sigma: float = 0.01,
    **kwargs,
) -> np.ndarray:
    """HED-space perturbation (safe no-op if skimage is unavailable)."""
    if not _HAS_SKIMAGE_COLOR:
        return image
    if image.dtype != np.uint8:
        image = _clip_u8(image)

    rgb = image.astype(np.float32) / 255.0
    hed = skcolor.rgb2hed(rgb)
    hed = hed + np.random.normal(0.0, float(sigma), size=hed.shape).astype(np.float32)
    hed = hed + np.random.normal(0.0, float(bias_sigma), size=(1, 1, 3)).astype(np.float32)
    out = skcolor.hed2rgb(hed)
    out = np.clip(out, 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


def _soft_stain_normalize(
    image: np.ndarray,
    blend_min: float = 0.15,
    blend_max: float = 0.35,
    **kwargs,
) -> np.ndarray:
    """
    Soft channel-percentile normalization blended with original image.

    This is lighter than strict stain normalization and reduces color outliers
    that often trigger false pathology on bright normal nuclei.
    """
    if image.dtype != np.uint8:
        image = _clip_u8(image)

    x = image.astype(np.float32)
    norm = np.empty_like(x)
    for ch in range(3):
        channel = x[..., ch]
        lo = float(np.percentile(channel, 2.0))
        hi = float(np.percentile(channel, 98.0))
        if hi <= lo + 1e-6:
            norm[..., ch] = channel
        else:
            norm[..., ch] = np.clip((channel - lo) * (255.0 / (hi - lo)), 0.0, 255.0)

    blend_lo = float(min(blend_min, blend_max))
    blend_hi = float(max(blend_min, blend_max))
    alpha = float(np.random.uniform(blend_lo, blend_hi))
    mixed = (1.0 - alpha) * x + alpha * norm
    return _clip_u8(mixed)


def _safe_prob(value: float, default: float) -> float:
    try:
        v = float(value)
    except Exception:
        v = float(default)
    return min(max(v, 0.0), 1.0)


def _get_color_augmentations(cfg: Config, noise_tf):
    """Common color/stain augmentations for both pathology and normal tiles."""
    global _PRINTED_HED_UNAVAILABLE_WARNING
    use_stain_aware = bool(getattr(cfg, 'PT_STAIN_AWARE_AUG', True))
    hsv_prob = _safe_prob(getattr(cfg, 'PT_STAIN_HSV_PROB', 0.35), 0.35)
    hed_prob = _safe_prob(getattr(cfg, 'PT_STAIN_HED_PROB', 0.25), 0.25)
    norm_prob = _safe_prob(getattr(cfg, 'PT_STAIN_NORM_PROB', 0.20), 0.20)
    blend_range = getattr(cfg, 'PT_STAIN_NORM_BLEND_RANGE', (0.15, 0.35))
    if not isinstance(blend_range, (list, tuple)) or len(blend_range) != 2:
        blend_range = (0.15, 0.35)
    blend_min = float(blend_range[0])
    blend_max = float(blend_range[1])

    stain_ops: list[tuple[A.BasicTransform, float]] = []
    if use_stain_aware:
        if hsv_prob > 0:
            stain_ops.append((
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=18,
                    val_shift_limit=12,
                    p=1.0,
                ),
                hsv_prob,
            ))
        if hed_prob > 0:
            if _HAS_SKIMAGE_COLOR:
                stain_ops.append((
                    A.Lambda(
                        image=partial(_hed_stain_jitter, sigma=0.05, bias_sigma=0.01),
                        mask=_identity_mask,
                        p=1.0,
                    ),
                    hed_prob,
                ))
            elif not _PRINTED_HED_UNAVAILABLE_WARNING:
                print('[aug] skimage is unavailable, disabling HED stain jitter.')
                _PRINTED_HED_UNAVAILABLE_WARNING = True
        if norm_prob > 0:
            stain_ops.append((
                A.Lambda(
                    image=partial(_soft_stain_normalize, blend_min=blend_min, blend_max=blend_max),
                    mask=_identity_mask,
                    p=1.0,
                ),
                norm_prob,
            ))

    if stain_ops:
        # Reweight transform selection by repeating operations according to configured priorities.
        weighted_ops = []
        def _repeat_by_prob(op, p):
            reps = max(1, int(round(10.0 * float(p))))
            for _ in range(reps):
                weighted_ops.append(op)
        for op, prob in stain_ops:
            _repeat_by_prob(op, prob)
        ops = weighted_ops if weighted_ops else [op for op, _ in stain_ops]
        stain_block = [A.OneOf(ops, p=0.7)]
    else:
        stain_block = [
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=12, sat_shift_limit=20, val_shift_limit=12, p=1.0),
                A.RGBShift(r_shift_limit=12, g_shift_limit=12, b_shift_limit=12, p=1.0),
                A.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.12, hue=0.04, p=1.0),
            ], p=0.6),
        ]

    return [
        # Stain-aware color perturbation (HSV/HED/soft-normalization).
        *stain_block,

        # Brightness/Contrast + CLAHE (lighting and contrast variation)
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.15, p=1.0),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            A.RandomGamma(gamma_limit=(90, 110), p=1.0),
        ], p=0.35),

        # Blur/Sharpness (focus varies across slides)
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=3, p=1.0),
            A.Sharpen(alpha=(0.1, 0.3), lightness=(0.7, 1.0), p=1.0),
        ], p=0.20),

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
        *_get_color_augmentations(cfg, noise_tf),

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
    rotate_limit = int(getattr(cfg, 'PT_PATHOLOGY_ROTATE_LIMIT', 30) or 30)
    rotate_limit = max(0, rotate_limit)
    strict_pathology_aug = bool(getattr(cfg, 'PT_PATHOLOGY_STRICT_AUG', True))

    # Geometric transforms - split into safe (rotation/flip) and risky (translate/scale)
    # Rotation and flips are safe - they don't remove content
    safe_geometric = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=rotate_limit, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
    ])

    transforms = [
        A.Resize(h, w),
        # Safe geometric augmentations (don't remove content)
        safe_geometric,
    ]

    if not strict_pathology_aug:
        # Optional risky geometry for non-strict mode.
        transforms.append(
            A.OneOf([
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
        )

    # Color augmentations are kept in both modes.
    transforms.extend(_get_color_augmentations(cfg, noise_tf))

    if not strict_pathology_aug:
        # Optional occlusion robustness for non-strict mode only.
        transforms.append(
            A.CoarseDropout(
                num_holes_range=(1, 4),  # Reduced from (1, 8)
                hole_height_range=(int(h * 0.03), int(h * 0.08)),  # Smaller holes
                hole_width_range=(int(w * 0.03), int(w * 0.08)),
                fill=0,
                p=0.2  # Reduced probability
            )
        )

    # Preprocessing (encoder-specific normalization)
    transforms.extend([
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

    return A.Compose(transforms)


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
    
    def __init__(
        self,
        lovasz_weight=1.0,
        ce_weight=1.0,
        label_smoothing: float = 0.0,
        class_weights: torch.Tensor | None = None,
    ):
        super(CombinedLoss, self).__init__()
        # Lovasz Softmax Loss (typically for multiclass, using logit inputs)
        self.lovasz_loss = smp.losses.LovaszLoss(mode='multiclass', per_image=True)
        # Cross Entropy with Label Smoothing
        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=float(label_smoothing),
        )
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


class FocalTverskyLoss(nn.Module):
    """Multi-class focal Tversky loss with optional class weighting."""

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 1.33,
        smooth: float = 1e-6,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.smooth = float(smooth)
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights.float())
        else:
            self.class_weights = None

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_true.dtype != torch.long:
            y_true = y_true.long()
        num_classes = y_pred.shape[1]
        y_true_onehot = F.one_hot(y_true, num_classes=num_classes).permute(0, 3, 1, 2).float()
        probs = F.softmax(y_pred, dim=1)

        dims = (0, 2, 3)
        tp = torch.sum(probs * y_true_onehot, dim=dims)
        fp = torch.sum(probs * (1.0 - y_true_onehot), dim=dims)
        fn = torch.sum((1.0 - probs) * y_true_onehot, dim=dims)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        focal_tversky = torch.pow(1.0 - tversky, self.gamma)

        if self.class_weights is not None and self.class_weights.numel() == focal_tversky.numel():
            weights = self.class_weights / self.class_weights.sum().clamp_min(self.smooth)
            return torch.sum(focal_tversky * weights)
        return focal_tversky.mean()


class FocalTverskyCELoss(nn.Module):
    """Recall-biased semantic segmentation loss: Focal-Tversky + CE."""

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 1.33,
        tversky_weight: float = 1.0,
        ce_weight: float = 0.5,
        label_smoothing: float = 0.0,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.tversky_weight = float(tversky_weight)
        self.ce_weight = float(ce_weight)
        self.tversky_loss = FocalTverskyLoss(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            class_weights=class_weights,
        )
        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=float(label_smoothing),
        )

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_true.dtype != torch.long:
            y_true = y_true.long()
        return self.tversky_weight * self.tversky_loss(y_pred, y_true) + self.ce_weight * self.ce_loss(y_pred, y_true)


class BoundaryLoss(nn.Module):
    """Edge-consistency loss on class boundaries (multi-class)."""

    def __init__(
        self,
        num_classes: int,
        kernel_size: int = 3,
        class_weights: torch.Tensor | None = None,
        ignore_background: bool = True,
        smooth: float = 1e-6,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        k = int(kernel_size)
        self.kernel_size = k if k % 2 == 1 else (k + 1)
        self.pad = self.kernel_size // 2
        self.ignore_background = bool(ignore_background)
        self.smooth = float(smooth)
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights.float())
        else:
            self.class_weights = None

    def _boundary_map(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C, H, W], values in [0,1]
        dilated = F.max_pool2d(x, kernel_size=self.kernel_size, stride=1, padding=self.pad)
        eroded = -F.max_pool2d(-x, kernel_size=self.kernel_size, stride=1, padding=self.pad)
        return (dilated - eroded).clamp(min=0.0, max=1.0)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_true.dtype != torch.long:
            y_true = y_true.long()

        probs = F.softmax(y_pred, dim=1)
        gt_onehot = F.one_hot(y_true, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        pred_b = self._boundary_map(probs)
        gt_b = self._boundary_map(gt_onehot)

        class_start = 1 if self.ignore_background and self.num_classes > 1 else 0
        pred_b = pred_b[:, class_start:, :, :]
        gt_b = gt_b[:, class_start:, :, :]

        dims = (0, 2, 3)
        intersection = torch.sum(pred_b * gt_b, dim=dims)
        denom = torch.sum(pred_b, dim=dims) + torch.sum(gt_b, dim=dims)
        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        per_class_loss = 1.0 - dice

        if self.class_weights is not None:
            cw = self.class_weights[class_start:]
            if cw.numel() == per_class_loss.numel():
                cw = cw / cw.sum().clamp_min(self.smooth)
                return torch.sum(per_class_loss * cw)
        return per_class_loss.mean()


class CompositeLoss(nn.Module):
    """Weighted sum of base segmentation loss and boundary loss."""

    def __init__(self, base_loss: nn.Module, boundary_loss: nn.Module, boundary_weight: float = 0.2):
        super().__init__()
        self.base_loss = base_loss
        self.boundary_loss = boundary_loss
        self.boundary_weight = float(max(0.0, boundary_weight))

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        base = self.base_loss(y_pred, y_true)
        if self.boundary_weight <= 0.0:
            return base
        return base + self.boundary_weight * self.boundary_loss(y_pred, y_true)


def _resolve_class_weights(cfg: Config, num_classes: int) -> torch.Tensor | None:
    raw_weights = getattr(cfg, 'PT_CLASS_WEIGHTS', None)
    if raw_weights is None:
        return None
    if not isinstance(raw_weights, (list, tuple)) or len(raw_weights) == 0:
        return None

    weights = [float(w) for w in raw_weights]
    if len(weights) < num_classes:
        weights.extend([1.0] * (num_classes - len(weights)))
    if len(weights) > num_classes:
        weights = weights[:num_classes]

    if all(w <= 0 for w in weights):
        return None
    weights = [max(0.0, w) for w in weights]
    if sum(weights) <= 0:
        return None
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def _iter_metric_classes(num_classes: int, include_background: bool) -> range:
    start = 0 if include_background else 1
    return range(start, num_classes)


def iou_score(y_pred, y_true, num_classes=3, *, include_background: bool = False):
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
    for cls in _iter_metric_classes(num_classes, include_background):
        pred_cls = (y_pred == cls)
        true_cls = (y_true == cls)
        
        intersection = (pred_cls & true_cls).float().sum()
        union = (pred_cls | true_cls).float().sum()
        
        if union.item() == 0:
            ious.append(1.0)  # If no pixels for this class, perfect score
        else:
            ious.append((intersection / union).item())
    
    return float(np.mean(ious)) if ious else 0.0


def f1_score(y_pred, y_true, num_classes=3, *, include_background: bool = False):
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
    for cls in _iter_metric_classes(num_classes, include_background):
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
    
    return float(np.mean(f1_scores)) if f1_scores else 0.0


def class_iou_score(y_pred, y_true, class_idx: int = 2):
    if y_true.dtype != torch.long:
        y_true = y_true.long()
    y_pred = torch.argmax(y_pred, dim=1)

    pred_cls = (y_pred == class_idx)
    true_cls = (y_true == class_idx)
    intersection = (pred_cls & true_cls).float().sum()
    union = (pred_cls | true_cls).float().sum()
    if union.item() == 0:
        return 1.0
    return (intersection / union).item()


def class_f1_score(y_pred, y_true, class_idx: int = 2):
    if y_true.dtype != torch.long:
        y_true = y_true.long()
    y_pred = torch.argmax(y_pred, dim=1)

    pred_cls = (y_pred == class_idx)
    true_cls = (y_true == class_idx)
    tp = (pred_cls & true_cls).float().sum()
    fp = (pred_cls & ~true_cls).float().sum()
    fn = (~pred_cls & true_cls).float().sum()
    denom = (2 * tp + fp + fn)
    if denom.item() == 0:
        return 1.0
    return ((2 * tp) / (denom + 1e-7)).item()


def binary_average_precision_score(y_prob: np.ndarray, y_true: np.ndarray) -> float:
    """
    Compute binary average precision (AP) without external deps.
    """
    y_prob = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    y_true = np.asarray(y_true, dtype=np.uint8).reshape(-1)
    positives = int(y_true.sum())
    if positives <= 0:
        return 0.0

    order = np.argsort(-y_prob, kind='mergesort')
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    precision = tp / np.maximum(tp + fp, 1)
    ap = float(precision[y_sorted == 1].sum() / positives)
    return ap


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
        pathology_min_keep_ratio=float(getattr(cfg, 'PT_PATHOLOGY_MIN_KEEP_RATIO', 0.9) or 0.9),
        pathology_min_pixels_after_aug=int(getattr(cfg, 'MIN_PATHOLOGY_PIXELS', 10) or 10),
        aug_skip_prob=float(getattr(cfg, 'PT_AUG_SKIP_PROB', 0.2) or 0.0),
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
    scheduler_step_per_batch: bool,
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
    best_score = float('-inf')
    base_path = os.path.splitext(save_base_path or model_path)[0]
    pathology_class_idx = int(getattr(cfg, 'PT_PATHOLOGY_CLASS_INDEX', 2) or 2)
    include_background_metrics = bool(getattr(cfg, 'PT_INCLUDE_BACKGROUND_METRICS', False))
    checkpoint_metric = str(getattr(cfg, 'PT_CHECKPOINT_METRIC', 'pathology_iou') or 'pathology_iou').strip().lower()
    allowed_checkpoint_metrics = {
        'mean_iou',
        'mean_f1',
        'pathology_iou',
        'pathology_f1',
        'pathology_pr_auc',
    }
    if checkpoint_metric not in allowed_checkpoint_metrics:
        raise ValueError(
            f"PT_CHECKPOINT_METRIC must be one of {sorted(allowed_checkpoint_metrics)}, got {checkpoint_metric!r}"
        )
    
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
            did_optimizer_step = False

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
                scale_before = float(scaler.get_scale())
                scaler.scale(loss).backward()
                if clip_norm > 0:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), max_norm=clip_norm)
                scaler.step(optimizer)
                scaler.update()
                scale_after = float(scaler.get_scale())
                # GradScaler skips optimizer.step() on inf/nan gradients.
                # In that case we must not advance OneCycle per-batch scheduler.
                did_optimizer_step = scale_after >= scale_before
            else:
                loss.backward()
                if clip_norm > 0:
                    clip_grad_norm_(model.parameters(), max_norm=clip_norm)
                optimizer.step()
                did_optimizer_step = True

            if scheduler is not None and scheduler_step_per_batch and did_optimizer_step:
                try:
                    scheduler.step()
                except Exception:
                    pass
            
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
        val_path_iou = 0.0
        val_path_f1 = 0.0
        pr_probs: list[np.ndarray] = []
        pr_labels: list[np.ndarray] = []
        
        with torch.no_grad():
            vbar = tqdm(val_loader, desc=f"Val   {epoch+1}/{epochs}", leave=False, dynamic_ncols=True)
            for images, masks in vbar:
                images = images.to(DEVICE).float()
                masks = masks.to(DEVICE).long()
                
                outputs = model(images)
                
                # Calculate metrics
                iou = iou_score(
                    outputs,
                    masks,
                    output_classes_metrics,
                    include_background=include_background_metrics,
                )
                f1 = f1_score(
                    outputs,
                    masks,
                    output_classes_metrics,
                    include_background=include_background_metrics,
                )
                path_iou = class_iou_score(outputs, masks, pathology_class_idx)
                path_f1 = class_f1_score(outputs, masks, pathology_class_idx)
                
                val_iou += iou
                val_f1 += f1
                val_path_iou += path_iou
                val_path_f1 += path_f1

                if checkpoint_metric == 'pathology_pr_auc':
                    probs = torch.softmax(outputs, dim=1)[:, pathology_class_idx]
                    labels = (masks == pathology_class_idx)
                    pr_probs.append(probs.detach().float().cpu().numpy().reshape(-1))
                    pr_labels.append(labels.detach().cpu().numpy().astype(np.uint8).reshape(-1))
                if hasattr(vbar, 'set_postfix'):
                    seen = max(1, vbar.n)
                    vbar.set_postfix(
                        iou=f"{val_iou/seen:.4f}",
                        f1=f"{val_f1/seen:.4f}",
                        p_iou=f"{val_path_iou/seen:.4f}",
                    )
        
        val_batches = max(1, len(val_loader))
        avg_val_iou = val_iou / val_batches
        avg_val_f1 = val_f1 / val_batches
        avg_val_path_iou = val_path_iou / val_batches
        avg_val_path_f1 = val_path_f1 / val_batches
        avg_val_path_pr_auc = 0.0
        if checkpoint_metric == 'pathology_pr_auc':
            if pr_probs and pr_labels:
                avg_val_path_pr_auc = binary_average_precision_score(
                    np.concatenate(pr_probs),
                    np.concatenate(pr_labels),
                )
            else:
                avg_val_path_pr_auc = 0.0
        
        msg = (
            f'Epoch {epoch+1}/{epochs}: '
            f'Train Loss: {avg_train_loss:.4f}, '
            f'Val IoU: {avg_val_iou:.4f}, '
            f'Val F1: {avg_val_f1:.4f}, '
            f'Path IoU: {avg_val_path_iou:.4f}, '
            f'Path F1: {avg_val_path_f1:.4f}'
        )
        if checkpoint_metric == 'pathology_pr_auc':
            msg += f', Path PR AUC: {avg_val_path_pr_auc:.4f}'
        print(msg)

        # Step LR scheduler
        if scheduler is not None and not scheduler_step_per_batch:
            try:
                scheduler.step(epoch + 1)
            except Exception:
                pass

        # Optional per-epoch checkpointing (can generate many large files).
        if bool(getattr(cfg, 'PT_SAVE_EVERY_EPOCH', False)):
            epoch_weights = f"{base_path}_epoch{epoch+1:03d}.pth"
            torch.save(model.state_dict(), epoch_weights)
        torch.save(model.state_dict(), f"{base_path}_last.pth")

        metric_values = {
            'mean_iou': avg_val_iou,
            'mean_f1': avg_val_f1,
            'pathology_iou': avg_val_path_iou,
            'pathology_f1': avg_val_path_f1,
            'pathology_pr_auc': avg_val_path_pr_auc,
        }
        current_score = metric_values[checkpoint_metric]

        # Save best model
        if current_score > best_score:
            best_score = current_score
            torch.save(model.state_dict(), f"{base_path}_best.pth")
            print(f'New best model saved with {checkpoint_metric}: {best_score:.4f}')
    
    # Save final model
    torch.save(model.state_dict(), f"{base_path}_final.pth")
    print(f'Training completed. Best {checkpoint_metric}: {best_score:.4f}')


def _setup_training_components(
    cfg: Config,
    model,
    lr,
    use_amp,
    *,
    epochs: int,
    steps_per_epoch: int,
    scheduler_mode: str | None = None,
):
    """
    Setup optimizer, scheduler, criterion, and mixed precision components.
    """
    label_smoothing = float(getattr(cfg, 'PT_LABEL_SMOOTHING', 0.0) or 0.0)
    if not (0.0 <= label_smoothing <= 1.0):
        raise ValueError(f'PT_LABEL_SMOOTHING must be in [0, 1], got {label_smoothing}')

    num_classes = int(getattr(cfg, 'CLASSES', 3) or 3)
    class_weights = _resolve_class_weights(cfg, num_classes=num_classes)
    loss_mode = str(getattr(cfg, 'PT_LOSS', 'focal_tversky_ce') or 'focal_tversky_ce').strip().lower()
    if loss_mode in {'lovasz_ce', 'combined_loss', 'combined'}:
        criterion_base = CombinedLoss(
            label_smoothing=label_smoothing,
            class_weights=class_weights,
        )
    elif loss_mode in {'focal_tversky', 'focal_tversky_ce', 'tversky_ce'}:
        alpha = float(getattr(cfg, 'PT_TVERSKY_ALPHA', 0.3) or 0.3)
        beta = float(getattr(cfg, 'PT_TVERSKY_BETA', 0.7) or 0.7)
        gamma = float(getattr(cfg, 'PT_TVERSKY_GAMMA', 1.33) or 1.33)
        ce_w = float(getattr(cfg, 'PT_LOSS_CE_WEIGHT', 0.5) or 0.5)
        tv_w = float(getattr(cfg, 'PT_LOSS_TVERSKY_WEIGHT', 1.0) or 1.0)
        if alpha < 0 or beta < 0:
            raise ValueError(f'PT_TVERSKY_ALPHA/PT_TVERSKY_BETA must be >= 0, got alpha={alpha}, beta={beta}')
        if gamma <= 0:
            raise ValueError(f'PT_TVERSKY_GAMMA must be > 0, got {gamma}')
        criterion_base = FocalTverskyCELoss(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            tversky_weight=tv_w,
            ce_weight=ce_w,
            label_smoothing=label_smoothing,
            class_weights=class_weights,
        )
    else:
        raise ValueError(
            f"Unsupported PT_LOSS={loss_mode!r}. Use 'combined' or 'focal_tversky_ce'."
        )

    boundary_weight = float(getattr(cfg, 'PT_BOUNDARY_LOSS_WEIGHT', 0.0) or 0.0)
    if boundary_weight > 0.0:
        boundary_kernel_size = int(getattr(cfg, 'PT_BOUNDARY_KERNEL_SIZE', 3) or 3)
        boundary_ignore_bg = bool(getattr(cfg, 'PT_BOUNDARY_IGNORE_BACKGROUND', True))
        boundary_loss = BoundaryLoss(
            num_classes=num_classes,
            kernel_size=boundary_kernel_size,
            class_weights=class_weights,
            ignore_background=boundary_ignore_bg,
        )
        criterion = CompositeLoss(
            base_loss=criterion_base,
            boundary_loss=boundary_loss,
            boundary_weight=boundary_weight,
        )
    else:
        criterion = criterion_base

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
    if scheduler_mode is None:
        scheduler_mode = str(getattr(cfg, 'PT_SCHEDULER', 'onecycle') or 'onecycle')
    scheduler_mode = str(scheduler_mode).strip().lower()
    if scheduler_mode in {'cosine', 'cosinewarmrestarts', 'cosine_annealing_warm_restarts'}:
        scheduler_mode = 'cawr'
    if scheduler_mode not in {'onecycle', 'cawr'}:
        raise ValueError(f"Unsupported scheduler mode: {scheduler_mode!r}. Use 'onecycle' or 'cawr'.")
    scheduler_step_per_batch = False
    if scheduler_mode == 'onecycle':
        pct_start = float(getattr(cfg, 'PT_ONECYCLE_PCT_START', 0.1) or 0.1)
        pct_start = min(max(pct_start, 0.01), 0.99)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=max(1, int(epochs)),
            steps_per_epoch=max(1, int(steps_per_epoch)),
            pct_start=pct_start,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=1e4,
        )
        scheduler_step_per_batch = True
    else:
        # Note: T_mult=2 matches original train_new_model implementation.
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
        
    return criterion, optimizer, scheduler, scheduler_step_per_batch, scaler, amp_ctx


def _build_train_sampler(cfg: Config, train_subset) -> WeightedRandomSampler | None:
    sampler_mode = str(getattr(cfg, 'PT_TRAIN_SAMPLER', 'none') or 'none').strip().lower()
    if sampler_mode in {'none', 'off', 'false', 'shuffle'}:
        return None
    if sampler_mode not in {'pathology_balanced', 'pathology', 'balanced'}:
        raise ValueError(
            f"Unsupported PT_TRAIN_SAMPLER={sampler_mode!r}. Use 'none' or 'pathology_balanced'."
        )

    base_dataset = getattr(train_subset, 'dataset', None)
    subset_indices = list(getattr(train_subset, 'indices', []))
    if not isinstance(base_dataset, CytologyDataset) or not subset_indices:
        return None

    min_pixels = int(
        getattr(cfg, 'PT_SAMPLER_PATHOLOGY_MIN_PIXELS', getattr(cfg, 'MIN_PATHOLOGY_PIXELS', 10)) or 10
    )
    min_pixels = max(1, min_pixels)
    target_pos_fraction = float(getattr(cfg, 'PT_SAMPLER_TARGET_PATHOLOGY_FRACTION', 0.5) or 0.5)
    target_pos_fraction = min(max(target_pos_fraction, 0.05), 0.95)
    replacement = bool(getattr(cfg, 'PT_SAMPLER_REPLACEMENT', True))

    pos_flags = np.zeros(len(subset_indices), dtype=np.uint8)
    for i, src_idx in enumerate(subset_indices):
        pos_flags[i] = 1 if base_dataset.pathology_pixels(int(src_idx)) >= min_pixels else 0

    pos_count = int(pos_flags.sum())
    neg_count = int(len(pos_flags) - pos_count)
    if pos_count == 0 or neg_count == 0:
        print(
            '[sampler] pathology_balanced disabled: '
            f'pos_count={pos_count}, neg_count={neg_count}, min_pixels={min_pixels}'
        )
        return None

    pos_weight = target_pos_fraction / max(1, pos_count)
    neg_weight = (1.0 - target_pos_fraction) / max(1, neg_count)
    weights = np.where(pos_flags == 1, pos_weight, neg_weight).astype(np.float64)
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(weights),
        num_samples=len(weights),
        replacement=replacement,
    )
    print(
        '[sampler] pathology_balanced enabled: '
        f'pos={pos_count}, neg={neg_count}, target_pos_fraction={target_pos_fraction:.2f}, '
        f'min_pixels={min_pixels}, replacement={replacement}'
    )
    return sampler


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
        # On macOS/Windows, multiprocessing workers frequently add instability or
        # overhead for cv2/albumentations-heavy pipelines (WinError 1455 shared map,
        # spawn overhead on macOS). Default to 0 unless explicitly configured.
        if sys.platform in {'darwin', 'win32'}:
            num_workers = 0
        else:
            num_workers = min(4, os.cpu_count() or 4)
    pin_mem = torch.cuda.is_available()
    
    train_sampler = _build_train_sampler(cfg, train_dataset)
    train_shuffle = train_sampler is None

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=train_shuffle,
        sampler=train_sampler,
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


def _normalize_model_arch(raw_arch: str) -> str:
    arch = str(raw_arch or '').strip().lower()
    if arch in {'unet++', 'unet_plus_plus', 'unet-plus-plus', 'unetpp'}:
        return 'unetplusplus'
    if arch in {'deeplabv3+', 'deeplabv3_plus', 'deeplab-v3-plus'}:
        return 'deeplabv3plus'
    if arch == 'segformer':
        return 'segformer'
    return arch


def _resolve_base_lr(cfg: Config, *, explicit_lr: float | None) -> float:
    if explicit_lr is not None:
        return float(explicit_lr)

    fallback_lr = float(getattr(cfg, 'PT_LR', 1e-3))
    use_arch_defaults = bool(getattr(cfg, 'PT_USE_ARCH_LR_DEFAULTS', True))
    if not use_arch_defaults:
        return fallback_lr

    arch = _normalize_model_arch(getattr(cfg, 'PT_MODEL_ARCH', 'segformer'))
    arch_lr_map = {
        'unetplusplus': float(getattr(cfg, 'PT_LR_UNETPLUSPLUS', fallback_lr)),
        'deeplabv3plus': float(getattr(cfg, 'PT_LR_DEEPLABV3PLUS', fallback_lr)),
        'segformer': float(getattr(cfg, 'PT_LR_SEGFORMER', fallback_lr)),
    }
    return arch_lr_map.get(arch, fallback_lr)


def _build_segmentation_model(cfg: Config, classes: int):
    arch = _normalize_model_arch(getattr(cfg, 'PT_MODEL_ARCH', 'segformer'))
    encoder_name = str(cfg.PT_ENCODER_NAME)
    encoder_weights = cfg.PT_ENCODER_WEIGHTS
    kwargs = dict(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        classes=int(classes),
        activation=None,
    )

    if arch == 'unetplusplus':
        ctor = smp.UnetPlusPlus
    elif arch == 'deeplabv3plus':
        ctor = smp.DeepLabV3Plus
    elif arch == 'segformer':
        ctor = smp.Segformer
    else:
        raise ValueError(
            "PT_MODEL_ARCH must be one of {'unetplusplus', 'deeplabv3plus', 'segformer'}, "
            f"got {getattr(cfg, 'PT_MODEL_ARCH', None)!r}"
        )
    try:
        return ctor(**kwargs)
    except Exception as e:
        raise ValueError(
            f"Failed to build model for PT_MODEL_ARCH={arch!r} with PT_ENCODER_NAME={encoder_name!r}. "
            "Recommended encoders: "
            "unetplusplus -> resnet34/resnet50; "
            "deeplabv3plus -> resnet50/mit_b2; "
            "segformer -> mit_b2."
        ) from e


def train_new_model_pytorch(
    cfg: Config,
    model_path: str,
    output_classes: int,
    epochs: int,
    batch_size: int = 64,
    lr: float | None = None,
    use_amp: bool = True,
):
    """Train a new segmentation model using PyTorch."""
    set_seed(42)
    _log_device_selection(prefix="[train_new_model_pytorch]")
    image_shape = _resolve_hw_tuple(getattr(cfg, 'IMAGE_SHAPE', (256, 256)))
    image_chunk = _resolve_hw_tuple(getattr(cfg, 'IMAGE_CHUNK', image_shape), fallback=image_shape)
    print(f"[train_new_model_pytorch] image_shape={image_shape}, image_chunk={image_chunk}")
    print(
        f"[train_new_model_pytorch] aug_skip_prob="
        f"{float(getattr(cfg, 'PT_AUG_SKIP_PROB', 0.2) or 0.0):.2f}"
    )
    
    # Warm up loaders once (epoch 0 split): validates dataset and gives step count.
    train_loader_0, _, total_samples = _prepare_data_loaders(cfg, batch_size, epoch=0)
    steps_per_epoch = len(train_loader_0)
    print(f"Total samples: {total_samples}")
    
    model = _build_segmentation_model(cfg, output_classes)
    model = model.to(DEVICE)
    
    lr = _resolve_base_lr(cfg, explicit_lr=lr)
    print(f"[train_new_model_pytorch] base_lr={lr:.3e}")
    print(f"[train_new_model_pytorch] loss={str(getattr(cfg, 'PT_LOSS', 'focal_tversky_ce')).lower()}")
    print(
        f"[train_new_model_pytorch] boundary_loss_weight="
        f"{float(getattr(cfg, 'PT_BOUNDARY_LOSS_WEIGHT', 0.0) or 0.0):.3f}"
    )
    scheduler_mode_new = str(
        getattr(
            cfg,
            'PT_SCHEDULER_NEW',
            getattr(cfg, 'PT_SCHEDULER', 'onecycle'),
        ) or 'onecycle'
    ).strip().lower()
    print(f"[train_new_model_pytorch] scheduler={scheduler_mode_new}")
    criterion, optimizer, scheduler, scheduler_step_per_batch, scaler, amp_ctx = _setup_training_components(
        cfg,
        model,
        lr,
        use_amp,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        scheduler_mode=scheduler_mode_new,
    )
    
    _train_model_loop(
        cfg, model, criterion, optimizer, scheduler, scheduler_step_per_batch, scaler, amp_ctx,
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
    image_shape = _resolve_hw_tuple(getattr(cfg, 'IMAGE_SHAPE', (256, 256)))
    image_chunk = _resolve_hw_tuple(getattr(cfg, 'IMAGE_CHUNK', image_shape), fallback=image_shape)
    print(f"[train_current_model_pytorch] image_shape={image_shape}, image_chunk={image_chunk}")
    print(
        f"[train_current_model_pytorch] aug_skip_prob="
        f"{float(getattr(cfg, 'PT_AUG_SKIP_PROB', 0.2) or 0.0):.2f}"
    )
    
    # Warm up loaders (epoch 0 split) to validate dataset and capture step count.
    train_loader_0, _, _ = _prepare_data_loaders(cfg, batch_size, epoch=0)
    steps_per_epoch = len(train_loader_0)
    
    # Load existing architecture selected in config.
    model = _build_segmentation_model(cfg, cfg.CLASSES)
    try:
        state = torch.load(model_path, map_location=DEVICE, weights_only=True)
    except TypeError:  # older torch
        state = torch.load(model_path, map_location=DEVICE)
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        raise RuntimeError(
            f'Failed to load checkpoint {model_path!r} into {getattr(cfg, "PT_MODEL_ARCH", "unknown")!r}. '
            'Most likely architecture/encoder in config does not match checkpoint. '
            'Use matching PT_MODEL_ARCH/PT_ENCODER_NAME or start training from scratch.'
        ) from e
    model = model.to(DEVICE)
    
    lr = _resolve_base_lr(cfg, explicit_lr=lr)
    print(f"[train_current_model_pytorch] base_lr={lr:.3e}")
    print(f"[train_current_model_pytorch] loss={str(getattr(cfg, 'PT_LOSS', 'focal_tversky_ce')).lower()}")
    print(
        f"[train_current_model_pytorch] boundary_loss_weight="
        f"{float(getattr(cfg, 'PT_BOUNDARY_LOSS_WEIGHT', 0.0) or 0.0):.3f}"
    )
    scheduler_mode_finetune = str(
        getattr(
            cfg,
            'PT_SCHEDULER_FINETUNE',
            getattr(cfg, 'PT_SCHEDULER', 'cawr'),
        ) or 'cawr'
    ).strip().lower()
    print(f"[train_current_model_pytorch] scheduler={scheduler_mode_finetune}")
    criterion, optimizer, scheduler, scheduler_step_per_batch, scaler, amp_ctx = _setup_training_components(
        cfg,
        model,
        lr,
        use_amp,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        scheduler_mode=scheduler_mode_finetune,
    )
    
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
        cfg, model, criterion, optimizer, scheduler, scheduler_step_per_batch, scaler, amp_ctx,
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
