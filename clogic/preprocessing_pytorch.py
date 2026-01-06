"""PyTorch preprocessing utilities.

Goal: keep *identical* preprocessing between training (Albumentations)
and inference (NumPy/OpenCV + torch), while optionally supporting
encoder-specific preprocessing recommended by segmentation_models_pytorch
for pretrained encoders.

By default this module preserves the project's current behavior:
- RGB uint8 input
- resize to config.IMAGE_SHAPE
- scale to [0, 1] via /255

If `use_encoder_preprocessing=True`, it applies SMP's
`get_preprocessing_fn(encoder_name, pretrained=encoder_weights)`.
"""

from __future__ import annotations

from typing import Callable, Tuple

import cv2
import numpy as np

try:
    from segmentation_models_pytorch.encoders import get_preprocessing_fn
except Exception:  # pragma: no cover
    get_preprocessing_fn = None  # type: ignore[assignment]


def resize_rgb_image(image_rgb: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    """Resize RGB image to (H, W) using bilinear interpolation."""
    h, w = int(hw[0]), int(hw[1])
    return cv2.resize(image_rgb, (w, h), interpolation=cv2.INTER_LINEAR)


def get_encoder_preprocess_fn(
    encoder_name: str,
    encoder_weights: str | None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return SMP preprocessing callable for encoder/weights.

    The callable expects an HWC image and returns an HWC float image.
    """
    if get_preprocessing_fn is None:
        raise RuntimeError(
            "segmentation_models_pytorch is required for encoder preprocessing"
        )
    if not encoder_weights:
        raise ValueError("encoder_weights must be non-empty to use encoder preprocessing")

    return get_preprocessing_fn(encoder_name, pretrained=encoder_weights)


def preprocess_rgb_image(
    image_rgb: np.ndarray,
    *,
    use_encoder_preprocessing: bool,
    encoder_name: str,
    encoder_weights: str | None,
) -> np.ndarray:
    """Preprocess RGB uint8 image to float32 HWC.

    - If use_encoder_preprocessing=False: returns image / 255.0
    - If True: uses SMP's encoder preprocessing function.

    Input is assumed to be RGB.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB image, got shape={image_rgb.shape}")

    if use_encoder_preprocessing:
        preprocess_fn = get_encoder_preprocess_fn(encoder_name, encoder_weights)
        # SMP preprocessing works with float/uint8; enforce float32 for stability.
        image_f = image_rgb.astype(np.float32)
        out = preprocess_fn(image_f)
        return np.asarray(out, dtype=np.float32)

    return (image_rgb.astype(np.float32) / 255.0).astype(np.float32)
