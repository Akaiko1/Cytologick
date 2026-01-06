"""
PyTorch implementation of inference pipeline for Cytologick.

This module provides the primary inference implementation for local model execution.
All functions return probability maps (H, W, C) with softmax applied.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import segmentation_models_pytorch as smp

import config


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# Shared Utilities
# =============================================================================

def _get_default_batch_size() -> int:
    """Get optimal batch size based on available hardware."""
    return 32 if torch.cuda.is_available() else 16


def _pad_image(source: np.ndarray, shapes: tuple) -> tuple[np.ndarray, int, int]:
    """
    Pad image to be evenly divisible by tile shapes.

    Args:
        source: Input image (H, W, C)
        shapes: Tile size (height, width)

    Returns:
        Tuple of (padded_image, original_height, original_width)
    """
    orig_h, orig_w = source.shape[:2]
    pad_h = (shapes[0] - (orig_h % shapes[0])) % shapes[0]
    pad_w = (shapes[1] - (orig_w % shapes[1])) % shapes[1]
    padded = cv2.copyMakeBorder(source, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
    return padded, orig_h, orig_w


def _extract_tiles(source: np.ndarray, shapes: tuple) -> tuple[list, list]:
    """
    Extract non-overlapping tiles from image.

    Args:
        source: Padded image (H, W, C)
        shapes: Tile size (height, width)

    Returns:
        Tuple of (list of tile arrays, list of (x, y) coordinates)
    """
    tiles = []
    coords = []
    for x in range(0, source.shape[0], shapes[0]):
        for y in range(0, source.shape[1], shapes[1]):
            tile = source[x:x + shapes[0], y:y + shapes[1]]
            tiles.append(tile)
            coords.append((x, y))
    return tiles, coords


def image_to_tensor_pytorch(image: np.ndarray, add_dim: bool = True) -> torch.Tensor:
    """
    Convert image to PyTorch tensor with proper preprocessing.

    Args:
        image: Input image as numpy array (H, W, C), values 0-255
        add_dim: Whether to add batch dimension

    Returns:
        Preprocessed tensor (B, C, H, W) or (C, H, W) if add_dim=False
    """
    # Normalize to [0, 1] and convert to tensor
    tensor = torch.from_numpy(image / 255.0).float()

    # Permute from (H, W, C) to (C, H, W) and add batch dim for resize
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)

    # Resize to model input shape
    tensor = F.interpolate(
        tensor,
        size=config.IMAGE_SHAPE,
        mode='bilinear',
        align_corners=False
    )

    if not add_dim:
        tensor = tensor.squeeze(0)

    return tensor


def _run_batched_inference(
    model: torch.nn.Module,
    tiles: list[np.ndarray],
    batch_size: int | None = None,
    return_probs: bool = True
) -> np.ndarray:
    """
    Run batched inference on a list of tiles.

    Args:
        model: PyTorch model (already on device and in eval mode)
        tiles: List of image tiles (H, W, C)
        batch_size: Batch size for inference (None for auto)
        return_probs: If True, return softmax probabilities; if False, return argmax

    Returns:
        Array of predictions, shape depends on return_probs:
        - return_probs=True: (N, H, W, C) probabilities
        - return_probs=False: (N, H, W) class indices
    """
    if batch_size is None:
        batch_size = _get_default_batch_size()

    outputs = []

    with torch.no_grad():
        for i in range(0, len(tiles), batch_size):
            batch_tiles = tiles[i:i + batch_size]
            # Build tensor batch [B, C, H, W]
            batch_t = torch.cat(
                [image_to_tensor_pytorch(t) for t in batch_tiles], dim=0
            ).to(DEVICE)

            logits = model(batch_t)

            if return_probs:
                probs = F.softmax(logits, dim=1).cpu().numpy()  # [B, C, H, W]
                probs = np.transpose(probs, (0, 2, 3, 1))       # [B, H, W, C]
                outputs.append(probs)
            else:
                classes = torch.argmax(logits, dim=1).cpu().numpy()  # [B, H, W]
                outputs.append(classes)

    return np.concatenate(outputs, axis=0)


def create_mask_pytorch(pred_mask: torch.Tensor) -> np.ndarray:
    """
    Create a mask from PyTorch model predictions by taking the argmax.

    Args:
        pred_mask: Model prediction tensor (B, C, H, W)

    Returns:
        Predicted mask as numpy array (H, W)
    """
    pred_mask = torch.argmax(pred_mask, dim=1)
    return pred_mask[0].cpu().numpy()


# =============================================================================
# Main Inference Functions
# =============================================================================

def apply_model_pytorch(
    source: np.ndarray,
    model: torch.nn.Module,
    shapes: tuple = config.IMAGE_SHAPE
) -> np.ndarray:
    """
    Apply PyTorch model to image using sliding window approach.

    Returns discrete class indices (sparse map). For probability maps,
    use apply_model_raw_pytorch instead.

    Args:
        source: Input image as numpy array (H, W, C)
        model: PyTorch model
        shapes: Window size tuple

    Returns:
        Pathology map as numpy array (H, W) with class indices
    """
    model.eval()
    model = model.to(DEVICE)

    # Pad and extract tiles
    source_padded, orig_h, orig_w = _pad_image(source, shapes)
    tiles, coords = _extract_tiles(source_padded, shapes)

    # Run batched inference (return class indices, not probs)
    predictions = _run_batched_inference(model, tiles, return_probs=False)

    # Reconstruct map
    pathology_map = np.zeros(source_padded.shape[:2], dtype=np.uint8)
    for idx, (x, y) in enumerate(coords):
        pred = predictions[idx]
        # Resize prediction back to tile size using nearest neighbor
        pred_resized = cv2.resize(
            pred.astype(np.uint8), shapes, interpolation=cv2.INTER_NEAREST
        )
        pathology_map[x:x + shapes[0], y:y + shapes[1]] = pred_resized

    return pathology_map[:orig_h, :orig_w]


def apply_model_raw_pytorch(
    source: np.ndarray,
    model: torch.nn.Module,
    classes: int,
    shapes: tuple = config.IMAGE_SHAPE,
    batch_size: int | None = None
) -> np.ndarray:
    """
    Apply PyTorch model and return probability maps.

    This is the primary inference function for most use cases.
    Returns softmax probabilities suitable for threshold-based detection.

    Args:
        source: Input image as numpy array (H, W, C)
        model: PyTorch model
        classes: Number of output classes
        shapes: Window size tuple
        batch_size: Batch size for inference (None for auto)

    Returns:
        Probability map as numpy array (H, W, C) with values in [0, 1]
    """
    model.eval()
    model = model.to(DEVICE)

    # Pad and extract tiles
    source_padded, orig_h, orig_w = _pad_image(source, shapes)
    tiles, coords = _extract_tiles(source_padded, shapes)

    # Run batched inference (return probabilities)
    predictions = _run_batched_inference(model, tiles, batch_size, return_probs=True)

    # Reconstruct probability map
    pathology_map = np.zeros(
        (source_padded.shape[0], source_padded.shape[1], classes),
        dtype=np.float32
    )
    for idx, (x, y) in enumerate(coords):
        # Predictions are already at model output size (IMAGE_SHAPE)
        pathology_map[x:x + shapes[0], y:y + shapes[1]] = predictions[idx]

    return pathology_map[:orig_h, :orig_w]


def load_pytorch_model(model_path: str, num_classes: int = 3) -> torch.nn.Module:
    """
    Load a trained PyTorch model from file.

    Args:
        model_path: Path to the saved model state dict
        num_classes: Number of output classes

    Returns:
        Loaded PyTorch model in eval mode
    """
    model = smp.Unet(
        encoder_name='efficientnet-b3',
        encoder_weights=None,
        classes=num_classes,
        activation=None  # return logits; softmax applied in inference
    )

    state = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    return model


def apply_model_smooth_pytorch(
    source: np.ndarray,
    model: torch.nn.Module,
    shape: int = config.IMAGE_SHAPE[0]
) -> np.ndarray:
    """
    Apply PyTorch model with smooth windowing to reduce edge artifacts.

    Uses overlapping windows with spline blending for seamless predictions.
    Optionally applies Test Time Augmentation (8x rotation/flip averaging).

    Args:
        source: Input image as numpy array (H, W, C)
        model: PyTorch model
        shape: Window size (single int, assumes square windows)

    Returns:
        Probability map as numpy array (H, W, C) with values in [0, 1]
    """
    import clogic.smooth as smooth

    model.eval()
    model = model.to(DEVICE)

    def _pred_func(img_batch: np.ndarray) -> np.ndarray:
        """Prediction wrapper for smooth.py - handles batch of tiles."""
        # Use shared batched inference (returns probs in NHWC format)
        return _run_batched_inference(model, list(img_batch), return_probs=True)

    predictions_smooth = smooth.predict_img_with_smooth_windowing(
        source,
        window_size=shape,
        subdivisions=2,  # 50% overlap
        nb_classes=config.CLASSES,
        pred_func=_pred_func,
        use_tta=config.USE_TTA
    )

    return predictions_smooth.astype(np.float32)
