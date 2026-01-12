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

from config import Config
from clogic.preprocessing_pytorch import preprocess_rgb_image, resize_rgb_image


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


def image_to_tensor_pytorch(cfg: Config, image: np.ndarray, add_dim: bool = True) -> torch.Tensor:
    """
    Convert image to PyTorch tensor with proper preprocessing.

    Args:
        image: Input image as numpy array (H, W, C), values 0-255
        add_dim: Whether to add batch dimension

    Returns:
        Preprocessed tensor (B, C, H, W) or (C, H, W) if add_dim=False
    """
    if image.shape[:2] != tuple(cfg.IMAGE_SHAPE):
        image = resize_rgb_image(image, cfg.IMAGE_SHAPE)

    image_f = preprocess_rgb_image(
        image,
        use_encoder_preprocessing=bool(cfg.PT_USE_ENCODER_PREPROCESSING),
        encoder_name=str(cfg.PT_ENCODER_NAME),
        encoder_weights=cfg.PT_ENCODER_WEIGHTS,
    )

    tensor = torch.from_numpy(image_f).float().permute(2, 0, 1)
    if add_dim:
        tensor = tensor.unsqueeze(0)
    return tensor


def _run_batched_inference(
    cfg: Config,
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
                [image_to_tensor_pytorch(cfg, t) for t in batch_tiles], dim=0
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
    cfg: Config,
    source: np.ndarray,
    model: torch.nn.Module,
    shapes: tuple | None = None
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

    shapes = tuple(cfg.IMAGE_SHAPE) if shapes is None else shapes

    # Pad and extract tiles
    source_padded, orig_h, orig_w = _pad_image(source, shapes)
    tiles, coords = _extract_tiles(source_padded, shapes)

    # Run batched inference (return class indices, not probs)
    predictions = _run_batched_inference(cfg, model, tiles, return_probs=False)

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
    cfg: Config,
    source: np.ndarray,
    model: torch.nn.Module,
    classes: int | None = None,
    shapes: tuple | None = None,
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

    shapes = tuple(cfg.IMAGE_SHAPE) if shapes is None else shapes
    classes = int(cfg.CLASSES) if classes is None else int(classes)

    # Pad and extract tiles
    source_padded, orig_h, orig_w = _pad_image(source, shapes)
    tiles, coords = _extract_tiles(source_padded, shapes)

    # Run batched inference (return probabilities)
    predictions = _run_batched_inference(cfg, model, tiles, batch_size, return_probs=True)

    # Reconstruct probability map
    pathology_map = np.zeros(
        (source_padded.shape[0], source_padded.shape[1], classes),
        dtype=np.float32
    )
    for idx, (x, y) in enumerate(coords):
        # Predictions are already at model output size (IMAGE_SHAPE)
        pathology_map[x:x + shapes[0], y:y + shapes[1]] = predictions[idx]

    return pathology_map[:orig_h, :orig_w]


def load_pytorch_model(cfg: Config, model_path: str, num_classes: int | None = None) -> torch.nn.Module:
    """
    Load a trained PyTorch model from file.

    Args:
        model_path: Path to the saved model state dict
        num_classes: Number of output classes

    Returns:
        Loaded PyTorch model in eval mode
    """
    n_classes = int(cfg.CLASSES) if num_classes is None else int(num_classes)

    model = smp.Unet(
        encoder_name=str(cfg.PT_ENCODER_NAME),
        encoder_weights=None,
        classes=n_classes,
        activation=None  # return logits; softmax applied in inference
    )

    state = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    return model


def apply_model_smooth_pytorch(
    cfg: Config,
    source: np.ndarray,
    model: torch.nn.Module,
    shape: int | None = None
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
        Probability map as numpy array (H, W, C) with values in [0, 1].
    """
    shape = int(cfg.IMAGE_SHAPE[0]) if shape is None else int(shape)
    import clogic.smooth as smooth

    model.eval()
    model = model.to(DEVICE)

    def _pred_func(img_batch: np.ndarray) -> np.ndarray:
        """Prediction wrapper for smooth.py - handles batch of tiles."""
        # Use shared batched inference (returns probs in NHWC format)
        return _run_batched_inference(cfg, model, list(img_batch), return_probs=True)

    predictions_smooth = smooth.predict_img_with_smooth_windowing(
        source,
        window_size=shape,
        subdivisions=2,  # 50% overlap
        nb_classes=int(cfg.CLASSES),
        pred_func=_pred_func,
        use_tta=bool(cfg.USE_TTA)
    )

    return predictions_smooth.astype(np.float32)


def apply_model_region_pytorch(
    cfg: Config,
    source: np.ndarray,
    model: torch.nn.Module,
    classes: int | None = None,
    shapes: tuple | None = None,
    batch_size: int | None = None,
    atypical_threshold: float = 0.5
) -> tuple[np.ndarray, list[tuple[int, int, int, int, float]]]:
    """
    Apply PyTorch model with region-wise refinement.

    Two-pass approach:
    1. First pass: fast tiled inference to find non-background regions
    2. Merge classes 1 and 2 (non-background) into a single mask
    3. For each connected region, compute bounding box, center a tile on it,
       and re-run inference for accurate segmentation

    This improves accuracy for cells split across tile boundaries.

    Args:
        source: Input image as numpy array (H, W, C)
        model: PyTorch model
        classes: Number of output classes
        shapes: Window size tuple (height, width)
        batch_size: Batch size for inference (None for auto)
        atypical_threshold: Threshold for considering region as atypical

    Returns:
        Tuple of:
        - Probability map as numpy array (H, W, C) with values in [0, 1]
        - List of atypical region bboxes: (x, y, width, height, avg_probability)
    """
    model.eval()
    model = model.to(DEVICE)

    shapes = tuple(cfg.IMAGE_SHAPE) if shapes is None else shapes
    classes = int(cfg.CLASSES) if classes is None else int(classes)

    # --- Pass 1: Fast tiled inference ---
    initial_map = apply_model_raw_pytorch(
        cfg, source, model, classes=classes, shapes=shapes, batch_size=batch_size
    )

    # Get predicted class per pixel (argmax)
    predicted_class = np.argmax(initial_map, axis=2)

    # Create binary mask: 1 where class is 1 or 2 (non-background)
    non_background_mask = ((predicted_class == 1) | (predicted_class == 2)).astype(np.uint8)

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        non_background_mask, connectivity=8
    )

    if num_labels <= 1:
        # No non-background regions found
        return initial_map, []

    # --- Pass 2: Region-wise refinement ---
    # Extract each region, resize to model input size, run inference
    region_tiles = []
    region_bboxes = []  # (bx, by, bw, bh, label_idx)

    for label_idx in range(1, num_labels):  # Skip background (0)
        # Get bounding box: [x, y, width, height, area]
        bx, by, bw, bh, area = stats[label_idx]

        # Skip only single-pixel noise
        if area < 3:
            continue

        # Extract region from source image (will be resized in image_to_tensor_pytorch)
        region = source[by:by + bh, bx:bx + bw]

        region_tiles.append(region)
        region_bboxes.append((bx, by, bw, bh, label_idx))

    if not region_tiles:
        return initial_map, []

    # Run inference on resized regions
    region_predictions = _run_batched_inference(
        cfg, model, region_tiles, batch_size, return_probs=True
    )

    # Create output map - keep green overlay from initial fast inference
    # Merge class 1 and 2 into class 1 (green) as base layer
    output_map = np.zeros_like(initial_map)
    output_map[:, :, 0] = initial_map[:, :, 0]  # Keep background
    # Merge class 1 + class 2 probabilities into class 1 (will show as green)
    output_map[:, :, 1] = initial_map[:, :, 1] + initial_map[:, :, 2]

    # Collect atypical bounding boxes after refinement
    atypical_bboxes = []

    # Check each region for atypical cells, only overwrite class 2 pixels
    for idx, (bx, by, bw, bh, label_idx) in enumerate(region_bboxes):
        pred = region_predictions[idx]  # (IMAGE_SHAPE[0], IMAGE_SHAPE[1], classes)

        # Resize prediction back to original region size
        pred_resized = cv2.resize(pred, (bw, bh), interpolation=cv2.INTER_LINEAR)

        # Check if any pixel has class 2 as dominant
        pred_class = np.argmax(pred_resized, axis=2)
        atypical_mask = pred_class == 2

        if np.any(atypical_mask):
            # Only overwrite pixels where class 2 dominates (red areas)
            output_map[by:by + bh, bx:bx + bw][atypical_mask] = pred_resized[atypical_mask]

            # Calculate max class 2 probability
            max_prob = float(np.max(pred_resized[:, :, 2]))
            atypical_bboxes.append((bx, by, bw, bh, max_prob))

    return output_map, atypical_bboxes
