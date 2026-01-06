"""
Inference utilities shared between Desktop GUI and Web GUI.

This module provides common functions for:
- Model inference selection (local vs remote)
- Remote endpoint health checking
- Inference result processing
"""

import os
import socket
from urllib.parse import urlparse
from typing import Optional, Tuple, Any
import numpy as np
import cv2

from config import Config


# -----------------------------------------------------------------------------
# Remote Endpoint Utilities
# -----------------------------------------------------------------------------

def check_remote_available(cfg: Config, timeout: Optional[float] = None) -> bool:
    """
    Check if the remote inference endpoint is reachable via TCP.
    
    Args:
        timeout: Connection timeout in seconds. Uses config.HEALTH_TIMEOUT if None.
        
    Returns:
        True if endpoint is reachable, False otherwise.
    """
    endpoint = getattr(cfg, 'ENDPOINT_URL', None)
    if not endpoint:
        return False
    
    try:
        parsed = urlparse(endpoint)
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == 'https' else 80)
        
        if not host:
            return False
            
        connection_timeout = timeout if timeout is not None else float(getattr(cfg, 'HEALTH_TIMEOUT', 1.5))
        with socket.create_connection((host, port), timeout=connection_timeout):
            return True
    except Exception:
        return False


def get_endpoint_url(cfg: Config) -> str:
    """
    Get the configured remote inference endpoint URL.
    
    Returns:
        Endpoint URL string.
    """
    return str(getattr(cfg, 'ENDPOINT_URL', 'http://127.0.0.1:8501'))


# -----------------------------------------------------------------------------
# Inference Dispatch
# -----------------------------------------------------------------------------

def run_inference(
    cfg: Config,
    image: np.ndarray,
    model: Any,
    mode: str = 'direct',
    classes: int = 3,
    shapes: Tuple[int, int] = (128, 128)
) -> np.ndarray:
    """
    Run inference on an image using the specified mode.
    
    Args:
        image: Input image as numpy array (H, W, C) in RGB format.
        model: Loaded model object (PyTorch or TensorFlow).
        mode: Inference mode - 'direct', 'smooth', or 'remote'.
        classes: Number of output classes.
        shapes: Model input shape (height, width).
        
    Returns:
        Pathology map as numpy array.
    """
    # Import framework-specific modules lazily to avoid import errors
    if str(cfg.FRAMEWORK).lower() != 'pytorch':
        raise RuntimeError('TensorFlow inference is deprecated; set FRAMEWORK=pytorch')

    import clogic.inference_pytorch as inference

    if mode == 'remote':
        import tfs_connector as tfs

        resize_opts = tfs.ResizeOptions(
            chunk_size=(256, 256),
            model_input_size=tuple(shapes),
        )
        endpoint = get_endpoint_url(cfg)
        maps = tfs.apply_segmentation_model_parallel(
            [image],
            endpoint_url=endpoint,
            model_name=str(getattr(cfg, 'MODEL_NAME', 'demetra')),
            batch_size=1,
            resize_options=resize_opts,
            normalization=lambda x: x / 255,
            parallelism_mode=1,
            thread_count=4,
        )
        return maps[0]
    elif mode == 'smooth':
        # Use TTA + smooth windowing
        return inference.apply_model_smooth_pytorch(cfg, image, model, shape=shapes[0])
    else:
        # Standard fast inference
        return inference.apply_model_raw_pytorch(cfg, image, model, classes=classes, shapes=shapes)


# -----------------------------------------------------------------------------
# Result Processing
# -----------------------------------------------------------------------------

def process_pathology_map(
    pathology_map: np.ndarray,
    threshold: float = 0.6,
    class_index: int = 2
) -> Tuple[np.ndarray, dict]:
    """
    Process a pathology probability map into visualization and stats.
    
    Args:
        pathology_map: Probability map (H, W, C) or class indices (H, W).
        threshold: Confidence threshold for detection (0-1).
        class_index: Which class index represents 'atypical' cells.
        
    Returns:
        Tuple of (visualization_image, stats_dict).
    """
    import clogic.graphics as drawing
    
    # If we received a probability map (H, W, C), use dense processing
    if isinstance(pathology_map, np.ndarray) and pathology_map.ndim == 3:
        return drawing.process_dense_pathology_map(pathology_map, threshold=threshold)
    else:
        return drawing.process_sparse_pathology_map(pathology_map)


def format_detection_stats(stats: dict) -> str:
    """
    Format detection statistics into a human-readable string.
    
    Args:
        stats: Statistics dictionary from process_pathology_map.
        
    Returns:
        Formatted string for display.
    """
    # Dense stats: dict of lesion_idx -> probability
    if stats and all(isinstance(k, int) for k in stats.keys()):
        vals = list(stats.values())
        n = len(vals)
        avg = int(round((sum(vals) / max(1, n)) * 100))
        return f'Detections: {n}\nAvg conf: {avg}%'
    else:
        # Sparse stats: dict of label -> count
        lines = [f'{key}: {val}' for key, val in stats.items()]
        return '\n'.join(lines)
