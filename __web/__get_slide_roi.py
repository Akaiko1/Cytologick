"""
ROI extraction from whole-slide images using AI inference.

This module provides functions for:
- Extracting regions of interest (ROIs) from slide images
- Running AI inference (local PyTorch or remote TensorFlow Serving)
- Processing inference results into contour overlays

The primary entry point is get_slide_rois() which analyzes a slide
and returns detected abnormal regions.
"""

import os
import random
import time
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

import config
import clogic.graphics as graphics

# OpenSlide import with DLL handling
if hasattr(os, 'add_dll_directory'):
    with os.add_dll_directory(config.OPENSLIDE_PATH):
        import openslide
else:
    import openslide

# Optional PyTorch inference support
_PYTORCH_AVAILABLE = False
try:
    import torch
    from clogic import inference_pytorch as inference_pt
    _PYTORCH_AVAILABLE = True
except ImportError:
    pass


# =============================================================================
# Slide Access
# =============================================================================

def get_slide(slide_path: str) -> openslide.OpenSlide:
    """
    Open a slide file.
    
    Args:
        slide_path: Path to the slide file (MRXS, SVS, etc.)
        
    Returns:
        OpenSlide object for the slide
    """
    return openslide.OpenSlide(slide_path)


# =============================================================================
# Main Entry Point
# =============================================================================

def get_slide_rois(slide_path: str) -> Dict[str, Any]:
    """
    Extract regions of interest from a slide using AI inference.
    
    This function:
    1. Opens the slide and generates a low-resolution preview
    2. Identifies tissue regions using adaptive thresholding
    3. Samples random tissue regions for analysis
    4. Runs AI inference to detect abnormalities
    5. Returns contour data for overlay rendering
    
    Args:
        slide_path: Path to the slide file
        
    Returns:
        Dictionary mapping ROI names to (shift, rect, contour) tuples
    """
    slide = get_slide(slide_path)
    
    print(f"Slide levels: {slide.level_count}")
    print(f"Level dimensions: {slide.level_dimensions}")
    print(f"Level downsamples: {slide.level_downsamples}")
    
    # Get a low-resolution preview for tissue detection
    preview_level = slide.level_count - 2
    preview_dims = slide.level_dimensions[preview_level]
    downsample_factor = slide.level_downsamples[preview_level]
    
    preview = np.array(slide.read_region((0, 0), preview_level, preview_dims))
    
    # Find tissue regions using adaptive thresholding
    tissue_regions = _find_tissue_regions(preview, preview_dims)
    
    # Scale coordinates back to level 0
    tissue_regions_scaled = [
        [int(coord * downsample_factor) for coord in region]
        for region in tissue_regions
    ]
    
    print(f"Found {len(tissue_regions_scaled)} tissue regions")
    
    # Sample and extract probe images from tissue regions
    probe_images, probe_coords = _extract_probes(
        slide, 
        tissue_regions_scaled, 
        downsample_factor
    )
    
    print(f"Extracted {len(probe_images)} probe images")
    
    # Run AI inference on probes
    pathology_maps = _run_inference(probe_images)
    
    # Convert inference results to contour overlays
    results = _extract_contours(pathology_maps, probe_coords)
    
    print(f"Detected {len(results)} regions of interest")
    return results


# =============================================================================
# Tissue Detection
# =============================================================================

def _find_tissue_regions(
    preview: np.ndarray, 
    preview_dims: Tuple[int, int],
    sample_size: Tuple[int, int] = (25, 25),
    min_tissue_pixels: int = 225
) -> List[List[int]]:
    """
    Find tissue regions in a slide preview using adaptive thresholding.
    
    Args:
        preview: RGBA preview image as numpy array
        preview_dims: Dimensions of the preview (width, height)
        sample_size: Size of sampling grid cells
        min_tissue_pixels: Minimum non-background pixels to consider tissue
        
    Returns:
        List of [x, y] coordinates for tissue regions
    """
    # Convert to grayscale and threshold
    preview_gray = cv2.cvtColor(preview, cv2.COLOR_RGBA2GRAY)
    preview_binary = cv2.adaptiveThreshold(
        preview_gray, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        11, 2
    )
    
    # Sample grid to find tissue regions
    tissue_regions = []
    
    for x in range(0, preview_dims[0], sample_size[0]):
        for y in range(0, preview_dims[1], sample_size[1]):
            sample = preview_binary[y:y + sample_size[1], x:x + sample_size[0]]
            pixel_sum = np.sum(sample / 255)
            
            if pixel_sum <= min_tissue_pixels:
                # Too few pixels, mark as background
                preview_binary[y:y + sample_size[1], x:x + sample_size[0]] = 0
            else:
                tissue_regions.append([x, y])
    
    return tissue_regions


# =============================================================================
# Probe Extraction
# =============================================================================

def _extract_probes(
    slide: openslide.OpenSlide,
    tissue_regions: List[List[int]],
    downsample_factor: float,
    num_samples: int = 1,
    chunk_size: int = 1024
) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """
    Extract probe images from tissue regions for inference.
    
    Args:
        slide: OpenSlide object
        tissue_regions: List of tissue region coordinates
        downsample_factor: Downsampling factor for coordinate conversion
        num_samples: Number of regions to sample
        chunk_size: Size of each probe image
        
    Returns:
        Tuple of (list of probe images, list of coordinates)
    """
    probe_images = []
    probe_coords = []
    
    # Sample random tissue regions
    sample_regions = random.sample(
        tissue_regions, 
        min(num_samples, len(tissue_regions))
    )
    
    for region in sample_regions:
        # Calculate probe dimensions
        probe_w = int(25 * downsample_factor)
        probe_h = int(25 * downsample_factor)
        probe_w, probe_h = graphics.get_corrected_size(probe_w, probe_h, chunk_size)
        
        print(f"Probe size: {probe_w}x{probe_h}")
        
        # Extract probe chunks from the region
        for x in range(region[0], region[0] + probe_w, chunk_size):
            for y in range(region[1], region[1] + probe_h, chunk_size):
                probe = slide.read_region((x, y), 0, (chunk_size, chunk_size))
                probe_rgb = cv2.cvtColor(np.array(probe), cv2.COLOR_RGBA2RGB)
                
                probe_images.append(probe_rgb)
                probe_coords.append((x, y))
    
    return probe_images, probe_coords


# =============================================================================
# Inference
# =============================================================================

def _run_inference(probe_images: List[np.ndarray]) -> List[np.ndarray]:
    """
    Run AI inference on probe images.
    
    Attempts local PyTorch inference first, falls back to remote
    TensorFlow Serving endpoint if local model unavailable.
    
    Args:
        probe_images: List of probe images as numpy arrays
        
    Returns:
        List of pathology probability maps
    """
    # Try local PyTorch model first
    if _PYTORCH_AVAILABLE and config.FRAMEWORK.lower() == 'pytorch':
        result = _run_pytorch_inference(probe_images)
        if result is not None:
            return result
    
    # Fall back to remote inference
    return _run_remote_inference(probe_images)


def _run_pytorch_inference(probe_images: List[np.ndarray]) -> Optional[List[np.ndarray]]:
    """
    Run inference using local PyTorch model.
    
    Args:
        probe_images: List of probe images
        
    Returns:
        List of pathology maps, or None if model unavailable
    """
    model_path = _find_local_model()
    if not model_path:
        print("No local PyTorch model found")
        return None
    
    try:
        print(f"Using local PyTorch model: {model_path}")
        model = inference_pt.load_pytorch_model(model_path, num_classes=config.CLASSES)
        
        pathology_maps = []
        use_fast_mode = getattr(config, 'WEB_FAST_TILES', True)
        
        for probe in probe_images:
            if use_fast_mode:
                # Downscale for faster inference
                h, w = probe.shape[:2]
                probe_small = cv2.resize(probe, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
                pm_small = inference_pt.apply_model_raw_pytorch(
                    probe_small, model, 
                    classes=config.CLASSES, 
                    shapes=config.IMAGE_SHAPE
                )
                pm = cv2.resize(pm_small, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                pm = inference_pt.apply_model_raw_pytorch(
                    probe, model,
                    classes=config.CLASSES,
                    shapes=config.IMAGE_SHAPE
                )
            pathology_maps.append(pm)
        
        return pathology_maps
        
    except Exception as e:
        print(f"Local PyTorch inference failed: {e}")
        return None


def _run_remote_inference(probe_images: List[np.ndarray]) -> List[np.ndarray]:
    """
    Run inference using remote TensorFlow Serving endpoint.
    
    Args:
        probe_images: List of probe images
        
    Returns:
        List of pathology maps (empty list on failure)
    """
    try:
        import tfs_connector as tfs
        
        start_time = time.time()
        
        resize_opts = tfs.ResizeOptions(
            chunk_size=(256, 256),
            model_input_size=tuple(config.IMAGE_SHAPE)
        )
        
        endpoint = getattr(config, 'ENDPOINT_URL', 'http://51.250.28.160:7500')
        
        pathology_maps = tfs.apply_segmentation_model_parallel(
            probe_images,
            endpoint_url=endpoint,
            model_name='demetra',
            batch_size=4,
            resize_options=resize_opts,
            normalization=lambda x: x / 255,
            parallelism_mode=1,
            thread_count=8
        )
        
        print(f"Remote inference took {time.time() - start_time:.2f}s")
        return pathology_maps
        
    except Exception as e:
        print(f"Remote inference failed: {e}")
        return []


# =============================================================================
# Contour Extraction
# =============================================================================

def _extract_contours(
    pathology_maps: List[np.ndarray],
    probe_coords: List[Tuple[int, int]]
) -> Dict[str, Any]:
    """
    Extract contours from pathology probability maps.
    
    Args:
        pathology_maps: List of probability maps (H, W, C)
        probe_coords: Corresponding coordinates for each map
        
    Returns:
        Dictionary mapping ROI names to overlay data
    """
    results = {}
    
    conf_threshold = float(getattr(config, 'WEB_CONF_THRESHOLD', 0.5))
    class_index = int(getattr(config, 'WEB_ATYPICAL_CLASS_INDEX', 2))
    
    for idx, pmap in enumerate(pathology_maps):
        # Extract atypical class probability
        atypical_probs = pmap[..., class_index]
        max_prob = np.max(atypical_probs)
        print(f"Probe {idx}: max atypical probability = {max_prob:.3f}")
        
        # Find contours above threshold
        contours = _find_high_confidence_contours(
            atypical_probs, 
            conf_threshold
        )
        
        if not contours:
            continue
        
        # Filter by average probability
        filtered_contours = [
            cnt for cnt in contours
            if _get_contour_probability(atypical_probs, cnt) >= conf_threshold
        ]
        
        if not filtered_contours:
            continue
        
        # Shift contours to slide coordinates and add to results
        shift = probe_coords[idx]
        for cnt in filtered_contours:
            shifted = [point + shift for point in cnt]
            rect = cv2.boundingRect(np.array(shifted))
            results[f'cnt_{idx}'] = [
                shift,
                [rect],
                [point.tolist() for point in shifted]
            ]
    
    return results


def _find_high_confidence_contours(
    probability_map: np.ndarray,
    threshold: float
) -> List[np.ndarray]:
    """
    Find contours in a probability map above the threshold.
    
    Args:
        probability_map: 2D probability map
        threshold: Confidence threshold
        
    Returns:
        List of contours
    """
    # Primary threshold
    binary_map = (probability_map > threshold).astype(np.uint8) * 255
    contours = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    
    # Fallback: relax threshold if nothing found
    if not contours:
        relaxed_threshold = max(0.3, threshold * 0.75)
        binary_map = (probability_map > relaxed_threshold).astype(np.uint8) * 255
        contours = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    
    return contours


def _get_contour_probability(
    probability_map: np.ndarray,
    contour: np.ndarray
) -> float:
    """
    Calculate average probability inside a contour.
    
    Args:
        probability_map: 2D probability map
        contour: Contour points
        
    Returns:
        Average probability (0-1)
    """
    mask = np.zeros(probability_map.shape)
    cv2.drawContours(mask, [contour], -1, 1, -1)
    
    total_pixels = max(1.0, float(np.sum(mask)))
    total_probability = float(np.sum(np.where(mask > 0, probability_map, 0)))
    
    return total_probability / total_pixels


# =============================================================================
# Model Discovery
# =============================================================================

def _find_local_model() -> Optional[str]:
    """
    Locate a local PyTorch model file.
    
    Search order (matching desktop GUI):
    1. PYTORCH_MODEL_PATH environment variable
    2. _main/new_best.pth (primary training output)
    3. _main/new_final.pth
    4. _main/new_last.pth
    5. _main/model.pth (legacy)
    6. Any .pth file in _main/
    
    Returns:
        Path to model file, or None if not found
    """
    # Check environment variable first
    env_path = os.getenv('PYTORCH_MODEL_PATH')
    if env_path and os.path.exists(env_path):
        return env_path
    
    # Preferred model paths (in order of priority, matching desktop GUI)
    preferred_paths = [
        os.path.join('_main', 'new_best.pth'),      # Primary: best model from training
        os.path.join('_main', 'new_final.pth'),     # Secondary: final model from training
        os.path.join('_main', 'new_last.pth'),      # Tertiary: last checkpoint
        os.path.join('_main', 'model.pth'),         # Legacy paths
        os.path.join('_main', 'model_best.pth'),
        os.path.join('_main', 'model_final.pth'),
    ]
    
    for path in preferred_paths:
        if os.path.exists(path):
            return path
    
    # Fallback: search for any .pth file in _main directory
    main_dir = '_main'
    if os.path.isdir(main_dir):
        for filename in os.listdir(main_dir):
            if filename.lower().endswith('.pth'):
                return os.path.join(main_dir, filename)
    
    return None
