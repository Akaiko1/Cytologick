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

from config import Config
import clogic.graphics as graphics

_OPENSLIDE = None


def _import_openslide(cfg: Config):
    global _OPENSLIDE
    if _OPENSLIDE is not None:
        return _OPENSLIDE

    # OpenSlide import with DLL handling (Windows only)
    if hasattr(os, 'add_dll_directory'):
        with os.add_dll_directory(cfg.OPENSLIDE_PATH):
            import openslide as _oslide
    else:
        import openslide as _oslide

    _OPENSLIDE = _oslide
    return _OPENSLIDE

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

def get_slide(cfg: Config, slide_path: str):
    """
    Open a slide file.
    
    Args:
        slide_path: Path to the slide file (MRXS, SVS, etc.)
        
    Returns:
        OpenSlide object for the slide
    """
    oslide = _import_openslide(cfg)
    return oslide.OpenSlide(slide_path)


# =============================================================================
# Main Entry Point
# =============================================================================

def get_slide_rois(cfg: Config, slide_path: str) -> Dict[str, Any]:
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
    slide = get_slide(cfg, slide_path)
    
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
    
    scan_all_tissue = bool(cfg.WEB_SCAN_ALL_TISSUE)
    progress_every = int(cfg.WEB_PROGRESS_EVERY)
    if scan_all_tissue:
        print("Scanning all tissue regions for probes")
    else:
        print("Sampling tissue regions for probes")

    # Sample and extract probe images from tissue regions
    probe_images, probe_coords = _extract_probes(
        cfg,
        slide,
        tissue_regions_scaled,
        downsample_factor,
        scan_all=scan_all_tissue,
        progress_every=progress_every,
    )
    
    print(f"Extracted {len(probe_images)} probe images")
    
    # Run AI inference on probes
    pathology_maps = _run_inference(cfg, probe_images)
    
    # Convert inference results to contour overlays
    results = _extract_contours(cfg, pathology_maps, probe_coords)
    
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
    cfg: Config,
    slide: Any,
    tissue_regions: List[List[int]],
    downsample_factor: float,
    num_samples: int = 1,
    chunk_size: int = 1024,
    scan_all: bool = False,
    progress_every: int = 25,
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
    
    # Sample random tissue regions or scan all
    if scan_all:
        sample_regions = list(tissue_regions)
    else:
        sample_regions = random.sample(
            tissue_regions,
            min(num_samples, len(tissue_regions)),
        )

    total_regions = len(sample_regions)
    total_probes = 0
    skipped_probes = 0
    min_tissue_fraction = float(cfg.WEB_MIN_TISSUE_FRACTION)
    tissue_gray_threshold = int(cfg.WEB_TISSUE_GRAY_THRESHOLD)
    
    for idx, region in enumerate(sample_regions, start=1):
        if progress_every > 0 and (idx == 1 or idx % progress_every == 0 or idx == total_regions):
            print(f"Scanning tissue region {idx}/{total_regions} at ({region[0]}, {region[1]})")
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

                if min_tissue_fraction > 0:
                    tissue_fraction = _estimate_tissue_fraction(
                        probe_rgb, tissue_gray_threshold
                    )
                    if tissue_fraction < min_tissue_fraction:
                        skipped_probes += 1
                        continue

                probe_images.append(probe_rgb)
                probe_coords.append((x, y))
                total_probes += 1
                if progress_every > 0 and total_probes % progress_every == 0:
                    print(f"Extracted {total_probes} probes so far")

    if min_tissue_fraction > 0:
        print(f"Skipped {skipped_probes} low-tissue probes")

    return probe_images, probe_coords


# =============================================================================
# Inference
# =============================================================================

def _run_inference(cfg: Config, probe_images: List[np.ndarray]) -> List[np.ndarray]:
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
    if _PYTORCH_AVAILABLE and str(cfg.FRAMEWORK).lower() == 'pytorch':
        result = _run_pytorch_inference(cfg, probe_images)
        if result is not None:
            return result
    
    # Fall back to remote inference
    return _run_remote_inference(cfg, probe_images)


def _run_pytorch_inference(cfg: Config, probe_images: List[np.ndarray]) -> Optional[List[np.ndarray]]:
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
        model = inference_pt.load_pytorch_model(cfg, model_path, num_classes=cfg.CLASSES)
        
        pathology_maps = []
        use_fast_mode = bool(cfg.WEB_FAST_TILES)
        
        progress_every = int(cfg.WEB_PROGRESS_EVERY)
        total_probes = len(probe_images)
        batch_size = int(cfg.WEB_PT_BATCH_SIZE)
        if batch_size <= 0:
            batch_size = None
        for idx, probe in enumerate(probe_images, start=1):
            pm = inference_pt.apply_model_raw_pytorch(
                cfg,
                probe,
                model,
                classes=cfg.CLASSES,
                shapes=cfg.IMAGE_SHAPE,
                batch_size=batch_size,
            )
            pathology_maps.append(pm)
            if progress_every > 0 and (idx == 1 or idx % progress_every == 0 or idx == total_probes):
                print(f"Local inference progress: {idx}/{total_probes}")
        
        return pathology_maps
        
    except Exception as e:
        print(f"Local PyTorch inference failed: {e}")
        return None


def _run_remote_inference(cfg: Config, probe_images: List[np.ndarray]) -> List[np.ndarray]:
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
            model_input_size=tuple(cfg.IMAGE_SHAPE)
        )

        endpoint = str(cfg.ENDPOINT_URL)
        
        total_probes = len(probe_images)
        if total_probes == 0:
            return []
        print(f"Running remote inference on {total_probes} probes...")

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


def _estimate_tissue_fraction(
    image_rgb: np.ndarray,
    gray_threshold: int,
    sample_size: int = 128
) -> float:
    """
    Estimate tissue fraction in an RGB image by counting dark pixels.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    if max(gray.shape[:2]) > sample_size:
        gray = cv2.resize(gray, (sample_size, sample_size), interpolation=cv2.INTER_AREA)
    tissue = gray < gray_threshold
    return float(np.count_nonzero(tissue)) / float(tissue.size)


# =============================================================================
# Contour Extraction
# =============================================================================

def _extract_contours(
    cfg: Config,
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
    roi_index = 0
    
    conf_threshold = float(cfg.WEB_CONF_THRESHOLD)
    class_index = int(cfg.WEB_ATYPICAL_CLASS_INDEX)
    
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
        for cnt_idx, cnt in enumerate(filtered_contours):
            shifted = [point + shift for point in cnt]
            rect = cv2.boundingRect(np.array(shifted))
            roi_index += 1
            key = f'cnt_{idx}_{cnt_idx}_{roi_index}'
            results[key] = [
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
