from collections import namedtuple
import json
import glob
import math
import cv2
import os
import random

import numpy as np

from config import Config, load_config


Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')


def _resolve_cfg(cfg: Config | None) -> Config:
    return cfg if cfg is not None else load_config()


def extract_atlas(slidepath, jsonpath, openslide_path):
    if hasattr(os, 'add_dll_directory'):
        with os.add_dll_directory(openslide_path):
            import openslide
    else:
        import openslide

    slide = openslide.OpenSlide(slidepath)

    with open(jsonpath, 'r') as f:
        rois = json.load(f)
    
    if not os.path.exists('atlas'):
        os.mkdir('atlas')
    
    regions = __get_regions(rois)

    for idx, region in enumerate(regions):
        name, _, max_x, max_y, min_x, min_y = region
        name = name.strip('?)')

        folder_name = os.path.join('atlas', name)

        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
        
        crop = slide.read_region((min_x, min_y), 0, (max_x - min_x, max_y - min_y))
        crop = np.asarray(crop)

        # OpenSlide returns RGBA, convert to BGR for cv2.imwrite/imread compatibility
        # TODO randomize name if file exists
        cv2.imwrite(os.path.join(folder_name, f'{idx}_{name}_coords_{min_x}_{min_y}.bmp'), cv2.cvtColor(crop, cv2.COLOR_RGBA2BGR))


def extract_all_cells(
    slides_folder,
    json_folder,
    openslide_path,
    classes,
    cfg: Config | None = None,
    broaden_individual_rect: int | None = None,
    exclude_duplicates: bool | None = None,
):
    cfg = _resolve_cfg(cfg)
    if broaden_individual_rect is None:
        broaden_individual_rect = int(getattr(cfg, 'BROADEN_INDIVIDUAL_RECT', 0))
    if exclude_duplicates is None:
        exclude_duplicates = bool(getattr(cfg, 'EXCLUDE_DUPLICATES', False))

    if hasattr(os, 'add_dll_directory'):
        with os.add_dll_directory(openslide_path):
            import openslide
    else:
        import openslide

    slides_list = glob.glob(os.path.join(slides_folder, '**', '*.mrxs'), recursive=True)
    print('Total slides: ', len(slides_list))

    json_list = glob.glob(os.path.join(json_folder, '**', '*.json'), recursive=True)

    roi_path = os.path.join('dataset', 'rois')
    masks_path = os.path.join('dataset', 'masks')
    __make_dirs(roi_path, masks_path)

    for slidepath in slides_list:
        slide_id = os.path.splitext(os.path.basename(slidepath))[0]
        # Derive expected JSON name robustly from slide filename
        base_name = os.path.splitext(os.path.basename(slidepath))[0]
        json_name = f"{base_name}.json"
        json_path = [f for f in json_list if json_name in f]

        slide = openslide.OpenSlide(slidepath)

        if not json_path:
            print(f'JSON for {slidepath} not found, skipping')
            continue
        
        with open(json_path[0], 'r') as f:
            rois = json.load(f)
        
        regions = __get_regions(rois, exclude_duplicates=exclude_duplicates)

        print(f'Parsing {slide}: {len(regions)} regions total')

        for _, region in enumerate(regions):
            name, points, max_x, max_y, min_x, min_y = region
            name = name.strip('?)')

            if 'rect' in name.lower():
                continue

            if not 100 <= max_x - min_x <= 200 or not 100 <= max_y - min_y <= 200:  # This sets min and max size for resulting roi crop (+- 2)
                continue
            
            span = broaden_individual_rect
            w_max_x, w_max_y, w_min_x, w_min_y = max_x + span, max_y + span, min_x - span, min_y - span

            crop = slide.read_region((w_min_x, w_min_y), 0, (w_max_x - w_min_x, w_max_y - w_min_y))
            roi = np.asarray(crop)
            mask = np.zeros(roi.shape[:2], dtype=np.uint8)

            __draw_masks(
                classes,
                __get_rect_regions(
                    rois,
                    (w_min_x, w_min_y),
                    (w_max_x, w_max_y),
                    exclude_duplicates=exclude_duplicates,
                ),
                mask,
            )

            roi = roi[span + 2:span + (max_y - min_y) - 2, span + 2: span + (max_x - min_x) - 2]
            mask = mask[span + 2:span + (max_y - min_y) - 2, span + 2: span + (max_x - min_x) - 2]

            # OpenSlide returns RGBA, convert to BGR for cv2.imwrite/imread compatibility
            prefix = f'{slide_id}__{name}'
            cv2.imwrite(os.path.join(roi_path, f'{prefix}_coords_{max_x}_{max_y}.bmp'), cv2.cvtColor(roi, cv2.COLOR_RGBA2BGR))
            cv2.imwrite(os.path.join(masks_path, f'{prefix}_coords_{max_x}_{max_y}.bmp'), mask)


def extract_all_slides(
    slides_folder,
    json_folder,
    openslide_path,
    classes,
    zoom_levels=[256],
    cfg: Config | None = None,
    exclude_duplicates: bool | None = None,
    overlap: float | None = None,
    centered_crop: bool | None = None,
):
    """
    Extract training tiles from all slides.

    Args:
        centered_crop: If True, use pathology-centered cropping that avoids
            truncating pathology objects at tile edges. Generates both group tiles
            (multiple pathologies) and individual tiles (one pathology each).
            If None, reads from cfg.CENTERED_CROP (default False).
    """
    cfg = _resolve_cfg(cfg)
    if exclude_duplicates is None:
        exclude_duplicates = bool(getattr(cfg, 'EXCLUDE_DUPLICATES', False))
    if overlap is None:
        overlap = float(getattr(cfg, 'TILE_OVERLAP', 0.0))
    if centered_crop is None:
        centered_crop = bool(getattr(cfg, 'CENTERED_CROP', False))
    centered_algo = str(getattr(cfg, 'CENTERED_CROP_ALGO', 'heuristic'))
    edge_margin = int(getattr(cfg, 'CENTERED_CROP_EDGE_MARGIN', 3))
    normal_limit_mode = str(getattr(cfg, 'NORMAL_TILES_LIMIT_MODE', 'same'))
    normal_limit_multiplier = float(getattr(cfg, 'NORMAL_TILES_MULTIPLIER', 1.0))
    dataset_debug = bool(getattr(cfg, 'DATASET_DEBUG', False))
    rect_border_padding = int(getattr(cfg, 'RECT_BORDER_PADDING', 0))
    force_extraction = bool(getattr(cfg, 'FORCE_EXTRACTION', True))
    debug_root = os.path.abspath('debug_info') if dataset_debug else None
    stats_root = os.path.abspath('debug_info' if dataset_debug else '.')
    algo_label = centered_algo.strip().lower()
    if algo_label not in {'heuristic', 'ring'}:
        algo_label = 'heuristic'

    slides_list = glob.glob(os.path.join(slides_folder, '**', '*.mrxs'), recursive=True)
    print('Total slides: ', len(slides_list))
    if centered_crop:
        edge_margin = max(int(edge_margin), 0)
        print('Using pathology-centered cropping')
        if algo_label == 'ring':
            print(f'Centered-crop algo: ring (edge margin {edge_margin}px; exhaustive, slower)')
        else:
            print(f'Centered-crop algo: heuristic (edge margin {edge_margin}px; fast, may miss)')
        _reset_crop_stats()
    elif overlap > 0:
        print(f'Using tile overlap: {overlap:.0%}')

    json_list = glob.glob(os.path.join(json_folder, '**', '*.json'), recursive=True)

    for slide in slides_list:
        slide_id = os.path.splitext(os.path.basename(slide))[0]
        # Derive expected JSON name robustly from slide filename
        base_name = os.path.splitext(os.path.basename(slide))[0]
        json_name = f"{base_name}.json"
        json_path = [f for f in json_list if json_name in f]

        if not json_path:
            print(f'JSON for {slide} not found, skipping')
            continue

        with open(json_path[0], 'r') as f:
            rois = json.load(f)

        rects = __extract_rects(rois)
        print(f'Parsing {slide}: {len(rects)} rectangles total')

        for rect in rects:
            rect_name, rect_coords = next(iter(rect.items()))

            top, bot = rect_coords
            if bot[0] - top[0] > 8000:
                print(f'{rect_name} larger than 8000px - skipping')
                continue

            __extract_rect_regions(
                rect_coords,
                slide,
                json_path[0],
                openslide_path,
                classes=classes,
                zoom_levels=zoom_levels,
                rect_name=rect_name,
                slide_id=slide_id,
                exclude_duplicates=exclude_duplicates,
                overlap=overlap,
                centered_crop=centered_crop,
                centered_algo=algo_label,
                edge_margin=edge_margin,
                normal_limit_mode=normal_limit_mode,
                normal_limit_multiplier=normal_limit_multiplier,
                dataset_debug=dataset_debug,
                debug_root=debug_root,
                rect_border_padding=rect_border_padding,
                force_extraction=force_extraction,
            )

    if centered_crop:
        stats = _get_crop_stats()
        if stats_root:
            stats.dump_details(stats_root)
        if dataset_debug and debug_root:
            stats.debug_root = debug_root
        stats.print_summary()


def __extract_rect_regions(
    rect,
    slidepath,
    jsonpath,
    openslide_path,
    rect_name='roi',
    slide_id: str | None = None,
    zoom_levels=[128, 256, 512],
    classes={},
    exclude_duplicates: bool = False,
    overlap: float = 0.0,
    centered_crop: bool = False,
    centered_algo: str = 'heuristic',
    edge_margin: int = 3,
    normal_limit_mode: str = 'same',
    normal_limit_multiplier: float = 1.0,
    dataset_debug: bool = False,
    debug_root: str | None = None,
    rect_border_padding: int = 0,
    force_extraction: bool = True,
):
    if hasattr(os, 'add_dll_directory'):
        with os.add_dll_directory(openslide_path):
            import openslide
    else:
        import openslide

    slide = openslide.OpenSlide(slidepath)

    with open(jsonpath, 'r') as f:
        rois = json.load(f)

    top, bot = rect
    regions = __get_rect_regions(rois, top, bot, exclude_duplicates=exclude_duplicates)

    if not regions:
        print(f'regions is none for {rect_name}, skipping')
        return
    print(f'{rect_name} contains {len(regions)} regions')

    rectangle = np.asarray(slide.read_region(top, 0, (bot[0] - top[0], bot[1] - top[1])))
    masks = np.zeros(rectangle.shape[:2], dtype=np.uint8)

    roi_path = os.path.join('dataset', 'rois')
    masks_path = os.path.join('dataset', 'masks')

    __make_dirs(roi_path, masks_path)
    __draw_masks(classes, regions, masks)

    name_prefix = f'{slide_id}__{rect_name}' if slide_id else rect_name
    pad = max(int(rect_border_padding), 0)
    pathology_records = []
    if regions and classes:
        height, width = rectangle.shape[:2]
        for region in regions:
            anno, _, max_x, max_y, min_x, min_y = region
            if anno not in classes:
                continue
            local_min_x = max(0, min_x - top[0])
            local_max_x = min(width, max_x - top[0])
            local_min_y = max(0, min_y - top[1])
            local_max_y = min(height, max_y - top[1])
            pathology_records.append({
                'label': anno,
                'bbox': (local_min_y + pad, local_min_x + pad, local_max_y + pad, local_max_x + pad),
                'group_tiles': 0,
                'ind_tiles': 0,
                'forced_tiles': 0,
            })

    if pad > 0:
        border_value = (255, 255, 255, 255) if rectangle.ndim == 3 and rectangle.shape[2] == 4 else (255, 255, 255)
        rectangle = cv2.copyMakeBorder(
            rectangle,
            pad,
            pad,
            pad,
            pad,
            cv2.BORDER_CONSTANT,
            value=border_value,
        )
        masks = np.pad(masks, ((pad, pad), (pad, pad)), mode='constant', constant_values=0)
    if centered_crop:
        # Use pathology-centered cropping (avoids truncating pathologies)
        # Tile size is determined by object bbox + padding, not by zoom_levels
        __crop_dataset_centered(
            name_prefix,
            rectangle,
            masks,
            roi_path,
            masks_path,
            centered_algo=centered_algo,
            edge_margin=edge_margin,
            normal_limit_mode=normal_limit_mode,
            normal_limit_multiplier=normal_limit_multiplier,
            slide_id=slide_id,
            rect_label=rect_name,
            pathology_records=pathology_records,
            force_extraction=force_extraction,
        )
    else:
        __crop_dataset(name_prefix, zoom_levels, rectangle, masks, roi_path, masks_path, overlap=overlap)

    if dataset_debug and centered_crop:
        _write_debug_rect_image(
            debug_root,
            slide_id,
            rect_name,
            rectangle,
            pathology_records,
            pad=rect_border_padding,
        )


def __make_dirs(roi_path, masks_path):
    if not os.path.exists('dataset'):
        os.mkdir('dataset')
    
    if not os.path.exists(roi_path):
        os.mkdir(roi_path)
    
    if not os.path.exists(masks_path):
        os.mkdir(masks_path)


# Mask values for saving (visible in image viewers)
# These get converted back to class indices (0, 1, 2) when loading for training
MASK_VALUE_BACKGROUND = 0      # Class 0: background (black)
MASK_VALUE_NORMAL = 127        # Class 1: normal cells (gray)
MASK_VALUE_ABNORMAL = 255      # Class 2: abnormal cells (white)


def __draw_masks(classes, regions, masks):
    top_layer = []  # TODO priority should vary by level value
    for region in regions:
        name, points, _, _, _, _ = region
        name = name.strip('?) ')

        if name in classes.keys():
            top_layer.append(region)
            continue

        # Normal cells -> class 1 (saved as 127 for visibility)
        cv2.drawContours(masks, [points], 0, MASK_VALUE_NORMAL, -1)

    # double pass for class priority
    for region in top_layer:
        name, points, _, _, _, _ = region
        name = name.strip('?) ')
        # Abnormal cells -> class 2 (saved as 255 for visibility)
        cv2.drawContours(masks, [points], 0, MASK_VALUE_ABNORMAL, -1)


def _get_pathology_objects(masks):
    """
    Find all pathology objects in mask and return their properties.

    Returns:
        tuple: (labeled_array, list of object dicts with id, cx, cy, bbox)
    """
    from scipy import ndimage

    pathology = (masks == MASK_VALUE_ABNORMAL).astype(np.uint8)
    labeled, num_objects = ndimage.label(pathology)

    objects = []
    for obj_id in range(1, num_objects + 1):
        obj_mask = (labeled == obj_id)
        ys, xs = np.where(obj_mask)
        if len(ys) == 0:
            continue
        objects.append({
            'id': obj_id,
            'cy': int(np.mean(ys)),
            'cx': int(np.mean(xs)),
            'y1': int(ys.min()),
            'y2': int(ys.max()),
            'x1': int(xs.min()),
            'x2': int(xs.max()),
            'area': len(ys),
        })

    return labeled, objects


def _get_normal_objects(masks):
    """
    Find all normal cell objects in mask and return their properties.

    Returns:
        tuple: (labeled_array, list of object dicts with id, cx, cy, bbox)
    """
    from scipy import ndimage

    normal = (masks == MASK_VALUE_NORMAL).astype(np.uint8)
    labeled, num_objects = ndimage.label(normal)

    objects = []
    for obj_id in range(1, num_objects + 1):
        obj_mask = (labeled == obj_id)
        ys, xs = np.where(obj_mask)
        if len(ys) == 0:
            continue
        objects.append({
            'id': obj_id,
            'cy': int(np.mean(ys)),
            'cx': int(np.mean(xs)),
            'y1': int(ys.min()),
            'y2': int(ys.max()),
            'x1': int(xs.min()),
            'x2': int(xs.max()),
            'area': len(ys),
        })

    return labeled, objects


def _bbox_intersects_tile(obj, y1, x1, y2, x2):
    """Check if object bbox intersects with tile bounds."""
    return not (obj['x2'] < x1 or obj['x1'] > x2 or obj['y2'] < y1 or obj['y1'] > y2)


def _bbox_fully_inside(obj, y1, x1, y2, x2, margin=2):
    """Check if object bbox is fully inside tile bounds."""
    return (obj['y1'] >= y1 + margin and obj['y2'] <= y2 - margin and
            obj['x1'] >= x1 + margin and obj['x2'] <= x2 - margin)


def _has_truncated_objects(tile_coords, labeled, margin=3):
    """
    Check if any pathology objects are truncated (cut by tile edge).

    Args:
        tile_coords: (y1, x1, y2, x2) tile bounds
        labeled: labeled array from scipy.ndimage.label
        margin: edge margin in pixels

    Returns:
        True if any object touches tile edge
    """
    y1, x1, y2, x2 = tile_coords
    h, w = y2 - y1, x2 - x1

    tile_labeled = labeled[y1:y2, x1:x2]
    objects_in_tile = set(np.unique(tile_labeled)) - {0}

    for obj_id in objects_in_tile:
        obj_pixels = (tile_labeled == obj_id)

        # Check if object touches any edge
        if (np.any(obj_pixels[:margin, :]) or      # top
            np.any(obj_pixels[-margin:, :]) or     # bottom
            np.any(obj_pixels[:, :margin]) or      # left
            np.any(obj_pixels[:, -margin:])):      # right
            return True

    return False


def _compute_tile_for_objects(objects, h, w, padding=30):
    """
    Compute square tile coordinates that fits all objects with padding.

    Returns:
        tuple: (y1, x1, y2, x2) or None if objects don't fit
    """
    if not objects:
        return None

    # Combined bbox of all objects
    min_y1 = min(obj['y1'] for obj in objects)
    max_y2 = max(obj['y2'] for obj in objects)
    min_x1 = min(obj['x1'] for obj in objects)
    max_x2 = max(obj['x2'] for obj in objects)

    obj_h = max_y2 - min_y1
    obj_w = max_x2 - min_x1
    tile_size = max(obj_h, obj_w) + 2 * padding

    # Center tile on combined bbox center
    cy = (min_y1 + max_y2) // 2
    cx = (min_x1 + max_x2) // 2
    half = tile_size // 2

    y1 = cy - half
    x1 = cx - half
    y2 = y1 + tile_size
    x2 = x1 + tile_size

    # Clamp to image bounds
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y2 > h:
        y1 -= (y2 - h)
        y2 = h
    if x2 > w:
        x1 -= (x2 - w)
        x2 = w

    # Final bounds check
    if y1 < 0 or x1 < 0 or y2 > h or x2 > w:
        return None

    return (y1, x1, y2, x2)


def _tile_truncates_pathology(y1, x1, y2, x2, labeled, edge_margin=3):
    """
    Check if any pathology object is truncated (cut by tile edge).

    Args:
        y1, x1, y2, x2: tile bounds
        labeled: labeled array from scipy.ndimage.label
        edge_margin: pixels from edge to check

    Returns:
        True if any pathology touches tile edge (would be truncated)
    """
    tile_labeled = labeled[y1:y2, x1:x2]
    objects_in_tile = set(np.unique(tile_labeled)) - {0}

    for obj_id in objects_in_tile:
        obj_pixels = (tile_labeled == obj_id)

        # Check if object touches any edge
        if (np.any(obj_pixels[:edge_margin, :]) or      # top
            np.any(obj_pixels[-edge_margin:, :]) or     # bottom
            np.any(obj_pixels[:, :edge_margin]) or      # left
            np.any(obj_pixels[:, -edge_margin:])):      # right
            return True

    return False


def _find_nearby_pathologies(center_obj, all_objects, max_distance=100):
    """
    Find pathology objects close to center object.

    Returns:
        list of objects within max_distance of center object's bbox
    """
    nearby = [center_obj]
    cx, cy = center_obj['cx'], center_obj['cy']

    for obj in all_objects:
        if obj['id'] == center_obj['id']:
            continue

        # Distance from center object's center to other object's center
        dist = math.hypot(obj['cx'] - cx, obj['cy'] - cy)
        if dist <= max_distance:
            nearby.append(obj)

    return nearby


def _try_fit_tile_without_truncation(
    objects,
    labeled,
    h,
    w,
    padding=30,
    max_expand=200,
    min_padding=10,
    padding_step=5,
    edge_margin=3,
):
    """
    Try to fit objects into a tile without truncating any pathology.

    Starts with tight bbox + padding, expands if needed to avoid truncation.

    Returns:
        tuple: (y1, x1, y2, x2) or None if cannot fit without truncation
    """
    # Start with combined bbox
    min_y1 = min(obj['y1'] for obj in objects)
    max_y2 = max(obj['y2'] for obj in objects)
    min_x1 = min(obj['x1'] for obj in objects)
    max_x2 = max(obj['x2'] for obj in objects)

    obj_h = max_y2 - min_y1
    obj_w = max_x2 - min_x1

    cy = (min_y1 + max_y2) // 2
    cx = (min_x1 + max_x2) // 2

    pad_start = max(int(padding), int(min_padding))
    pad_end = max(int(min_padding), 0)
    step = max(int(padding_step), 1)

    for pad in range(pad_start, pad_end - 1, -step):
        base_size = max(obj_h, obj_w) + 2 * pad

        # Try increasing sizes until no truncation
        for extra in range(0, max_expand + 1, 10):
            tile_size = base_size + extra
            half = tile_size // 2

            y1 = cy - half
            x1 = cx - half
            y2 = y1 + tile_size
            x2 = x1 + tile_size

            # Clamp to bounds
            if y1 < 0:
                y2 -= y1
                y1 = 0
            if x1 < 0:
                x2 -= x1
                x1 = 0
            if y2 > h:
                y1 -= (y2 - h)
                y2 = h
            if x2 > w:
                x1 -= (x2 - w)
                x2 = w

            # Check bounds
            if y1 < 0 or x1 < 0 or y2 > h or x2 > w:
                continue

            # Check no truncation
            if not _tile_truncates_pathology(y1, x1, y2, x2, labeled, edge_margin=edge_margin):
                return (y1, x1, y2, x2)

    return None


def _find_intruding_pathologies(y1, x1, y2, x2, target_obj, all_pathology_objects):
    """
    Find other pathologies that intrude into the tile.

    Returns:
        list of (obj, relative_position) where relative_position is
        ('top', 'bottom', 'left', 'right') indicating where the intruder is
    """
    intruders = []
    tile_cy = (y1 + y2) // 2
    tile_cx = (x1 + x2) // 2

    for other_obj in all_pathology_objects:
        if other_obj['id'] == target_obj['id']:
            continue

        # Check if other object's bbox intersects tile
        if not (other_obj['x2'] >= x1 and other_obj['x1'] <= x2 and
                other_obj['y2'] >= y1 and other_obj['y1'] <= y2):
            continue

        # Determine relative position of intruder
        dy = other_obj['cy'] - tile_cy
        dx = other_obj['cx'] - tile_cx

        if abs(dy) > abs(dx):
            position = 'bottom' if dy > 0 else 'top'
        else:
            position = 'right' if dx > 0 else 'left'

        intruders.append((other_obj, position))

    return intruders


def _compute_shift_away_from_intruders(intruders, step=10):
    """
    Compute shift direction to move away from intruding pathologies.

    Returns:
        (dy, dx) shift to apply
    """
    dy, dx = 0, 0

    for intruder, position in intruders:
        if position == 'top':
            dy += step  # Move down (away from top intruder)
        elif position == 'bottom':
            dy -= step  # Move up (away from bottom intruder)
        elif position == 'left':
            dx += step  # Move right (away from left intruder)
        elif position == 'right':
            dx -= step  # Move left (away from right intruder)

    return dy, dx


def _integral_image(mask):
    """Compute integral image with 1px zero padding (top/left)."""
    if mask.dtype != np.int64 and mask.dtype != np.int32:
        mask = mask.astype(np.int32)
    summed = mask.cumsum(axis=0).cumsum(axis=1)
    return np.pad(summed, ((1, 0), (1, 0)), mode='constant', constant_values=0)


def _sum_rect(integral, y1, x1, y2, x2):
    return integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1]


def _ring_sum(integral, y1, x1, size, edge_margin):
    if edge_margin <= 0:
        return 0
    y2 = y1 + size
    x2 = x1 + size
    outer = _sum_rect(integral, y1, x1, y2, x2)
    if size <= 2 * edge_margin:
        return outer
    inner = _sum_rect(integral, y1 + edge_margin, x1 + edge_margin, y2 - edge_margin, x2 - edge_margin)
    return outer - inner


def _try_fit_single_pathology_heuristic(
    obj,
    all_pathology_objects,
    labeled,
    h,
    w,
    padding=30,
    edge_margin=3,
    max_iterations=40,
    min_padding=10,
    padding_step=5,
):
    """
    Heuristic: shift a tile around the target pathology until no pathology
    pixels touch the tile edge (allows intruders in the interior).
    """
    obj_h = obj['y2'] - obj['y1']
    obj_w = obj['x2'] - obj['x1']
    cy, cx = obj['cy'], obj['cx']

    pad_start = max(int(padding), int(min_padding))
    pad_end = max(int(min_padding), 0)
    step_pad = max(int(padding_step), 1)

    for pad in range(pad_start, pad_end - 1, -step_pad):
        tile_size = max(obj_h, obj_w) + 2 * pad
        half = tile_size // 2

        # Start with centered tile
        total_dy, total_dx = 0, 0

        for iteration in range(max_iterations):
            y1 = cy - half + total_dy
            x1 = cx - half + total_dx
            y2 = y1 + tile_size
            x2 = x1 + tile_size

            # Clamp to bounds
            if y1 < 0:
                y2 -= y1
                y1 = 0
            if x1 < 0:
                x2 -= x1
                x1 = 0
            if y2 > h:
                y1 -= (y2 - h)
                y2 = h
            if x2 > w:
                x1 -= (x2 - w)
                x2 = w

            # Check bounds
            if y1 < 0 or x1 < 0 or y2 > h or x2 > w:
                break

            # Check target object is fully inside (with edge margin)
            if not (obj['y1'] >= y1 + edge_margin and obj['y2'] <= y2 - edge_margin and
                    obj['x1'] >= x1 + edge_margin and obj['x2'] <= x2 - edge_margin):
                break  # Shifted too far, target object out of frame

            # Accept if no pathology touches the tile edge band
            if not _tile_truncates_pathology(y1, x1, y2, x2, labeled, edge_margin=edge_margin):
                return (y1, x1, y2, x2)

            # Find intruding pathologies to guide shift direction (heuristic)
            intruders = _find_intruding_pathologies(y1, x1, y2, x2, obj, all_pathology_objects)
            if not intruders:
                break

            # Calculate shift based on where intruders are
            # Shift step proportional to intruder size for efficiency
            max_intruder_size = max(
                max(intr['y2'] - intr['y1'], intr['x2'] - intr['x1'])
                for intr, _ in intruders
            )
            step = max(10, min(max_intruder_size // 2, 30))

            shift_dy, shift_dx = _compute_shift_away_from_intruders(intruders, step)

            if shift_dy == 0 and shift_dx == 0:
                break  # No clear direction to shift

            total_dy += shift_dy
            total_dx += shift_dx

            # Safety: don't shift too far from original center
            max_total_shift = tile_size
            if abs(total_dy) > max_total_shift or abs(total_dx) > max_total_shift:
                break

    return None


def _try_fit_single_pathology_ring(
    obj,
    integral,
    h,
    w,
    padding=30,
    edge_margin=3,
    min_padding=10,
    padding_step=5,
):
    """
    Ring-sum search: exhaustively find a tile where no pathology pixels
    touch the edge band (allows intruders in the interior).
    """
    obj_h = obj['y2'] - obj['y1']
    obj_w = obj['x2'] - obj['x1']
    cy, cx = obj['cy'], obj['cx']

    pad_start = max(int(padding), int(min_padding))
    pad_end = max(int(min_padding), 0)
    step_pad = max(int(padding_step), 1)

    edge_margin = max(int(edge_margin), 0)

    for pad in range(pad_start, pad_end - 1, -step_pad):
        tile_size = max(obj_h, obj_w) + 2 * pad
        if tile_size <= 0:
            continue
        if tile_size > h or tile_size > w:
            continue

        y1_min = max(0, obj['y2'] + edge_margin - tile_size)
        y1_max = min(obj['y1'] - edge_margin, h - tile_size)
        x1_min = max(0, obj['x2'] + edge_margin - tile_size)
        x1_max = min(obj['x1'] - edge_margin, w - tile_size)

        if y1_min > y1_max or x1_min > x1_max:
            continue

        y1_center = min(max(cy - tile_size // 2, y1_min), y1_max)
        x1_center = min(max(cx - tile_size // 2, x1_min), x1_max)

        best = None
        best_dist = None
        for y1 in range(int(y1_min), int(y1_max) + 1):
            y2 = y1 + tile_size
            for x1 in range(int(x1_min), int(x1_max) + 1):
                if _ring_sum(integral, y1, x1, tile_size, edge_margin) != 0:
                    continue
                dist = (y1 - y1_center) ** 2 + (x1 - x1_center) ** 2
                if best is None or dist < best_dist:
                    best = (y1, x1)
                    best_dist = dist
                    if dist == 0:
                        return (y1, x1, y2, x1 + tile_size)

        if best is not None:
            y1, x1 = best
            return (y1, x1, y1 + tile_size, x1 + tile_size)

    return None


def _try_shift_tile_away_from_pathology(obj, masks, h, w, padding=30, max_iterations=15):
    """
    Try to shift a normal cell tile to avoid pathology.

    Uses smart shifting: analyzes where pathology pixels are located
    and shifts tile in the opposite direction.

    Returns:
        tuple: (y1, x1, y2, x2) or None if cannot avoid pathology
    """
    obj_h = obj['y2'] - obj['y1']
    obj_w = obj['x2'] - obj['x1']
    tile_size = max(obj_h, obj_w) + 2 * padding

    cy, cx = obj['cy'], obj['cx']
    half = tile_size // 2

    total_dy, total_dx = 0, 0

    for iteration in range(max_iterations):
        y1 = cy - half + total_dy
        x1 = cx - half + total_dx
        y2 = y1 + tile_size
        x2 = x1 + tile_size

        # Clamp to bounds
        if y1 < 0:
            y2 -= y1
            y1 = 0
        if x1 < 0:
            x2 -= x1
            x1 = 0
        if y2 > h:
            y1 -= (y2 - h)
            y2 = h
        if x2 > w:
            x1 -= (x2 - w)
            x2 = w

        # Check bounds
        if y1 < 0 or x1 < 0 or y2 > h or x2 > w:
            break

        tile_mask = masks[y1:y2, x1:x2]
        pathology_pixels = (tile_mask == MASK_VALUE_ABNORMAL)

        if not np.any(pathology_pixels):
            # Success! No pathology in tile
            return (y1, x1, y2, x2)

        # Find where pathology is located in the tile
        path_ys, path_xs = np.where(pathology_pixels)
        if len(path_ys) == 0:
            return (y1, x1, y2, x2)

        # Calculate centroid of pathology pixels relative to tile center
        tile_h, tile_w = tile_mask.shape[:2]
        path_center_y = np.mean(path_ys) - tile_h // 2
        path_center_x = np.mean(path_xs) - tile_w // 2

        # Calculate shift step based on pathology extent
        path_extent_y = path_ys.max() - path_ys.min() + 1
        path_extent_x = path_xs.max() - path_xs.min() + 1
        step = max(10, min(max(path_extent_y, path_extent_x), 40))

        # Shift away from pathology
        shift_dy = -int(np.sign(path_center_y) * step) if abs(path_center_y) > 5 else 0
        shift_dx = -int(np.sign(path_center_x) * step) if abs(path_center_x) > 5 else 0

        if shift_dy == 0 and shift_dx == 0:
            # Pathology is centered, try all directions
            if iteration == 0:
                shift_dy, shift_dx = step, 0
            elif iteration == 1:
                shift_dy, shift_dx = -step, 0
            elif iteration == 2:
                shift_dy, shift_dx = 0, step
            elif iteration == 3:
                shift_dy, shift_dx = 0, -step
            else:
                break

        total_dy += shift_dy
        total_dx += shift_dx

        # Safety: don't shift too far
        max_total_shift = tile_size
        if abs(total_dy) > max_total_shift or abs(total_dx) > max_total_shift:
            break

    return None


def _try_fit_single_pathology_force(
    obj,
    labeled,
    h,
    w,
    padding=30,
    edge_margin=3,
    min_padding=10,
    padding_step=5,
    coarse_step=8,
):
    obj_h = obj['y2'] - obj['y1']
    obj_w = obj['x2'] - obj['x1']
    cy, cx = obj['cy'], obj['cx']

    pad_start = max(int(padding), int(min_padding))
    pad_end = max(int(min_padding), 0)
    step_pad = max(int(padding_step), 1)
    coarse_step = max(int(coarse_step), 1)

    edge_margin = max(int(edge_margin), 0)

    other_mask = (labeled != 0) & (labeled != obj['id'])
    integral = _integral_image(other_mask.astype(np.uint8))

    best = None
    best_score = None
    best_dist = None

    for pad in range(pad_start, pad_end - 1, -step_pad):
        tile_size = max(obj_h, obj_w) + 2 * pad
        if tile_size <= 0 or tile_size > h or tile_size > w:
            continue

        y1_min = max(0, obj['y2'] + edge_margin - tile_size)
        y1_max = min(obj['y1'] - edge_margin, h - tile_size)
        x1_min = max(0, obj['x2'] + edge_margin - tile_size)
        x1_max = min(obj['x1'] - edge_margin, w - tile_size)

        if y1_min > y1_max or x1_min > x1_max:
            continue

        y1_center = min(max(cy - tile_size // 2, y1_min), y1_max)
        x1_center = min(max(cx - tile_size // 2, x1_min), x1_max)

        def evaluate_range(y_start, y_end, x_start, x_end, step):
            nonlocal best, best_score, best_dist
            for y1 in range(int(y_start), int(y_end) + 1, step):
                y2 = y1 + tile_size
                for x1 in range(int(x_start), int(x_end) + 1, step):
                    x2 = x1 + tile_size
                    score = _sum_rect(integral, y1, x1, y2, x2)
                    dist = (y1 - y1_center) ** 2 + (x1 - x1_center) ** 2
                    if (best_score is None or score < best_score or
                            (score == best_score and dist < best_dist)):
                        best_score = score
                        best_dist = dist
                        best = (y1, x1, y2, x2)

        evaluate_range(y1_min, y1_max, x1_min, x1_max, coarse_step)

        if best is not None:
            refine = coarse_step
            y_start = max(y1_min, best[0] - refine)
            y_end = min(y1_max, best[0] + refine)
            x_start = max(x1_min, best[1] - refine)
            x_end = min(x1_max, best[1] + refine)
            evaluate_range(y_start, y_end, x_start, x_end, 1)
            if best_score == 0 and best_dist == 0:
                return best

    return best


def _pathologies_in_tile(pathology_records, y1, x1, y2, x2, edge_margin=0):
    if not pathology_records:
        return []
    hits = []
    for rec in pathology_records:
        py1, px1, py2, px2 = rec['bbox']
        if (py1 >= y1 + edge_margin and py2 <= y2 - edge_margin and
                px1 >= x1 + edge_margin and px2 <= x2 - edge_margin):
            hits.append(rec)
    return hits


def _bbox_overlaps_mask(bbox, intruder_mask, tile_y1, tile_x1):
    by1, bx1, by2, bx2 = bbox
    sy1 = max(0, by1 - tile_y1)
    sx1 = max(0, bx1 - tile_x1)
    sy2 = min(intruder_mask.shape[0], by2 - tile_y1)
    sx2 = min(intruder_mask.shape[1], bx2 - tile_x1)
    if sy1 >= sy2 or sx1 >= sx2:
        return False
    return bool(np.any(intruder_mask[sy1:sy2, sx1:sx2]))


def _increment_pathology_records(pathology_records, y1, x1, y2, x2, edge_margin=0, field='ind_tiles', intruder_mask=None):
    for rec in pathology_records or []:
        by1, bx1, by2, bx2 = rec['bbox']
        if (by1 >= y1 + edge_margin and by2 <= y2 - edge_margin and
                bx1 >= x1 + edge_margin and bx2 <= x2 - edge_margin):
            if intruder_mask is not None and _bbox_overlaps_mask(rec['bbox'], intruder_mask, y1, x1):
                continue
            rec[field] = rec.get(field, 0) + 1


def _bbox_intersects_tile(bbox, y1, x1, y2, x2):
    by1, bx1, by2, bx2 = bbox
    return not (bx2 <= x1 or bx1 >= x2 or by2 <= y1 or by1 >= y2)


def _rect_intersection_area(a, b):
    ay1, ax1, ay2, ax2 = a
    by1, bx1, by2, bx2 = b
    dx = min(ax2, bx2) - max(ax1, bx1)
    dy = min(ay2, by2) - max(ay1, by1)
    if dx <= 0 or dy <= 0:
        return 0
    return dx * dy


def _match_record_for_object(obj, pathology_records):
    if not pathology_records:
        return None
    obj_bbox = (obj['y1'], obj['x1'], obj['y2'], obj['x2'])
    best_idx = None
    best_area = 0
    for idx, rec in enumerate(pathology_records):
        area = _rect_intersection_area(obj_bbox, rec['bbox'])
        if area > best_area:
            best_area = area
            best_idx = idx
    if best_idx is not None and best_area > 0:
        return best_idx

    # Fallback: nearest center
    ocx, ocy = obj['cx'], obj['cy']
    best_idx = None
    best_dist = None
    for idx, rec in enumerate(pathology_records):
        by1, bx1, by2, bx2 = rec['bbox']
        rcx = (bx1 + bx2) / 2.0
        rcy = (by1 + by2) / 2.0
        dist = (rcx - ocx) ** 2 + (rcy - ocy) ** 2
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_idx = idx
    return best_idx


def _increment_forced_pathology_records(pathology_records, y1, x1, y2, x2, intruder_mask=None):
    for rec in pathology_records or []:
        if not _bbox_intersects_tile(rec['bbox'], y1, x1, y2, x2):
            continue
        if intruder_mask is not None and _bbox_overlaps_mask(rec['bbox'], intruder_mask, y1, x1):
            continue
        rec['forced_tiles'] = rec.get('forced_tiles', 0) + 1


def _apply_intruder_artifact(roi, mask, intruder_mask):
    if intruder_mask is None or not np.any(intruder_mask):
        return
    if roi.ndim == 3 and roi.shape[2] == 4:
        colors = [(0, 0, 0, 255), (255, 255, 255, 255)]
    elif roi.ndim == 3 and roi.shape[2] == 3:
        colors = [(0, 0, 0), (255, 255, 255)]
    else:
        colors = [0, 255]
    color = random.choice(colors)
    roi[intruder_mask] = color
    mask[intruder_mask] = MASK_VALUE_BACKGROUND


def _center_tile_for_object(obj, padding):
    obj_h = obj['y2'] - obj['y1']
    obj_w = obj['x2'] - obj['x1']
    tile_size = max(obj_h, obj_w) + 2 * int(padding)
    half = tile_size // 2
    cy, cx = int(obj['cy']), int(obj['cx'])
    y1 = cy - half
    x1 = cx - half
    y2 = y1 + tile_size
    x2 = x1 + tile_size
    return (y1, x1, y2, x2)


def _extract_tile_with_padding(rectangle, masks, labeled_path, y1, x1, y2, x2):
    h, w = masks.shape[:2]
    iy1 = max(0, y1)
    ix1 = max(0, x1)
    iy2 = min(h, y2)
    ix2 = min(w, x2)

    roi = rectangle[iy1:iy2, ix1:ix2]
    mask = masks[iy1:iy2, ix1:ix2]
    labels = labeled_path[iy1:iy2, ix1:ix2]

    pad_top = max(0, -y1)
    pad_left = max(0, -x1)
    pad_bottom = max(0, y2 - h)
    pad_right = max(0, x2 - w)

    if pad_top or pad_bottom or pad_left or pad_right:
        border_value = (255, 255, 255, 255) if roi.ndim == 3 and roi.shape[2] == 4 else (255, 255, 255)
        roi = cv2.copyMakeBorder(
            roi,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=border_value,
        )
        mask = np.pad(mask, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
        labels = np.pad(labels, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)

    return roi, mask, labels


def _safe_debug_name(text):
    name = str(text or 'unknown').strip()
    name = name.replace(os.sep, '_')
    name = name.replace(' ', '_')
    return name or 'unknown'


def _write_debug_rect_image(debug_root, slide_id, rect_label, rectangle, pathology_records, pad=0):
    if not debug_root:
        return
    os.makedirs(debug_root, exist_ok=True)
    slide_name = _safe_debug_name(slide_id)
    rect_name = _safe_debug_name(rect_label)
    out_dir = os.path.join(debug_root, slide_name)
    os.makedirs(out_dir, exist_ok=True)

    debug_img = cv2.cvtColor(rectangle, cv2.COLOR_RGBA2BGR)
    pad = max(int(pad), 0)
    if pad > 0:
        h, w = debug_img.shape[:2]
        # Visualize padded rect boundary clearly in debug overlays.
        cv2.rectangle(debug_img, (pad, pad), (w - pad - 1, h - pad - 1), (255, 0, 255), 2)

    for rec in pathology_records or []:
        y1, x1, y2, x2 = [int(v) for v in rec.get('bbox', (0, 0, 0, 0))]
        forced = rec.get('forced_tiles', 0) > 0
        extracted = (rec.get('group_tiles', 0) + rec.get('ind_tiles', 0)) > 0
        if extracted:
            color = (0, 200, 0)
        elif forced:
            color = (0, 200, 200)
        else:
            color = (0, 0, 200)
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)
        label = (
            f"{rec.get('label', 'unknown')} "
            f"g{rec.get('group_tiles', 0)} "
            f"i{rec.get('ind_tiles', 0)} "
            f"f{rec.get('forced_tiles', 0)}"
        )
        text_y = y1 - 5 if y1 > 10 else y1 + 15
        cv2.putText(debug_img, label, (max(0, x1), max(0, text_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    out_path = os.path.join(out_dir, f'{rect_name}.jpg')
    cv2.imwrite(out_path, debug_img)


class CropStats:
    """Statistics collector for centered crop dataset generation."""

    def __init__(self):
        self.total_pathologies = 0
        self.total_normals = 0
        self.total_pathology_regions = 0
        self.extracted_pathology_regions = 0
        self.inextractable_pathology_regions = 0

        # Greedy phase
        self.greedy_saved = 0
        self.greedy_failed_truncation = 0
        self.greedy_failed_duplicate = 0

        # Individual phase
        self.ind_saved = 0
        self.ind_isolated = 0  # Saved using primary algo
        self.ind_fallback = 0  # Used fallback placement
        self.ind_forced = 0    # Forced extraction with intruder masking
        self.ind_failed_truncation = 0
        self.ind_failed_duplicate = 0

        # Normal phase
        self.norm_saved = 0
        self.norm_shifted = 0  # Required shifting
        self.norm_failed_pathology = 0  # Could not avoid pathology
        self.norm_failed_duplicate = 0
        self.norm_skipped_limit = 0  # Skipped due to target limit

        # Quality stats
        self.file_rects = {}
        self.rect_pathologies = {}
        self.inextractable = []
        self.stats_output_path = None
        self.debug_root = None

    def register_rect(self, slide_id, rect_name, pathologies):
        file_key = slide_id or 'unknown'
        self.file_rects[file_key] = self.file_rects.get(file_key, 0) + 1
        rect_key = (file_key, rect_name or 'unknown')
        self.rect_pathologies[rect_key] = pathologies or []
        self.total_pathology_regions += len(pathologies or [])

    def finalize_rect(self, slide_id, rect_name):
        rect_key = (slide_id or 'unknown', rect_name or 'unknown')
        for rec in self.rect_pathologies.get(rect_key, []):
            total_tiles = (
                rec.get('group_tiles', 0)
                + rec.get('ind_tiles', 0)
                + rec.get('forced_tiles', 0)
            )
            if total_tiles > 0:
                self.extracted_pathology_regions += 1
            else:
                self.inextractable_pathology_regions += 1
                self.inextractable.append({
                    'file': rect_key[0],
                    'rect': rect_key[1],
                    'label': rec.get('label', 'unknown'),
                    'bbox': rec.get('bbox'),
                })

    def dump_details(self, output_dir):
        if not output_dir:
            return
        os.makedirs(output_dir, exist_ok=True)
        payload = {
            'totals': {
                'pathologies_detected': self.total_pathologies,
                'normals_detected': self.total_normals,
                'annotated_pathologies': self.total_pathology_regions,
                'extracted_pathologies': self.extracted_pathology_regions,
                'inextractable_pathologies': self.inextractable_pathology_regions,
                'greedy_saved': self.greedy_saved,
            'greedy_failed_truncation': self.greedy_failed_truncation,
            'greedy_failed_duplicate': self.greedy_failed_duplicate,
            'individual_saved': self.ind_saved,
            'individual_primary': self.ind_isolated,
            'individual_fallback': self.ind_fallback,
            'individual_forced': self.ind_forced,
            'individual_failed_truncation': self.ind_failed_truncation,
            'individual_failed_duplicate': self.ind_failed_duplicate,
                'normal_saved': self.norm_saved,
                'normal_shifted': self.norm_shifted,
                'normal_failed_pathology': self.norm_failed_pathology,
                'normal_failed_duplicate': self.norm_failed_duplicate,
                'normal_skipped_limit': self.norm_skipped_limit,
            },
            'file_rects': self.file_rects,
            'rect_pathologies': {
                f'{file_key}::{rect_name}': records
                for (file_key, rect_name), records in self.rect_pathologies.items()
            },
            'inextractable': self.inextractable,
        }
        out_path = os.path.join(output_dir, 'quality_stats.json')
        with open(out_path, 'w') as f:
            json.dump(payload, f, indent=2)
        self.stats_output_path = out_path

    def print_summary(self):
        """Print final statistics summary."""
        print("\n" + "=" * 60)
        print("DATASET GENERATION SUMMARY")
        print("=" * 60)

        print(f"\nInput objects:")
        print(f"  Pathologies detected:  {self.total_pathologies}")
        print(f"  Normal cells detected: {self.total_normals}")

        print(f"\nAnnotated pathology regions:")
        print(f"  Total:                 {self.total_pathology_regions}")
        print(f"  Extracted (>=1 tile):  {self.extracted_pathology_regions}")
        print(f"  Inextractable:         {self.inextractable_pathology_regions}")

        print(f"\nGreedy tiles (groups):")
        print(f"  Saved:                 {self.greedy_saved}")
        print(f"  Failed (truncation):   {self.greedy_failed_truncation}")
        print(f"  Failed (duplicate):    {self.greedy_failed_duplicate}")

        print(f"\nIndividual tiles (pathology):")
        print(f"  Saved:                 {self.ind_saved}")
        print(f"    - Primary algo:      {self.ind_isolated}")
        print(f"    - Fallback:          {self.ind_fallback}")
        print(f"    - Forced:            {self.ind_forced}")
        print(f"  Failed (truncation):   {self.ind_failed_truncation}")
        print(f"  Failed (duplicate):    {self.ind_failed_duplicate}")

        print(f"\nNormal tiles:")
        print(f"  Saved:                 {self.norm_saved}")
        print(f"    - Direct:            {self.norm_saved - self.norm_shifted}")
        print(f"    - After shift:       {self.norm_shifted}")
        print(f"  Failed (pathology):    {self.norm_failed_pathology}")
        print(f"  Failed (duplicate):    {self.norm_failed_duplicate}")
        print(f"  Skipped (limit):       {self.norm_skipped_limit}")

        total_saved = self.greedy_saved + self.ind_saved + self.norm_saved
        total_failed = (self.greedy_failed_truncation + self.greedy_failed_duplicate +
                        self.ind_failed_truncation + self.ind_failed_duplicate +
                        self.norm_failed_pathology + self.norm_failed_duplicate)

        print(f"\nTOTAL:")
        print(f"  Tiles saved:           {total_saved}")
        print(f"  Tiles failed:          {total_failed}")
        if total_saved + total_failed > 0:
            success_rate = total_saved / (total_saved + total_failed) * 100
            print(f"  Success rate:          {success_rate:.1f}%")

        if self.file_rects:
            print(f"\nRects per file:")
            for file_key in sorted(self.file_rects.keys()):
                print(f"  {file_key}: {self.file_rects[file_key]}")

        if self.stats_output_path:
            rel_path = os.path.relpath(self.stats_output_path, os.getcwd())
            print(f"\nDetails written: {rel_path}")
        if self.debug_root:
            rel_root = os.path.relpath(self.debug_root, os.getcwd())
            print(f"Debug overlays:  {rel_root}{os.sep}<slide>{os.sep}<rect>.jpg")
        print("=" * 60 + "\n")


# Global stats collector (reset per extract_all_slides call)
_crop_stats: CropStats | None = None


def _get_crop_stats() -> CropStats:
    global _crop_stats
    if _crop_stats is None:
        _crop_stats = CropStats()
    return _crop_stats


def _reset_crop_stats():
    global _crop_stats
    _crop_stats = CropStats()


def __crop_dataset_centered(
    name_prefix,
    rectangle,
    masks,
    roi_path,
    masks_path,
    padding=30,
    centered_algo='heuristic',
    edge_margin=3,
    normal_limit_mode='same',
    normal_limit_multiplier=1.0,
    slide_id=None,
    rect_label=None,
    pathology_records=None,
    force_extraction=True,
):
    """
    Crop dataset by centering tiles on pathology objects.

    Three-phase approach:
    1. Greedy: fit nearby pathologies together (context with multiple objects)
    2. Individual: each pathology separately, edge-safe tiles
    3. Normal: normal cells, shifted away from any pathology

    Ensures no pathology is truncated at tile edges.

    Args:
        rect_name: Name prefix for saved files
        rectangle: RGBA image from OpenSlide
        masks: Mask array
        roi_path: Output path for ROI images
        masks_path: Output path for mask images
        padding: Padding around object bbox in pixels
        centered_algo: 'heuristic' or 'ring'
        edge_margin: Edge band width to keep pathology away from tile borders
        normal_limit_mode: 'same' | 'all' | 'multiplier'
        normal_limit_multiplier: Multiplier when normal_limit_mode == 'multiplier'
        slide_id: slide filename stem for stats
        rect_label: rectangle label from annotation
        pathology_records: list of pathology region dicts for per-rect stats
        force_extraction: Force extraction by masking intruders when individual fails
    """
    stats = _get_crop_stats()
    if pathology_records is None:
        pathology_records = []
    stats.register_rect(slide_id, rect_label or name_prefix, pathology_records)

    labeled_path, pathology_objects = _get_pathology_objects(masks)
    labeled_normal, normal_objects = _get_normal_objects(masks)

    stats.total_pathologies += len(pathology_objects)
    stats.total_normals += len(normal_objects)

    h, w = masks.shape[:2]
    centered_algo = (centered_algo or 'heuristic').strip().lower()
    if centered_algo not in {'heuristic', 'ring'}:
        centered_algo = 'heuristic'
    edge_margin = max(int(edge_margin), 0)
    normal_limit_mode = (normal_limit_mode or 'same').strip().lower()
    if normal_limit_mode not in {'same', 'all', 'multiplier'}:
        normal_limit_mode = 'same'
    try:
        normal_limit_multiplier = float(normal_limit_multiplier)
    except (TypeError, ValueError):
        normal_limit_multiplier = 1.0
    pathology_integral = None
    if centered_algo == 'ring':
        pathology_integral = _integral_image((labeled_path > 0).astype(np.uint8))
    saved_tiles = set()

    for obj in pathology_objects:
        obj['rec_idx'] = _match_record_for_object(obj, pathology_records)

    # Phase 1: Greedy tiles — group nearby pathologies (context)
    greedy_count = 0
    used_in_greedy = set()

    for obj in pathology_objects:
        if obj['id'] in used_in_greedy:
            continue

        # Find nearby pathologies
        nearby = _find_nearby_pathologies(obj, pathology_objects, max_distance=150)

        if len(nearby) > 1:
            # Try to fit all nearby together
            tile = _try_fit_tile_without_truncation(
                nearby, labeled_path, h, w, padding, edge_margin=edge_margin
            )

            if tile is None:
                stats.greedy_failed_truncation += 1
                continue

            y1, x1, y2, x2 = tile
            tile_key = (y1, x1, y2, x2)

            if tile_key in saved_tiles:
                stats.greedy_failed_duplicate += 1
                # Still mark as used
                for nearby_obj in nearby:
                    used_in_greedy.add(nearby_obj['id'])
                continue

            roi = rectangle[y1:y2, x1:x2]
            mask = masks[y1:y2, x1:x2]
            tile_size = y2 - y1

            try:
                cv2.imwrite(
                    os.path.join(roi_path, f'{name_prefix}_grp_{greedy_count}_coords_{y1}_{x1}_{tile_size}.bmp'),
                    cv2.cvtColor(roi, cv2.COLOR_RGBA2BGR)
                )
                cv2.imwrite(
                    os.path.join(masks_path, f'{name_prefix}_grp_{greedy_count}_coords_{y1}_{x1}_{tile_size}.bmp'),
                    mask
                )
                saved_tiles.add(tile_key)
                greedy_count += 1
                stats.greedy_saved += 1
                _increment_pathology_records(
                    pathology_records, y1, x1, y2, x2, edge_margin=edge_margin, field='group_tiles'
                )

                # Mark all nearby as used in greedy (but still process individually!)
                for nearby_obj in nearby:
                    used_in_greedy.add(nearby_obj['id'])
            except cv2.error:
                print(f'ROI empty: {y1}, {x1}')

    # Phase 2: Individual tiles — EVERY pathology gets its own edge-safe tile
    # Even those in greedy groups get individual tiles for focused learning
    individual_count = 0
    for obj in pathology_objects:
        tile = None
        used_primary = False
        used_forced = False
        intruder_mask = None
        if centered_algo == 'ring' and pathology_integral is not None:
            tile = _try_fit_single_pathology_ring(
                obj, pathology_integral, h, w, padding, edge_margin=edge_margin
            )
            used_primary = tile is not None
        else:
            tile = _try_fit_single_pathology_heuristic(
                obj, pathology_objects, labeled_path, h, w, padding, edge_margin=edge_margin
            )
            used_primary = tile is not None

        if tile is None:
            # Fallback: fit without shifting (still enforces edge margin)
            tile = _try_fit_tile_without_truncation(
                [obj], labeled_path, h, w, padding, edge_margin=edge_margin
            )

        if tile is None and force_extraction:
            tile = _try_fit_single_pathology_force(
                obj,
                labeled_path,
                h,
                w,
                padding,
                edge_margin=0,
            )
            if tile is None:
                tile = _center_tile_for_object(obj, padding)
            used_forced = True

        if tile is None:
            stats.ind_failed_truncation += 1
            continue

        y1, x1, y2, x2 = tile
        tile_key = (y1, x1, y2, x2)

        if tile_key in saved_tiles and not used_forced:
            if force_extraction:
                # Duplicate is treated as a failure; force keeps a tile guaranteed.
                used_forced = True
            else:
                stats.ind_failed_duplicate += 1
                continue

        roi, mask, tile_labels = _extract_tile_with_padding(rectangle, masks, labeled_path, y1, x1, y2, x2)
        if used_forced:
            intruder_mask = (tile_labels != 0) & (tile_labels != obj['id'])
            if np.any(intruder_mask):
                roi = roi.copy()
                mask = mask.copy()
                _apply_intruder_artifact(roi, mask, intruder_mask)
        tile_size = y2 - y1

        try:
            cv2.imwrite(
                os.path.join(roi_path, f'{name_prefix}_ind_{individual_count}_coords_{y1}_{x1}_{tile_size}.bmp'),
                cv2.cvtColor(roi, cv2.COLOR_RGBA2BGR)
            )
            cv2.imwrite(
                os.path.join(masks_path, f'{name_prefix}_ind_{individual_count}_coords_{y1}_{x1}_{tile_size}.bmp'),
                mask
            )
            saved_tiles.add(tile_key)
            individual_count += 1
            stats.ind_saved += 1
            if used_primary:
                stats.ind_isolated += 1
            elif used_forced:
                stats.ind_forced += 1
            else:
                stats.ind_fallback += 1
            rec_idx = obj.get('rec_idx')
            if used_forced:
                if rec_idx is not None and 0 <= rec_idx < len(pathology_records):
                    rec = pathology_records[rec_idx]
                    rec['forced_tiles'] = rec.get('forced_tiles', 0) + 1
                else:
                    _increment_forced_pathology_records(
                        pathology_records,
                        y1,
                        x1,
                        y2,
                        x2,
                        intruder_mask=intruder_mask,
                    )
            else:
                if rec_idx is not None and 0 <= rec_idx < len(pathology_records):
                    rec = pathology_records[rec_idx]
                    rec['ind_tiles'] = rec.get('ind_tiles', 0) + 1
                else:
                    _increment_pathology_records(
                        pathology_records,
                        y1,
                        x1,
                        y2,
                        x2,
                        edge_margin=edge_margin,
                        field='ind_tiles',
                        intruder_mask=intruder_mask,
                    )
        except cv2.error:
            print(f'ROI empty: {y1}, {x1}')

    # Phase 3: Normal cell tiles (for class balance)
    # Try to shift tiles away from pathology instead of just skipping
    normal_count = 0
    base_normals = greedy_count + individual_count
    if normal_limit_mode == 'all':
        target_normals = len(normal_objects)
    elif normal_limit_mode == 'multiplier':
        target_normals = max(int(round(base_normals * normal_limit_multiplier)), 0)
    else:
        target_normals = max(base_normals, 5)

    for obj in normal_objects:
        if normal_count >= target_normals:
            stats.norm_skipped_limit += 1
            continue

        tile = _compute_tile_for_objects([obj], h, w, padding)
        if tile is None:
            continue

        y1, x1, y2, x2 = tile
        tile_key = (y1, x1, y2, x2)

        if tile_key in saved_tiles:
            stats.norm_failed_duplicate += 1
            continue

        # Check if tile contains pathology
        tile_mask = masks[y1:y2, x1:x2]
        shifted = False
        if np.any(tile_mask == MASK_VALUE_ABNORMAL):
            # Try to shift tile away from pathology
            shifted_tile = _try_shift_tile_away_from_pathology(obj, masks, h, w, padding)
            if shifted_tile is None:
                stats.norm_failed_pathology += 1
                continue
            y1, x1, y2, x2 = shifted_tile
            tile_key = (y1, x1, y2, x2)
            if tile_key in saved_tiles:
                stats.norm_failed_duplicate += 1
                continue
            tile_mask = masks[y1:y2, x1:x2]
            shifted = True

        roi = rectangle[y1:y2, x1:x2]
        tile_size = y2 - y1

        try:
            cv2.imwrite(
                os.path.join(roi_path, f'{name_prefix}_norm_{normal_count}_coords_{y1}_{x1}_{tile_size}.bmp'),
                cv2.cvtColor(roi, cv2.COLOR_RGBA2BGR)
            )
            cv2.imwrite(
                os.path.join(masks_path, f'{name_prefix}_norm_{normal_count}_coords_{y1}_{x1}_{tile_size}.bmp'),
                tile_mask
            )
            saved_tiles.add(tile_key)
            normal_count += 1
            stats.norm_saved += 1
            if shifted:
                stats.norm_shifted += 1
        except cv2.error:
            print(f'ROI empty: {y1}, {x1}')

    stats.finalize_rect(slide_id, rect_label or name_prefix)
    print(f'  {name_prefix}: {greedy_count} grp, {individual_count} ind, {normal_count} norm')


def __crop_dataset(rect_name, zoom_levels, rectangle, masks, roi_path, masks_path, overlap: float = 0.0):
    """
    Crop dataset with optional overlap.

    Args:
        rectangle: RGBA image from OpenSlide
        overlap: Overlap ratio (0.0 = no overlap, 0.5 = 50% overlap)
    """
    for zoom in zoom_levels:
        # Calculate stride based on overlap
        stride = max(1, int(zoom * (1.0 - overlap)))

        for x in range(0, rectangle.shape[0], stride):
            for y in range(0, rectangle.shape[1], stride):
                roi = rectangle[x: x + zoom, y: y + zoom]
                mask = masks[x: x + zoom, y: y + zoom]

                if mask.shape[0] < 100 or mask.shape[1] < 100:  # skip small crops
                    continue

                try:
                    # OpenSlide returns RGBA, convert to BGR for cv2.imwrite/imread compatibility
                    cv2.imwrite(os.path.join(roi_path, f'{rect_name}_coords_{x}_{y}_{zoom}.bmp'), cv2.cvtColor(roi, cv2.COLOR_RGBA2BGR))
                    cv2.imwrite(os.path.join(masks_path, f'{rect_name}_coords_{x}_{y}_{zoom}.bmp'), mask)
                except cv2.error:
                    print('ROI empty: ', x, y)


def __points_to_rect(points, min_x, min_y):
    return [[[p[0] - min_x, p[1] - min_y]] for p in points]


def __intersection(a: Rectangle, b: Rectangle):  
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    if (dx >= 0) and (dy >= 0):
        return dx * dy
    return 0


def __index_close(fxy, sxy_list: list, thrsh = 15):
    fxy_rect = Rectangle(fxy[0][0], fxy[0][1], fxy[1][0], fxy[1][1])
    area_fxy = (fxy[1][0] - fxy[0][0]) * (fxy[1][1] - fxy[0][1])

    for sxy in sxy_list:
        sxy_rect = Rectangle(sxy[1][0], sxy[1][1], sxy[2][0], sxy[2][1])
        area_sxy = (sxy[2][0] - sxy[1][0]) * (sxy[2][1] - sxy[1][1])

        if ((math.hypot(fxy[0][0] - sxy[1][0], fxy[0][1] - sxy[1][1]) <= thrsh) or \
            (math.hypot(fxy[1][0] - sxy[2][0], fxy[0][1] - sxy[1][1]) <= thrsh) or \
            (math.hypot(fxy[0][0] - sxy[1][0], fxy[1][1] - sxy[2][1]) <= thrsh) or \
            (math.hypot(fxy[1][0] - sxy[2][0], fxy[1][1] - sxy[2][1]) <= thrsh)) and \
                ((__intersection(fxy_rect, sxy_rect) >= 0.2 * max(area_fxy, area_sxy))):

            return sxy[0]
    
    return None


def __extract_rects(rois):
    rects = []

    for roi in rois:
        anno, points, b_rect = roi['label'], roi['points'], roi['rect']
        if 'rect' not in anno.lower():
            continue

        points = [[int(p[0]), int(p[1])] for p in points]
        max_x, min_x = int(b_rect[2]), int(b_rect[0])
        max_y, min_y = int(b_rect[3]), int(b_rect[1])

        rects.append({anno: [[min_x, min_y], [max_x, max_y]]})
    
    return rects


def __get_rect_regions(rois, top, bot, exclude_duplicates: bool = False):
    regions, registry = [], []
    
    for idx, roi in enumerate(rois):

        anno, points, b_rect = roi['label'], roi['points'], roi['rect']

        if 'rect' in anno.lower():
            regions.append(None)
            continue

        points = [[int(p[0]), int(p[1])] for p in points]

        max_x, min_x = int(b_rect[2]), int(b_rect[0])
        max_y, min_y = int(b_rect[3]), int(b_rect[1])

        norm_points = [[[p[0] - top[0], p[1] - top[1]]] for p in points
                        if -100 < p[0] - top[0] < bot[0] - top[0] + 100
                          and -100 < p[1] - top[1] < bot[1] - top[1] + 100]

        if not norm_points:
            regions.append(None)
            continue
        
        if exclude_duplicates:
            close_idx = __index_close(((min_x, min_y), (max_x, max_y)), registry)
            if close_idx is not None:
                regions[close_idx] = None
                registry = [r for r in registry if r[0] != close_idx]

            registry.append((idx, (min_x, min_y), (max_x, max_y)))

        regions.append((anno, np.asarray(norm_points), max_x, max_y, min_x, min_y))

    return [r for r in regions if r is not None]


def __get_regions(rois, local_coords: bool = True, exclude_duplicates: bool = False):
    regions, registry = [], []

    for idx, roi in enumerate(rois):
        anno, points, b_rect = roi['label'], roi['points'], roi['rect']

        if 'rect' in anno.lower():
            regions.append(None)
            continue

        points = [[int(p[0]), int(p[1])] for p in points]
        max_x, min_x = int(b_rect[2]), int(b_rect[0])
        max_y, min_y = int(b_rect[3]), int(b_rect[1])

        norm_points = [[[p[0] - min_x, p[1] - min_y]] for p in points] if local_coords else points

        if exclude_duplicates:
            close_idx = __index_close(((min_x, min_y), (max_x, max_y)), registry)
            if close_idx is not None:

                # _, _, o_max_x, o_max_y, o_min_x, o_min_y = regions[close_idx]
                # crop = slide.read_region((min_x, min_y), 0, (max_x - min_x, max_y - min_y))
                # orig = slide.read_region((o_min_x, o_min_y), 0, (o_max_x - o_min_x, o_max_y - o_min_y))
                # orig, roi = np.asarray(orig), np.asarray(crop)
                # cv2.imwrite(os.path.join(os.path.join('dataset', 'dupes'), f'{close_idx}_2_{idx}_coords_{max_x}_{max_y}.bmp'),
                #              np.hstack((cv2.resize(orig,(256, 256)), cv2.resize(roi,(256, 256)))))

                regions[close_idx] = None
                registry = [r for r in registry if r[0] != close_idx]

            registry.append((idx, (min_x, min_y), (max_x, max_y)))

        regions.append((anno, np.asarray(norm_points), max_x, max_y, min_x, min_y))

    return [r for r in regions if r is not None]
