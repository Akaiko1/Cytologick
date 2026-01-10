from collections import namedtuple
import json
import glob
import math
import cv2
import os

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
    debug: bool = False,
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
            mask = np.zeros(roi.shape) if debug else np.zeros(roi.shape[:2], dtype=np.uint8)

            __draw_masks(
                classes,
                debug,
                __get_rect_regions(
                    rois,
                    (w_min_x, w_min_y),
                    (w_max_x, w_max_y),
                    exclude_duplicates=exclude_duplicates,
                ),
                mask,
            )
            if debug:
                 __debug_draw_contours(
                     classes,
                     __get_rect_regions(
                         rois,
                         (w_min_x, w_min_y),
                         (w_max_x, w_max_y),
                         exclude_duplicates=exclude_duplicates,
                     ),
                     roi,
                 )

            roi = roi[span + 2:span + (max_y - min_y) - 2, span + 2: span + (max_x - min_x) - 2]
            mask = mask[span + 2:span + (max_y - min_y) - 2, span + 2: span + (max_x - min_x) - 2]

            if debug and np.min(mask[:, :, 1]) > 0: # Check in green channel if all the image filled with green color
                continue

            # OpenSlide returns RGBA, convert to BGR for cv2.imwrite/imread compatibility
            cv2.imwrite(os.path.join(roi_path, f'{name}_coords_{max_x}_{max_y}.bmp'), cv2.cvtColor(roi, cv2.COLOR_RGBA2BGR))
            cv2.imwrite(os.path.join(masks_path, f'{name}_coords_{max_x}_{max_y}.bmp'), mask)


def extract_all_slides(
    slides_folder,
    json_folder,
    openslide_path,
    classes,
    zoom_levels=[256],
    debug: bool = False,
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

    slides_list = glob.glob(os.path.join(slides_folder, '**', '*.mrxs'), recursive=True)
    print('Total slides: ', len(slides_list))
    if centered_crop:
        print('Using pathology-centered cropping')
    elif overlap > 0:
        print(f'Using tile overlap: {overlap:.0%}')

    json_list = glob.glob(os.path.join(json_folder, '**', '*.json'), recursive=True)

    for slide in slides_list:
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
                debug=debug,
                exclude_duplicates=exclude_duplicates,
                overlap=overlap,
                centered_crop=centered_crop,
            )


def __extract_rect_regions(
    rect,
    slidepath,
    jsonpath,
    openslide_path,
    rect_name='roi',
    zoom_levels=[128, 256, 512],
    classes={},
    debug: bool = False,
    exclude_duplicates: bool = False,
    overlap: float = 0.0,
    centered_crop: bool = False,
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
    masks = np.zeros(rectangle.shape if debug else rectangle.shape[:2], dtype=np.uint8)

    roi_path = os.path.join('dataset', 'rois')
    masks_path = os.path.join('dataset', 'masks')

    __make_dirs(roi_path, masks_path)
    __draw_masks(classes, debug, regions, masks)

    if debug:
        __debug_draw_contours(classes, regions, rectangle)
        # OpenSlide returns RGBA, convert to BGR for cv2.imwrite/imread compatibility
        cv2.imwrite(f"{jsonpath.split(os.sep)[-1].replace('.json', '')}_{rect_name}.jpg", cv2.cvtColor(rectangle, cv2.COLOR_RGBA2BGR))
        cv2.imwrite(f"{jsonpath.split(os.sep)[-1].replace('.json', '')}_{rect_name}_mask.jpg", masks)

    if centered_crop:
        # Use pathology-centered cropping (avoids truncating pathologies)
        # Tile size is determined by object bbox + padding, not by zoom_levels
        __crop_dataset_centered(rect_name, rectangle, masks, roi_path, masks_path)
    elif debug:
        __crop_debug_dataset(rect_name, zoom_levels, rectangle, masks, roi_path, masks_path, jsonpath.split(os.sep)[-1].replace('.json', ''), overlap=overlap)
    else:
        __crop_dataset(rect_name, zoom_levels, rectangle, masks, roi_path, masks_path, overlap=overlap)


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


def __draw_masks(classes, debug, regions, masks):
    top_layer = []  # TODO priority should vary by level value
    for region in regions:
        name, points, _, _, _, _ = region
        name = name.strip('?) ')

        if name in classes.keys():
            top_layer.append(region)
            continue

        # Normal cells -> class 1 (saved as 127 for visibility)
        cv2.drawContours(masks, [points], 0, (0, 255, 0) if debug else MASK_VALUE_NORMAL, -1)

    # double pass for class priority
    for region in top_layer:
        name, points, _, _, _, _ = region
        name = name.strip('?) ')
        # Abnormal cells -> class 2 (saved as 255 for visibility)
        cv2.drawContours(masks, [points], 0, (0, 0, 255) if debug else MASK_VALUE_ABNORMAL, -1)


def __debug_draw_contours(classes, regions, image):
    top_layer = []
    for region in regions:
        name, points, _, _, _, _ = region
        name = name.strip('?) ')

        if name in classes.keys():
            top_layer.append(region)
            continue
        
        cv2.drawContours(image, [points], 0, (0, 255, 0), 3)  # 1 class background info
    
    for region in top_layer:
        name, points, _, _, _, _ = region
        name = name.strip('?) ')
        cv2.drawContours(image, [points], 0, (255, 0, 0), 3)


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


def _try_fit_tile_without_truncation(objects, labeled, h, w, padding=30, max_expand=100):
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
    base_size = max(obj_h, obj_w) + 2 * padding

    cy = (min_y1 + max_y2) // 2
    cx = (min_x1 + max_x2) // 2

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
        if not _tile_truncates_pathology(y1, x1, y2, x2, labeled):
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


def _try_fit_single_pathology_isolated(obj, all_pathology_objects, labeled, h, w, padding=30, max_iterations=20):
    """
    Try to fit a single pathology into a tile WITHOUT other pathologies.

    Uses smart shifting: analyzes where intruding pathologies are located
    and shifts tile in the opposite direction.

    Returns:
        tuple: (y1, x1, y2, x2) or None if cannot isolate
    """
    obj_h = obj['y2'] - obj['y1']
    obj_w = obj['x2'] - obj['x1']
    tile_size = max(obj_h, obj_w) + 2 * padding

    cy, cx = obj['cy'], obj['cx']
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

        # Check target object is fully inside
        if not (obj['y1'] >= y1 and obj['y2'] <= y2 and
                obj['x1'] >= x1 and obj['x2'] <= x2):
            break  # Shifted too far, target object out of frame

        # Check our object doesn't touch edges
        tile_labeled = labeled[y1:y2, x1:x2]
        if np.any(tile_labeled == obj['id']):
            our_pixels = (tile_labeled == obj['id'])
            if (np.any(our_pixels[:3, :]) or np.any(our_pixels[-3:, :]) or
                np.any(our_pixels[:, :3]) or np.any(our_pixels[:, -3:])):
                break  # Our object would be truncated

        # Find intruding pathologies
        intruders = _find_intruding_pathologies(y1, x1, y2, x2, obj, all_pathology_objects)

        if not intruders:
            # Success! No other pathologies in tile
            return (y1, x1, y2, x2)

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
        max_total_shift = tile_size // 2
        if abs(total_dy) > max_total_shift or abs(total_dx) > max_total_shift:
            break

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


class CropStats:
    """Statistics collector for centered crop dataset generation."""

    def __init__(self):
        self.total_pathologies = 0
        self.total_normals = 0

        # Greedy phase
        self.greedy_saved = 0
        self.greedy_failed_truncation = 0
        self.greedy_failed_duplicate = 0

        # Individual phase
        self.ind_saved = 0
        self.ind_isolated = 0  # Successfully isolated from neighbors
        self.ind_fallback = 0  # Used fallback (not isolated)
        self.ind_failed_truncation = 0
        self.ind_failed_duplicate = 0

        # Normal phase
        self.norm_saved = 0
        self.norm_shifted = 0  # Required shifting
        self.norm_failed_pathology = 0  # Could not avoid pathology
        self.norm_failed_duplicate = 0
        self.norm_skipped_limit = 0  # Skipped due to target limit

    def print_summary(self):
        """Print final statistics summary."""
        print("\n" + "=" * 60)
        print("DATASET GENERATION SUMMARY")
        print("=" * 60)

        print(f"\nInput objects:")
        print(f"  Pathologies detected:  {self.total_pathologies}")
        print(f"  Normal cells detected: {self.total_normals}")

        print(f"\nGreedy tiles (groups):")
        print(f"  Saved:                 {self.greedy_saved}")
        print(f"  Failed (truncation):   {self.greedy_failed_truncation}")
        print(f"  Failed (duplicate):    {self.greedy_failed_duplicate}")

        print(f"\nIndividual tiles (pathology):")
        print(f"  Saved:                 {self.ind_saved}")
        print(f"    - Isolated:          {self.ind_isolated}")
        print(f"    - Fallback:          {self.ind_fallback}")
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


def __crop_dataset_centered(rect_name, rectangle, masks, roi_path, masks_path, padding=30):
    """
    Crop dataset by centering tiles on pathology objects.

    Three-phase approach:
    1. Greedy: fit nearby pathologies together (context with multiple objects)
    2. Individual: each pathology separately, isolated from neighbors
    3. Normal: normal cells, shifted away from any pathology

    Ensures no pathology is truncated at tile edges.

    Args:
        rect_name: Name prefix for saved files
        rectangle: RGBA image from OpenSlide
        masks: Mask array
        roi_path: Output path for ROI images
        masks_path: Output path for mask images
        padding: Padding around object bbox in pixels
    """
    stats = _get_crop_stats()

    labeled_path, pathology_objects = _get_pathology_objects(masks)
    labeled_normal, normal_objects = _get_normal_objects(masks)

    stats.total_pathologies += len(pathology_objects)
    stats.total_normals += len(normal_objects)

    h, w = masks.shape[:2]
    saved_tiles = set()

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
            tile = _try_fit_tile_without_truncation(nearby, labeled_path, h, w, padding)

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
                    os.path.join(roi_path, f'{rect_name}_grp_{greedy_count}_coords_{y1}_{x1}_{tile_size}.bmp'),
                    cv2.cvtColor(roi, cv2.COLOR_RGBA2BGR)
                )
                cv2.imwrite(
                    os.path.join(masks_path, f'{rect_name}_grp_{greedy_count}_coords_{y1}_{x1}_{tile_size}.bmp'),
                    mask
                )
                saved_tiles.add(tile_key)
                greedy_count += 1
                stats.greedy_saved += 1

                # Mark all nearby as used in greedy (but still process individually!)
                for nearby_obj in nearby:
                    used_in_greedy.add(nearby_obj['id'])
            except cv2.error:
                print(f'ROI empty: {y1}, {x1}')

    # Phase 2: Individual tiles — EVERY pathology gets its own isolated tile
    # Even those in greedy groups get individual tiles for focused learning
    individual_count = 0
    for obj in pathology_objects:
        # Try to isolate this pathology from neighbors
        tile = _try_fit_single_pathology_isolated(
            obj, pathology_objects, labeled_path, h, w, padding
        )

        isolated = tile is not None

        if tile is None:
            # Fallback: fit without isolation check
            tile = _try_fit_tile_without_truncation([obj], labeled_path, h, w, padding)

        if tile is None:
            stats.ind_failed_truncation += 1
            continue

        y1, x1, y2, x2 = tile
        tile_key = (y1, x1, y2, x2)

        if tile_key in saved_tiles:
            stats.ind_failed_duplicate += 1
            continue

        roi = rectangle[y1:y2, x1:x2]
        mask = masks[y1:y2, x1:x2]
        tile_size = y2 - y1

        try:
            cv2.imwrite(
                os.path.join(roi_path, f'{rect_name}_ind_{individual_count}_coords_{y1}_{x1}_{tile_size}.bmp'),
                cv2.cvtColor(roi, cv2.COLOR_RGBA2BGR)
            )
            cv2.imwrite(
                os.path.join(masks_path, f'{rect_name}_ind_{individual_count}_coords_{y1}_{x1}_{tile_size}.bmp'),
                mask
            )
            saved_tiles.add(tile_key)
            individual_count += 1
            stats.ind_saved += 1
            if isolated:
                stats.ind_isolated += 1
            else:
                stats.ind_fallback += 1
        except cv2.error:
            print(f'ROI empty: {y1}, {x1}')

    # Phase 3: Normal cell tiles (for class balance)
    # Try to shift tiles away from pathology instead of just skipping
    normal_count = 0
    target_normals = max(greedy_count + individual_count, 5)

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
                os.path.join(roi_path, f'{rect_name}_norm_{normal_count}_coords_{y1}_{x1}_{tile_size}.bmp'),
                cv2.cvtColor(roi, cv2.COLOR_RGBA2BGR)
            )
            cv2.imwrite(
                os.path.join(masks_path, f'{rect_name}_norm_{normal_count}_coords_{y1}_{x1}_{tile_size}.bmp'),
                tile_mask
            )
            saved_tiles.add(tile_key)
            normal_count += 1
            stats.norm_saved += 1
            if shifted:
                stats.norm_shifted += 1
        except cv2.error:
            print(f'ROI empty: {y1}, {x1}')

    print(f'  {rect_name}: {greedy_count} grp, {individual_count} ind, {normal_count} norm')


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


def __crop_debug_dataset(rect_name, zoom_levels, rectangle, masks, roi_path, masks_path, slide_name, overlap: float = 0.0):
    """
    Crop dataset with optional overlap (debug mode with slide subdirectories).

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
                    os.makedirs(os.path.join(roi_path, slide_name), exist_ok=True)
                    os.makedirs(os.path.join(masks_path, slide_name), exist_ok=True)

                    # OpenSlide returns RGBA, convert to BGR for cv2.imwrite/imread compatibility
                    cv2.imwrite(os.path.join(roi_path, slide_name, f'{rect_name}_coords_{x}_{y}_{zoom}.bmp'), cv2.cvtColor(roi, cv2.COLOR_RGBA2BGR))
                    cv2.imwrite(os.path.join(masks_path, slide_name, f'{rect_name}_coords_{x}_{y}_{zoom}.bmp'), mask)
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
