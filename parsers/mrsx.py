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
        # base_zoom is the middle value from zoom_levels, or first if only one
        base_zoom = zoom_levels[len(zoom_levels) // 2] if zoom_levels else 256
        min_zoom = min(zoom_levels) if zoom_levels else 128
        max_zoom = max(zoom_levels) if zoom_levels else 512
        __crop_dataset_centered(rect_name, base_zoom, rectangle, masks, roi_path, masks_path,
                                min_zoom=min_zoom, max_zoom=max_zoom)
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


def _try_greedy_tile(center_obj, all_objects, labeled, masks_shape,
                     base_zoom, min_zoom, max_zoom, margin=10):
    """
    Try to fit center object + neighbors into one tile.

    Returns:
        tuple: (y1, x1, y2, x2, set of included object ids) or None
    """
    h, w = masks_shape[:2]
    half = base_zoom // 2

    cy, cx = center_obj['cy'], center_obj['cx']
    y1, y2 = cy - half, cy + half
    x1, x2 = cx - half, cx + half

    # Find intersecting objects
    intersecting = [obj for obj in all_objects if _bbox_intersects_tile(obj, y1, x1, y2, x2)]

    # Single object case
    if len(intersecting) == 1:
        if y1 >= 0 and x1 >= 0 and y2 <= h and x2 <= w:
            if not _has_truncated_objects((y1, x1, y2, x2), labeled):
                return (y1, x1, y2, x2, {center_obj['id']})
        return None

    # Try to fit all intersecting objects by expanding tile
    combined_y1 = min(obj['y1'] for obj in intersecting) - margin
    combined_y2 = max(obj['y2'] for obj in intersecting) + margin
    combined_x1 = min(obj['x1'] for obj in intersecting) - margin
    combined_x2 = max(obj['x2'] for obj in intersecting) + margin

    needed_h = combined_y2 - combined_y1
    needed_w = combined_x2 - combined_x1
    tile_size = max(needed_h, needed_w)

    if tile_size <= max_zoom:
        # Can fit all — center tile on combined bbox
        center_y = (combined_y1 + combined_y2) // 2
        center_x = (combined_x1 + combined_x2) // 2
        half_tile = tile_size // 2

        new_y1 = max(0, min(center_y - half_tile, h - tile_size))
        new_x1 = max(0, min(center_x - half_tile, w - tile_size))
        new_y2 = new_y1 + tile_size
        new_x2 = new_x1 + tile_size

        if new_y2 <= h and new_x2 <= w:
            if not _has_truncated_objects((new_y1, new_x1, new_y2, new_x2), labeled):
                included = {obj['id'] for obj in intersecting}
                return (new_y1, new_x1, new_y2, new_x2, included)

    # Try shifting to exclude distant neighbors
    for dy in [0, -30, 30, -60, 60]:
        for dx in [0, -30, 30, -60, 60]:
            shifted_y1 = max(0, cy - half + dy)
            shifted_x1 = max(0, cx - half + dx)
            shifted_y2 = min(h, shifted_y1 + base_zoom)
            shifted_x2 = min(w, shifted_x1 + base_zoom)

            # Ensure correct size
            if shifted_y2 - shifted_y1 != base_zoom or shifted_x2 - shifted_x1 != base_zoom:
                continue

            if not _has_truncated_objects((shifted_y1, shifted_x1, shifted_y2, shifted_x2), labeled):
                included = {obj['id'] for obj in all_objects
                           if _bbox_fully_inside(obj, shifted_y1, shifted_x1, shifted_y2, shifted_x2)}
                if center_obj['id'] in included:
                    return (shifted_y1, shifted_x1, shifted_y2, shifted_x2, included)

    return None


def _fit_single_object(obj, labeled, masks_shape, base_zoom, min_zoom, max_zoom, margin=15):
    """
    Find optimal tile for a single object by varying size and position.

    Returns:
        tuple: (y1, x1, y2, x2) or None
    """
    h, w = masks_shape[:2]

    obj_h = obj['y2'] - obj['y1'] + 2 * margin
    obj_w = obj['x2'] - obj['x1'] + 2 * margin
    min_size = max(min_zoom, obj_h, obj_w)

    # Try sizes from minimum to base_zoom
    for zoom in range(min_size, min(base_zoom + 1, max_zoom + 1), 16):
        # Try different positions within valid range
        max_dy = zoom - obj_h
        max_dx = zoom - obj_w

        for dy in range(0, max(1, max_dy), max(1, max_dy // 5)):
            for dx in range(0, max(1, max_dx), max(1, max_dx // 5)):
                y1 = obj['y1'] - margin - dy
                x1 = obj['x1'] - margin - dx
                y2 = y1 + zoom
                x2 = x1 + zoom

                # Check bounds
                if y1 < 0 or x1 < 0 or y2 > h or x2 > w:
                    continue

                if not _has_truncated_objects((y1, x1, y2, x2), labeled):
                    return (y1, x1, y2, x2)

    return None


def __crop_dataset_centered(rect_name, base_zoom, rectangle, masks, roi_path, masks_path,
                            min_zoom=128, max_zoom=512, margin=15):
    """
    Crop dataset by centering tiles on pathology objects.

    Generates two types of tiles:
    1. Greedy tiles — fit multiple nearby pathologies into one tile (context)
    2. Individual tiles — one pathology per tile with optimal bbox (detail)

    Args:
        rect_name: Name prefix for saved files
        base_zoom: Target tile size
        rectangle: RGBA image from OpenSlide
        masks: Mask array
        roi_path: Output path for ROI images
        masks_path: Output path for mask images
        min_zoom: Minimum tile size
        max_zoom: Maximum tile size
        margin: Margin around pathology objects
    """
    labeled, objects = _get_pathology_objects(masks)

    if not objects:
        return

    h, w = masks.shape[:2]
    saved_tiles = set()  # Track saved tiles to avoid duplicates

    # Phase 1: Greedy tiles — group nearby pathologies
    used_in_greedy = set()
    greedy_count = 0

    for obj in objects:
        if obj['id'] in used_in_greedy:
            continue

        tile = _try_greedy_tile(obj, objects, labeled, masks.shape,
                                base_zoom, min_zoom, max_zoom, margin)

        if tile is not None:
            y1, x1, y2, x2, included_ids = tile
            tile_key = (y1, x1, y2, x2)

            if tile_key not in saved_tiles:
                roi = rectangle[y1:y2, x1:x2]
                mask = masks[y1:y2, x1:x2]

                zoom = y2 - y1
                try:
                    cv2.imwrite(
                        os.path.join(roi_path, f'{rect_name}_grp_{greedy_count}_coords_{y1}_{x1}_{zoom}.bmp'),
                        cv2.cvtColor(roi, cv2.COLOR_RGBA2BGR)
                    )
                    cv2.imwrite(
                        os.path.join(masks_path, f'{rect_name}_grp_{greedy_count}_coords_{y1}_{x1}_{zoom}.bmp'),
                        mask
                    )
                    saved_tiles.add(tile_key)
                    greedy_count += 1
                except cv2.error:
                    print(f'ROI empty: {y1}, {x1}')

            used_in_greedy.update(included_ids)

    # Phase 2: Individual tiles — each pathology separately
    individual_count = 0

    for obj in objects:
        tile = _fit_single_object(obj, labeled, masks.shape,
                                  base_zoom, min_zoom, max_zoom, margin)

        if tile is not None:
            y1, x1, y2, x2 = tile
            tile_key = (y1, x1, y2, x2)

            if tile_key not in saved_tiles:
                roi = rectangle[y1:y2, x1:x2]
                mask = masks[y1:y2, x1:x2]

                zoom = y2 - y1
                try:
                    cv2.imwrite(
                        os.path.join(roi_path, f'{rect_name}_ind_{individual_count}_coords_{y1}_{x1}_{zoom}.bmp'),
                        cv2.cvtColor(roi, cv2.COLOR_RGBA2BGR)
                    )
                    cv2.imwrite(
                        os.path.join(masks_path, f'{rect_name}_ind_{individual_count}_coords_{y1}_{x1}_{zoom}.bmp'),
                        mask
                    )
                    saved_tiles.add(tile_key)
                    individual_count += 1
                except cv2.error:
                    print(f'ROI empty: {y1}, {x1}')

    # Phase 3: Negative tiles — regions without pathology (normal cells + background)
    negative_count = 0
    num_pathology_tiles = greedy_count + individual_count
    # Generate roughly equal number of negative samples
    target_negatives = max(num_pathology_tiles, 5)
    max_attempts = target_negatives * 20

    for attempt in range(max_attempts):
        if negative_count >= target_negatives:
            break

        # Random position
        zoom = base_zoom
        if h <= zoom or w <= zoom:
            break

        y1 = np.random.randint(0, h - zoom)
        x1 = np.random.randint(0, w - zoom)
        y2 = y1 + zoom
        x2 = x1 + zoom

        tile_key = (y1, x1, y2, x2)
        if tile_key in saved_tiles:
            continue

        # Check no pathology in this tile
        tile_mask = masks[y1:y2, x1:x2]
        if np.any(tile_mask == MASK_VALUE_ABNORMAL):
            continue

        # Require at least some normal cells (not just empty background)
        normal_ratio = np.sum(tile_mask == MASK_VALUE_NORMAL) / tile_mask.size
        if normal_ratio < 0.01:  # At least 1% normal cells
            continue

        roi = rectangle[y1:y2, x1:x2]
        try:
            cv2.imwrite(
                os.path.join(roi_path, f'{rect_name}_neg_{negative_count}_coords_{y1}_{x1}_{zoom}.bmp'),
                cv2.cvtColor(roi, cv2.COLOR_RGBA2BGR)
            )
            cv2.imwrite(
                os.path.join(masks_path, f'{rect_name}_neg_{negative_count}_coords_{y1}_{x1}_{zoom}.bmp'),
                tile_mask
            )
            saved_tiles.add(tile_key)
            negative_count += 1
        except cv2.error:
            print(f'ROI empty: {y1}, {x1}')

    print(f'  {rect_name}: {greedy_count} group, {individual_count} individual, {negative_count} negative tiles')


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
