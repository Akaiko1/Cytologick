from collections import namedtuple
import json
import glob
import math
import cv2
import os

import numpy as np

import config


Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')


def extract_atlas(slidepath, jsonpath, openslide_path):
    with os.add_dll_directory(openslide_path):
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

        # TODO randomize name if file exists
        cv2.imwrite(os.path.join(folder_name, f'{idx}_{name}_coords_{min_x}_{min_y}.bmp'),cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))


def extract_all_cells(slides_folder, json_folder, openslide_path, classes, debug=False):
    with os.add_dll_directory(openslide_path):
        import openslide

    slides_list = glob.glob(os.path.join(slides_folder, '**', '*.mrxs'), recursive=True)
    print('Total slides: ', len(slides_list))

    json_list = glob.glob(os.path.join(json_folder, '**', '*.json'), recursive=True)

    roi_path = os.path.join('dataset', 'rois')
    masks_path = os.path.join('dataset', 'masks')
    __make_dirs(roi_path, masks_path)

    for slidepath in slides_list:
        json_name = slidepath.split('\\')[-1].rstrip('.mrsx') + '.json'
        json_path = [f for f in json_list if json_name in f]

        slide = openslide.OpenSlide(slidepath)

        if not json_path:
            print(f'JSON for {slide} not found')
        
        with open(json_path[0], 'r') as f:
            rois = json.load(f)
        
        regions = __get_regions(rois)

        print(f'Parsing {slide}: {len(regions)} regions total')

        for _, region in enumerate(regions):
            name, points, max_x, max_y, min_x, min_y = region
            name = name.strip('?)')

            if 'rect' in name.lower():
                continue
            
            crop = slide.read_region((min_x, min_y), 0, (max_x - min_x, max_y - min_y))
            roi = np.asarray(crop)
            mask = np.zeros(roi.shape)

            __draw_other_rois(classes, debug, __get_rect_regions(rois, (min_x, min_y), (max_x, max_y)), name, mask)

            if name in classes:
                cv2.drawContours(mask, [points], 0, (0, 0, 255) if debug else int(classes[name]), -1)
            else:
                cv2.drawContours(mask, [points], 0, (0, 255, 0) if debug else 1, -1)

            cv2.imwrite(os.path.join(roi_path, f'{name}_coords_{max_x}_{max_y}.bmp'), cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            cv2.imwrite(os.path.join(masks_path, f'{name}_coords_{max_x}_{max_y}.bmp'), mask)


def extract_all_slides(slides_folder, json_folder, openslide_path, classes, zoom_levels=[256], debug=False):

    slides_list = glob.glob(os.path.join(slides_folder, '**', '*.mrxs'), recursive=True)
    print('Total slides: ', len(slides_list))

    json_list = glob.glob(os.path.join(json_folder, '**', '*.json'), recursive=True)

    for slide in slides_list:
        json_name = slide.split('\\')[-1].rstrip('.mrsx') + '.json'
        json_path = [f for f in json_list if json_name in f]

        if not json_path:
            print(f'JSON for {slide} not found')
        
        with open(json_path[0], 'r') as f:
            rois = json.load(f)
        
        rects = __extract_rects(rois)
        print(f'Parsing {slide}: {len(rects)} rectangles total')

        for rect in rects:
            rect_name, rect_coords= next(iter(rect.items()))

            top, bot = rect_coords
            if bot[0] - top[0] > 8000:
                print(f'{rect_name} larger than 8000px - skipping')
                continue
            
            __extract_rect_regions(rect_coords, slide, json_path[0], openslide_path,
             classes=classes, zoom_levels=zoom_levels, rect_name=rect_name, debug=debug)



def __extract_rect_regions(rect, slidepath, jsonpath, openslide_path, rect_name='roi', zoom_levels=[128, 256, 512], classes={}, debug=False):
    with os.add_dll_directory(openslide_path):
        import openslide
    
    slide = openslide.OpenSlide(slidepath)

    with open(jsonpath, 'r') as f:
        rois = json.load(f)

    top, bot = rect
    regions = __get_rect_regions(rois, top, bot)

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
        cv2.imwrite(f"{jsonpath.split(os.sep)[-1].replace('.json', '')}_{rect_name}.jpg", rectangle)
        cv2.imwrite(f"{jsonpath.split(os.sep)[-1].replace('.json', '')}_{rect_name}_mask.jpg", masks)

    __crop_dataset(rect_name, zoom_levels, rectangle, masks, roi_path, masks_path)


def __make_dirs(roi_path, masks_path):
    if not os.path.exists('dataset'):
        os.mkdir('dataset')
    
    if not os.path.exists(roi_path):
        os.mkdir(roi_path)
    
    if not os.path.exists(masks_path):
        os.mkdir(masks_path)


def __draw_masks(classes, debug, regions, masks):
    top_layer = []  # TODO priority should vary by level value
    for region in regions:
        name, points, _, _, _, _ = region
        name = name.strip('?) ')

        if name in classes.keys():
            top_layer.append(region)
            continue
        
        # print(f'Normal: {name}')
        cv2.drawContours(masks, [points], 0, (0, 255, 0) if debug else 1, -1)  # 1 class background info
    
    # double pass for class priority
    for region in top_layer:
        name, points, _, _, _, _ = region
        name = name.strip('?) ')
        # print(f'Atypical: {name}')
        cv2.drawContours(masks, [points], 0, (0, 0, 255) if debug else int(classes[name]), -1)


def __crop_dataset(rect_name, zoom_levels, rectangle, masks, roi_path, masks_path):
    for zoom in zoom_levels:
        for x in range(0, rectangle.shape[0], zoom):
            for y in range(0, rectangle.shape[1], zoom):
                roi = rectangle[x: x + zoom, y: y + zoom]
                mask = masks[x: x + zoom, y: y + zoom]

                if mask.shape[0] < int(zoom/3) or mask.shape[1] < int(zoom/3):  # skip small crops
                    continue

                try:
                    cv2.imwrite(os.path.join(roi_path, f'{rect_name}_coords_{x}_{y}_{zoom}.bmp'), cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                    cv2.imwrite(os.path.join(masks_path, f'{rect_name}_coords_{x}_{y}_{zoom}.bmp'), mask)
                except cv2.error:
                    print('ROI empty: ', x, y)


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


def __draw_other_rois(classes, debug, regions, name, mask):
    for other in regions:
        name, points, _, _, _, _ = other
        name = name.strip('?)')

        if 'rect' in name.lower():
            continue

        if name in classes:
            cv2.drawContours(mask, [points], 0, (0, 0, 255) if debug else int(classes[name]), -1)
        else:
            cv2.drawContours(mask, [points], 0, (0, 255, 0) if debug else 1, -1)


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


def __get_rect_regions(rois, top, bot):
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
        
        if config.EXCLUDE_DUPLICATES:
            close_idx = __index_close(((min_x, min_y), (max_x, max_y)), registry)
            if close_idx is not None:
                regions[close_idx] = None
                registry = [r for r in registry if r[0] != close_idx]

            registry.append((idx, (min_x, min_y), (max_x, max_y)))

        regions.append((anno, np.asarray(norm_points), max_x, max_y, min_x, min_y))

    return [r for r in regions if r is not None]


def __get_regions(rois, local_coords=True):
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

        if config.EXCLUDE_DUPLICATES:
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