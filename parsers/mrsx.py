import json
import glob
import cv2
import os

import numpy as np


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


def extract_rects(rois):
    rects = []

    for roi in rois:
        for anno, points in roi.items():
            if 'rect' not in anno.lower():
                continue

            points = [[int(p[0]), int(p[1])] for p in points]
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            max_x, min_x = max(xs), min(xs)
            max_y, min_y = max(ys), min(ys)

            rects.append({anno: [[min_x, min_y], [max_x, max_y]]})
    
    return rects


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
        
        rects = extract_rects(rois)
        print(f'Parsing {slide}: {len(rects)} rectangles total')

        for rect in rects:
            rect_name, rect_coords= next(iter(rect.items()))

            top, bot = rect_coords
            if bot[0] - top[0] > 8000:
                print(f'{rect_name} larger than 8000px - skipping')
                continue
            
            extract_rect_regions(rect_coords, slide, json_path[0], openslide_path,
             classes=classes, zoom_levels=zoom_levels, rect_name=rect_name, debug=debug)


def extract_rect_regions(rect, slidepath, jsonpath, openslide_path, rect_name='roi', zoom_levels=[128, 256, 512], classes={}, debug=False):
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
    for region in regions:
        name, points, _, _, _, _ = region
        name = name.strip('?)')

        if name in classes:
            cv2.drawContours(masks, [points], 0, (0, 0, 255) if debug else int(classes[name]), -1)
        else:
            cv2.drawContours(masks, [points], 0, (0, 255, 0) if debug else 1, -1)


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


def __get_rect_regions(rois, top, bot):
    regions = []
    
    for roi in rois:
        for anno, points in roi.items():
            if 'rect' in anno.lower():
                continue

            points = [[int(p[0]), int(p[1])] for p in points]
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            max_x, min_x = max(xs), min(xs)
            max_y, min_y = max(ys), min(ys)

            norm_points = [[[p[0] - top[0], p[1] - top[1]]] for p in points
            if -100 < p[0] - top[0] < bot[0] - top[0] + 100
            and -100 < p[1] - top[1] < bot[1] - top[1] + 100]

            if not norm_points:
                continue

            regions.append((anno, np.asarray(norm_points), max_x, max_y, min_x, min_y))

    return regions


def __get_regions(rois):
    regions = []

    for roi in rois:
        for anno, points in roi.items():
            points = [[int(p[0]), int(p[1])] for p in points]
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            max_x, min_x = max(xs), min(xs)
            max_y, min_y = max(ys), min(ys)

            norm_points = [[[p[0] - min_x, p[1] - min_y]] for p in points]
            regions.append((anno, np.asarray(norm_points), max_x, max_y, min_x, min_y))

    return regions