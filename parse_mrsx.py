import config
import json
import math
import cv2
import os

import numpy as np


def extract_atlas(slidepath, jsonpath, OPENSLIDE_PATH, CLASSES={}):

    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide

    slide = openslide.OpenSlide(slidepath)

    with open(jsonpath, 'r') as f:
        rois = json.load(f)
    
    if not os.path.exists('atlas'):
        os.mkdir('atlas')
    
    # if not os.path.exists('masks'):
    #     os.mkdir('masks')
    
    regions = get_regions(rois)

    for idx, region in enumerate(regions):
        name, points, max_x, max_y, min_x, min_y = region
        name = name.strip('?)')

        folder_name = os.path.join('atlas', name)
        # masks_name = os.path.join('masks', name)

        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
        
        # if not os.path.exists(masks_name):
        #     os.mkdir(masks_name)

        crop = slide.read_region((min_x, min_y), 0, (max_x - min_x, max_y - min_y))
        crop = np.asarray(crop)

        # mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        # if name in CLASSES:
        #     cv2.drawContours(mask, [points], 0, int(CLASSES[name]), -1)
        # else:
        #     cv2.drawContours(mask, [points], 0, 1, -1)

        # TODO randomize name if file exists
        cv2.imwrite(os.path.join(folder_name, f'{idx}_{name}_coords_{min_x}_{min_y}.bmp'),cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        # cv2.imwrite(os.path.join(masks_name, f'{idx}_{name}_coords_{min_x}_{min_y}.bmp'),cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))

def extract_rect(rect, slidepath, jsonpath, OPENSLIDE_PATH, rect_name='roi', ZOOM_LEVELS=[128, 256, 512], CLASSES={}):

    with open(jsonpath, 'r') as f:
        rois = json.load(f)

    top, bot = rect
    regions = get_rect_regions(rois, top)

    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide

    slide = openslide.OpenSlide(slidepath)

    rectangle = np.asarray(slide.read_region(top, 0, (bot[0] - top[0], bot[1] - top[1])))
    masks = np.zeros(rectangle.shape[:2], dtype=np.uint8)

    if not os.path.exists('dataset'):
        os.mkdir('dataset')

    roi_path = os.path.join('dataset', 'rois')
    masks_path = os.path.join('dataset', 'masks')

    if not os.path.exists(roi_path):
        os.mkdir(roi_path)
    
    if not os.path.exists(masks_path):
        os.mkdir(masks_path)

    for region in regions:
        name, points, _, _, _, _ = region
        name = name.strip('?)')

        if name in CLASSES:
            cv2.drawContours(masks, [points], 0, int(CLASSES[name]), -1)
        else:
            cv2.drawContours(masks, [points], 0, 1, -1)

    for zoom in ZOOM_LEVELS:
        for x in range(0, rectangle.shape[0], zoom):
            for y in range(0, rectangle.shape[1], zoom):
                roi = rectangle[x: x + zoom, y: y + zoom]
                mask = masks[x: x + zoom, y: y + zoom]

                try:
                    cv2.imwrite(os.path.join(roi_path, f'{rect_name}_coords_{x}_{y}_{zoom}.bmp'), cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                    cv2.imwrite(os.path.join(masks_path, f'{rect_name}_coords_{x}_{y}_{zoom}.bmp'), mask)
                except cv2.error:
                    print('ROI empty: ', x, y)


def get_rects(rois):
    rects = []

    for roi in rois:
        for anno, points in roi.items():
            if 'rect' not in anno:
                continue

            points = [[int(p[0]), int(p[1])] for p in points]
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            max_x, min_x = max(xs), min(xs)
            max_y, min_y = max(ys), min(ys)

            rects.append({anno: [[min_x, min_y], [max_x, max_y]]})
    
    return rects


def get_rect_regions(rois, top_left):
    regions = []
    
    for roi in rois:
        for anno, points in roi.items():
            if 'rect' in anno:
                continue

            points = [[int(p[0]), int(p[1])] for p in points]
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            max_x, min_x = max(xs), min(xs)
            max_y, min_y = max(ys), min(ys)

            norm_points = [[[p[0] - top_left[0], p[1] - top_left[1]]] for p in points]
            regions.append((anno, np.asarray(norm_points), max_x, max_y, min_x, min_y))

    
    return regions


def get_regions(rois):
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


def main():
    # extract_dataset(config.CURRENT_SLIDE, 'rois.json', config.OPENSLIDE_PATH, {
    #     'Artifact': 1,
    #     'cylindrical': 2,
    #     'Normal superficial': 2
    # })
    extract_atlas(config.CURRENT_SLIDE, 'rois.json', config.OPENSLIDE_PATH, config.LABELS)

    with open('rois.json', 'r') as f:
        rois = json.load(f)
    
    rects = get_rects(rois)
    rect = rects[0]['rect 1']

    extract_rect(rect, config.CURRENT_SLIDE, 'rois.json', config.OPENSLIDE_PATH, CLASSES=config.LABELS)


if __name__ == '__main__':
    main()