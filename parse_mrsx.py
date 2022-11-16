import config
import json
import cv2
import os

import numpy as np

def main():
    OPENSLIDE_PATH = 'E:\\Github\\DemetraAI\\openslide\\bin'

    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide

    slide = openslide.OpenSlide(config.CURRENT_SLIDE)
    print(slide.level_count)

    with open('rois.json', 'r') as f:
        rois = json.load(f)
    
    if not os.path.exists('results'):
        os.mkdir('results')
    
    regions = get_regions(rois)

    for idx, region in enumerate(regions):
        name, points, max_x, max_y, min_x, min_y = region
        name = name.strip('?)')
        folder_name = os.path.join('results', name)

        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

        crop = slide.read_region((min_x, min_y), 0, (max_x - min_x, max_y - min_y))
        crop = np.asarray(crop)
        cv2.drawContours(crop, [points], 0, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(folder_name, f'{idx}_{name}_coords_{min_x}_{min_y}.bmp'),cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

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
    


if __name__ == '__main__':
    main()