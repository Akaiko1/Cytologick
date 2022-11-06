import json
import cv2
import os

import numpy as np

def main():
    OPENSLIDE_PATH = 'E:\\Github\\DemetraAI\\openslide\\bin'

    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide

    slide = openslide.OpenSlide('current\slide-2022-09-12T15-38-25-R1-S2.mrxs')
    print(slide.level_count)

    with open('rois.json', 'r') as f:
        rois = json.load(f)

    for roi in rois[173:]:
        for anno, points in roi.items():
            points = [[int(p[0]), int(p[1])] for p in points]
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            max_x, min_x = max(xs), min(xs)
            max_y, min_y = max(ys), min(ys)

            norm_points = [[[p[0] - min_x, p[1] - min_y]] for p in points]

            region = np.asarray(norm_points)

    for lvl in range(slide.level_count):
        crop = slide.read_region((min_x, min_y), lvl, (max_x - min_x, max_y - min_y))
        crop = np.asarray(crop)
        cv2.drawContours(crop, [region], 0, (0, 255, 0), 2)
        cv2.imwrite(f'p{lvl}.bmp', crop)

        # crop.save(f'p{lvl}.bmp')            


if __name__ == '__main__':
    main()