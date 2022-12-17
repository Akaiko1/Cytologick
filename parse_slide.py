import config
import json

from parsers import mrsx

def main():

    mrsx.extract_atlas(config.CURRENT_SLIDE, 'rois.json', config.OPENSLIDE_PATH)

    with open('rois.json', 'r') as f:
        rois = json.load(f)
    
    rects = mrsx.extract_rects(rois)
    rect = rects[0]['rect 1']

    mrsx.extract_rect_regions(rect, config.CURRENT_SLIDE, 'rois.json', config.OPENSLIDE_PATH, CLASSES=config.LABELS)


if __name__ == '__main__':
    main()