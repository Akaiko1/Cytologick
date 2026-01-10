from parsers import mrsx

from config import load_config

DEBUG=False

def main():
    cfg = load_config()
    mrsx.extract_all_slides(cfg.HDD_SLIDES, cfg.TEMP_FOLDER, cfg.OPENSLIDE_PATH, cfg.LABELS, zoom_levels=[256, 384, 512], debug=DEBUG, cfg=cfg)

    # extract_all_cells only needed for grid mode (not centered_crop)
    # In centered_crop mode, individual cells are already captured via _ind_ tiles
    if not cfg.CENTERED_CROP:
        mrsx.extract_all_cells(cfg.HDD_SLIDES, cfg.TEMP_FOLDER, cfg.OPENSLIDE_PATH, cfg.LABELS, debug=DEBUG, cfg=cfg)


if __name__ == '__main__':
    main()
