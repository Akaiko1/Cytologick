import config

from parsers import mrsx

DEBUG=False

def main():
    mrsx.extract_all_slides(config.HDD_SLIDES, config.TEMP_FOLDER, config.OPENSLIDE_PATH, config.LABELS, zoom_levels=[256, 384, 512], debug=DEBUG)
    mrsx.extract_all_cells(config.HDD_SLIDES, config.TEMP_FOLDER, config.OPENSLIDE_PATH, config.LABELS, debug=DEBUG)


if __name__ == '__main__':
    main()
