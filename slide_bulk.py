import config

from parsers import mrsx

def main():
    mrsx.extract_all_slides(config.HDD_SLIDES, config.TEMP_FOLDER, config.OPENSLIDE_PATH, config.LABELS, zoom_levels=[256], debug=False)


if __name__ == '__main__':
    main()