import config

from parsers import mrsx

def main():
    mrsx.extract_all_slides(config.HDD_SLIDES, config.TEMP_FOLDER, config.OPENSLIDE_PATH, config.LABELS)


if __name__ == '__main__':
    main()