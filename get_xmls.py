import parsers.xml as demetra_xml
import parsers.helpers as hp

from config import load_config


def main():
    cfg = load_config()
    jsons = demetra_xml.all_asap_to_xml(cfg.HDD_SLIDES, cfg.TEMP_FOLDER)
    hp.markdown_data(jsons)


if __name__ == '__main__':
    main()
