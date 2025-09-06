import parsers.xml as demetra_xml
import parsers.helpers as hp

import config


def main():
    jsons = demetra_xml.all_asap_to_xml(config.HDD_SLIDES, config.TEMP_FOLDER)
    hp.markdown_data(jsons)


if __name__ == '__main__':
    main()
