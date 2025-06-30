import parsers.xml as demetra_xml
import parsers.helpers as hp

import config


def main():
    # That part is deprecated and commented out
    # jsons = demetra_xml.all_slides_to_xml(config.HDD_SLIDES, config.TEMP_FOLDER)
    # hp.markdown_data(jsons)

    jsons = demetra_xml.all_asap_to_xml('/Volumes/My Passport/CYTOLOGY', config.TEMP_FOLDER)
    hp.markdown_data(jsons)


if __name__ == '__main__':
    main()
