import parsers.xml as demetra_xml

import config
import json

def main():
    demetra_xml.extract_xml(config.CURRENT_SLIDE_XML, 'temp.xml')
    nodes = demetra_xml.get_xml_rois('temp.xml')

    with open('rois.json', 'w') as f:
        json.dump(nodes, f)


if __name__ == '__main__':
    main()