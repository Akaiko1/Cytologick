import glob
import os
import json
import pprint

import config
import xml.etree.ElementTree as ET


def main(temp_folder=config.TEMP_FOLDER, slides_folder=config.HDD_SLIDES_SVS):
    if not os.path.exists(temp_folder):
        os.mkdir(temp_folder)

    slides_list = glob.glob(os.path.join(slides_folder, '**', '*.svs'), recursive=True)
    print('Total slides: ', len(slides_list))

    xml_list = glob.glob(os.path.join(slides_folder, '**', '*.xml'), recursive=True)
    print('Total .xml files: ', len(xml_list))

    print(slides_list, xml_list)

    svs_report = {'Total objects': 0}

    for entry in xml_list:
        filename = os.path.splitext(os.path.basename(entry))[0]
        tree = ET.parse(entry)
        root = tree.getroot()
        annotations = root.find('Annotations')

        for child in annotations:
            name = child.get('Name')

            if 'rect' in name.lower():
                continue

            if name not in svs_report:
                svs_report[name] = 1
                svs_report['Total objects'] += 1
            else:
                svs_report[name] += 1
                svs_report['Total objects'] += 1
    
    pprint.pprint(svs_report)

    # Aggregate report for all XMLs in the folder.
    out_name = 'svs_report.json'
    with open(os.path.join(temp_folder, out_name), 'w') as f:
        json.dump(svs_report, f)


if __name__ == '__main__':
    main()
