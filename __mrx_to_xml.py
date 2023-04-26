import glob
import json
import os

import xml.etree.ElementTree as ET
import config

from io import BytesIO


def __get_json_list(folder_path='temp'):
    return glob.glob(os.path.join(folder_path, '**', '*.json'), recursive=True)


def __get_slidename(json_path):
    return json_path.split(os.sep)[-1].rstrip('.json')


def __get_type(name):
    if 'rect' in name.lower():
        return'Rectangle'
    
    return 'Spline'


def __get_color(name):
    if 'rect' in name.lower():
        return '#F4FA58'

    if name in config.LABELS:
        return '#aa0000'

    return '#00aa00'


def __generate_asap_file(name, rois):
    asap = ET.Element('ASAP_Annotations')
    root = ET.SubElement(asap, 'Annotations')

    for roi in rois:
        annotation = ET.SubElement(root, 'Annotation',
                                        Name=roi['label'],
                                          Type=__get_type(roi['label']),
                                            PartOfGroup='None',
                                              Color=__get_color(roi['label']))

        coords = ET.SubElement(annotation, 'Coordinates')

        for idx, point in enumerate(roi['points']):
            ET.SubElement(coords, 'Coordinate', Order=str(idx), X=str(point[0]), Y=str(point[1]))

    compiled = ET.ElementTree(asap)
    compiled.write(f'{name}.xml', encoding='utf-8', xml_declaration=True)


def main(folder_path='temp'):

    for j_file in __get_json_list():
        name = __get_slidename(j_file)

        with open(j_file, 'r') as f:
            rois = json.load(f)
        
        __generate_asap_file(name, rois)


if __name__ == '__main__':
    main()
