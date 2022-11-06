from pprint import pprint
from lxml import etree

import json

def extract_xml(): 
    lines = []
    with open('current\slide-2022-09-12T15-38-25-R1-S2\Data0021.dat', 'r') as f:
        content = f.readlines()

        lines.append('<data>')

        for line in content:
            if len(line) < 10:
                continue
            if '<header>' in line:
                continue
            
            line = line.replace('<data>', '<node>')
            line = line.replace('</data>', '</node>')
            lines.append(line)

        lines.append('</data>')


    with open('temp.xml', 'w') as f:
        f.writelines(lines)


def get_xml_rois():
    parser = etree.XMLParser(recover=True)
    tree = etree.parse('temp.xml', parser=parser)

    root = tree.getroot()
    all_nodes = []

    for node in root.findall('node'):
        content = node.getchildren()[1]
        bookmark = content.find('SimpleBookmark')
        label = bookmark.get('Caption')
        points = []

        xml_points = content.find('polygon_data')

        for point in xml_points.find('polygon_points').getchildren():
            points.append([point.get('polypointX'), point.get('polypointY')])

        all_nodes.append({
            f"{label}": points
        })

    return all_nodes


def main():
    extract_xml()
    nodes = get_xml_rois()

    with open('rois.json', 'w') as f:
        json.dump(nodes, f)


if __name__ == '__main__':
    main()