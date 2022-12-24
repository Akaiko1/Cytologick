import json
import glob
import os

from lxml import etree


def __get_markup_files(list_to_filter):
    result = []
    for dat_file in list_to_filter:
        with open(dat_file, 'r') as f:
            filesize = os.path.getsize(dat_file)/1048576 
            if filesize > 100:
                continue
            try:
                content = f.read(5000)
                if '<polygon_point' in content and '<Diff' not in content:
                    result.append(dat_file)
            except UnicodeDecodeError:
                continue
    return result


def __get_info_tuples(slides_list, xmls_list):
    return [(f.split('\\')[-1].rstrip('.mrsx'), f, [x for x in xmls_list if f.split('\\')[-1].rstrip('.mrsx') in x][0]) for f in slides_list]


def all_slides_to_xml(slides_folder: str, temp_folder='temp') -> list[tuple]:
    if not os.path.exists(temp_folder):
        os.mkdir(temp_folder)

    slides_list = glob.glob(os.path.join(slides_folder, '**', '*.mrxs'), recursive=True)
    print('Total slides: ', len(slides_list))

    dat_list = glob.glob(os.path.join(slides_folder, '**', '*.dat'), recursive=True)
    print('Total .dat files: ', len(dat_list))

    xmls_list = __get_markup_files(dat_list)
    print('Total .xml files with markup data in .dat files: ', len(xmls_list))

    slides_with_names = __get_info_tuples(slides_list, xmls_list)

    json_list = []
    for slide_name, _, xml_path in slides_with_names:
        extract_xml(xml_path, os.path.join(temp_folder, f'{slide_name}.xml'))
        nodes = get_xml_rois(os.path.join(temp_folder, f'{slide_name}.xml'))
        json_list.append((slide_name, nodes))

        with open(os.path.join(temp_folder, f'{slide_name}.json'), 'w') as f:
            json.dump(nodes, f)

    return json_list

def extract_xml(slidename, filename) -> None: 
    
    lines = []
    with open(slidename, 'r') as f:
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


    with open(filename, 'w') as f:
        f.writelines(lines)


def get_xml_rois(filename) -> list:
    parser = etree.XMLParser(recover=True)
    tree = etree.parse(filename, parser=parser)

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
        
        if 'Annotation' not in label:
            all_nodes.append({
                f"{label}": points
            })

    return all_nodes
