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
    info = []
    for f in slides_list:
        base = os.path.splitext(os.path.basename(f))[0]
        matches = [x for x in xmls_list if base in x]
        if not matches:
            continue
        info.append((base, f, matches[0]))
    return info


def all_asap_to_xml(slides_folder: str, temp_folder='temp'):
    if not os.path.exists(temp_folder):
        os.mkdir(temp_folder)

    slides_list = glob.glob(os.path.join(slides_folder, '**', '*.mrxs'), recursive=True)
    print('Total slides: ', len(slides_list))

    xmls_list = glob.glob(os.path.join(slides_folder, '**', '*.xml'), recursive=True)
    xmls_list = [e for e in xmls_list if '_old' not in e]
    print('Total .xml files with markup data in .dat files: ', len(xmls_list))

    for slide, xml in zip(slides_list, xmls_list):
        print(slide, xml)

    slides_with_names = __get_info_tuples(slides_list, xmls_list)

    json_list = []
    for slide_name, _, xml_path in slides_with_names:
        print(f'Processing slide {slide_name} ... xml path: {xml_path}')
        nodes = get_asap_rois(xml_path)
        json_list.append((slide_name, nodes))

        with open(os.path.join(temp_folder, f'{slide_name}.json'), 'w') as f:
            json.dump(nodes, f)

    return json_list


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
        print(f'Processing slide {slide_name} ...')
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
        if len(line) < 100:
            continue
        if '<header>' in line:
            continue
        
        line = line.replace('<data>', '<node>')
        line = line.replace('</data>', '</node>')
        lines.append(line)

    lines.append('</data>')


    with open(filename, 'w') as f:
        f.writelines(lines)


def get_asap_rois(filename) -> list:
    parser = etree.XMLParser(recover=True)
    tree = etree.parse(filename, parser=parser)

    root = tree.getroot()
    annotations = root.find('Annotations')
    all_nodes = []

    for roi in annotations.findall('Annotation'):
        coords = roi.find('Coordinates')
        label = roi.get('Name')
        points = []

        for point in coords.findall('Coordinate'):
            points.append([int(float(point.get('X'))), int(float(point.get('Y')))])
        
        if len(points) < 3:
            print(f'Corrupted contour: {label}')
            continue
        
        points_xs, points_ys = [p[0] for p in points], [p[1] for p in points]

        if 'Annotation' not in label:
            all_nodes.append({
                'label': label,
                'points': points,
                'rect': [min(points_xs), min(points_ys), 
                         max(points_xs), max(points_ys)]
            })

    return all_nodes


def get_xml_rois(filename) -> list:
    parser = etree.XMLParser(recover=True)
    tree = etree.parse(filename, parser=parser)

    root = tree.getroot()
    all_nodes = []

    for node in root.findall('node'):
        content = node.getchildren()[1]
        bookmark = content.find('SimpleBookmark')
        slide_flag = content.find('slide_flag')

        if bookmark is None:  # TODO debug extraction
            continue

        label = bookmark.get('Caption')
        points = []

        xml_points = content.find('polygon_data')

        for point in xml_points.find('polygon_points').getchildren():
            points.append([point.get('polypointX'), point.get('polypointY')])
        
        if 'Annotation' not in label:
            all_nodes.append({
                'label': label,
                'points': points,
                'rect': [slide_flag.get('brLeft'), slide_flag.get('brTop'), 
                         slide_flag.get('brRight'), slide_flag.get('brBottom')]
            })

    return all_nodes
