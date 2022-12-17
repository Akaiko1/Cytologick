from lxml import etree

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
