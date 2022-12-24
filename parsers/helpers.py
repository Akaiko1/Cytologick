def __get_names(data):
    return [list(i.keys())[0] for i in data if 'rect' not in list(i.keys())[0].lower()]


def __get_names_count(data):
    names = __get_names(data)

    result = {}
    for name in names:
        if name in result:
            result[name] += 1
        else:
            result[name] = 0

    return result


def markdown_data(jsons, md_file='stats.md'):
    markdown = '|Наименование слайда|Всего объектов (включая сторонние)|Тип объекта|Количество|\n'
    markdown += '|----------|:--------:|----------|:--------:|\n'

    for name, data in jsons:
        info = __get_names_count(data)
        markdown += f'|{name}|{len(data)}|     |     |\n' # template = markdown += f'|     |     |     |     |\n'

        for obj, total in info.items():
            markdown += f'|     |     |{obj}|{total}|\n' 
    
    with open(md_file, 'w') as f:
        f.writelines(markdown)