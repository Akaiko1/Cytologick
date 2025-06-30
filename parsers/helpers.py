def __get_names(data):
    return [i['label'] for i in data if 'rect' not in i['label'].lower()]


def __get_names_count(data):
    names = __get_names(data)

    result = {}
    for name in names:
        if name in result:
            result[name] += 1
        else:
            result[name] = 1

    return result


def markdown_data(jsons, md_file='stats.md'):
    markdown = '|Наименование слайда|Всего объектов (включая сторонние)|Тип объекта|Количество|\n'
    markdown += '|----------|:--------:|----------|:--------:|\n'

    cells_count = 0
    objects_total = {}

    for name, data in jsons:
        info = __get_names_count(data)
        cells_count += len(data)
        markdown += f'|{name}|{len(data)}|     |     |\n' # template = markdown += f'|     |     |     |     |\n'

        for obj, total in info.items():
            obj = obj.strip('?) ')

            if obj in objects_total:
                objects_total[obj] += total
            else:
                objects_total[obj] = total

            markdown += f'|     |     |{obj}|{total}|\n' 
    
    markdown += f'|  Всего клеток размечено:   |  {cells_count}   |     |    |\n' 
    markdown += f'|  Статистика   |  по   |  всем   |  слайдам  |\n' 
    for obj, total in objects_total.items(): 
        markdown += f'|     |     |{obj}|{total}|\n'
    
    with open(md_file, 'w') as f:
        f.writelines(markdown)