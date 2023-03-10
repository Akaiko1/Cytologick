import config
import glob
import os

from pathlib import Path
from pprint import pprint

with os.add_dll_directory(config.OPENSLIDE_PATH):
    import openslide

PROPERTIES = [('Количество слоев в файле (максимум, минимум)', 'openslide.level-count'),
                ('Ширина изображения в пикселях (максимум, минимум)', 'openslide.level[0].width'),
                ('Высота изображения в пикселях (максимум, минимум)', 'openslide.level[0].height'),
                ('Разрешение (максимум, минимум)', 'openslide.mpp-x')]


def __fill_properties(report, slide_data):
    # for entry, val in slide_data.properties.items():
    #     print(entry, val)

    for entry, key in PROPERTIES:
        value = float(slide_data.properties[key])

        if entry not in report.keys():
            report[entry] = [value]
        else:
            report[entry].append(value)


def __pretty_bytes(size_in_bytes):
    size_in_bytes = float(size_in_bytes)
    Kilo = float(1024)
    Mega = float(Kilo ** 2)
    Giga = float(Kilo ** 3)
    Tera = float(Kilo ** 4)

    if size_in_bytes < Kilo:
        return '{0} {1}'.format(size_in_bytes,'Bytes' if 0 == size_in_bytes > 1 else 'Byte')
    elif Kilo <= size_in_bytes < Mega:
        return '{0:.2f} KB'.format(size_in_bytes / Kilo)
    elif Mega <= size_in_bytes < Giga:
        return '{0:.2f} MB'.format(size_in_bytes / Mega)
    elif Giga <= size_in_bytes < Tera:
        return '{0:.2f} GB'.format(size_in_bytes / Giga)
    elif Tera <= size_in_bytes:
        return '{0:.2f} TB'.format(size_in_bytes / Tera)


def main():
    slides_list = glob.glob(os.path.join(config.HDD_SLIDES, '**', '*.mrxs'), recursive=True)
    slide_names = [s.rstrip('.mrxs').split(os.sep)[-1] for s in slides_list]

    sizes = __get_sizes(slides_list)
    summary = __get_summary(slides_list, sizes)

    if not os.path.exists('meta'):
        os.mkdir('meta')

    for idx, slide_name in enumerate(slide_names):
        info = f'Наименование слайда: {slide_name}\nРазмер файла: {sizes[idx]}\n'

        for key, _ in PROPERTIES:
            info += f'{key.rstrip(" (максимум, минимум)")}: {summary[key][idx]}\n'

        with open(os.path.join('meta', f'{slide_name}.md'), 'w') as md_file:
            md_file.write(info)
        
    __print_summary(summary)


def __print_summary(summary):
    for key, _ in PROPERTIES:
        if key in summary:
            summary[key] = (max(summary[key]), min(summary[key]))

    pprint(summary)


def __get_summary(slides_list, sizes):
    summary = {
        'всего микропрепаратов размечено': len(slides_list),
        'размер файла (максимум, минимум, средний)': (__pretty_bytes(max(sizes)),
                                                       __pretty_bytes(min(sizes)),
                                                         __pretty_bytes((max(sizes) + min(sizes))/2) )
    }

    for slide in slides_list:
        slide_data = openslide.OpenSlide(slide)
        __fill_properties(summary, slide_data)
    return summary


def __get_sizes(slides_list):
    sizes = []
    for slide_path in slides_list:
        size = os.path.getsize(slide_path)
        info_size = sum(f.stat().st_size for f in Path(slide_path.rstrip('.mrxs')).glob('**/*') if f.is_file())
        sizes.append((size + info_size))
    return sizes


if __name__ == '__main__':
    main()
