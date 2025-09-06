# Portions of this code are based on:
# deepzoom_server - Example web application for serving whole-slide images
# https://github.com/openslide/openslide-python/blob/main/examples/deepzoom/deepzoom_server.py

from io import BytesIO
from optparse import OptionParser
import os
import re

import cv2
import PIL
import shutil
import numpy as np
import __web.__get_slide_roi as gsr

from unicodedata import normalize
from flask import Flask, abort, make_response, render_template, url_for

OPENSLIDE_PATH = os.path.abspath('..\\DemetraAI\\openslide\\bin')
if hasattr(os, 'add_dll_directory'):
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

from openslide import ImageSlide, open_slide
from openslide.deepzoom import DeepZoomGenerator
from fpdf import FPDF, HTMLMixin

class MyFPDF(FPDF, HTMLMixin):
	pass

SLIDE_NAME = 'slide'

GLOBAL_PARAMS = dict(
    level=0,
    coords=[]
)


def __process_global(level, coords):
    global GLOBAL_PARAMS

    if level != GLOBAL_PARAMS['level']:
        GLOBAL_PARAMS['level'] = level
        GLOBAL_PARAMS['coords'] = []
    else:
        if coords not in GLOBAL_PARAMS['coords']:
            GLOBAL_PARAMS['coords'].append(coords)
        print(GLOBAL_PARAMS)


def create_app(slide_path, config=None, config_file=None):
    # Create and configure app
    app = Flask(__name__)
    app.drawing_list={}
    app.meta={}
    app.config.from_mapping(
        DEEPZOOM_SLIDE=slide_path,  # 'G:\Github\DemetraAI\current\slide-2022-09-12T15-38-25-R1-S2.mrxs',
        DEEPZOOM_FORMAT='jpeg',
        DEEPZOOM_TILE_SIZE=254,
        DEEPZOOM_OVERLAP=1,
        DEEPZOOM_LIMIT_BOUNDS=True,
        DEEPZOOM_TILE_QUALITY=75,
    )
    app.config.from_envvar('DEEPZOOM_TILER_SETTINGS', silent=True)
    if config_file is not None:
        app.config.from_pyfile(config_file)
    if config is not None:
        app.config.from_mapping(config)

    # Open slide
    slidefile = app.config['DEEPZOOM_SLIDE']
    if slidefile is None:
        raise ValueError('No slide file specified')
    config_map = {
        'DEEPZOOM_TILE_SIZE': 'tile_size',
        'DEEPZOOM_OVERLAP': 'overlap',
        'DEEPZOOM_LIMIT_BOUNDS': 'limit_bounds',
    }
    opts = {v: app.config[k] for k, v in config_map.items()}
    slide = open_slide(slidefile)
    app.slides = {SLIDE_NAME: DeepZoomGenerator(slide, **opts)}
    app.associated_images = []
    app.slide_properties = slide.properties
    for name, image in slide.associated_images.items():
        app.associated_images.append(name)
        slug = slugify(name)
        app.slides[slug] = DeepZoomGenerator(ImageSlide(image), **opts)
    try:
        mpp_x = slide.properties[openslide.PROPERTY_NAME_MPP_X]
        mpp_y = slide.properties[openslide.PROPERTY_NAME_MPP_Y]
        app.slide_mpp = (float(mpp_x) + float(mpp_y)) / 2
    except (KeyError, ValueError):
        app.slide_mpp = 0

    # Set up routes
    @app.route('/')
    def index():
        slide_url = url_for('dzi', slug=SLIDE_NAME)
        associated_urls = {
            name: url_for('dzi', slug=slugify(name)) for name in app.associated_images
        }
        return render_template(
            'slide-multipane.html',
            slide_url=slide_url,
            associated=associated_urls,
            properties=app.slide_properties,
            slide_mpp=app.slide_mpp,
            regions=app.drawing_list,
            details=app.render_list,
            meta=app.meta,
        )

    @app.route('/<slug>.dzi')
    def dzi(slug):
        format = app.config['DEEPZOOM_FORMAT']
        try:
            resp = make_response(app.slides[slug].get_dzi(format))
            resp.mimetype = 'application/xml'
            return resp
        except KeyError:
            # Unknown slug
            abort(404)

    @app.route('/<slug>_files/<int:level>/<int:col>_<int:row>.<format>')
    def tile(slug, level, col, row, format):

        # __process_global(level, (col, row))

        format = format.lower()
        if format != 'jpeg' and format != 'png':
            # Not supported by Deep Zoom
            abort(404)
        try:
            tile = np.array(app.slides[slug].get_tile(level, (col, row)))
            (tile_x, tile_y), level, (tile_w, tile_h) = app.slides[slug].get_tile_coordinates(level, (col, row))

            for _, drawing_stats in app.drawing_list.items():
                (_, _), rect, cnt = drawing_stats

                if level > 0:
                    coeff = slide.level_downsamples[level]
                    sized_tile_w, sized_tile_h = int(tile_w * coeff), int(tile_h * coeff)
                else:
                    sized_tile_w, sized_tile_h = tile_w, tile_h

                if __check_cnt_tile(sized_tile_w, sized_tile_h, tile_x, tile_y, rect[0]):
                    tile_cnt = __get_tile_cnt(level, tile_x, tile_y, cnt)
                    cv2.drawContours(tile, [tile_cnt], -1, (255, 0, 0), -1)

            tile = PIL.Image.fromarray(np.uint8(tile))
        except KeyError:
            # Unknown slug
            abort(404)
        except ValueError:
            # Invalid level or coordinates
            abort(404)
            
        buf = BytesIO()
        tile.save(buf, format, quality=app.config['DEEPZOOM_TILE_QUALITY'])
        resp = make_response(buf.getvalue())
        resp.mimetype = 'image/%s' % format
        return resp
    
    def __check_cnt_tile(tile_w, tile_h, tile_x, tile_y, rect):
        cnt_x, cnt_y, cnt_w, cnt_h = rect
        if (tile_x < cnt_x < tile_x + tile_w) and (tile_y < cnt_y < tile_y + tile_h):
            return True
        if (tile_x < cnt_x + cnt_w < tile_x + tile_w) and (tile_y < cnt_y + cnt_h < tile_y + tile_h):
            return True
        return False

    def __get_tile_cnt(level, tile_x, tile_y, cnt):
        tile_cnt = []
        for point in cnt:
            point_x, point_y = point[0][0] - tile_x, point[0][1] - tile_y
            if level > 0:
                point_x, point_y = int(point_x/slide.level_downsamples[level]), int(point_y/slide.level_downsamples[level])
            tile_cnt.append([[point_x, point_y]])

        tile_cnt = np.array(tile_cnt)
        return tile_cnt

    return app


def slugify(text):
    text = normalize('NFKD', text.lower()).encode('ascii', 'ignore').decode()
    return re.sub('[^a-z0-9]+', '-', text)


def start_web(slide_path, drawing_list, index, outside_port):
    parser = OptionParser(usage='Usage: %prog [options] [slide]')
    __fill_parser_default(parser)

    (opts, args) = parser.parse_args()
    config = {}
    config_file = opts.config
    # Set only those settings specified on the command line
    for k in dir(opts):
        v = getattr(opts, k)
        if not k.startswith('_') and v is not None:
            config[k] = v
    # Set slide file if specified
    try:
        config['DEEPZOOM_SLIDE'] = args[0]
    except IndexError:
        pass
    app = create_app(slide_path, config, config_file)
    app.drawing_list = drawing_list
    app.render_list = []
    slide = open_slide(slide_path)

    temp_index_path = os.path.join('__web', 'static', 'temp', str(index))
    __prepare_folders(temp_index_path)
    __render_images(drawing_list, index, app, slide, temp_index_path)
    __render_pdf(app, index, temp_index_path)

    app.run(host=opts.host, port=outside_port, threaded=True)

def __render_images(drawing_list, index, app, slide, temp_index_path):
    for key, entry in drawing_list.items():
        (_, _), rect, cnt = entry
        ooi_image_rgba = np.array(slide.read_region((rect[0][0], rect[0][1]), 0, (rect[0][2], rect[0][3])))
        ooi_image = cv2.cvtColor(ooi_image_rgba, cv2.COLOR_RGBA2BGR)
        cv2.imwrite(os.path.join(temp_index_path, f'{key}.jpg'), ooi_image)
        app.render_list.append((key, os.path.join('static', 'temp', str(index), f'{key}.jpg')))


def __render_pdf(app, index, save_folder):
    pdf = MyFPDF()
    pdf.add_page()
    # Try to use bundled Roboto if present; fallback to core font
    try:
        pdf.add_font('Roboto', '', 'Roboto-Regular.ttf', uni=True)
        pdf.set_font('Roboto', '', 14)
    except Exception:
        pdf.set_font('Helvetica', '', 14)
    
    offset = 0
    pdf.text(x=25, y=15, txt='Automatically generated report')
    pdf.text(x=25, y=30, txt=f'Findings total: {len(app.render_list)}')

    for name, image in app.render_list:
        # Check if adding another image will exceed the page height
        if (60 + int(offset * 35) + 35) > 290:  # assuming a page height of 297mm and a bottom margin
            pdf.add_page()
            offset = 0
        
        pdf.text(x=25, y=60 + int(offset * 35), txt=f'Finding name: {name}')
        pdf.image(os.path.join('__web', image), x=50, y=60 + int(offset * 35) + 5, w=25, h=25)
        offset += 1
        
    app.meta['PDF'] = os.path.join('static', 'temp', str(index), 'report.pdf')
    pdf.output(os.path.join(save_folder, 'report.pdf'))


def __prepare_folders(temp_index_path):
    if not os.path.exists(temp_index_path):
        os.makedirs(temp_index_path)
    else:
        shutil.rmtree(temp_index_path)
        os.makedirs(temp_index_path)

def __fill_parser_default(parser):
    parser.add_option(
        '-B',
        '--ignore-bounds',
        dest='DEEPZOOM_LIMIT_BOUNDS',
        default=True,
        action='store_false',
        help='display entire scan area',
    )
    parser.add_option(
        '-c', '--config', metavar='FILE', dest='config', help='config file'
    )
    parser.add_option(
        '-d',
        '--debug',
        dest='DEBUG',
        action='store_true',
        help='run in debugging mode (insecure)',
    )
    parser.add_option(
        '-e',
        '--overlap',
        metavar='PIXELS',
        dest='DEEPZOOM_OVERLAP',
        type='int',
        help='overlap of adjacent tiles [1]',
    )
    parser.add_option(
        '-f',
        '--format',
        metavar='{jpeg|png}',
        dest='DEEPZOOM_FORMAT',
        help='image format for tiles [jpeg]',
    )
    parser.add_option(
        '-l',
        '--listen',
        metavar='ADDRESS',
        dest='host',
        default='0.0.0.0',
        help='address to listen on [127.0.0.1]',
    )
    parser.add_option(
        '-p',
        '--port',
        metavar='PORT',
        dest='port',
        type='int',
        default=5001,
        help='port to listen on [5000]',
    )
    parser.add_option(
        '-Q',
        '--quality',
        metavar='QUALITY',
        dest='DEEPZOOM_TILE_QUALITY',
        type='int',
        help='JPEG compression quality [75]',
    )
    parser.add_option(
        '-s',
        '--size',
        metavar='PIXELS',
        dest='DEEPZOOM_TILE_SIZE',
        type='int',
        help='tile size [254]',
    )
