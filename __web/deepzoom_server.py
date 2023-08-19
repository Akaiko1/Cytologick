#!/usr/bin/env python
#
# deepzoom_server - Example web application for serving whole-slide images
#
# Copyright (c) 2010-2015 Carnegie Mellon University
#
# This library is free software; you can redistribute it and/or modify it
# under the terms of version 2.1 of the GNU Lesser General Public License
# as published by the Free Software Foundation.
#
# This library is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
# License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this library; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#

from io import BytesIO
from optparse import OptionParser
import os
import re

import cv2
import PIL
import numpy as np


from unicodedata import normalize

from flask import Flask, abort, make_response, render_template, url_for

OPENSLIDE_PATH = os.path.abspath('..\\DemetraAI\\openslide\\bin')
with os.add_dll_directory(OPENSLIDE_PATH):
    import openslide

from openslide import ImageSlide, open_slide
from openslide.deepzoom import DeepZoomGenerator

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


DRAWING_LIST = {
    'cnt_1': [[53844, 123139], [99, 99], [[[53844, 123139]], [[53944, 123239]], [[53874, 123179]], [[53844, 123139]]]]
}


def create_app(config=None, config_file=None):
    # Create and configure app
    app = Flask(__name__)
    app.config.from_mapping(
        DEEPZOOM_SLIDE=os.path.abspath('..\\DemetraAI\\current\\slide-2022-09-12T15-38-25-R1-S2.mrxs'),  # 'G:\Github\DemetraAI\current\slide-2022-09-12T15-38-25-R1-S2.mrxs',
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

            for _, drawing_stats in DRAWING_LIST.items():
                (_, _), (_, _), cnt = drawing_stats

                if level > 0:
                    tile_w, tile_h = int(tile_w/slide.level_downsamples[level]), int(tile_h/slide.level_downsamples[level])

                tile_cnt = __get_tile_cnt(level, tile_x, tile_y, cnt)
                cv2.drawContours(tile, [tile_cnt], -1, (255, 25 * level, 0), -1)

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


if __name__ == '__main__':
    parser = OptionParser(usage='Usage: %prog [options] [slide]')
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
        default='127.0.0.1',
        help='address to listen on [127.0.0.1]',
    )
    parser.add_option(
        '-p',
        '--port',
        metavar='PORT',
        dest='port',
        type='int',
        default=5000,
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
    app = create_app(config, config_file)

    app.run(host=opts.host, port=opts.port, threaded=True)
