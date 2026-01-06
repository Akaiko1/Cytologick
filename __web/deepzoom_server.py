"""
DeepZoom server for whole-slide image viewing.

This module provides a Flask-based deep zoom image server based on
OpenSlide's deepzoom example. It serves:
- DZI (Deep Zoom Image) metadata
- Individual tiles at various zoom levels
- Overlay drawings for detected regions of interest

Based on: https://github.com/openslide/openslide-python/blob/main/examples/deepzoom/deepzoom_server.py
"""

import os
import re
import shutil
from io import BytesIO
from optparse import OptionParser
from unicodedata import normalize
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np
import PIL
from flask import Flask, abort, make_response, render_template, url_for
from fpdf import FPDF, HTMLMixin

import config
import __web.__get_slide_roi as gsr

# OpenSlide import with DLL handling for Windows
OPENSLIDE_PATH = os.path.abspath(os.path.join('..', 'DemetraAI', 'openslide', 'bin'))
if hasattr(os, 'add_dll_directory'):
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

from openslide import ImageSlide, open_slide
from openslide.deepzoom import DeepZoomGenerator


# =============================================================================
# PDF Report Generator
# =============================================================================

class ReportPDF(FPDF, HTMLMixin):
    """PDF generator with HTML support for creating analysis reports."""
    pass


# =============================================================================
# Constants
# =============================================================================

SLIDE_NAME = 'slide'


# =============================================================================
# Helper Functions
# =============================================================================

def slugify(text: str) -> str:
    """
    Convert text to a URL-safe slug.
    
    Args:
        text: Input text to convert
        
    Returns:
        Lowercase alphanumeric slug with hyphens
    """
    text = normalize('NFKD', text.lower()).encode('ascii', 'ignore').decode()
    return re.sub('[^a-z0-9]+', '-', text)


def _check_contour_intersects_tile(
    tile_w: int, tile_h: int, 
    tile_x: int, tile_y: int, 
    rect: Tuple[int, int, int, int]
) -> bool:
    """
    Check if a contour bounding rect intersects with a tile region.
    
    Args:
        tile_w, tile_h: Tile dimensions
        tile_x, tile_y: Tile position in slide coordinates
        rect: Contour bounding rect (x, y, w, h)
        
    Returns:
        True if the contour intersects the tile
    """
    cnt_x, cnt_y, cnt_w, cnt_h = rect
    
    # Check if any corner of contour is in tile
    if (tile_x < cnt_x < tile_x + tile_w) and (tile_y < cnt_y < tile_y + tile_h):
        return True
    if (tile_x < cnt_x + cnt_w < tile_x + tile_w) and (tile_y < cnt_y + cnt_h < tile_y + tile_h):
        return True
    
    return False


def _transform_contour_to_tile(
    tile_x: int,
    tile_y: int,
    contour: List,
    scale_x: float,
    scale_y: float,
) -> np.ndarray:
    """
    Transform a contour from slide coordinates (level 0) to tile coordinates.

    Args:
        tile_x, tile_y: Tile position in slide coordinates (level 0)
        contour: Contour points in slide coordinates (level 0)
        scale_x, scale_y: Scale factors from slide coords to tile pixels

    Returns:
        Contour points in tile coordinates as numpy array
    """
    tile_contour = []

    for point in contour:
        point_x = (point[0][0] - tile_x) * scale_x
        point_y = (point[0][1] - tile_y) * scale_y
        tile_contour.append([[int(point_x), int(point_y)]])

    return np.array(tile_contour)


# =============================================================================
# Application Factory
# =============================================================================

def create_app(
    slide_path: str, 
    config_dict: Optional[Dict] = None, 
    config_file: Optional[str] = None
) -> Flask:
    """
    Create a Flask app for serving deep zoom tiles of a slide.
    
    Args:
        slide_path: Path to the slide file
        config_dict: Optional configuration overrides
        config_file: Optional path to configuration file
        
    Returns:
        Configured Flask application
    """
    app = Flask(__name__)
    
    # Initialize app state
    app.drawing_list = {}   # ROI contours to overlay
    app.render_list = []    # Pre-rendered ROI images
    app.meta = {}           # Metadata (PDF path, etc.)
    
    # Default DeepZoom configuration
    app.config.from_mapping(
        DEEPZOOM_SLIDE=slide_path,
        DEEPZOOM_FORMAT='jpeg',
        DEEPZOOM_TILE_SIZE=254,
        DEEPZOOM_OVERLAP=1,
        DEEPZOOM_LIMIT_BOUNDS=True,
        DEEPZOOM_TILE_QUALITY=75,
    )
    
    # Load additional configuration
    app.config.from_envvar('DEEPZOOM_TILER_SETTINGS', silent=True)
    if config_file is not None:
        app.config.from_pyfile(config_file)
    if config_dict is not None:
        app.config.from_mapping(config_dict)
    
    # Open the slide
    slidefile = app.config['DEEPZOOM_SLIDE']
    if slidefile is None:
        raise ValueError('No slide file specified')
    
    # Configure DeepZoom generator
    dz_opts = {
        'tile_size': app.config['DEEPZOOM_TILE_SIZE'],
        'overlap': app.config['DEEPZOOM_OVERLAP'],
        'limit_bounds': app.config['DEEPZOOM_LIMIT_BOUNDS'],
    }
    
    slide = open_slide(slidefile)
    app.slides = {SLIDE_NAME: DeepZoomGenerator(slide, **dz_opts)}
    app.associated_images = []
    app.slide_properties = slide.properties
    
    # Add associated images (thumbnail, label, etc.)
    for name, image in slide.associated_images.items():
        app.associated_images.append(name)
        slug = slugify(name)
        app.slides[slug] = DeepZoomGenerator(ImageSlide(image), **dz_opts)
    
    # Get microns per pixel for scale bar
    try:
        mpp_x = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
        mpp_y = float(slide.properties[openslide.PROPERTY_NAME_MPP_Y])
        app.slide_mpp = (mpp_x + mpp_y) / 2
    except (KeyError, ValueError):
        app.slide_mpp = 0
    
    # -------------------------------------------------------------------------
    # Routes
    # -------------------------------------------------------------------------
    
    @app.route('/')
    def index():
        """Render the main slide viewer page."""
        slide_url = url_for('dzi', slug=SLIDE_NAME)
        associated_urls = {
            name: url_for('dzi', slug=slugify(name)) 
            for name in app.associated_images
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
    def dzi(slug: str):
        """Serve DZI XML descriptor for an image."""
        format_ext = app.config['DEEPZOOM_FORMAT']
        try:
            resp = make_response(app.slides[slug].get_dzi(format_ext))
            resp.mimetype = 'application/xml'
            return resp
        except KeyError:
            abort(404)
    
    @app.route('/<slug>_files/<int:level>/<int:col>_<int:row>.<format>')
    def tile(slug: str, level: int, col: int, row: int, format: str):
        """
        Serve a single tile with optional ROI overlays.
        
        This route handles the core deep zoom tile serving with the addition
        of drawing detected regions of interest on tiles where they appear.
        """
        format = format.lower()
        if format not in ('jpeg', 'png'):
            abort(404)
        
        try:
            # Get the base tile
            tile_img = np.array(app.slides[slug].get_tile(level, (col, row)))
            (tile_x, tile_y), slide_level, (tile_w, tile_h) = app.slides[slug].get_tile_coordinates(level, (col, row))
            z_w, z_h = app.slides[slug].get_tile_dimensions(level, (col, row))
            
            # Calculate tile dimensions in slide level 0 coordinates
            if slide_level > 0:
                coeff = slide.level_downsamples[slide_level]
                sized_tile_w = int(tile_w * coeff)
                sized_tile_h = int(tile_h * coeff)
            else:
                sized_tile_w, sized_tile_h = tile_w, tile_h

            scale_x = (z_w / sized_tile_w) if sized_tile_w else 1.0
            scale_y = (z_h / sized_tile_h) if sized_tile_h else 1.0

            # Draw any overlapping contours
            for _, drawing_stats in app.drawing_list.items():
                (_, _), rect, contour = drawing_stats

                # Check if contour intersects this tile
                if _check_contour_intersects_tile(
                    sized_tile_w, sized_tile_h, tile_x, tile_y, rect[0]
                ):
                    tile_contour = _transform_contour_to_tile(
                        tile_x, tile_y, contour, scale_x, scale_y
                    )
                    cv2.drawContours(tile_img, [tile_contour], -1, (255, 0, 0), -1)
            
            tile_pil = PIL.Image.fromarray(np.uint8(tile_img))
            
        except KeyError:
            abort(404)
        except ValueError:
            abort(404)
        
        # Encode and return
        buf = BytesIO()
        tile_pil.save(buf, format, quality=app.config['DEEPZOOM_TILE_QUALITY'])
        resp = make_response(buf.getvalue())
        resp.mimetype = f'image/{format}'
        return resp
    
    return app


# =============================================================================
# Server Entry Point
# =============================================================================

def start_web(
    slide_path: str, 
    drawing_list: Dict, 
    index: int, 
    port: int
) -> None:
    """
    Start a deep zoom server for a specific slide.
    
    This function:
    1. Creates the Flask app for the slide
    2. Pre-renders ROI images
    3. Generates a PDF report
    4. Starts the Flask development server
    
    Args:
        slide_path: Path to the slide file
        drawing_list: Dictionary of detected ROIs to overlay
        index: Slide index (used for temp folder naming)
        port: Port to run the server on
    """
    # Parse command line options (for config compatibility)
    parser = OptionParser(usage='Usage: %prog [options] [slide]')
    _add_parser_options(parser)
    opts, args = parser.parse_args()
    
    # Build config dict from options
    config_dict = {}
    for key in dir(opts):
        value = getattr(opts, key)
        if not key.startswith('_') and value is not None:
            config_dict[key] = value
    
    # Override with explicit slide path
    config_dict['DEEPZOOM_SLIDE'] = slide_path
    
    # Create the app
    app = create_app(slide_path, config_dict, opts.config)
    app.drawing_list = drawing_list
    app.render_list = []
    
    # Open slide for image rendering
    slide = open_slide(slide_path)
    
    # Prepare output folder
    temp_folder = os.path.join('__web', 'static', 'temp', str(index))
    _prepare_output_folder(temp_folder)
    
    # Render ROI previews and PDF report
    _render_roi_images(drawing_list, index, app, slide, temp_folder)
    _render_pdf_report(app, index, temp_folder)
    
    # Start the server
    print(f"Starting slide viewer on port {port}")
    app.run(host=opts.host, port=port, threaded=True)


# =============================================================================
# Helper Functions for Rendering
# =============================================================================

def _prepare_output_folder(folder_path: str) -> None:
    """Create or clean the output folder."""
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)


def _render_roi_images(
    drawing_list: Dict, 
    index: int, 
    app: Flask, 
    slide: openslide.OpenSlide,
    output_folder: str
) -> None:
    """Render preview images for each detected ROI."""
    for key, entry in drawing_list.items():
        (_, _), rect, _ = entry
        x, y, w, h = rect[0]
        
        # Read the region from the slide
        roi_rgba = np.array(slide.read_region((x, y), 0, (w, h)))
        roi_bgr = cv2.cvtColor(roi_rgba, cv2.COLOR_RGBA2BGR)
        
        # Save preview image
        image_path = os.path.join(output_folder, f'{key}.jpg')
        cv2.imwrite(image_path, roi_bgr)
        
        # Add to render list for template
        static_path = os.path.join('static', 'temp', str(index), f'{key}.jpg')
        app.render_list.append((key, static_path))


def _render_pdf_report(app: Flask, index: int, output_folder: str) -> None:
    """Generate a PDF summary report of findings."""
    pdf = ReportPDF()
    pdf.add_page()
    
    # Try to use bundled font, fall back to Helvetica
    try:
        pdf.add_font('Roboto', '', 'Roboto-Regular.ttf', uni=True)
        pdf.set_font('Roboto', '', 14)
    except Exception:
        pdf.set_font('Helvetica', '', 14)
    
    # Report header
    pdf.text(x=25, y=15, txt='Automatically generated report')
    pdf.text(x=25, y=30, txt=f'Findings total: {len(app.render_list)}')
    
    # Add each finding
    PAGE_HEIGHT = 290  # Approximate usable page height in mm
    y_offset = 0
    
    for name, image_path in app.render_list:
        current_y = 60 + int(y_offset * 35)
        
        # Check if we need a new page
        if (current_y + 35) > PAGE_HEIGHT:
            pdf.add_page()
            y_offset = 0
            current_y = 60
        
        pdf.text(x=25, y=current_y, txt=f'Finding name: {name}')
        pdf.image(
            os.path.join('__web', image_path), 
            x=50, y=current_y + 5, w=25, h=25
        )
        y_offset += 1
    
    # Save PDF
    pdf_filename = 'report.pdf'
    pdf_path = os.path.join(output_folder, pdf_filename)
    pdf.output(pdf_path)
    
    # Store path for template access
    app.meta['PDF'] = os.path.join('static', 'temp', str(index), pdf_filename)


def _add_parser_options(parser: OptionParser) -> None:
    """Add command-line options for the deep zoom server."""
    parser.add_option(
        '-B', '--ignore-bounds',
        dest='DEEPZOOM_LIMIT_BOUNDS',
        default=True,
        action='store_false',
        help='display entire scan area',
    )
    parser.add_option(
        '-c', '--config', 
        metavar='FILE', 
        dest='config', 
        help='config file'
    )
    parser.add_option(
        '-d', '--debug',
        dest='DEBUG',
        action='store_true',
        help='run in debugging mode (insecure)',
    )
    parser.add_option(
        '-e', '--overlap',
        metavar='PIXELS',
        dest='DEEPZOOM_OVERLAP',
        type='int',
        help='overlap of adjacent tiles [1]',
    )
    parser.add_option(
        '-f', '--format',
        metavar='{jpeg|png}',
        dest='DEEPZOOM_FORMAT',
        help='image format for tiles [jpeg]',
    )
    parser.add_option(
        '-l', '--listen',
        metavar='ADDRESS',
        dest='host',
        default='0.0.0.0',
        help='address to listen on [0.0.0.0]',
    )
    parser.add_option(
        '-p', '--port',
        metavar='PORT',
        dest='port',
        type='int',
        default=5001,
        help='port to listen on [5001]',
    )
    parser.add_option(
        '-Q', '--quality',
        metavar='QUALITY',
        dest='DEEPZOOM_TILE_QUALITY',
        type='int',
        help='JPEG compression quality [75]',
    )
    parser.add_option(
        '-s', '--size',
        metavar='PIXELS',
        dest='DEEPZOOM_TILE_SIZE',
        type='int',
        help='tile size [254]',
    )
