"""
Web application entry point for Cytologick slide viewer.

This module provides a Flask-based web interface for:
- Browsing and selecting slide files
- Uploading new slide archives
- Launching per-slide deep zoom viewers

The web interface runs on port 5001 and spawns additional Flask
instances on ports 5002+ for each opened slide.
"""

import glob
import os
import shutil
import threading
import webbrowser

from flask import Flask, flash, redirect, render_template, request, url_for

import config
import __web.deepzoom_server as deep
import __web.__get_slide_roi as gsr


# Supported archive formats for upload
ARCHIVE_EXTENSIONS = ['.zip', '.rar']


def _refresh_slide_list(app: Flask) -> None:
    """
    Refresh the list of available slide files.
    
    Scans the configured slides folder for MRXS files and updates
    the app's file list.
    
    Args:
        app: Flask application instance with slides_folder attribute
    """
    app.files = glob.glob(
        os.path.join(app.slides_folder, '**', '*.mrxs'), 
        recursive=True
    )
    app.file_names = [
        os.path.splitext(os.path.basename(f))[0] 
        for f in app.files
    ]


def create_app(slides_folder: str) -> Flask:
    """
    Create and configure the main Flask application.
    
    Args:
        slides_folder: Path to directory containing slide files
        
    Returns:
        Configured Flask application instance
    """
    app = Flask(__name__)
    app.secret_key = 'cytologick-secret-key'  # Required for flash messages
    
    # Application state
    app.files = []          # List of slide file paths
    app.file_names = []     # Display names for slides
    app.threads = {}        # Mapping of slide index -> viewer thread
    app.slides_folder = slides_folder
    
    _refresh_slide_list(app)
    
    # -------------------------------------------------------------------------
    # Routes
    # -------------------------------------------------------------------------
    
    @app.route("/")
    def main_page():
        """Render the main slide selection page."""
        _refresh_slide_list(app)
        return render_template(
            'menu.html', 
            files=app.files,
            file_names=app.file_names,
            state=app.threads,
            exp_ip=config.IP_EXPOSED
        )
    
    @app.route('/upload', methods=['GET', 'POST'])
    def upload_file():
        """Handle slide file uploads (supports archives)."""
        if request.method != 'POST':
            print('Upload: Not a POST request')
            return redirect(request.url)
        
        # Validate file presence
        if 'file' not in request.files:
            flash('No file part in the request')
            return redirect(request.url)
        
        file = request.files.get('file')
        if not file or not file.filename:
            flash('No file selected')
            return redirect(url_for('main_page'))
        
        # Handle archive vs direct file upload
        is_archive = any(ext in file.filename.lower() for ext in ARCHIVE_EXTENSIONS)
        
        if is_archive:
            # Save temporarily, extract, then clean up
            temp_path = file.filename
            file.save(temp_path)
            try:
                shutil.unpack_archive(temp_path, app.slides_folder)
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        else:
            # Save directly to slides folder
            save_path = os.path.join(app.slides_folder, file.filename)
            file.save(save_path)
        
        _refresh_slide_list(app)
        return render_template(
            'menu.html',
            files=app.files,
            file_names=app.file_names,
            state=app.threads,
            exp_ip=config.IP_EXPOSED
        )
    
    @app.route("/<int:index>")
    def start_file_inspection(index: int = 0):
        """
        Launch or navigate to the deep zoom viewer for a slide.
        
        Each slide gets its own Flask instance running on a unique port.
        
        Args:
            index: Index into the slide file list
        """
        target_port = 5002 + index
        
        if index not in app.threads:
            # Start new viewer thread for this slide
            slide_path = app.files[index]
            rois = gsr.get_slide_rois(slide_path)
            
            viewer_thread = threading.Thread(
                name=f'cytologick_viewer_{index}',
                target=deep.start_web,
                args=[slide_path, rois, index, target_port],
                daemon=True  # Modern Python daemon thread syntax
            )
            viewer_thread.start()
            app.threads[index] = viewer_thread
        
        # Open browser tab for viewer
        viewer_url = f"http://{config.IP_EXPOSED}:{target_port}"
        webbrowser.open_new_tab(viewer_url)
        
        return render_template(
            'menu.html',
            files=app.files,
            file_names=app.file_names,
            state=app.threads,
            exp_ip=config.IP_EXPOSED
        )
    
    return app


def run(slides_folder: str, host: str = '0.0.0.0', port: int = 5001) -> None:
    """
    Start the web application server.
    
    Args:
        slides_folder: Path to directory containing slide files
        host: Host address to bind to (default: all interfaces)
        port: Port number to listen on (default: 5001)
    """
    app = create_app(slides_folder)
    print(f"Starting Cytologick web server on http://{host}:{port}")
    app.run(host=host, port=port, threaded=True)
