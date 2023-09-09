import os
import glob
import threading
import webbrowser

import config
import __web.deepzoom_server as deep
import __web.__get_slide_roi as gsr

from flask import Flask, redirect, render_template


def get_app():
    app = Flask(__name__)
    app.files = []
    app.threads = {}

    @app.route("/")
    def main_page():
        return render_template('menu.html', files=app.files, state=app.threads, exp_ip=config.IP_EXPOSED)

    @app.route("/<int:index>")
    def start_file_inspection(index=0):
        target_port = 5002 + int(index)

        if index not in app.threads.keys():
            preview_window = threading.Thread(name=f'demetra_{index}',
                                        target=deep.start_web,
                                        args=[app.files[index],
                                               gsr.get_slide_rois(app.files[index]),
                                                 index, target_port])
            preview_window.setDaemon(True)
            preview_window.start()
            app.threads[index] = preview_window
            webbrowser.open_new_tab(f"http://{config.IP_EXPOSED}:{target_port}")
            return render_template('menu.html', files=app.files, state=app.threads, exp_ip=config.IP_EXPOSED)
        else:
            webbrowser.open_new_tab(f"http://{config.IP_EXPOSED}:{target_port}")
            return render_template('menu.html', files=app.files, state=app.threads, exp_ip=config.IP_EXPOSED)

    return app


def run(slides_folder):
    app = get_app()
    app.files = glob.glob(os.path.join(slides_folder, '**', '*.mrxs'), recursive=True)
    app.run(host='0.0.0.0', port=5001, threaded=True)
