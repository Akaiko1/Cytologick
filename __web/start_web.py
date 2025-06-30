import os
import glob
from pprint import pprint
import threading
import webbrowser

import shutil

import config
import __web.deepzoom_server as deep
import __web.__get_slide_roi as gsr

from flask import Flask, flash, redirect, render_template, request, url_for


ARCHIVES = ['.zip', '.rar']


def set_names(app):
    app.files = glob.glob(os.path.join(app.slides_folder, '**', '*.mrxs'), recursive=True)
    app.file_names = [f.split(os.sep)[-1].split('.')[-2] for f in app.files]


def get_app(slides_folder: str):
    app = Flask(__name__)
    app.files = []
    app.threads = {}
    app.slides_folder = slides_folder
    set_names(app)


    @app.route("/")
    def main_page():
        set_names(app)
        return render_template('menu.html', files=app.files,
                                file_names=app.file_names,
                                  state=app.threads,
                                    exp_ip=config.IP_EXPOSED)
    

    @app.route('/upload', methods=['GET', 'POST'])
    def upload_file():
        if request.method == 'POST':
            if 'file' not in request.files:
                flash('Нет файла в приложении')
                return redirect(request.url)
            file = request.files.get('file')
            if any(ext in file.filename for ext in ARCHIVES):
                file.save(file.filename)
                shutil.unpack_archive(file.filename, app.slides_folder)
                os.remove(file.filename)
            else:
                file.save(os.path.join(app.slides_folder, file.filename))
            set_names(app)
            return render_template('menu.html', files=app.files,
                                    file_names=app.file_names,
                                      state=app.threads,
                                        exp_ip=config.IP_EXPOSED)
        else:
            print('Not a post request')
            return redirect(request.url)


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
        else:
            webbrowser.open_new_tab(f"http://{config.IP_EXPOSED}:{target_port}")

        return render_template('menu.html', files=app.files,
                                file_names=app.file_names,
                                    state=app.threads,
                                    exp_ip=config.IP_EXPOSED)

    return app


def run(slides_folder):
    app = get_app(slides_folder)
    app.run(host='0.0.0.0', port=5001, threaded=True)
