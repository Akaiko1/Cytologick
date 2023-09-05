import os
import glob
import threading
import webbrowser

import __web.deepzoom_server as deep
import __web.__get_slide_roi as gsr

from flask import Flask, redirect, render_template

app = Flask(__name__)
files = []

@app.route("/")
def main_page():
    return render_template('menu.html', files=files)

@app.route("/<int:index>")
def start_file_inspection(index=0):
    target_port = 5002 + int(index)

    drawing_list = gsr.get_slide_rois(files[index])

    t_webApp = threading.Thread(name='Web App', target=deep.start_web, args=[files[index], drawing_list, target_port])
    t_webApp.setDaemon(True)
    t_webApp.start()
    
    webbrowser.open_new_tab(f"http://127.0.0.1:{target_port}")

    return render_template('menu.html', files=files)

def run(slides_folder):
    global files    
    files = glob.glob(os.path.join(slides_folder, '**', '*.mrxs'), recursive=True)

    app.run(host='0.0.0.0', port=5001, threaded=True)