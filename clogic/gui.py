import glob
from PyQt5.QtWidgets import QApplication, QWidget, QComboBox, QHBoxLayout,\
     QVBoxLayout, QLabel, QScrollArea, QStackedLayout, QPushButton, QRadioButton, QSlider
from PyQt5.QtGui import QPixmap, QPalette, QPainter, QBrush, QPen
from PyQt5.QtCore import Qt
import socket
from urllib.parse import urlparse

import cv2
import os

import config
import clogic.contours as contours
import clogic.graphics as drawing

import numpy as np

# Import framework-specific modules based on config
if config.FRAMEWORK.lower() == 'pytorch':
    import clogic.inference_pytorch as inference
    import torch
else:
    import clogic.inference as inference
    import tensorflow as tf

if hasattr(os, 'add_dll_directory'):
    with os.add_dll_directory(config.OPENSLIDE_PATH):
        import openslide
else:
    import openslide


class Preview(QWidget):
    modes = ['smooth', 'direct', 'remote']
    mode_names = ['Local: Comprehensive', 'Local: Fast', 'Cloud: Fast']

    def __init__(self, parent, pixmap):
        super().__init__()

        self.parent = parent
        self.image = pixmap
        self.map = None
        self.conf_threshold = 0.6  # default 60%
        
        self.setWindowTitle("Preview")
        self.setPreviewLayout()
        self.setMaximumSize(2000, 1200)
    
    def _remote_available(self, timeout: float | None = None) -> bool:
        """Quick TCP reachability check for the configured remote endpoint.

        Uses the default endpoint from the TF inference module if available.
        """
        # Prefer endpoint from config
        endpoint = getattr(config, 'ENDPOINT_URL', None)
        if not endpoint:
            try:
                # Import here to avoid TF import when using PyTorch-only mode
                import clogic.inference as tf_inf
                endpoint = tf_inf.apply_remote.__defaults__[2] if len(tf_inf.apply_remote.__defaults__) >= 3 else 'http://127.0.0.1:8501'
            except Exception:
                endpoint = 'http://127.0.0.1:8501'

        try:
            parsed = urlparse(endpoint)
            host = parsed.hostname
            port = parsed.port or (443 if parsed.scheme == 'https' else 80)
            if not host:
                return False
            to = timeout if timeout is not None else getattr(config, 'HEALTH_TIMEOUT', 1.5)
            with socket.create_connection((host, port), timeout=to):
                return True
        except Exception:
            return False

    def setPreviewLayout(self):
        main_layout = QHBoxLayout()

        display_layout = QVBoxLayout()
        display_widget = QWidget()
        display_widget.setMaximumWidth(200)
        display_widget.setLayout(display_layout)

        remote_ok = self._remote_available()
        for idx, mode in enumerate(self.modes):
            if mode == 'remote' and not remote_ok:
                # Skip cloud option if endpoint not reachable
                continue
            
            # guard disabling offline modes
            if self.parent.model is None and 'remote' not in mode:
                continue

            radiobutton = QRadioButton(self.mode_names[idx])
            # Fallback to a local mode if remote is not available
            if config.UNET_PRED_MODE == 'remote' and not remote_ok and self.parent.model is not None:
                config.UNET_PRED_MODE = 'direct'
            if mode == config.UNET_PRED_MODE:
                radiobutton.setChecked(True)
            radiobutton.mode = mode
            radiobutton.toggled.connect(self.__modeSelected)
            display_layout.addWidget(radiobutton)

        self.display = QLabel()
        self.display.paintEvent = self.printImage

        self.info = QLabel()
        self.info.setText('Analysis \nResults')

        # Cloud status label
        self.cloud_status = QLabel()
        self.cloud_status.setText('Cloud: Available' if remote_ok else 'Cloud: Unavailable')
        self.cloud_status.setStyleSheet('color: #2ecc71;' if remote_ok else 'color: #e74c3c;')

        # Confidence slider controls
        self.conf_label = QLabel()
        self._set_conf_label()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setSingleStep(1)
        self.slider.setValue(int(self.conf_threshold * 100))
        self.slider.valueChanged.connect(self._conf_changed)

        self.button = QPushButton('Analyze')
        self.button.clicked.connect(self.runModel)

        display_layout.addWidget(self.cloud_status)
        display_layout.addWidget(self.conf_label)
        display_layout.addWidget(self.slider)
        display_layout.addWidget(self.info)
        display_layout.addWidget(self.button)
        main_layout.addWidget(self.display)
        main_layout.addWidget(display_widget)

        self.setLayout(main_layout)
        self.resize(self.image.width(), self.image.height())

    def _set_conf_label(self):
        self.conf_label.setText(f'Min confidence: {int(self.conf_threshold*100)}%')

    def _conf_changed(self, val):
        self.conf_threshold = float(val) / 100.0
        self._set_conf_label()
    
    def runModel(self):
        source = cv2.imread('gui_preview.bmp', 1)
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        source_resized = cv2.resize(source, (int(source.shape[1]/2), int(source.shape[0]/2)))

        if config.FRAMEWORK.lower() == 'pytorch':
            match config.UNET_PRED_MODE:
                case 'direct':
                    # Use probability map to compute per-lesion confidence
                    pathology_map = inference.apply_model_raw_pytorch(source_resized, self.parent.model, classes=config.CLASSES, shapes=(128, 128))
                case 'smooth':
                    # Smooth currently falls back to regular; still return probabilities
                    pathology_map = inference.apply_model_raw_pytorch(source_resized, self.parent.model, classes=config.CLASSES, shapes=(128, 128))
                case 'remote':
                    # Remote inference still uses TensorFlow serving
                    import clogic.inference as tf_inference
                    pathology_map = tf_inference.apply_remote(source)
        else:
            match config.UNET_PRED_MODE:
                case 'direct':
                    pathology_map = inference.apply_model(source_resized, self.parent.model, shapes=(128, 128))
                case 'smooth':
                    pathology_map = inference.apply_model_smooth(source_resized, self.parent.model, shape=128)
                case 'remote':
                    pathology_map = inference.apply_remote(source)

        map_to_display = self.process_pathology_map(pathology_map, config.UNET_PRED_MODE)
        cv2.imwrite('gui_map.png', map_to_display)

        self.map = QPixmap('gui_map.png')
        self.display.repaint()

    def process_pathology_map(self, pathology_map, mode):
        # If we received a probability map (H, W, C), use dense processing with threshold
        if isinstance(pathology_map, np.ndarray) and pathology_map.ndim == 3:
            markup, stats = drawing.process_dense_pathology_map(pathology_map, threshold=self.conf_threshold)
        else:
            markup, stats = drawing.process_sparse_pathology_map(pathology_map)

        self.set_info_text(stats)
        return markup

    def set_info_text(self, stats):
        # Dense stats: dict of lesion_idx -> prob, show summary
        if stats and all(isinstance(k, int) for k in stats.keys()):
            vals = list(stats.values())
            n = len(vals)
            avg = int(round((sum(vals) / max(1, n)) * 100))
            text = f'Detections: {n}\nAvg conf: {avg}%'
        else:
            text = ''
            for key, val in stats.items():
                text += f'{key}: {val}\n'
        self.info.setText(text)

    def __modeSelected(self):
        radioButton = self.sender()
        config.UNET_PRED_MODE = radioButton.mode
    
    def printImage(self, event):
        painter = QPainter(self.display)
        painter.drawPixmap(self.display.rect(), self.image)

        if self.map is not None:
            painter.drawPixmap(self.display.rect(), self.map)
        

class Menu(QWidget):

    def __init__(self, parent):
        super().__init__()

        self.slide_list=glob.glob(os.path.join(config.HDD_SLIDES, '**', '*.mrxs'), recursive=True)
        self.levels = []
        self.parent = parent

        self.setWindowTitle("Select Slide")
        self.setMenuLayout()

    def setMenuLayout(self):
        layout = QVBoxLayout()

        self.slide_select = QComboBox()
        self.slide_select.addItems(self.slide_list)
        self.slide_select.currentIndexChanged.connect(self.slide_selected)
        self.slide_select.activated.connect(self.slide_selected)

        self.level_select = QComboBox()
        self.level_select.currentIndexChanged.connect(self.level_selected)

        layout.addWidget(self.slide_select)
        layout.addWidget(self.level_select)

        self.setLayout(layout)
    
    def slide_selected(self, i):
        self.parent.current_slide = openslide.OpenSlide(os.path.join(config.SLIDE_DIR, self.slide_list[i]))
        self.levels = self.parent.current_slide.level_dimensions
        self.level_select.clear()
        # set to < 5 because 5 or more will cause software to crash due to image size being too large
        self.level_select.addItems([str(e[0]) for e in enumerate(self.levels) if e[0] < 4])

    def level_selected(self, i):
        if i < 0:
            return
        position = len(self.levels) - i - 1
        self.parent.prop = int(self.levels[0][0]/self.levels[position][0])  # passing coefficient for region previev
        width, height = self.levels[position]
        slide = self.parent.current_slide.read_region((0, 0), position, (width, height))
        slide.save('gui_temp.bmp', 'bmp')

        self.parent.slideImage = QPixmap('gui_temp.bmp')
        self.parent.image.resize(self.parent.slideImage.width(), self.parent.slideImage.height())
        self.parent.scrollArea.horizontalScrollBar().setValue(int(self.parent.image.width()/3))  # code moves the slider closer to center
        self.parent.scrollArea.verticalScrollBar().setValue(int(self.parent.image.height()/2.5))  # code moves the slider closer to center
        self.parent.image.repaint()


class Viewer(QWidget):
    doDraw = False
    drag = False
    slideImage = None
    model = None

    def __init__(self):
        super().__init__()

        self.appw, self.apph = 800, 800
        self.current_slide = None

        self.setWindowTitle("Cytologick")
        self.setGeometry(100, 100, self.appw, self.apph)
        self.setViewerLayout()
        self.show()

        self.slide_menu = Menu(self)
        self.slide_menu.show()

        self._load_local_model()

    def _load_local_model(self):
        """Load local model based on configured framework."""
        if config.FRAMEWORK.lower() == 'pytorch':
            self._load_pytorch_model()
        else:
            self._load_tensorflow_model()
    
    def _load_pytorch_model(self):
        """Load PyTorch model from _main folder."""
        model_files = [
            '_main/model.pth',
            '_main/model_best.pth', 
            '_main/model_final.pth'
        ]
        
        for model_path in model_files:
            if os.path.exists(model_path):
                print(f'Local PyTorch model located at {model_path}, loading')
                try:
                    self.model = inference.load_pytorch_model(model_path, config.CLASSES)
                    print('PyTorch model loaded successfully')
                    return
                except Exception as e:
                    print(f'Failed to load PyTorch model: {e}')
                    continue
        
        print('No PyTorch model found in _main/ folder')
        self.model = None
    
    def _load_tensorflow_model(self):
        """Load TensorFlow model from _main folder."""
        if os.path.exists('_main'):
            print('Local TensorFlow model located, loading')
            try:
                self.model = tf.keras.models.load_model('_main', compile=False)
                print('TensorFlow model loaded successfully')
            except Exception as e:
                print(f'Failed to load TensorFlow model: {e}')
                self.model = None
        else:
            print('No TensorFlow model found in _main/ folder')
            self.model = None

    def setViewerLayout(self):
        display = QHBoxLayout()

        self.image = QLabel()
        self.image.mousePressEvent = self.pressPos
        self.image.mouseMoveEvent = self.movePos
        self.image.mouseReleaseEvent = self.releasePos
        self.image.paintEvent = self.printRect

        self.scrollArea = QScrollArea()
        self.scrollArea.setWidget(self.image)

        display.addWidget(self.scrollArea)

        wrapping = QVBoxLayout()

        self.terminal = QLabel()
        self.terminal.setStyleSheet("background-color: black; color: white; padding: 5px")
        self.terminal.resize(25, 200)

        wrapping.addLayout(display)
        wrapping.addWidget(self.terminal)

        self.setLayout(wrapping)
    
    def showPreview(self):
        x, y, width, height = self.p_x * self.prop, self.p_y * self.prop, (self.r_x - self.p_x) * self.prop, (self.r_y - self.p_y) * self.prop

        if not width or not height:
            return

        if width < 0:
            x, width = x + width, abs(width)
        if height < 0:
            y, height = y + height, abs(height)

        width, height = drawing.get_corrected_size(width, height, config.IMAGE_CHUNK[0])
        region = self.current_slide.read_region((x, y), 0, (width, height))
        region.save('gui_preview.bmp', 'bmp')

        preview = Preview(self, QPixmap('gui_preview.bmp'))
        preview.show()
    
    def pressPos(self, event):
        self.p_x = event.pos().x()
        self.p_y = event.pos().y()

        self.drag = True

        self.terminal.setText(f'Click coordinates: {self.p_x}, {self.p_y}')

    def movePos(self, event):
        if not self.drag:
            return

        self.r_x = event.pos().x()
        self.r_y = event.pos().y()

        self.doDraw = True

        self.image.repaint()
    
    def releasePos(self , event):
        self.r_x = event.pos().x()
        self.r_y = event.pos().y()

        self.doDraw = False
        self.drag = False

        self.terminal.setText(f'Release coordinates: {self.r_x}, {self.r_y}')
        self.showPreview()
    
    def printRect(self, event):
        if self.slideImage is None:
            return
        
        painter = QPainter(self.image)
        painter.drawPixmap(self.image.rect(), self.slideImage)

        if not self.doDraw:
            return

        painter.setPen(QPen(Qt.black, 2, Qt.SolidLine))
        painter.setBrush(QBrush(Qt.green, Qt.DiagCrossPattern))
        painter.drawRect(self.p_x, self.p_y , self.r_x - self.p_x, self.r_y - self.p_y)
    
