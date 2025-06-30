import glob
from PyQt5.QtWidgets import QApplication, QWidget, QComboBox, QHBoxLayout,\
     QVBoxLayout, QLabel, QScrollArea, QStackedLayout, QPushButton, QRadioButton
from PyQt5.QtGui import QPixmap, QPalette, QPainter, QBrush, QPen
from PyQt5.QtCore import Qt

import cv2
import os

import config
import clogic.inference as inference
import clogic.contours as contours
import clogic.graphics as drawing

import numpy as np
import tensorflow as tf

if hasattr(os, 'add_dll_directory'):
    with os.add_dll_directory(config.OPENSLIDE_PATH):
        import openslide
else:
    import openslide


class Preview(QWidget):
    modes = ['smooth', 'direct', 'remote']
    mode_names = ['Локально: избыточно', 'Локально: быстрая', 'Облако: быстрая']

    def __init__(self, parent, pixmap):
        super().__init__()

        self.parent = parent
        self.image = pixmap
        self.map = None
        
        self.setWindowTitle("Предпросмотр")
        self.setPreviewLayout()
        self.setMaximumSize(2000, 1200)
    
    def setPreviewLayout(self):
        main_layout = QHBoxLayout()

        display_layout = QVBoxLayout()
        display_widget = QWidget()
        display_widget.setMaximumWidth(200)
        display_widget.setLayout(display_layout)

        for idx, mode in enumerate(self.modes):
            
            # guard disabling offline modes
            if self.parent.model is None and 'remote' not in mode:
                continue

            radiobutton = QRadioButton(self.mode_names[idx])
            if mode == config.UNET_PRED_MODE:
                radiobutton.setChecked(True)
            radiobutton.mode = mode
            radiobutton.toggled.connect(self.__modeSelected)
            display_layout.addWidget(radiobutton)

        self.display = QLabel()
        self.display.paintEvent = self.printImage

        self.info = QLabel()
        self.info.setText('Подсчёт \nрезультатов')

        self.button = QPushButton('Разметка')
        self.button.clicked.connect(self.runModel)

        display_layout.addWidget(self.info)
        display_layout.addWidget(self.button)
        main_layout.addWidget(self.display)
        main_layout.addWidget(display_widget)

        self.setLayout(main_layout)
        self.resize(self.image.width(), self.image.height())
    
    def runModel(self):
        source = cv2.imread('gui_preview.bmp', 1)
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        source_resized = cv2.resize(source, (int(source.shape[1]/2), int(source.shape[0]/2)))

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
        if mode in ['remote']:
            markup, stats = drawing.process_dense_pathology_map(pathology_map)
        else:
            markup, stats = drawing.process_sparse_pathology_map(pathology_map)

        self.set_info_text(stats)
        return markup

    def set_info_text(self, stats):
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

        self.setWindowTitle("Выбор слайда")
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

        self.setWindowTitle("Demetra AI")
        self.setGeometry(100, 100, self.appw, self.apph)
        self.setViewerLayout()
        self.show()

        self.slide_menu = Menu(self)
        self.slide_menu.show()

        if os.path.exists('_main'):
            print('Local model located, loading')
            self.model = tf.keras.models.load_model('_main', compile=False)

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
    