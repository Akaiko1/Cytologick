from PyQt5.QtWidgets import QApplication, QWidget, QComboBox, QHBoxLayout,\
     QVBoxLayout, QLabel, QScrollArea, QStackedLayout, QPushButton
from PyQt5.QtGui import QPixmap, QPalette, QPainter, QBrush, QPen
from PyQt5.QtCore import Qt

import cv2
import os

import config
import demetra.inference as inference
import numpy as np
import tensorflow as tf

with os.add_dll_directory(config.OPENSLIDE_PATH):
    import openslide


class Preview(QWidget):

    def __init__(self, parent, pixmap):
        super().__init__()

        self.parent = parent
        self.image = pixmap
        self.map = None
        
        self.setWindowTitle("Предпросмотр")
        self.setPreviewLayout()
    
    def setPreviewLayout(self):
        layout = QVBoxLayout()

        self.display = QLabel()
        self.display.paintEvent = self.printImage

        self.button = QPushButton('Разметить HSIL, LSIL')
        self.button.clicked.connect(self.runModel)

        layout.addWidget(self.display)
        layout.addWidget(self.button)
        self.setLayout(layout)

        self.resize(self.image.width(), self.image.height())
    
    def runModel(self):
        source = cv2.imread('gui_preview.bmp', 1)
        source = cv2.resize(source, (int(source.shape[1]/2), int(source.shape[0]/2)))

        pathology_map = inference.apply_model(source, self.parent.model)
        alpha_map = self.apply_colors(pathology_map)

        cv2.imwrite('gui_map.png', alpha_map)
        self.map = QPixmap('gui_map.png')
        self.display.repaint()

    def apply_colors(self, pathology_map):
        alpha_map = np.zeros((pathology_map.shape[0], pathology_map.shape[1], 4))

        alpha_map[:, :, 0] = np.where(pathology_map == 1, 255, 0)
        alpha_map[:, :, 2] = np.where(pathology_map == 2, 255, 0)
        alpha_map[:, :, 3] = np.where(pathology_map == 1, 50, alpha_map[:, :, 3])
        alpha_map[:, :, 3] = np.where(pathology_map == 2, 125, alpha_map[:, :, 3])
        return alpha_map
    
    def printImage(self, event):
        painter = QPainter(self.display)
        painter.drawPixmap(self.display.rect(), self.image)

        if self.map is not None:
            painter.drawPixmap(self.display.rect(), self.map)
        

class Menu(QWidget):

    def __init__(self, parent):
        super().__init__()

        self.slide_list=[f for f in os.listdir(config.SLIDE_DIR) if '.mrxs' in f]
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
        self.level_select.addItems([str(e[0]) for e in enumerate(self.levels)])
        self.parent.label.setText('\n'.join(map(str, self.levels)))

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
        self.parent.image.repaint()


class Viewer(QWidget):
    doDraw = False
    drag = False
    slideImage = None
    model = tf.keras.models.load_model('demetra_main')

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

    def setViewerLayout(self):
        display = QHBoxLayout()

        self.image = QLabel()
        self.image.mousePressEvent = self.pressPos
        self.image.mouseMoveEvent = self.movePos
        self.image.mouseReleaseEvent = self.releasePos
        self.image.paintEvent = self.printRect

        self.scrollArea = QScrollArea()
        self.scrollArea.setWidget(self.image)

        self.label = QLabel()
        self.label.setText('Слайд не выбран')

        display.addWidget(self.scrollArea)
        display.addWidget(self.label)

        wrapping = QVBoxLayout()

        self.terminal = QLabel()
        self.terminal.setStyleSheet("background-color: black; color: white; padding: 5px")
        self.terminal.resize(25, 200)

        wrapping.addLayout(display)
        wrapping.addWidget(self.terminal)

        self.setLayout(wrapping)
    
    def showPreview(self):
        x, y, width, height = self.p_x * self.prop, self.p_y * self.prop, (self.r_x - self.p_x) * self.prop, (self.r_y - self.p_y) * self.prop
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
    