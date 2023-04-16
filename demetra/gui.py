from PyQt5.QtWidgets import QApplication, QWidget, QComboBox, QHBoxLayout,\
     QVBoxLayout, QLabel, QScrollArea, QStackedLayout, QPushButton
from PyQt5.QtGui import QPixmap, QPalette, QPainter, QBrush, QPen
from PyQt5.QtCore import Qt

import cv2
import os

import config
import demetra.inference as inference
import demetra.contours as contours
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
        self.setMaximumSize(2000, 1200)
    
    def setPreviewLayout(self):
        main_layout = QHBoxLayout()

        display_layout = QVBoxLayout()
        display_widget = QWidget()
        display_widget.setMaximumWidth(200)
        display_widget.setLayout(display_layout)

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
        source = cv2.resize(source, (int(source.shape[1]/2), int(source.shape[0]/2)))

        match config.UNET_PRED_MODE:
            case 'direct':
                pathology_map = inference.apply_model(source, self.parent.model, shapes=(128, 128))
            case 'smooth':
                pathology_map = inference.apply_model_smooth(source, self.parent.model, shape=128)

        map_to_display = self.process_pathology_map(pathology_map)
        cv2.imwrite('gui_map.png', map_to_display)

        self.map = QPixmap('gui_map.png')
        self.display.repaint()

    def apply_colors(self, pathology_map):
        alpha_map = np.zeros((pathology_map.shape[0], pathology_map.shape[1], 4))

        alpha_map[:, :, 0] = np.where(pathology_map == 1, 255, 0)
        alpha_map[:, :, 2] = np.where(pathology_map == 2, 255, 0)
        alpha_map[:, :, 3] = np.where(pathology_map == 1, 50, alpha_map[:, :, 3])
        alpha_map[:, :, 3] = np.where(pathology_map == 2, 125, alpha_map[:, :, 3])
        return alpha_map
    
    def process_pathology_map(self, pathology_map):
        markup = np.zeros((pathology_map.shape + (4,)))
        stats = {
            'Всего объектов':0,
            'Групп':0,
            'Предупреждений':0,
            'Атипичных':0,
        }

        all_cells = np.where((pathology_map == 1) | (pathology_map == 2) , 1, 0)
        atypical_parts = np.where(pathology_map == 2, 1, 0)

        cell_contours = cv2.findContours(all_cells.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
        cell_contours = [c for c in cell_contours if cv2.contourArea(c) > 300]
        stats['Всего объектов'] = len(cell_contours)
        cv2.drawContours(markup, cell_contours, -1, (0, 255, 0), 2)

        single_cell_contours = [c for c in cell_contours if cv2.contourArea(c) <= 6500]
        cluster_contours = [c for c in cell_contours if cv2.contourArea(c) > 6500]
        stats['Групп'] = len(cluster_contours)

        atypical_contours = cv2.findContours(atypical_parts.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
        atypical_contours = [a for a in atypical_contours if cv2.contourArea(a) > 300]
        
        # atypical_contours = contours.smooth_contours(atypical_contours, shape=markup.shape[:2])
        
        self.process_cells(markup, stats,
                            single_cell_contours, atypical_contours)
        self.process_clusters(markup, stats,
                               cluster_contours, atypical_contours)
        self.set_info_text(stats)

        markup[:, :, 3] = np.where(markup[:, :, 2] > 0, 127, markup[:, :, 3])  # red alpha channel transparency
        markup[:, :, 3] = np.where(markup[:, :, 1] > 0, 200, markup[:, :, 3])  # green alpha channel transparency

        return markup

    def process_cells(self, markup, stats, single_cell_contours, atypical_contours):
        for cell in single_cell_contours:
            proportion = 0
            for a_cell in atypical_contours:
                if cv2.pointPolygonTest(cell, (int(a_cell[0][0][0]), int(a_cell[0][0][1])), False) <= 0:
                    continue

                proportion += int(cv2.contourArea(a_cell) * 100/cv2.contourArea(cell))

            if not proportion:
                continue

            stats['Атипичных'] += 1

            M = cv2.moments(cell)
            t_x, t_y = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

            cv2.drawContours(markup, [cell], -1, (0, 0, 255), -1)
            cv2.drawContours(markup, [cell], -1, (0, 125, 255), 2)
            cv2.putText(markup, f'{proportion} %', 
                        (t_x, t_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.55,
                        (0, 1, 0),
                        2,
                        -1)

    def process_clusters(self, markup, stats, cluster_contours, atypical_contours):
        for cell in cluster_contours:
            detection = False
            for a_cell in atypical_contours:
                if cv2.pointPolygonTest(cell, (int(a_cell[0][0][0]), int(a_cell[0][0][1])), False) <= 0:
                    continue
                
                detection = True
                cv2.drawContours(markup, [a_cell], -1, (0, 0, 255), -1)
            
            if detection:
                stats['Предупреждений'] += 1
                cv2.drawContours(markup, [cell], -1, (0, 255, 255), 2)

    def set_info_text(self, stats):
        text = ''
        for key, val in stats.items():
            text += f'{key}: {val}\n'
        self.info.setText(text)
    
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
        # set to < 5 because 5 or more will cause software to crash due to image size being too large
        self.level_select.addItems([str(e[0]) for e in enumerate(self.levels) if e[0] < 5])

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
    model = tf.keras.models.load_model('demetra_main', compile=False)

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
    