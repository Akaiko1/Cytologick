from PyQt5.QtWidgets import QApplication, QWidget, QComboBox, QHBoxLayout,\
     QVBoxLayout, QLabel, QScrollArea
from PyQt5.QtGui import QPixmap, QPalette

import sys
import os

import config

with os.add_dll_directory(config.OPENSLIDE_PATH):
    import openslide

class Menu(QWidget):

    def __init__(self, parent):
        super().__init__()

        self.slide_list=[f for f in os.listdir(config.SLIDE_DIR) if '.mrxs' in f]
        self.levels = []
        self.parent = parent
        self.setWindowTitle("Выбор слайда")

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
        width, height = self.levels[position]
        slide = self.parent.current_slide.read_region((0, 0), position, (width, height))
        slide.save('gui_temp.bmp', 'bmp')

        self.parent.image.setPixmap(QPixmap('gui_temp.bmp'))
        self.parent.image.adjustSize()


class Viewer(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Demetra AI")
        self.appw, self.apph = 800, 800
        self.setGeometry(100, 100, self.appw, self.apph)
        self.slide_menu = Menu(self)
        self.current_slide = None

        display = QHBoxLayout()

        self.image = QLabel()
        self.image.mousePressEvent = self.getPos

        self.scrollArea = QScrollArea()
        self.scrollArea.setWidget(self.image)

        self.label = QLabel()
        self.label.setText('Слайд не выбран')

        display.addWidget(self.scrollArea)
        display.addWidget(self.label)

        wrapping = QVBoxLayout()

        self.terminal = QLabel()

        wrapping.addLayout(display)
        wrapping.addWidget(self.terminal)

        self.setLayout(wrapping)

        self.show()
        self.slide_menu.show()
    
    def getPos(self , event):
        x = event.pos().x()
        y = event.pos().y()

        self.terminal.setText(f'click coordinates: {x}, {y}')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainView = Viewer()
    sys.exit(app.exec_())
