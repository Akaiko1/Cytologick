from PyQt5.QtWidgets import QApplication, QWidget, QComboBox, QHBoxLayout, QLabel
from PyQt5.QtGui import QPixmap

import sys
import os

import config

with os.add_dll_directory(config.OPENSLIDE_PATH):
    import openslide

class Menu(QWidget):

    def __init__(self, parent):
        super().__init__()

        self.slide_list=[f for f in os.listdir(config.SLIDE_DIR) if '.mrxs' in f]
        self.parent = parent
        self.setWindowTitle("Выбор слайда")

        layout = QHBoxLayout()

        self.combo_box = QComboBox()
        self.combo_box.addItems(self.slide_list)
        self.combo_box.currentIndexChanged.connect(self.slide_selected)
        self.combo_box.activated.connect(self.slide_selected)

        layout.addWidget(self.combo_box)
        self.setLayout(layout)
    
    def slide_selected(self, i):
        self.parent.current_slide = openslide.OpenSlide(os.path.join(config.SLIDE_DIR, self.slide_list[i]))
        levels = self.parent.current_slide.level_dimensions

        self.parent.label.setText('\n'.join(map(str, levels)))

        width, height = levels[-1]
        slide = self.parent.current_slide.read_region((0, 0), len(levels) - 1, (width, height))
        slide.save('gui_temp.bmp', 'bmp')

        self.parent.image.setPixmap(QPixmap('gui_temp.bmp'))



class Viewer(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Demetra AI")
        self.appw, self.apph = 1000, 1000
        self.setGeometry(100, 100, self.appw, self.apph)
        self.slide_menu = Menu(self)
        self.current_slide = None

        layout = QHBoxLayout()

        self.image = QLabel()

        self.label = QLabel()
        self.label.setText('Слайд не выбран')

        layout.addWidget(self.image)
        layout.addWidget(self.label)
        self.setLayout(layout)

        self.show()
        self.slide_menu.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainView = Viewer()
    sys.exit(app.exec_())
