from clogic.gui import *
from qt_material import apply_stylesheet
import sys

if __name__ == '__main__':
    app = QApplication(sys.argv)
    apply_stylesheet(app, theme='dark_cyan.xml')
    mainView = Viewer()
    sys.exit(app.exec_())
