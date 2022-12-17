from demetra.gui import *
import sys

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainView = Viewer()
    sys.exit(app.exec_())
