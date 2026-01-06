from clogic.gui import *
import sys
from PyQt5.QtWidgets import QStyleFactory

if __name__ == '__main__':
    app = QApplication(sys.argv)

    # Use Fusion style for consistent cross-platform appearance
    app.setStyle(QStyleFactory.create('Fusion'))

    mainView = Viewer()
    sys.exit(app.exec_())
