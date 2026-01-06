import sys

from PyQt5.QtWidgets import QApplication, QStyleFactory

from config import load_config
from clogic.gui import Viewer

if __name__ == '__main__':
    app = QApplication(sys.argv)

    # Use Fusion style for consistent cross-platform appearance
    app.setStyle(QStyleFactory.create('Fusion'))

    cfg = load_config()
    mainView = Viewer(cfg)
    sys.exit(app.exec_())
