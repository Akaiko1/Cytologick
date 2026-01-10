import sys

# IMPORTANT: On Windows, PyTorch must be imported BEFORE PyQt5
# due to DLL loading order conflicts (fbgemm.dll / c10.dll)
import torch  # noqa: F401 - must be first

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
