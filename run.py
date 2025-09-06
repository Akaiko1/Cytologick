from clogic.gui import *
try:
    from qt_material import apply_stylesheet
except Exception:
    def apply_stylesheet(app, theme='dark_teal.xml', **kwargs):
        pass
import sys
import os

if __name__ == '__main__':
    app = QApplication(sys.argv)
    # Apply theme: 'auto' (platform QSS), 'qt' (qt_material), 'windows', 'mac', 'qdarkstyle', or 'qss'
    theme = getattr(config, 'GUI_THEME', 'auto')
    base_dir = os.path.dirname(os.path.abspath(__file__))
    styles_dir = os.path.join(base_dir, 'styles')

    def load_qss(path):
        try:
            with open(path, 'r') as f:
                app.setStyleSheet(f.read())
            return True
        except Exception:
            return False

    # Resolve theme path
    qss_path = None
    if theme == 'windows' or (theme == 'auto' and sys.platform.startswith('win')):
        qss_path = os.path.join(styles_dir, 'windows.qss')
    elif theme == 'mac' or (theme == 'auto' and sys.platform == 'darwin'):
        qss_path = os.path.join(styles_dir, 'mac.qss')

    applied = False
    # Try QSS for platform or explicit 'qss'
    if theme == 'qss':
        custom = getattr(config, 'GUI_CUSTOM_QSS', '')
        if custom and os.path.exists(custom):
            applied = load_qss(custom)
    elif qss_path and os.path.exists(qss_path):
        applied = load_qss(qss_path)

    # Try qdarkstyle if requested
    if not applied and theme == 'qdarkstyle':
        try:
            import qdarkstyle
            app.setStyleSheet(qdarkstyle.load_stylesheet())
            applied = True
        except Exception:
            applied = False

    # Fallback to qt_material
    if not applied:
        material = getattr(config, 'GUI_MATERIAL_THEME', 'dark_teal.xml')
        apply_stylesheet(app, theme=material)
    mainView = Viewer()
    sys.exit(app.exec_())
