"""Application entry point for the HAGRID Industrial SOP Monitor GUI."""

from __future__ import annotations

import argparse
import os
import sys

# Ensure the project root and gui directory are importable when run directly.
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "gui"))


def main() -> None:
    parser = argparse.ArgumentParser(description="HAGRID Industrial SOP Monitor")
    parser.add_argument("-p", "--config", default="configs/gesture.yaml", help="Path to gesture YAML config")
    parser.add_argument("--source", type=int, default=0, help="Camera index")
    args = parser.parse_args()

    from qt_compat import QtWidgets, QT_BINDING
    from main_window import MainWindow

    # Remove the variable set by OpenCV to prevent conflicts with PyQt5
    if "QT_QPA_PLATFORM_PLUGIN_PATH" in os.environ:
        os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setApplicationName("HAGRID SOP Monitor")
    app.setStyle("Fusion")

    # Apply a clean Fusion palette.
    from qt_compat import QtGui

    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor("#f0f3f7"))
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor("#ffffff"))
    app.setPalette(palette)

    GLOBAL_QSS = """
    QWidget { font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif; color: #1f2933; }
    QMainWindow { background: #f0f3f7; }
    QStatusBar { background: #eef2f6; color: #5b6776; }
    QMenuBar { background: #1f2a44; color: #ffffff; }
    QMenuBar::item:selected { background: #2b3a63; }
    QMenu { background: #ffffff; border: 1px solid #d8dee4; }
    QMenu::item:selected { background: #eaf4ff; color: #1456a8; }
    QPushButton { border-radius: 6px; }
    QLineEdit, QComboBox, QSpinBox {
        background: #ffffff; border: 1px solid #d8dee4; border-radius: 6px; padding: 6px 8px;
    }
    QLineEdit:focus { border: 1px solid #2b7de9; }
    """
    app.setStyleSheet(GLOBAL_QSS)

    window = MainWindow(config_path=args.config, source=args.source)
    window.show()
    print(f"[main] using {QT_BINDING} backend")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
