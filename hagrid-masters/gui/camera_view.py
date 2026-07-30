"""Camera display widget that converts OpenCV frames to Qt pixmaps."""

from __future__ import annotations

import cv2
import numpy as np

from qt_compat import QtCore, QtGui, QtWidgets, Qt


class CameraView(QtWidgets.QWidget):
    """Renders BGR numpy frames, scaled to the widget size, letterboxed."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(480, 360)
        self._pixmap: QtGui.QPixmap | None = None
        self._placeholder = True
        self._placeholder_text_color = QtGui.QColor("#6b7785")
        self.setAutoFillBackground(True)
        bg = self.palette()
        bg.setColor(self.backgroundRole(), QtGui.QColor("#f1f5f9"))
        self.setPalette(bg)

    def update_frame(self, frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, w * 3, QtGui.QImage.Format_RGB888)
        self._pixmap = QtGui.QPixmap.fromImage(qimg).copy()
        self._placeholder = False
        self.update()

    def show_placeholder(self, message: str = "Camera offline") -> None:
        self._placeholder = True
        self._placeholder_msg = message
        self._pixmap = None
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QtGui.QColor("#0d1117"))
        if self._pixmap is not None and not self._pixmap.isNull():
            scaled = self._pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
        else:
            painter.setPen(self._placeholder_text_color)
            painter.setFont(QtGui.QFont("Arial", 14))
            painter.drawText(self.rect(), Qt.AlignCenter,
                             getattr(self, "_placeholder_msg", "Camera offline"))