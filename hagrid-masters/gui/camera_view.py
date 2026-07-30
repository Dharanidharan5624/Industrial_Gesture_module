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
        self._scaled_cache: QtGui.QPixmap | None = None   # cached scaled pixmap
        self._cache_size: QtCore.QSize = QtCore.QSize()   # size for which cache is valid
        self._placeholder = True
        self._placeholder_msg = "Camera offline"
        self._placeholder_text_color = QtGui.QColor("#6b7785")
        self.setAutoFillBackground(True)
        bg = self.palette()
        bg.setColor(self.backgroundRole(), QtGui.QColor("#0d1117"))
        self.setPalette(bg)

    def update_frame(self, frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, w * 3, QtGui.QImage.Format_RGB888)
        new_pixmap = QtGui.QPixmap.fromImage(qimg)
        # Invalidate scale cache only when frame dimensions actually change
        if (self._pixmap is None or
                new_pixmap.width() != (self._pixmap.width() if self._pixmap else 0) or
                new_pixmap.height() != (self._pixmap.height() if self._pixmap else 0)):
            self._scaled_cache = None
        self._pixmap = new_pixmap
        self._placeholder = False
        self.update()

    def show_placeholder(self, message: str = "Camera offline") -> None:
        self._placeholder = True
        self._placeholder_msg = message
        self._pixmap = None
        self._scaled_cache = None
        self.update()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        """Invalidate the scale cache when the widget resizes."""
        self._scaled_cache = None
        super().resizeEvent(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor("#0d1117"))
        if self._pixmap is not None and not self._pixmap.isNull():
            # Rebuild scale cache only when widget size changed or pixmap changed
            if self._scaled_cache is None or self._cache_size != self.size():
                # FastTransformation is hardware-accelerated and imperceptible
                # at real-time framerates — SmoothTransformation is 3-5x slower.
                self._scaled_cache = self._pixmap.scaled(
                    self.size(), Qt.KeepAspectRatio, Qt.FastTransformation
                )
                self._cache_size = self.size()
            x = (self.width() - self._scaled_cache.width()) // 2
            y = (self.height() - self._scaled_cache.height()) // 2
            painter.drawPixmap(x, y, self._scaled_cache)
        else:
            painter.setPen(self._placeholder_text_color)
            painter.setFont(QtGui.QFont("Arial", 14))
            painter.drawText(self.rect(), Qt.AlignCenter, self._placeholder_msg)