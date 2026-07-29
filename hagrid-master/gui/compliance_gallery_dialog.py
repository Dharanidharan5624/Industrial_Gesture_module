"""Compliance Monitor Screenshot Gallery Dialog.

Provides a rich visual gallery for viewing, previewing, and managing
all compliance evidence screenshots captured during safety violations.
Includes single screenshot deletion and batch 'Delete All' capabilities.
"""

from __future__ import annotations

import os
import glob
from datetime import datetime
from typing import List, Optional

from qt_compat import QtCore, QtGui, QtWidgets, Qt
from constants import COMPLIANCE_SCREENSHOT_DIR, LOG_DIR


class ImagePreviewDialog(QtWidgets.QDialog):
    """Full-resolution image preview modal for compliance screenshots."""

    def __init__(self, image_path: str, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.setWindowTitle(f"Screenshot Preview — {os.path.basename(image_path)}")
        self.setMinimumSize(850, 600)
        self.resize(1024, 720)
        self.setModal(True)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Header with filename and close button
        header = QtWidgets.QHBoxLayout()
        title_lbl = QtWidgets.QLabel(os.path.basename(self.image_path))
        title_lbl.setStyleSheet("font-size: 16px; font-weight: 700; color: #0f172a;")
        header.addWidget(title_lbl)
        header.addStretch(1)

        close_btn = QtWidgets.QPushButton("Close")
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #e2e8f0;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                padding: 6px 16px;
                font-weight: 600;
                color: #334155;
            }
            QPushButton:hover {
                background-color: #cbd5e1;
            }
        """)
        close_btn.clicked.connect(self.accept)
        header.addWidget(close_btn)
        layout.addLayout(header)

        # Image display container
        self.image_label = QtWidgets.QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        self.image_label.setStyleSheet(
            "background-color: #0f172a; border-radius: 8px; border: 1px solid #1e293b;"
        )

        pixmap = QtGui.QPixmap(self.image_path)
        if not pixmap.isNull():
            self._pixmap = pixmap
            self._update_scaled_pixmap()
        else:
            self.image_label.setText("Failed to load image screenshot.")
            self.image_label.setStyleSheet("color: #ef4444; font-size: 14px;")

        layout.addWidget(self.image_label, stretch=1)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "_pixmap") and not self._pixmap.isNull():
            self._update_scaled_pixmap()

    def _update_scaled_pixmap(self) -> None:
        target_size = self.image_label.size()
        if target_size.width() > 50 and target_size.height() > 50:
            scaled = self._pixmap.scaled(
                target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled)


class ComplianceGalleryDialog(QtWidgets.QDialog):
    """Grid gallery of all captured evidence screenshots with search & delete actions."""

    def __init__(self, parent=None, screenshot_dir: Optional[str] = None):
        super().__init__(parent)
        self.screenshot_dir = screenshot_dir or COMPLIANCE_SCREENSHOT_DIR
        self.setWindowTitle("Compliance Monitor Screenshot Gallery")
        self.setMinimumSize(950, 650)
        self.resize(1100, 750)
        self.setModal(True)
        self._image_paths: List[str] = []
        self._build_ui()
        self.reload_gallery()

    def _build_ui(self) -> None:
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)

        # 1. Top Header Bar
        top_bar = QtWidgets.QHBoxLayout()
        title_box = QtWidgets.QVBoxLayout()
        title_box.setSpacing(2)

        title = QtWidgets.QLabel("Compliance Evidence Screenshot Gallery")
        title.setStyleSheet("font-size: 20px; font-weight: 800; color: #0f172a;")
        sub = QtWidgets.QLabel(
            "Review and manage evidence screenshots captured during compliance events."
        )
        sub.setStyleSheet("font-size: 12px; color: #64748b;")
        title_box.addWidget(title)
        title_box.addWidget(sub)
        top_bar.addLayout(title_box)
        top_bar.addStretch(1)

        self.count_badge = QtWidgets.QLabel("Total: 0 Screenshots")
        self.count_badge.setStyleSheet("""
            QLabel {
                background-color: #f1f5f9;
                color: #334155;
                font-size: 12px;
                font-weight: 700;
                padding: 6px 14px;
                border-radius: 12px;
                border: 1px solid #cbd5e1;
            }
        """)
        top_bar.addWidget(self.count_badge)

        # Delete All Button
        self.delete_all_btn = QtWidgets.QPushButton("Delete All Screenshots")
        self.delete_all_btn.setCursor(Qt.PointingHandCursor)
        self.delete_all_btn.setStyleSheet("""
            QPushButton {
                background-color: #ef4444;
                border: 1px solid #dc2626;
                color: #ffffff;
                font-weight: 700;
                font-size: 12px;
                border-radius: 8px;
                padding: 7px 16px;
            }
            QPushButton:hover {
                background-color: #dc2626;
            }
            QPushButton:disabled {
                background-color: #fca5a5;
                border-color: #fca5a5;
            }
        """)
        self.delete_all_btn.clicked.connect(self.confirm_delete_all)
        top_bar.addWidget(self.delete_all_btn)

        # Refresh Button
        refresh_btn = QtWidgets.QPushButton("Refresh Gallery")
        refresh_btn.setCursor(Qt.PointingHandCursor)
        refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #2563eb;
                border: 1px solid #1d4ed8;
                color: #ffffff;
                font-weight: 700;
                font-size: 12px;
                border-radius: 8px;
                padding: 7px 16px;
            }
            QPushButton:hover {
                background-color: #1d4ed8;
            }
        """)
        refresh_btn.clicked.connect(self.reload_gallery)
        top_bar.addWidget(refresh_btn)

        main_layout.addLayout(top_bar)

        # 2. Filter & Search Bar
        filter_bar = QtWidgets.QFrame()
        filter_bar.setStyleSheet("""
            QFrame {
                background-color: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 10px;
                padding: 4px;
            }
        """)
        filt_layout = QtWidgets.QHBoxLayout(filter_bar)
        filt_layout.setContentsMargins(10, 6, 10, 6)
        filt_layout.setSpacing(10)

        filt_layout.addWidget(QtWidgets.QLabel("Search:"))
        self.search_input = QtWidgets.QLineEdit()
        self.search_input.setPlaceholderText("Filter by timestamp, event, or filename...")
        self.search_input.setStyleSheet("""
            QLineEdit {
                background-color: #ffffff;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                padding: 5px 10px;
                font-size: 12px;
                color: #0f172a;
            }
            QLineEdit:focus {
                border-color: #2563eb;
            }
        """)
        self.search_input.textChanged.connect(self._apply_filter)
        filt_layout.addWidget(self.search_input, stretch=1)

        main_layout.addWidget(filter_bar)

        # 3. Gallery Scroll Area
        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                border: 1px solid #e2e8f0;
                border-radius: 10px;
                background-color: #ffffff;
            }
        """)

        self.grid_container = QtWidgets.QWidget()
        self.grid_layout = QtWidgets.QGridLayout(self.grid_container)
        self.grid_layout.setContentsMargins(16, 16, 16, 16)
        self.grid_layout.setSpacing(16)
        self.grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        self.scroll_area.setWidget(self.grid_container)
        main_layout.addWidget(self.scroll_area, stretch=1)

        # 4. Bottom Close Bar
        bottom_bar = QtWidgets.QHBoxLayout()
        bottom_bar.addStretch(1)
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #f1f5f9;
                border: 1px solid #cbd5e1;
                color: #334155;
                font-weight: 700;
                font-size: 12px;
                border-radius: 8px;
                padding: 7px 20px;
            }
            QPushButton:hover {
                background-color: #e2e8f0;
            }
        """)
        close_btn.clicked.connect(self.accept)
        bottom_bar.addWidget(close_btn)
        main_layout.addLayout(bottom_bar)

    def scan_screenshots(self) -> List[str]:
        """Scan all evidence screenshot files sorted by newest first."""
        search_dirs = [
            self.screenshot_dir,
            os.path.join(LOG_DIR, "compliance_screenshots"),
            "compliance_screenshots",
            "logs/compliance_screenshots",
        ]
        found_files = set()
        for d in search_dirs:
            if d and os.path.exists(d):
                for ext in ("*.png", "*.jpg", "*.jpeg"):
                    for filepath in glob.glob(os.path.join(d, ext)):
                        found_files.add(os.path.abspath(filepath))

        # Sort by modification time (newest first)
        sorted_files = sorted(
            list(found_files),
            key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0,
            reverse=True,
        )
        return sorted_files

    def reload_gallery(self) -> None:
        """Reload all screenshots from disk and populate grid."""
        self._image_paths = self.scan_screenshots()
        self.count_badge.setText(f"Total: {len(self._image_paths)} Screenshots")
        self.delete_all_btn.setEnabled(len(self._image_paths) > 0)
        self._populate_grid(self._image_paths)

    def _apply_filter(self) -> None:
        query = self.search_input.text().strip().lower()
        if not query:
            filtered = self._image_paths
        else:
            filtered = [
                p for p in self._image_paths if query in os.path.basename(p).lower()
            ]
        self._populate_grid(filtered)

    def _clear_grid(self) -> None:
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def _populate_grid(self, image_list: List[str]) -> None:
        self._clear_grid()

        if not image_list:
            empty_lbl = QtWidgets.QLabel("No compliance evidence screenshots found.")
            empty_lbl.setAlignment(Qt.AlignCenter)
            empty_lbl.setStyleSheet(
                "font-size: 14px; font-weight: 600; color: #94a3b8; padding: 40px;"
            )
            self.grid_layout.addWidget(empty_lbl, 0, 0, 1, 3)
            return

        cols = 3
        for idx, img_path in enumerate(image_list):
            card = self._create_image_card(img_path)
            row = idx // cols
            col = idx % cols
            self.grid_layout.addWidget(card, row, col)

    def _create_image_card(self, img_path: str) -> QtWidgets.QFrame:
        card = QtWidgets.QFrame()
        card.setFixedSize(310, 260)
        card.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 10px;
            }
            QFrame:hover {
                border-color: #93c5fd;
                background-color: #f8fafc;
            }
        """)

        layout = QtWidgets.QVBoxLayout(card)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # Thumbnail Image Button
        thumb_btn = QtWidgets.QPushButton()
        thumb_btn.setFixedSize(290, 160)
        thumb_btn.setCursor(Qt.PointingHandCursor)
        thumb_btn.setStyleSheet(
            "border: none; background-color: #0f172a; border-radius: 6px;"
        )

        pixmap = QtGui.QPixmap(img_path)
        if not pixmap.isNull():
            scaled = pixmap.scaled(
                290, 160, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            icon = QtGui.QIcon(scaled)
            thumb_btn.setIcon(icon)
            thumb_btn.setIconSize(scaled.size())
        else:
            thumb_btn.setText("Image Error")

        thumb_btn.clicked.connect(lambda _, p=img_path: self.preview_image(p))
        layout.addWidget(thumb_btn)

        # File Meta Info
        fname = os.path.basename(img_path)
        try:
            mtime = os.path.getmtime(img_path)
            date_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            date_str = "Unknown Date"

        info_lbl = QtWidgets.QLabel(f"{fname}\n{date_str}")
        info_lbl.setStyleSheet(
            "font-size: 11px; font-weight: 600; color: #475569; border: none; background: transparent;"
        )
        info_lbl.setToolTip(img_path)
        layout.addWidget(info_lbl)

        # Action Buttons Row (View & Delete)
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(8)

        view_btn = QtWidgets.QPushButton("View")
        view_btn.setCursor(Qt.PointingHandCursor)
        view_btn.setStyleSheet("""
            QPushButton {
                background-color: #eff6ff;
                border: 1px solid #bfdbfe;
                color: #1d4ed8;
                font-weight: 700;
                font-size: 11px;
                border-radius: 6px;
                padding: 4px 12px;
            }
            QPushButton:hover {
                background-color: #dbeafe;
            }
        """)
        view_btn.clicked.connect(lambda _, p=img_path: self.preview_image(p))
        btn_row.addWidget(view_btn)

        del_btn = QtWidgets.QPushButton("Delete")
        del_btn.setCursor(Qt.PointingHandCursor)
        del_btn.setStyleSheet("""
            QPushButton {
                background-color: #fef2f2;
                border: 1px solid #fecaca;
                color: #dc2626;
                font-weight: 700;
                font-size: 11px;
                border-radius: 6px;
                padding: 4px 12px;
            }
            QPushButton:hover {
                background-color: #fee2e2;
            }
        """)
        del_btn.clicked.connect(lambda _, p=img_path: self.delete_single_screenshot(p))
        btn_row.addWidget(del_btn)

        layout.addLayout(btn_row)
        return card

    def preview_image(self, img_path: str) -> None:
        if os.path.exists(img_path):
            dlg = ImagePreviewDialog(img_path, parent=self)
            dlg.exec_()

    def delete_single_screenshot(self, img_path: str) -> None:
        """Delete a single evidence screenshot file."""
        fname = os.path.basename(img_path)
        reply = QtWidgets.QMessageBox.question(
            self,
            "Confirm Delete",
            f"Are you sure you want to delete this screenshot?\n\n{fname}",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if reply == QtWidgets.QMessageBox.Yes:
            try:
                if os.path.exists(img_path):
                    os.remove(img_path)
                self.reload_gallery()
            except Exception as err:
                QtWidgets.QMessageBox.critical(
                    self, "Error", f"Failed to delete screenshot:\n{err}"
                )

    def confirm_delete_all(self) -> None:
        """Delete all evidence screenshots with confirmation dialog."""
        total = len(self._image_paths)
        if total == 0:
            return

        reply = QtWidgets.QMessageBox.warning(
            self,
            "Delete All Screenshots",
            f"Are you sure you want to delete ALL {total} compliance screenshots?\n\n"
            "This action cannot be undone.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if reply == QtWidgets.QMessageBox.Yes:
            deleted_count = 0
            for img_path in self._image_paths:
                try:
                    if os.path.exists(img_path):
                        os.remove(img_path)
                        deleted_count += 1
                except Exception:
                    pass

            QtWidgets.QMessageBox.information(
                self,
                "Deleted All",
                f"Successfully deleted {deleted_count} compliance screenshots.",
            )
            self.reload_gallery()
