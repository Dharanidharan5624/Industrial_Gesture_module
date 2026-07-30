"""Recent operations log table styled for the dark theme."""

from __future__ import annotations

import csv
import os
from typing import List

from qt_compat import QtCore, QtGui, QtWidgets

from constants import CSV_COLUMNS


class LogTableModel(QtCore.QAbstractTableModel):
    def __init__(self, rows: List[list] = None):
        super().__init__()
        self._rows = rows or []
        self._headers = CSV_COLUMNS

    def rowCount(self, parent=QtCore.QModelIndex()) -> int:
        return len(self._rows)

    def columnCount(self, parent=QtCore.QModelIndex()) -> int:
        return len(self._headers)

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid() or role != QtCore.Qt.DisplayRole:
            return None
        try:
            return str(self._rows[index.row()][index.column()])
        except (IndexError, KeyError):
            return None

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return self._headers[section]
        return None

    def set_rows(self, rows: List[list]) -> None:
        self.beginResetModel()
        self._rows = rows
        self.endResetModel()


class LogPanel(QtWidgets.QWidget):
    """Bottom panel showing recent operations from Supabase + local CSV."""

    def __init__(self, parent=None):
        super().__init__(parent)
        # Flexible height so Live Monitor panels keep usable space on smaller windows.
        self.setMinimumHeight(140)
        self.setMaximumHeight(240)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        self.is_dark = False
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        header_row = QtWidgets.QHBoxLayout()
        self.title = QtWidgets.QLabel("Recent Operations Log")
        self.title.setStyleSheet("font-size: 14px; font-weight: 700; color: #0f172a;")
        
        self.refresh_btn = QtWidgets.QPushButton("Refresh Logs")
        self.refresh_btn.setObjectName("secondaryBtn")
        self.refresh_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.refresh_btn.setStyleSheet("QPushButton { background-color: #fff; border: 1px solid #ccc; border-radius: 4px; padding: 5px 10px; } QPushButton:hover { background-color: #f0f0f0; }")
        
        self.clear_btn = QtWidgets.QPushButton("Clear Logs")
        self.clear_btn.setObjectName("dangerBtn")
        self.clear_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.clear_btn.setStyleSheet("QPushButton { background-color: #ef4444; color: #fff; border: none; border-radius: 4px; padding: 5px 10px; font-weight: 600; } QPushButton:hover { background-color: #dc2626; }")
        
        header_row.addWidget(self.title)
        header_row.addStretch(1)
        header_row.addWidget(self.refresh_btn)
        header_row.addWidget(self.clear_btn)
        layout.addLayout(header_row)
 
        self.table = QtWidgets.QTableView()
        self.model = LogTableModel()
        self.table.setModel(self.model)
        self.table.setAlternatingRowColors(False)
        self.table.setStyleSheet("QTableView { background-color: #fff; border: 1px solid #ccc; }")
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table)

        self.refresh_btn.clicked.connect(self.refresh)
        self.clear_btn.clicked.connect(self.clear_logs)
        self.refresh()

    def clear_logs(self) -> None:
        reply = QtWidgets.QMessageBox.question(
            self,
            "Clear Log Table",
            "Are you sure you want to clear all operation and compliance log entries?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        if reply == QtWidgets.QMessageBox.Yes:
            if os.path.exists("screw_monitoring_log.csv"):
                try:
                    with open("screw_monitoring_log.csv", "w", newline="") as fh:
                        writer = csv.writer(fh)
                        writer.writerow(CSV_COLUMNS)
                except Exception:
                    pass
            try:
                from compliance.logger import ComplianceLogger
                ComplianceLogger().clear_all_logs()
            except Exception:
                pass
            self.refresh()
            QtWidgets.QMessageBox.information(self, "Cleared", "Log entries cleared successfully.")

    def set_theme(self, is_dark: bool) -> None:
        self.is_dark = is_dark
        title_col = "#f8fafc" if is_dark else "#0f172a"
        tbl_bg = "#1e293b" if is_dark else "#ffffff"
        tbl_border = "#334155" if is_dark else "#ccc"
        tbl_col = "#e2e8f0" if is_dark else "#1e293b"
        btn_bg = "#1e293b" if is_dark else "#fff"
        btn_border = "#475569" if is_dark else "#ccc"
        btn_col = "#cbd5e1" if is_dark else "#333"
        self.title.setStyleSheet(f"font-size: 14px; font-weight: 700; color: {title_col};")
        self.table.setStyleSheet(f"QTableView {{ background-color: {tbl_bg}; color: {tbl_col}; border: 1px solid {tbl_border}; }}")
        self.refresh_btn.setStyleSheet(f"QPushButton {{ background-color: {btn_bg}; color: {btn_col}; border: 1px solid {btn_border}; border-radius: 4px; padding: 5px 10px; }}")

    def refresh(self) -> None:
        rows = self._load_rows()
        self.model.set_rows(rows)

    def _load_rows(self) -> List[list]:
        # Prefer Supabase; fall back to the local CSV.
        try:
            from supabase_client import fetch_logs
            data = fetch_logs(limit=50)
            if data:
                return [
                    [
                        row.get("timestamp", ""),
                        row.get("worker_id", ""),
                        row.get("action", ""),
                        row.get("object", ""),
                        row.get("tool", ""),
                        row.get("hand", ""),
                        row.get("direction", ""),
                        row.get("rotation_count", ""),
                        row.get("confidence", ""),
                        row.get("status", ""),
                        row.get("screenshot_path", ""),
                    ]
                    for row in data
                ]
        except Exception:
            pass
            
        # CSV fallback
        rows: List[list] = []
        if os.path.exists("screw_monitoring_log.csv"):
            try:
                with open("screw_monitoring_log.csv", newline="") as fh:
                    reader = csv.reader(fh)
                    next(reader, None)  # header
                    rows = list(reader)[-50:]
            except Exception:
                pass
        return rows
