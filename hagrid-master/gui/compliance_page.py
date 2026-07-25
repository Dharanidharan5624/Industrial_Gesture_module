"""Supervisor Compliance Monitor page — live status, history, analytics export."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from qt_compat import QtCore, QtGui, QtWidgets, Qt

from compliance.logger import ComplianceLogger
try:
    from compliance_gallery_dialog import ComplianceGalleryDialog
except ImportError:
    from gui.compliance_gallery_dialog import ComplianceGalleryDialog
from constants import (
    COMPLIANCE_EVENT_BLUETOOTH,
    COMPLIANCE_EVENT_EARBUDS,
    COMPLIANCE_EVENT_MOBILE_PHONE,
    COMPLIANCE_EVENT_PASSED,
    COMPLIANCE_EVENT_SHIRT_BUTTON,
    COMPLIANCE_EVENT_SPECTACLES,
    COMPLIANCE_EVENT_WRITING,
)


class CompliancePage(QtWidgets.QWidget):
    """Full-page supervisor view for operator compliance."""

    def __init__(self, logger: Optional[ComplianceLogger] = None, parent=None):
        super().__init__(parent)
        self.setObjectName("page")
        self.logger = logger or ComplianceLogger()
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        title = QtWidgets.QLabel("Operator Compliance & Safety Monitor")
        title.setStyleSheet("font-size: 22px; font-weight: 800; color: #0f172a;")
        sub = QtWidgets.QLabel(
            "Real-time PPE / discipline status, warnings, and compliance history."
        )
        sub.setStyleSheet("color: #64748b; font-size: 12px;")
        layout.addWidget(title)
        layout.addWidget(sub)

        # KPI row
        kpi = QtWidgets.QHBoxLayout()
        kpi.setSpacing(12)
        self.kpi_warnings = self._kpi("Warnings", "0", "#dc2626")
        self.kpi_phone = self._kpi("Phone Alerts", "0", "#ea580c")
        self.kpi_shirt = self._kpi("Dress Code", "0", "#d97706")
        self.kpi_buds = self._kpi("Earbuds / BT", "0", "#7c3aed")
        self.kpi_pct = self._kpi("Compliance %", "100%", "#16a34a")
        for w in (self.kpi_warnings, self.kpi_phone, self.kpi_shirt, self.kpi_buds, self.kpi_pct):
            kpi.addWidget(w)
        layout.addLayout(kpi)

        # Live status grid
        status_card = QtWidgets.QFrame()
        status_card.setObjectName("card")
        s_lay = QtWidgets.QVBoxLayout(status_card)
        s_lay.setContentsMargins(16, 16, 16, 16)
        s_title = QtWidgets.QLabel("Live Detector Status")
        s_title.setStyleSheet("font-size: 15px; font-weight: 700; color: #0f172a;")
        s_lay.addWidget(s_title)
        grid = QtWidgets.QGridLayout()
        grid.setSpacing(10)
        self.live_labels: Dict[str, QtWidgets.QLabel] = {}
        items = [
            ("phone", "Mobile Phone"),
            ("shirt_button_open", "Shirt Button"),
            ("earbuds", "Earbuds"),
            ("bluetooth", "Bluetooth"),
            ("spectacles", "Spectacles"),
            ("writing", "Writing Activity"),
        ]
        for i, (key, label) in enumerate(items):
            box = QtWidgets.QFrame()
            box.setObjectName("card")
            bl = QtWidgets.QVBoxLayout(box)
            bl.setContentsMargins(12, 10, 12, 10)
            t = QtWidgets.QLabel(label)
            t.setStyleSheet("font-size: 11px; font-weight: 700; color: #64748b;")
            v = QtWidgets.QLabel("OK")
            v.setStyleSheet("font-size: 14px; font-weight: 800; color: #16a34a;")
            bl.addWidget(t)
            bl.addWidget(v)
            self.live_labels[key] = v
            grid.addWidget(box, i // 3, i % 3)
        s_lay.addLayout(grid)
        layout.addWidget(status_card)

        # Filters + history
        hist_card = QtWidgets.QFrame()
        hist_card.setObjectName("card")
        h_lay = QtWidgets.QVBoxLayout(hist_card)
        h_lay.setContentsMargins(16, 16, 16, 16)
        h_lay.setSpacing(12)

        # Smart filter bar — compact controls matching combo arrow style
        filter_bar = QtWidgets.QFrame()
        filter_bar.setObjectName("complianceFilterBar")
        _assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
        _down_arrow = os.path.join(_assets, "down_arrow.png").replace("\\", "/")
        filter_bar.setStyleSheet(f"""
            QFrame#complianceFilterBar {{
                background-color: #f1f5f9;
                border: 1px solid #e2e8f0;
                border-radius: 10px;
            }}
            QFrame#complianceFilterBar QLabel {{
                color: #475569;
                font-size: 12px;
                font-weight: 700;
                background: transparent;
                padding: 0 2px;
            }}
            QFrame#complianceFilterBar QComboBox,
            QFrame#complianceFilterBar QDateEdit {{
                background-color: #ffffff;
                border: 1px solid #cbd5e1;
                border-radius: 8px;
                padding: 2px 8px;
                padding-right: 30px;
                min-height: 28px;
                max-height: 28px;
                color: #0f172a;
                font-size: 12px;
                font-weight: 600;
            }}
            QFrame#complianceFilterBar QComboBox:hover,
            QFrame#complianceFilterBar QDateEdit:hover {{
                border: 1px solid #94a3b8;
                background-color: #f8fafc;
            }}
            QFrame#complianceFilterBar QComboBox:focus,
            QFrame#complianceFilterBar QDateEdit:focus {{
                border: 1px solid #2563eb;
                background-color: #ffffff;
            }}
            QFrame#complianceFilterBar QComboBox::drop-down,
            QFrame#complianceFilterBar QDateEdit::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 28px;
                border: none;
                border-left: 1px solid #e2e8f0;
                border-top-right-radius: 8px;
                border-bottom-right-radius: 8px;
                background-color: #eff6ff;
            }}
            QFrame#complianceFilterBar QComboBox::drop-down:hover,
            QFrame#complianceFilterBar QDateEdit::drop-down:hover {{
                background-color: #dbeafe;
            }}
            QFrame#complianceFilterBar QComboBox::down-arrow,
            QFrame#complianceFilterBar QDateEdit::down-arrow {{
                image: url({_down_arrow});
                width: 12px;
                height: 12px;
            }}
            QFrame#complianceFilterBar QDateEdit::up-button,
            QFrame#complianceFilterBar QDateEdit::down-button {{
                width: 0px;
                height: 0px;
                border: none;
                background: transparent;
            }}
            QFrame#complianceFilterBar QPushButton#secondaryBtn,
            QFrame#complianceFilterBar QPushButton#primaryBtn {{
                min-height: 28px;
                max-height: 28px;
                padding: 2px 12px;
                border-radius: 8px;
                font-size: 12px;
                font-weight: 700;
            }}
            QFrame#complianceFilterBar QPushButton#secondaryBtn {{
                background-color: #ffffff;
                border: 1px solid #cbd5e1;
                color: #0f172a;
            }}
            QFrame#complianceFilterBar QPushButton#secondaryBtn:hover {{
                background-color: #f8fafc;
                border: 1px solid #94a3b8;
            }}
            QFrame#complianceFilterBar QPushButton#primaryBtn {{
                background-color: #2563eb;
                border: 1px solid #1d4ed8;
                color: #ffffff;
            }}
            QFrame#complianceFilterBar QPushButton#primaryBtn:hover {{
                background-color: #1d4ed8;
            }}
        """)
        filt = QtWidgets.QHBoxLayout(filter_bar)
        filt.setContentsMargins(10, 8, 10, 8)
        filt.setSpacing(8)
        filt.setAlignment(Qt.AlignVCenter)

        _ctrl_h = 28

        filt.addWidget(QtWidgets.QLabel("Operator:"))
        self.op_combo = QtWidgets.QComboBox()
        self.op_combo.addItem("All")
        self.op_combo.setMinimumWidth(120)
        self.op_combo.setFixedHeight(_ctrl_h)
        filt.addWidget(self.op_combo)

        filt.addWidget(QtWidgets.QLabel("Event:"))
        self.event_combo = QtWidgets.QComboBox()
        self.event_combo.addItems([
            "All",
            COMPLIANCE_EVENT_MOBILE_PHONE,
            COMPLIANCE_EVENT_SHIRT_BUTTON,
            COMPLIANCE_EVENT_BLUETOOTH,
            COMPLIANCE_EVENT_EARBUDS,
            COMPLIANCE_EVENT_SPECTACLES,
            COMPLIANCE_EVENT_WRITING,
            COMPLIANCE_EVENT_PASSED,
        ])
        self.event_combo.setMinimumWidth(160)
        self.event_combo.setFixedHeight(_ctrl_h)
        filt.addWidget(self.event_combo)

        filt.addWidget(QtWidgets.QLabel("From:"))
        self.date_from = QtWidgets.QDateEdit(QtCore.QDate.currentDate().addDays(-7))
        self.date_from.setCalendarPopup(True)
        self.date_from.setDisplayFormat("yyyy-MM-dd")
        self.date_from.setFixedHeight(_ctrl_h)
        self.date_from.setMinimumWidth(158)
        self.date_from.setFixedWidth(158)
        filt.addWidget(self.date_from)

        filt.addWidget(QtWidgets.QLabel("To:"))
        self.date_to = QtWidgets.QDateEdit(QtCore.QDate.currentDate())
        self.date_to.setCalendarPopup(True)
        self.date_to.setDisplayFormat("yyyy-MM-dd")
        self.date_to.setFixedHeight(_ctrl_h)
        self.date_to.setMinimumWidth(158)
        self.date_to.setFixedWidth(158)
        filt.addWidget(self.date_to)

        _btn_w = 110  # uniform width for all action buttons

        refresh_btn = QtWidgets.QPushButton("Refresh")
        refresh_btn.setObjectName("secondaryBtn")
        refresh_btn.setCursor(Qt.PointingHandCursor)
        refresh_btn.setFixedHeight(_ctrl_h)
        refresh_btn.setFixedWidth(_btn_w)
        refresh_btn.clicked.connect(self.refresh_history)
        filt.addWidget(refresh_btn)

        export_btn = QtWidgets.QPushButton("Export CSV")
        export_btn.setObjectName("primaryBtn")
        export_btn.setCursor(Qt.PointingHandCursor)
        export_btn.setFixedHeight(_ctrl_h)
        export_btn.setFixedWidth(_btn_w)
        export_btn.clicked.connect(self.export_csv)
        filt.addWidget(export_btn)

        clear_btn = QtWidgets.QPushButton("Clear Logs")
        clear_btn.setObjectName("dangerBtn")
        clear_btn.setCursor(Qt.PointingHandCursor)
        clear_btn.setFixedHeight(_ctrl_h)
        clear_btn.setFixedWidth(_btn_w)
        clear_btn.clicked.connect(self.clear_logs)
        filt.addWidget(clear_btn)

        gallery_btn = QtWidgets.QPushButton("📷 Gallery")
        gallery_btn.setObjectName("galleryBtn")
        gallery_btn.setCursor(Qt.PointingHandCursor)
        gallery_btn.setFixedHeight(_ctrl_h)
        gallery_btn.setFixedWidth(_btn_w)
        gallery_btn.setStyleSheet("""
            QPushButton#galleryBtn {
                background-color: #7c3aed;
                border: 1px solid #6d28d9;
                color: #ffffff;
                border-radius: 8px;
                padding: 2px 8px;
                font-size: 12px;
                font-weight: 700;
            }
            QPushButton#galleryBtn:hover {
                background-color: #6d28d9;
            }
        """)
        gallery_btn.clicked.connect(self.open_gallery_dialog)
        filt.addWidget(gallery_btn)

        filt.addStretch(1)
        h_lay.addWidget(filter_bar)

        self.table = QtWidgets.QTableWidget(0, 9)
        self.table.setObjectName("complianceHistoryTable")
        self.table.setHorizontalHeaderLabels([
            "Timestamp", "Operator ID", "Name", "Event", "Result",
            "Camera", "Confidence", "Screenshot", "Status",
        ])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.setStyleSheet("""
            QTableWidget#complianceHistoryTable {
                background-color: #ffffff;
                alternate-background-color: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                gridline-color: transparent;
                selection-background-color: #dbeafe;
                selection-color: #0f172a;
                font-size: 12px;
                color: #0f172a;
            }
            QTableWidget#complianceHistoryTable::item {
                padding: 6px 8px;
                border-bottom: 1px solid #f1f5f9;
            }
            QHeaderView::section {
                background-color: #f1f5f9;
                color: #334155;
                font-weight: 700;
                font-size: 12px;
                padding: 8px;
                border: none;
                border-bottom: 1px solid #e2e8f0;
                border-right: 1px solid #e2e8f0;
            }
        """)
        h_lay.addWidget(self.table)
        layout.addWidget(hist_card, 1)

        self.refresh_history()
        self.refresh_summary()

    def _kpi(self, title: str, value: str, color: str) -> QtWidgets.QFrame:
        frame = QtWidgets.QFrame()
        frame.setObjectName("card")
        lay = QtWidgets.QVBoxLayout(frame)
        lay.setContentsMargins(14, 12, 14, 12)
        t = QtWidgets.QLabel(title)
        t.setStyleSheet("font-size: 11px; font-weight: 700; color: #64748b;")
        v = QtWidgets.QLabel(value)
        v.setObjectName("kpiValue")
        v.setStyleSheet(f"font-size: 20px; font-weight: 800; color: {color};")
        lay.addWidget(t)
        lay.addWidget(v)
        frame._value_lbl = v  # type: ignore[attr-defined]
        return frame

    def update_live_state(self, state: dict) -> None:
        mapping = [
            ("phone", True),
            ("shirt_button_open", True),
            ("earbuds", True),
            ("bluetooth", True),
            ("spectacles", False),
            ("writing", False),
        ]
        for key, warn in mapping:
            lbl = self.live_labels.get(key)
            if not lbl:
                continue
            data = state.get(key) or {}
            detected = bool(data.get("detected"))
            detail = data.get("detail") or ("Detected" if detected else "OK")
            if detected:
                lbl.setText(detail)
                lbl.setStyleSheet(
                    f"font-size: 14px; font-weight: 800; color: {'#dc2626' if warn else '#2563eb'};"
                )
            else:
                lbl.setText("OK")
                lbl.setStyleSheet("font-size: 14px; font-weight: 800; color: #16a34a;")

        # Append new events to table immediately
        for evt in state.get("new_events") or []:
            self._prepend_row(evt)
        self.kpi_warnings._value_lbl.setText(str(state.get("warning_count", 0)))  # type: ignore

    def _prepend_row(self, evt: dict) -> None:
        self.table.insertRow(0)
        vals = [
            evt.get("timestamp") or evt.get("Timestamp", ""),
            evt.get("operator_id") or evt.get("Operator_ID", ""),
            evt.get("operator_name") or evt.get("Operator_Name", ""),
            evt.get("event_type") or evt.get("Event_Type", ""),
            evt.get("detection_result") or evt.get("Detection_Result", ""),
            evt.get("camera_id") or evt.get("Camera_ID", ""),
            str(evt.get("confidence") if evt.get("confidence") is not None else evt.get("Confidence", "")),
            evt.get("screenshot_path") or evt.get("Screenshot_Path", ""),
            evt.get("status") or evt.get("Status", ""),
        ]
        for c, val in enumerate(vals):
            self.table.setItem(0, c, QtWidgets.QTableWidgetItem(str(val)))

    def refresh_summary(self) -> None:
        summary = self.logger.daily_summary(days=7)
        self.kpi_warnings._value_lbl.setText(str(summary.get("warnings", 0)))  # type: ignore
        by = summary.get("by_type") or {}
        self.kpi_phone._value_lbl.setText(str(by.get(COMPLIANCE_EVENT_MOBILE_PHONE, 0)))  # type: ignore
        self.kpi_shirt._value_lbl.setText(str(by.get(COMPLIANCE_EVENT_SHIRT_BUTTON, 0)))  # type: ignore
        buds = by.get(COMPLIANCE_EVENT_EARBUDS, 0) + by.get(COMPLIANCE_EVENT_BLUETOOTH, 0)
        self.kpi_buds._value_lbl.setText(str(buds))  # type: ignore
        self.kpi_pct._value_lbl.setText(f"{summary.get('compliance_pct', 100)}%")  # type: ignore

        # Refresh operator filter choices
        current = self.op_combo.currentText()
        ops = set()
        for row in self.logger.recent_events(500):
            oid = row.get("Operator_ID") or row.get("operator_id")
            if oid:
                ops.add(oid)
        self.op_combo.blockSignals(True)
        self.op_combo.clear()
        self.op_combo.addItem("All")
        for oid in sorted(ops):
            self.op_combo.addItem(oid)
        idx = self.op_combo.findText(current)
        self.op_combo.setCurrentIndex(max(0, idx))
        self.op_combo.blockSignals(False)

    def refresh_history(self) -> None:
        op = self.op_combo.currentText()
        et = self.event_combo.currentText()
        d0 = self.date_from.date().toString("yyyy-MM-dd")
        d1 = self.date_to.date().toString("yyyy-MM-dd")
        rows = self.logger.query_events(
            operator_id=None if op == "All" else op,
            event_type=None if et == "All" else et,
            date_from=d0,
            date_to=d1,
            limit=500,
        )
        self.table.setRowCount(0)
        for r in rows:
            self._prepend_row({
                "timestamp": r.get("timestamp") or r.get("Timestamp"),
                "operator_id": r.get("operator_id") or r.get("Operator_ID"),
                "operator_name": r.get("operator_name") or r.get("Operator_Name"),
                "event_type": r.get("event_type") or r.get("Event_Type"),
                "detection_result": r.get("detection_result") or r.get("Detection_Result"),
                "camera_id": r.get("camera_id") or r.get("Camera_ID"),
                "confidence": r.get("confidence") if r.get("confidence") is not None else r.get("Confidence"),
                "screenshot_path": r.get("screenshot_path") or r.get("Screenshot_Path"),
                "status": r.get("status") or r.get("Status"),
            })
        self.refresh_summary()

    def export_csv(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Compliance Log", "compliance_export.csv", "CSV Files (*.csv)"
        )
        if not path:
            return
        op = self.op_combo.currentText()
        et = self.event_combo.currentText()
        d0 = self.date_from.date().toString("yyyy-MM-dd")
        d1 = self.date_to.date().toString("yyyy-MM-dd")
        rows = self.logger.query_events(
            operator_id=None if op == "All" else op,
            event_type=None if et == "All" else et,
            date_from=d0,
            date_to=d1,
            limit=10000,
        )
        self.logger.export_csv(path, rows)
        QtWidgets.QMessageBox.information(self, "Export Complete", f"Exported {len(rows)} rows to:\n{path}")

    def open_gallery_dialog(self) -> None:
        """Open the compliance evidence screenshot gallery modal dialog."""
        dlg = ComplianceGalleryDialog(parent=self, screenshot_dir=self.logger.screenshot_dir)
        dlg.exec_()
        self.refresh_history()

    def clear_logs(self) -> None:
        """Clear all compliance event logs after user confirmation."""
        reply = QtWidgets.QMessageBox.question(
            self,
            "Clear Compliance Logs",
            "Are you sure you want to clear all recorded compliance logs?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        if reply == QtWidgets.QMessageBox.Yes:
            self.logger.clear_all_logs()
            self.table.setRowCount(0)
            self.refresh_summary()
            QtWidgets.QMessageBox.information(self, "Cleared", "All compliance logs have been cleared successfully.")
