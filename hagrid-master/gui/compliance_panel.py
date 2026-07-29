"""Live Monitor compliance status cards and toast notifications."""

from __future__ import annotations

from typing import Dict, List, Optional

from qt_compat import QtCore, QtGui, QtWidgets, Qt

from compliance.notifier import message_for


class ComplianceToast(QtWidgets.QFrame):
    """Non-blocking auto-dismiss notification."""

    def __init__(self, text: str, level: str = "warning", parent=None):
        super().__init__(parent)
        self.setObjectName("complianceToast")
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        lbl = QtWidgets.QLabel(text)
        lbl.setWordWrap(True)
        lbl.setStyleSheet("background: transparent; font-weight: 700; font-size: 12px;")
        layout.addWidget(lbl)
        if level == "warning":
            bg, fg, bd = "#fef2f2", "#b91c1c", "#fecaca"
        elif level == "info":
            bg, fg, bd = "#eff6ff", "#1d4ed8", "#bfdbfe"
        else:
            bg, fg, bd = "#f0fdf4", "#15803d", "#bbf7d0"
        self.setStyleSheet(
            f"QFrame#complianceToast {{ background:{bg}; color:{fg}; "
            f"border:1px solid {bd}; border-radius:8px; }}"
            f"QLabel {{ color:{fg}; }}"
        )
        QtCore.QTimer.singleShot(4500, self._fade)

    def _fade(self) -> None:
        self.hide()
        self.deleteLater()


class ComplianceStatusCard(QtWidgets.QFrame):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("compCard")
        self.setMinimumHeight(54)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(2)
        self.title = QtWidgets.QLabel(title)
        self.title.setStyleSheet("font-size: 11px; font-weight: 700; color: #64748b; background: transparent;")
        self.value = QtWidgets.QLabel("—")
        self.value.setStyleSheet("font-size: 12px; font-weight: 700; color: #0f172a; background: transparent;")
        self.value.setWordWrap(True)
        lay.addWidget(self.title)
        lay.addWidget(self.value)
        self._set_ok()

    def _set_ok(self) -> None:
        self.setStyleSheet(
            "QFrame#compCard { background:#ffffff; border:1px solid #e2e8f0; border-radius:8px; }"
        )
        self.value.setStyleSheet("font-size: 12px; font-weight: 700; color: #16a34a; background: transparent;")

    def _set_warn(self) -> None:
        self.setStyleSheet(
            "QFrame#compCard { background:#fef2f2; border:1px solid #fecaca; border-radius:8px; }"
        )
        self.value.setStyleSheet("font-size: 12px; font-weight: 700; color: #dc2626; background: transparent;")

    def _set_info(self) -> None:
        self.setStyleSheet(
            "QFrame#compCard { background:#eff6ff; border:1px solid #bfdbfe; border-radius:8px; }"
        )
        self.value.setStyleSheet("font-size: 12px; font-weight: 700; color: #2563eb; background: transparent;")

    def update_result(self, detected: bool, detail: str, warn: bool = True) -> None:
        self.value.setText(detail or ("Detected" if detected else "OK"))
        if detected and warn:
            self._set_warn()
        elif detected:
            self._set_info()
        else:
            self._set_ok()
            if not detail or detail in ("disabled", "model_unavailable", "no_phone", "no_face", "no_earbuds", "no_writing"):
                self.value.setText("OK")


class CompliancePanel(QtWidgets.QFrame):
    """Right-rail compliance snapshot for Live Monitor."""

    toast_requested = QtCore.pyqtSignal(str, str)  # text, level

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.is_dark = False
        # Natural content height — parent scroll area owns overflow scrolling.
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.header = QtWidgets.QLabel("Operator Compliance")
        self.header.setStyleSheet("font-size: 15px; font-weight: 700; color: #0f172a;")
        layout.addWidget(self.header)

        self.warn_lbl = QtWidgets.QLabel("Warnings today: 0")
        self.warn_lbl.setStyleSheet("color: #64748b; font-size: 11px; font-weight: 600;")
        layout.addWidget(self.warn_lbl)

        grid = QtWidgets.QGridLayout()
        grid.setSpacing(6)
        self.card_phone = ComplianceStatusCard("Mobile Phone")
        self.card_shirt = ComplianceStatusCard("Shirt Button")
        self.card_buds = ComplianceStatusCard("Earbuds / BT")
        self.card_specs = ComplianceStatusCard("Spectacles")
        self.card_write = ComplianceStatusCard("Writing")
        grid.addWidget(self.card_phone, 0, 0)
        grid.addWidget(self.card_shirt, 0, 1)
        grid.addWidget(self.card_buds, 1, 0)
        grid.addWidget(self.card_specs, 1, 1)
        grid.addWidget(self.card_write, 2, 0, 1, 2)
        layout.addLayout(grid)

        self.toast_host = QtWidgets.QVBoxLayout()
        self.toast_host.setSpacing(4)
        layout.addLayout(self.toast_host)

    def set_theme(self, is_dark: bool) -> None:
        self.is_dark = is_dark
        col = "#f8fafc" if is_dark else "#0f172a"
        self.header.setStyleSheet(f"font-size: 15px; font-weight: 700; color: {col};")

    def show_toast(self, text: str, level: str = "warning") -> None:
        toast = ComplianceToast(text, level=level, parent=self)
        self.toast_host.insertWidget(0, toast)

    def update_state(self, state: dict) -> None:
        phone = state.get("phone") or {}
        shirt = state.get("shirt_button_open") or {}
        earbuds = state.get("earbuds") or {}
        bluetooth = state.get("bluetooth") or {}
        specs = state.get("spectacles") or {}
        writing = state.get("writing") or {}

        self.card_phone.update_result(bool(phone.get("detected")), phone.get("detail", ""), warn=True)
        self.card_shirt.update_result(bool(shirt.get("detected")), shirt.get("detail", ""), warn=True)

        if bluetooth.get("detected"):
            self.card_buds.update_result(True, bluetooth.get("detail", "Bluetooth Device Detected"), warn=True)
        elif earbuds.get("detected"):
            detail = earbuds.get("detail") or "Earbuds Detected"
            self.card_buds.update_result(True, detail, warn=True)
        else:
            self.card_buds.update_result(False, earbuds.get("detail", ""), warn=True)

        self.card_specs.update_result(bool(specs.get("detected")), specs.get("detail", ""), warn=False)
        self.card_write.update_result(bool(writing.get("detected")), writing.get("detail", ""), warn=False)
        self.warn_lbl.setText(f"Warnings: {state.get('warning_count', 0)}")

        for evt in state.get("new_events") or []:
            et = evt.get("event_type") or evt.get("Event_Type", "")
            msg = evt.get("detection_result") or message_for(et)
            status = evt.get("status") or evt.get("Status", "Warning")
            level = "warning" if status == "Warning" else ("success" if status == "Passed" else "info")
            self.show_toast(msg, level=level)
