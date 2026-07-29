"""Main application window for the HAGRID Industrial SOP Monitor.

Redesigned with a modern, high-performance light-theme dashboard, featuring
tabbed views, live metrics, SOP builder, operator login, history quality inspect,
and parts checklist.
"""

from __future__ import annotations

import os
import csv
import sys
import time

# Quiet C++/OpenCV/MediaPipe log spam
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["FLAGS_stderrthreshold"] = "3"

import cv2
if hasattr(cv2, "setLogLevel"):
    cv2.setLogLevel(0)

from typing import List, Optional

_GUI_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_GUI_DIR)
for _path in (_PROJECT_ROOT, _GUI_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from qt_compat import QtCore, QtGui, QtWidgets, Qt, QAction

from camera_view import CameraView
from camera_worker import CameraWorker
from log_panel import LogPanel
from sop_panel import SopStepPanel, SopStepCard, SopStep, STEPS
from status_panel import StatusPanel
from compliance_panel import CompliancePanel
from compliance_page import CompliancePage
from compliance.logger import ComplianceLogger
from constants import (
    CSV_COLUMNS,
    LOG_CSV_PATH,
    COMPLIANCE_DEFAULT_SETTINGS,
)
from activity_logger import ActivityLogger, ACTIVITY_COLUMNS
from assembly_manager import AssemblyManager
import copy
import json

APP_NAME = "HAGRID Industrial SOP Monitor"
APP_VERSION = "2.0.0"


def _resolve_project_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(_PROJECT_ROOT, path))


class ThemeToggleSwitch(QtWidgets.QWidget):
    """Custom toggle switch for Light/Dark Theme toggle."""
    toggled = QtCore.pyqtSignal(bool)

    def __init__(self, is_dark: bool = False, parent=None):
        super().__init__(parent)
        self.is_dark = is_dark
        self.setFixedHeight(36)
        
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.label = QtWidgets.QLabel("Theme Mode:")
        self.label.setStyleSheet("font-size: 13px; font-weight: 700; background: transparent;")
        layout.addWidget(self.label)

        self.btn = QtWidgets.QPushButton()
        self.btn.setFixedSize(94, 30)
        self.btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn.setCheckable(True)
        self.btn.setChecked(is_dark)
        self.btn.clicked.connect(self._on_click)
        layout.addWidget(self.btn)

        self._update_style()

    def set_checked(self, checked: bool) -> None:
        self.is_dark = checked
        self.btn.setChecked(checked)
        self._update_style()

    def _on_click(self) -> None:
        self.is_dark = self.btn.isChecked()
        self._update_style()
        self.toggled.emit(self.is_dark)

    def _update_style(self) -> None:
        if self.is_dark:
            self.label.setStyleSheet("color: #f8fafc; font-size: 13px; font-weight: 700; background: transparent;")
            self.btn.setText("DARK ON")
            self.btn.setStyleSheet("""
                QPushButton {
                    background-color: #2563eb;
                    color: #ffffff;
                    border: 1px solid #3b82f6;
                    border-radius: 15px;
                    font-weight: 800;
                    font-size: 11px;
                    padding: 4px 8px;
                }
                QPushButton:hover {
                    background-color: #1d4ed8;
                }
            """)
        else:
            self.label.setStyleSheet("color: #0f172a; font-size: 13px; font-weight: 700; background: transparent;")
            self.btn.setText("LIGHT OFF")
            self.btn.setStyleSheet("""
                QPushButton {
                    background-color: #e2e8f0;
                    color: #475569;
                    border: 1px solid #cbd5e1;
                    border-radius: 15px;
                    font-weight: 800;
                    font-size: 11px;
                    padding: 4px 8px;
                }
                QPushButton:hover {
                    background-color: #cbd5e1;
                    color: #1e293b;
                }
            """)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, config_path: str = "configs/gesture.yaml", source: int = 0):
        super().__init__()
        self.is_dark_theme = False
        self.config_path = _resolve_project_path(config_path)
        self.source = source
        self.worker: Optional[CameraWorker] = None
        self.default_worker_id = "EMP001"

        # Initialize assembly manager and resolve active assembly
        self._assembly_manager = AssemblyManager()
        _active = self._assembly_manager.get_active_assembly()
        self.active_sop_name = _active.get("display_name", "Industrial Screw-Tightening Assembly")
        self.active_product_name = "Precision Assembly Module"
        self.active_variant = "PAM-V2"
        self.active_turn_target = float(_active.get("target_params", {}).get("turn_target", 2.5))
        self._active_detection_mode = _active.get("detection_mode", "screw_monitor")
        self._active_align_threshold = int(_active.get("target_params", {}).get("align_threshold_px", 50))
        self._compliance_settings = copy.deepcopy(COMPLIANCE_DEFAULT_SETTINGS)
        self._compliance_logger = ComplianceLogger()
        # Use real detectors by default; set True only for UI demos without models
        self._compliance_mock = False

        # Active settings defaults (overridden by _load_active_settings_early)
        self._active_cam_width   = 640
        self._active_cam_height  = 480
        self._active_rotation    = 0
        self._active_confidence  = 0.50
        self._active_sensitivity = 5.0

        # Pre-load saved settings from disk so _start_worker uses them
        self._load_active_settings_early()

        self.setWindowTitle(APP_NAME)
        self.resize(1920, 1080)
        self.setMinimumSize(1280, 800)

        # Load fonts
        for font_file in ("Poppins-Regular.ttf", "Poppins-Bold.ttf"):
            if os.path.exists(font_file):
                QtGui.QFontDatabase.addApplicationFont(font_file)

        # Set app stylesheet
        self._apply_theme()

        # Initialize activity logger (singleton) BEFORE building UI
        # so that _create_logs_page can reference it safely
        self._activity_logger = ActivityLogger()

        # Load registered operators database
        self._operators: List[dict] = []
        self._load_operators()

        # Setup UI
        self._build_ui()
        self._build_menu()

        # Initial theme application after UI widgets exist
        if self.is_dark_theme:
            self._apply_theme_mode(True)

        self._start_worker()

    def _load_active_settings_early(self) -> None:
        """Read app_settings.json at startup before UI is built."""
        path = os.path.join(_GUI_DIR, "app_settings.json")
        if not os.path.exists(path):
            return
        try:
            with open(path, "r") as f:
                s = json.load(f)
            res_map = {0: (1280, 720), 1: (1920, 1080), 2: (640, 480), 3: (3840, 2160)}
            rot_map = {0: 0, 1: 90, 2: 180, 3: 270}
            w, h = res_map.get(s.get("res_index", 0), (640, 480))
            self._active_cam_width   = w
            self._active_cam_height  = h
            self._active_rotation    = rot_map.get(s.get("rot_index", 0), 0)
            self._active_confidence  = s.get("confidence",  0.50)
            self._active_sensitivity = s.get("sensitivity", 5.0)
            self.is_dark_theme       = (s.get("theme", "light") == "dark")
            if isinstance(s.get("compliance"), dict):
                merged = copy.deepcopy(COMPLIANCE_DEFAULT_SETTINGS)
                merged.update({k: v for k, v in s["compliance"].items() if k != "detectors"})
                if isinstance(s["compliance"].get("detectors"), dict):
                    merged["detectors"] = {
                        **merged.get("detectors", {}),
                        **s["compliance"]["detectors"],
                    }
                self._compliance_settings = merged
        except Exception as e:
            print(f"[settings] early load error: {e}")


    def _on_theme_toggled(self, is_dark: bool) -> None:
        self.is_dark_theme = is_dark
        self._save_theme_setting(is_dark)
        self._apply_theme_mode(is_dark)
        mode_str = "Dark" if is_dark else "Light"
        if hasattr(self, "_activity_logger") and self._activity_logger:
            self._activity_logger.log("SETTINGS", "System Settings", "Theme Change", f"Switched to {mode_str} theme")

    def _save_theme_setting(self, is_dark: bool) -> None:
        import json
        path = os.path.join(_GUI_DIR, "app_settings.json")
        settings = {}
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    settings = json.load(f)
            except Exception:
                settings = {}
        settings["theme"] = "dark" if is_dark else "light"
        try:
            with open(path, "w") as f:
                json.dump(settings, f, indent=2)
        except Exception as e:
            print(f"[settings] failed to save theme: {e}")

    def _apply_theme_mode(self, is_dark: bool) -> None:
        self.is_dark_theme = is_dark
        self._apply_theme()

        # Helper colours
        txt      = "#f8fafc" if is_dark else "#0f172a"
        txt_dim  = "#cbd5e1" if is_dark else "#334155"
        txt_head = "#ffffff" if is_dark else "#0f172a"
        card_bg  = "#1e293b" if is_dark else "#ffffff"
        card_bdr = "#2d3f55" if is_dark else "#cbd5e1"
        page_bg  = "#0f172a" if is_dark else "#f8fafc"
        accent   = "#3b82f6" if is_dark else "#1d4ed8"

        # ── Sidebar subtitle label ──────────────────────────────────────────
        if hasattr(self, "sidebar_subtitle"):
            self.sidebar_subtitle.setStyleSheet(
                f"color: {'#60a5fa' if is_dark else '#334155'}; font-size: 11px; font-weight: 600; background: transparent;"
            )

        # ── Update subpanels ────────────────────────────────────────────────
        if hasattr(self, "status_panel") and self.status_panel:
            self.status_panel.set_theme(is_dark)
        if hasattr(self, "sop_panel") and self.sop_panel:
            self.sop_panel.set_theme(is_dark)
        if hasattr(self, "log_panel") and self.log_panel:
            self.log_panel.set_theme(is_dark)
        if hasattr(self, "compliance_panel") and self.compliance_panel:
            self.compliance_panel.set_theme(is_dark)

        # ── AI status indicator box in sidebar ──────────────────────────────
        if hasattr(self, "conn_status") and self.conn_status:
            if is_dark:
                self.conn_status.setStyleSheet(
                    "color: #4ade80; font-weight: 700; font-size: 11px; padding: 10px; margin: 10px;"
                    "background-color: #052e16; border-radius: 8px; border: 1px solid #166534;"
                )
            else:
                self.conn_status.setStyleSheet(
                    "color: #15803d; font-weight: 800; font-size: 11px; padding: 10px; margin: 10px;"
                    "background-color: #f0fdf4; border-radius: 8px; border: 1px solid #86efac;"
                )

        # ── Dashboard: title/subtitle labels ────────────────────────────────
        for attr, style in (
            ("dash_title_lbl",   f"font-size: 22px; font-weight: 800; color: {txt_head};"),
            ("dash_sub_lbl",     f"color: {txt_dim}; font-size: 12px; font-weight: 500;"),
            ("dash_time_lbl",    f"font-weight: 800; color: {accent}; font-size: 13px;"),
            ("dash_shift_lbl",   f"color: {txt_dim}; font-size: 11px; font-weight: 600;"),
            ("dash_health_title",f"font-size: 15px; font-weight: 800; color: {txt_head};"),
            ("dash_sys_details", f"color: {txt_dim}; font-size: 12px; font-weight: 600;"),
        ):
            if hasattr(self, attr):
                getattr(self, attr).setStyleSheet(style)

        # ── Health badges ───────────────────────────────────────────────────
        for badge_attr in ("ind_cam", "ind_ai", "ind_db", "ind_mes", "ind_gpu"):
            if hasattr(self, badge_attr):
                badge = getattr(self, badge_attr)
                if is_dark:
                    badge.setStyleSheet(
                        "background-color: #052e16; color: #4ade80; border: 1px solid #166534;"
                        "border-radius: 6px; padding: 6px 12px; font-weight: 700; font-size: 11px;"
                    )
                else:
                    badge.setStyleSheet(
                        "background-color: #dcfce7; color: #15803d; border: 1px solid #86efac;"
                        "border-radius: 6px; padding: 6px 12px; font-weight: 800; font-size: 11px;"
                    )

        # ── StatCard objects (Dashboard KPIs) ────────────────────────────────
        for card_attr in ("dash_kpi_total", "dash_kpi_ok", "dash_kpi_ng", "dash_kpi_yield",
                           "dash_kpi_acc", "dash_kpi_workers", "dash_kpi_sop", "dash_kpi_today",
                           "dash_kpi_status", "dash_kpi_fps", "dash_kpi_warnings", "dash_kpi_compliance"):
            if hasattr(self, card_attr):
                getattr(self, card_attr).set_theme(is_dark)

        # ── TrendChart objects ───────────────────────────────────────────────
        for chart_attr in ("chart_conf", "chart_turns", "chart_yield", "chart_cycle"):
            if hasattr(self, chart_attr):
                chart = getattr(self, chart_attr)
                chart.is_dark = is_dark
                chart.update()

        # ── Analytics page labels ────────────────────────────────────────────
        for attr, style in (
            ("analytics_title",    f"font-size: 22px; font-weight: 800; color: {txt_head};"),
            ("analytics_subtitle", f"color: {txt_dim}; font-size: 12px; font-weight: 500;"),
            ("analytics_acc_lbl",  f"font-size: 15px; font-weight: 800; color: {txt_head};"),
        ):
            if hasattr(self, attr):
                getattr(self, attr).setStyleSheet(style)

        # ── Operator Management page labels ──────────────────────────────────
        for attr, style in (
            ("op_title",          f"font-size: 22px; font-weight: 800; color: {txt_head};"),
            ("op_info_login",     f"color: {txt_dim}; font-size: 12px; font-weight: 600;"),
            ("op_info_working",   f"color: {txt_dim}; font-size: 12px; font-weight: 600;"),
            ("op_info_break",     f"color: {txt_dim}; font-size: 12px; font-weight: 600;"),
            ("op_kpi_title",      f"font-size: 14px; font-weight: 800; color: {accent};"),
            ("op_kpi_processed",  f"color: {txt_dim}; font-size: 12px; font-weight: 600;"),
            ("op_kpi_compliance", f"color: {txt_dim}; font-size: 12px; font-weight: 600;"),
            ("op_kpi_cycle",      f"color: {txt_dim}; font-size: 12px; font-weight: 600;"),
        ):
            if hasattr(self, attr):
                getattr(self, attr).setStyleSheet(style)

        # ── Settings page summary card ────────────────────────────────────
        if hasattr(self, "active_settings_card") and self.active_settings_card:
            if is_dark:
                self.active_settings_card.setStyleSheet(
                    "QFrame#card { background-color: #052e16; border: 1px solid #166534; border-radius: 10px; }"
                )
            else:
                self.active_settings_card.setStyleSheet(
                    "QFrame#card { background-color: #f0fdf4; border: 1px solid #86efac; border-radius: 10px; }"
                )
        for lbl_attr in ("acs_res_lbl", "acs_rot_lbl", "acs_conf_lbl", "acs_sens_lbl"):
            if hasattr(self, lbl_attr):
                col = "#4ade80" if is_dark else "#15803d"
                getattr(self, lbl_attr).setStyleSheet(
                    f"color: {col}; font-size: 12px; font-weight: 700; background: transparent;"
                )

        # ── Settings page section header labels ──────────────────────────────
        section_hdr_style = (
            f"font-weight: 800; color: {accent}; font-size: 13px;"
            f"background-color: {'#172554' if is_dark else '#eff6ff'}; border-radius: 6px; padding: 6px 10px; margin-top: 6px;"
        )
        for attr in ("settings_cam_hdr", "settings_ai_hdr", "settings_db_hdr"):
            if hasattr(self, attr):
                getattr(self, attr).setStyleSheet(section_hdr_style)

        # ── Settings status label ────────────────────────────────────────────
        if hasattr(self, "settings_status_lbl"):
            if is_dark:
                self.settings_status_lbl.setStyleSheet(
                    "background-color: #172554; color: #60a5fa; border: 1px solid #1e3a5f;"
                    "border-radius: 10px; padding: 4px 12px; font-size: 11px; font-weight: 700;"
                )
            else:
                self.settings_status_lbl.setStyleSheet(
                    "background-color: #fff7ed; color: #c2410c; border: 1px solid #ffedd5;"
                    "border-radius: 10px; padding: 4px 12px; font-size: 11px; font-weight: 700;"
                )

        # ── Quality Inspection Screenshot Preview Box ────────────────────────
        if hasattr(self, "logs_img_lbl") and self.logs_img_lbl:
            bg = "#0f172a" if is_dark else "#f8fafc"
            bdr = "#334155" if is_dark else "#cbd5e1"
            txt = "#cbd5e1" if is_dark else "#475569"
            if not self.logs_img_lbl.pixmap() or self.logs_img_lbl.pixmap().isNull():
                self.logs_img_lbl.setStyleSheet(
                    f"border: 1px dashed {bdr}; border-radius: 8px; background: {bg}; color: {txt}; min-height: 250px;"
                )

        # ── Reference Images Add Panel & Form Elements ───────────────────────
        if hasattr(self, "ref_add_panel") and self.ref_add_panel:
            bg = "#0f172a" if is_dark else "#ffffff"
            bdr = "#1e3a5f" if is_dark else "#cbd5e1"
            self.ref_add_panel.setStyleSheet(f"QFrame#sidebar {{ background-color: {bg}; border-right: 1px solid {bdr}; }}")

        if hasattr(self, "ref_panel_title") and self.ref_panel_title:
            col = "#f8fafc" if is_dark else "#0f172a"
            self.ref_panel_title.setStyleSheet(f"font-size: 15px; font-weight: 800; color: {col};")

        if hasattr(self, "ref_sep") and self.ref_sep:
            bdr = "#334155" if is_dark else "#cbd5e1"
            self.ref_sep.setStyleSheet(f"background-color: {bdr};")

        if hasattr(self, "ref_add_preview") and self.ref_add_preview:
            bg = "#0f172a" if is_dark else "#f8fafc"
            bdr = "#334155" if is_dark else "#cbd5e1"
            txt = "#94a3b8" if is_dark else "#475569"
            if not self.ref_add_preview.pixmap() or self.ref_add_preview.pixmap().isNull():
                self.ref_add_preview.setStyleSheet(
                    f"border: 2px dashed {bdr}; border-radius: 10px; background-color: {bg}; color: {txt}; font-size: 12px; min-height: 140px;"
                )

        if hasattr(self, "ref_form_labels"):
            col = "#e2e8f0" if is_dark else "#1e293b"
            for lbl in self.ref_form_labels:
                lbl.setStyleSheet(f"font-weight: 700; color: {col}; font-size: 13px;")

        if hasattr(self, "ref_store_info") and self.ref_store_info:
            col = "#94a3b8" if is_dark else "#475569"
            self.ref_store_info.setStyleSheet(f"color: {col}; font-size: 11px;")

        # ── Activity Logs Search & Filter Controls ───────────────────────────
        if hasattr(self, "activity_date_lbl") and self.activity_date_lbl:
            col = "#f8fafc" if is_dark else "#0f172a"
            self.activity_date_lbl.setStyleSheet(f"font-weight: 700; font-size: 13px; color: {col};")

        if hasattr(self, "search_frame") and self.search_frame:
            bg = "#1e293b" if is_dark else "#f8fafc"
            bdr = "#334155" if is_dark else "#e2e8f0"
            self.search_frame.setStyleSheet(f"QFrame {{ background-color: {bg}; border: 1px solid {bdr}; border-radius: 8px; }}")

        if hasattr(self, "activity_search_input") and self.activity_search_input:
            bg = "#0f172a" if is_dark else "#ffffff"
            bdr = "#334155" if is_dark else "#cbd5e1"
            txt = "#f1f5f9" if is_dark else "#1e293b"
            self.activity_search_input.setStyleSheet(
                f"QLineEdit {{ background-color: {bg}; border: 1px solid {bdr}; border-radius: 6px; padding: 8px 14px; font-size: 13px; color: {txt}; }}"
                f"QLineEdit:focus {{ border-color: #2563eb; background-color: {bg}; }}"
            )

        if hasattr(self, "activity_file_combo") and self.activity_file_combo:
            bg = "#0f172a" if is_dark else "#ffffff"
            bdr = "#334155" if is_dark else "#cbd5e1"
            txt = "#f1f5f9" if is_dark else "#1e293b"
            self.activity_file_combo.setStyleSheet(f"""
                QComboBox {{
                    background-color: {bg};
                    border: 1px solid {bdr};
                    border-radius: 6px;
                    padding: 6px 12px;
                    font-size: 12px;
                    font-weight: 600;
                    color: {txt};
                    min-height: 28px;
                }}
                QComboBox:hover {{ border-color: #2563eb; }}
                QComboBox::drop-down {{ border: none; width: 28px; }}
                QComboBox QAbstractItemView {{
                    background-color: {bg};
                    border: 1px solid {bdr};
                    color: {txt};
                    selection-background-color: #2563eb;
                    selection-color: #ffffff;
                    padding: 4px;
                }}
            """)

        # ── ThemeToggleSwitch widget state ───────────────────────────────────
        if hasattr(self, "theme_toggle") and self.theme_toggle:
            self.theme_toggle.set_checked(is_dark)

        # ── Generic pass: fix all inline label colors for theme ──────
        self._deep_update_labels(is_dark)

    def _deep_update_labels(self, is_dark: bool) -> None:
        """Walk all child QLabels and adjust text colors for active theme contrast.

        Always derives the new color from the label's ORIGINAL (baseline) stylesheet,
        cached on first run, instead of re-replacing an already-converted stylesheet.
        This prevents color drift after multiple theme toggles.
        """
        if not hasattr(self, "_base_label_styles"):
            self._base_label_styles: dict = {}

        # light-mode color -> dark-mode color
        to_dark = {
            "#0f172a": "#f1f5f9",
            "#1e293b": "#e2e8f0",
            "#475569": "#cbd5e1",
            "#374151": "#cbd5e1",
            "#6b7280": "#94a3b8",
            "#64748b": "#cbd5e1",
            "#334155": "#cbd5e1",
        }

        for lbl in self.findChildren(QtWidgets.QLabel):
            # Cache the ORIGINAL stylesheet the first time we see this label,
            # so every future toggle starts from the same known baseline.
            if lbl not in self._base_label_styles:
                self._base_label_styles[lbl] = lbl.styleSheet()

            base_ss = self._base_label_styles[lbl]

            if is_dark:
                ss = base_ss
                for light_c, dark_c in to_dark.items():
                    ss = ss.replace(f"color: {light_c}", f"color: {dark_c}")
            else:
                # Light theme = always the original baseline, no drift possible.
                ss = base_ss

            lbl.setStyleSheet(ss)
        else:
            dim_grays = {"#64748b", "#94a3b8", "#cbd5e1"}
            for lbl in self.findChildren(QtWidgets.QLabel):
                ss = lbl.styleSheet()
                if ss and any(g in ss for g in dim_grays):
                    ss = ss.replace("color: #64748b", "color: #334155")
                    ss = ss.replace("color: #94a3b8", "color: #334155")
                    ss = ss.replace("color: #cbd5e1", "color: #1e293b")
                    lbl.setStyleSheet(ss)

    def _apply_theme(self) -> None:
        assets_dir = os.path.join(_GUI_DIR, "assets")
        os.makedirs(assets_dir, exist_ok=True)
        down_arrow_path = os.path.join(assets_dir, "down_arrow.png").replace("\\", "/")
        up_arrow_path = os.path.join(assets_dir, "up_arrow.png").replace("\\", "/")

        # Generate down arrow PNG (blue)
        pix_down = QtGui.QPixmap(16, 16)
        pix_down.fill(QtCore.Qt.transparent)
        p = QtGui.QPainter(pix_down)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QColor("#2563eb"))
        path_down = QtGui.QPainterPath()
        path_down.moveTo(3, 6)
        path_down.lineTo(13, 6)
        path_down.lineTo(8, 11)
        path_down.closeSubpath()
        p.drawPath(path_down)
        p.end()
        pix_down.save(down_arrow_path)

        # Generate up arrow PNG (blue)
        pix_up = QtGui.QPixmap(16, 16)
        pix_up.fill(QtCore.Qt.transparent)
        p = QtGui.QPainter(pix_up)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QColor("#2563eb"))
        path_up = QtGui.QPainterPath()
        path_up.moveTo(3, 10)
        path_up.lineTo(13, 10)
        path_up.lineTo(8, 5)
        path_up.closeSubpath()
        p.drawPath(path_up)
        p.end()
        pix_up.save(up_arrow_path)

        if getattr(self, "is_dark_theme", False):
            qss = """
            QWidget {
                font-family: 'Poppins', 'Segoe UI', sans-serif;
                color: #f1f5f9;
                background-color: #0f172a;
            }
            QMainWindow {
                background-color: #0f172a;
            }
            QWidget#appRoot, QWidget#page, QStackedWidget {
                background-color: #0f172a;
                border: none;
            }
            QLabel {
                color: #f1f5f9;
                background-color: transparent;
            }
            QStatusBar {
                background-color: #1e293b;
                color: #cbd5e1;
                border-top: 1px solid #334155;
                font-size: 12px;
                font-weight: 600;
            }
            QMenuBar {
                background-color: #1e293b;
                color: #f1f5f9;
                border-bottom: 1px solid #334155;
            }
            QMenuBar::item:selected {
                background-color: #2563eb;
                color: #ffffff;
            }
            QMenu {
                background-color: #1e293b;
                border: 1px solid #334155;
                color: #f1f5f9;
            }
            QMenu::item:selected {
                background-color: #2563eb;
                color: #ffffff;
            }
            QFrame#sidebar {
                background-color: #0f172a;
                border-right: 1px solid #1e3a5f;
            }
            QLabel#sidebarTitle {
                font-weight: 800;
                font-size: 20px;
                color: #3b82f6;
                letter-spacing: 0.5px;
                padding: 10px 0px;
            }
            QPushButton#sidebarBtn {
                background-color: transparent;
                color: #cbd5e1;
                border: none;
                border-left: 3px solid transparent;
                border-radius: 0px;
                padding: 14px 20px;
                text-align: left;
                font-weight: 600;
                font-size: 13px;
            }
            QPushButton#sidebarBtn:hover {
                background-color: #1e293b;
                color: #ffffff;
            }
            QPushButton#sidebarBtn:checked {
                background-color: #172554;
                color: #60a5fa;
                border-left: 3px solid #3b82f6;
                font-weight: 700;
            }
            QFrame#card, QFrame#gestureCard, QFrame#statCard {
                background-color: #1e293b;
                border: 1px solid #2d3f55;
                border-radius: 12px;
            }
            QFrame#cardHeader {
                border-bottom: 1px solid #334155;
            }
            QLabel#cardTitle {
                font-size: 14px;
                font-weight: 700;
                color: #ffffff;
            }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                background-color: #0f172a;
                border: 1px solid #2d3f55;
                border-radius: 8px;
                padding: 8px 12px;
                color: #f1f5f9;
                font-size: 13px;
                selection-background-color: #2563eb;
            }
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
                border: 1px solid #3b82f6;
                background-color: #172554;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 32px;
                border-left-width: 1px;
                border-left-color: #2d3f55;
                border-left-style: solid;
                border-top-right-radius: 8px;
                border-bottom-right-radius: 8px;
                background-color: #1e293b;
            }
            QComboBox::drop-down:hover {
                background-color: #172554;
            }
            QComboBox::down-arrow {
                image: url({down_arrow_path});
                width: 12px;
                height: 12px;
            }
            QComboBox QAbstractItemView {
                background-color: #1e293b;
                border: 1px solid #334155;
                color: #f1f5f9;
                selection-background-color: #2563eb;
                selection-color: #ffffff;
                outline: none;
            }
            QSpinBox::up-button, QDoubleSpinBox::up-button {
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 24px;
                border-left: 1px solid #2d3f55;
                border-bottom: 1px solid #2d3f55;
                border-top-right-radius: 8px;
                background-color: #1e293b;
            }
            QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {
                background-color: #172554;
            }
            QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
                image: url({up_arrow_path});
                width: 10px;
                height: 10px;
            }
            QSpinBox::down-button, QDoubleSpinBox::down-button {
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: 24px;
                border-left: 1px solid #2d3f55;
                border-bottom-right-radius: 8px;
                background-color: #1e293b;
            }
            QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
                background-color: #172554;
            }
            QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
                image: url({down_arrow_path});
                width: 10px;
                height: 10px;
            }
            QCheckBox {
                spacing: 8px;
                font-weight: 600;
                color: #f1f5f9;
                background-color: transparent;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 4px;
                border: 1px solid #475569;
                background-color: #0f172a;
            }
            QCheckBox::indicator:checked {
                background-color: #2563eb;
                border: 1px solid #2563eb;
            }
            QPushButton#primaryBtn {
                background-color: #2563eb;
                color: #ffffff;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-weight: 700;
                font-size: 13px;
            }
            QPushButton#primaryBtn:hover {
                background-color: #1d4ed8;
            }
            QPushButton#dangerBtn {
                background-color: #ef4444;
                color: #ffffff;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-weight: 700;
                font-size: 13px;
            }
            QPushButton#dangerBtn:hover {
                background-color: #dc2626;
            }
            QPushButton#secondaryBtn {
                background-color: #1e293b;
                color: #e2e8f0;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 10px 20px;
                font-weight: 700;
                font-size: 13px;
            }
            QPushButton#secondaryBtn:hover {
                background-color: #2d3f55;
                color: #ffffff;
                border-color: #60a5fa;
            }
            QTableView {
                background-color: #1e293b;
                alternate-background-color: #243249;
                gridline-color: #2d3f55;
                border: 1px solid #2d3f55;
                border-radius: 12px;
                color: #f1f5f9;
            }
            QTableView::item {
                color: #f1f5f9;
                padding: 4px 8px;
            }
            QTableView::item:selected {
                background-color: #172554;
                color: #ffffff;
            }
            QHeaderView::section {
                background-color: #243249;
                color: #f1f5f9;
                font-weight: 700;
                padding: 10px;
                border: none;
                border-right: 1px solid #2d3f55;
                border-bottom: 1px solid #2d3f55;
                font-size: 12px;
            }
            QTableWidget {
                background-color: #1e293b;
                border: 1px solid #2d3f55;
                color: #f1f5f9;
            }
            QTableWidget::item {
                color: #f1f5f9;
                padding: 4px 8px;
            }
            QTabWidget::pane {
                background-color: #1e293b;
                border: 1px solid #2d3f55;
                border-radius: 8px;
            }
            QTabBar::tab {
                background-color: #0f172a;
                color: #94a3b8;
                border: 1px solid #2d3f55;
                border-bottom: none;
                border-radius: 6px 6px 0 0;
                padding: 8px 20px;
                font-weight: 600;
                font-size: 13px;
                min-width: 120px;
            }
            QTabBar::tab:selected {
                background-color: #1e293b;
                color: #ffffff;
                border-bottom: 2px solid #3b82f6;
                font-weight: 700;
            }
            QTabBar::tab:hover:!selected {
                background-color: #172554;
                color: #e2e8f0;
            }
            QProgressBar {
                background-color: #0f172a;
                border: 1px solid #2d3f55;
                border-radius: 8px;
                text-align: center;
                color: #ffffff;
                font-weight: 700;
                font-size: 12px;
                height: 22px;
            }
            QProgressBar::chunk {
                border-radius: 8px;
                background-color: #2563eb;
            }
            QScrollBar:vertical {
                background-color: #0f172a;
                width: 14px;
                margin: 0px;
                border: 1px solid #334155;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #475569;
                min-height: 30px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #3b82f6;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                background: none;
                border: none;
                height: 0px;
            }
            QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {
                background: none;
                border: none;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
            QScrollBar:horizontal {
                background-color: #0f172a;
                height: 14px;
                margin: 0px;
                border: 1px solid #334155;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal {
                background-color: #475569;
                min-width: 30px;
                border-radius: 5px;
            }
            QScrollBar::handle:horizontal:hover {
                background-color: #3b82f6;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                background: none;
                border: none;
                width: 0px;
            }
            QScrollBar::left-arrow:horizontal, QScrollBar::right-arrow:horizontal {
                background: none;
                border: none;
            }
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
                background: none;
            }
            QLabel#pageTitle {
                font-size: 22px;
                font-weight: 800;
                color: #ffffff;
            }
            QLabel#pageSubtitle {
                color: #94a3b8;
                font-size: 12px;
            }
            QLabel#sectionHeader {
                font-size: 15px;
                font-weight: 700;
                color: #ffffff;
            }
            QLabel#subText {
                color: #cbd5e1;
                font-size: 12px;
            }
            QSplitter::handle {
                background-color: #2d3f55;
            }
            QToolTip {
                background-color: #1e293b;
                color: #f1f5f9;
                border: 1px solid #334155;
                padding: 4px 8px;
                border-radius: 4px;
            }
            """
        else:
            qss = """
            QWidget {
                font-family: 'Poppins', 'Segoe UI', sans-serif;
                color: #1e293b;
                background-color: #f1f5f9;
            }
            QMainWindow {
                background-color: #f1f5f9;
            }
            QWidget#appRoot, QWidget#page, QStackedWidget {
                background-color: #f1f5f9;
                border: none;
            }
            QWidget {
                font-family: 'Poppins', 'Segoe UI', sans-serif;
                color: #0f172a;
                background-color: #f8fafc;
            }
            QMainWindow {
                background-color: #f8fafc;
            }
            QWidget#appRoot, QWidget#page, QStackedWidget {
                background-color: #f8fafc;
                border: none;
            }
            QLabel {
                color: #0f172a;
                background-color: transparent;
            }
            QStatusBar {
                background-color: #ffffff;
                color: #334155;
                border-top: 1px solid #cbd5e1;
                font-size: 12px;
                font-weight: 600;
            }
            QMenuBar {
                background-color: #ffffff;
                color: #0f172a;
                border-bottom: 1px solid #cbd5e1;
                font-weight: 600;
            }
            QMenuBar::item:selected {
                background-color: #eff6ff;
                color: #1d4ed8;
            }
            QMenu {
                background-color: #ffffff;
                border: 1px solid #cbd5e1;
                color: #0f172a;
            }
            QMenu::item:selected {
                background-color: #2563eb;
                color: #ffffff;
            }
            QFrame#sidebar {
                background-color: #ffffff;
                border-right: 1px solid #cbd5e1;
            }
            QLabel#sidebarTitle {
                font-weight: 800;
                font-size: 20px;
                color: #1d4ed8;
                letter-spacing: 0.5px;
                padding: 10px 0px;
            }
            QPushButton#sidebarBtn {
                background-color: transparent;
                color: #334155;
                border: none;
                border-left: 3px solid transparent;
                border-radius: 0px;
                padding: 14px 20px;
                text-align: left;
                font-weight: 700;
                font-size: 13px;
            }
            QPushButton#sidebarBtn:hover {
                background-color: #f1f5f9;
                color: #0f172a;
            }
            QPushButton#sidebarBtn:checked {
                background-color: #eff6ff;
                color: #1d4ed8;
                border-left: 3px solid #2563eb;
                font-weight: 800;
            }
            QFrame#card, QFrame#gestureCard, QFrame#statCard {
                background-color: #ffffff;
                border: 1px solid #cbd5e1;
                border-radius: 12px;
            }
            QFrame#cardHeader {
                border-bottom: 1px solid #cbd5e1;
            }
            QLabel#cardTitle {
                font-size: 14px;
                font-weight: 800;
                color: #0f172a;
            }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                background-color: #ffffff;
                border: 1px solid #94a3b8;
                border-radius: 8px;
                padding: 8px 12px;
                color: #0f172a;
                font-size: 13px;
                font-weight: 600;
            }
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
                border: 2px solid #2563eb;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 32px;
                border-left-width: 1px;
                border-left-color: #cbd5e1;
                border-left-style: solid;
                border-top-right-radius: 8px;
                border-bottom-right-radius: 8px;
                background-color: #f8fafc;
            }
            QComboBox::drop-down:hover {
                background-color: #eff6ff;
            }
            QComboBox::down-arrow {
                image: url({down_arrow_path});
                width: 12px;
                height: 12px;
            }
            QSpinBox::up-button, QDoubleSpinBox::up-button {
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 24px;
                border-left: 1px solid #cbd5e1;
                border-bottom: 1px solid #cbd5e1;
                border-top-right-radius: 8px;
                background-color: #f8fafc;
            }
            QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {
                background-color: #eff6ff;
            }
            QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
                image: url({up_arrow_path});
                width: 10px;
                height: 10px;
            }
            QSpinBox::down-button, QDoubleSpinBox::down-button {
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: 24px;
                border-left: 1px solid #cbd5e1;
                border-bottom-right-radius: 8px;
                background-color: #f8fafc;
            }
            QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
                background-color: #eff6ff;
            }
            QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
                image: url({down_arrow_path});
                width: 10px;
                height: 10px;
            }
            QCheckBox {
                spacing: 8px;
                font-weight: 700;
                color: #0f172a;
                background-color: transparent;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 4px;
                border: 1px solid #94a3b8;
                background-color: #ffffff;
            }
            QCheckBox::indicator:checked {
                background-color: #2563eb;
                border: 1px solid #2563eb;
            }
            QPushButton#primaryBtn {
                background-color: #2563eb;
                color: #ffffff;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-weight: 700;
                font-size: 13px;
            }
            QPushButton#primaryBtn:hover {
                background-color: #1d4ed8;
            }
            QPushButton#dangerBtn {
                background-color: #ef4444;
                color: #ffffff;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-weight: 700;
                font-size: 13px;
            }
            QPushButton#dangerBtn:hover {
                background-color: #dc2626;
            }
            QPushButton#secondaryBtn {
                background-color: #ffffff;
                color: #1e293b;
                border: 1px solid #94a3b8;
                border-radius: 8px;
                padding: 10px 20px;
                font-weight: 700;
                font-size: 13px;
            }
            QPushButton#secondaryBtn:hover {
                background-color: #eff6ff;
                color: #1d4ed8;
                border-color: #2563eb;
            }
            QTableView {
                background-color: #ffffff;
                alternate-background-color: #f8fafc;
                gridline-color: #cbd5e1;
                border: 1px solid #cbd5e1;
                border-radius: 12px;
                color: #0f172a;
            }
            QHeaderView::section {
                background-color: #f1f5f9;
                color: #0f172a;
                font-weight: 800;
                padding: 10px;
                border: none;
                border-right: 1px solid #cbd5e1;
                border-bottom: 2px solid #cbd5e1;
            }
            QTableWidget {
                background-color: #ffffff;
                border: 1px solid #cbd5e1;
                color: #0f172a;
            }
            QScrollBar:vertical {
                background-color: #f1f5f9;
                width: 14px;
                margin: 0px;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #94a3b8;
                min-height: 30px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #2563eb;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                background: none;
                border: none;
                height: 0px;
            }
            QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {
                background: none;
                border: none;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
            QScrollBar:horizontal {
                background-color: #f1f5f9;
                height: 14px;
                margin: 0px;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal {
                background-color: #94a3b8;
                min-width: 30px;
                border-radius: 5px;
            }
            QScrollBar::handle:horizontal:hover {
                background-color: #2563eb;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                background: none;
                border: none;
                width: 0px;
            }
            QScrollBar::left-arrow:horizontal, QScrollBar::right-arrow:horizontal {
                background: none;
                border: none;
            }
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
                background: none;
            }

            QLabel#pageTitle {
                font-size: 22px;
                font-weight: 800;
                color: #0f172a;
            }
            QLabel#pageSubtitle {
                color: #334155;
                font-size: 13px;
                font-weight: 500;
            }
            QLabel#sectionHeader {
                font-size: 15px;
                font-weight: 800;
                color: #0f172a;
            }
            QLabel#subText {
                color: #334155;
                font-size: 12px;
                font-weight: 500;
            }
            QLabel#sectionHeader {
                font-size: 15px;
                font-weight: 700;
                color: #0f172a;
            }
            QLabel#subText {
                color: #475569;
                font-size: 12px;
            }
            """
        qss = qss.replace("{down_arrow_path}", down_arrow_path).replace("{up_arrow_path}", up_arrow_path)
        self.setStyleSheet(qss)

    def _build_ui(self) -> None:
        # Top-level central widget
        central = QtWidgets.QWidget()
        central.setObjectName("appRoot")
        self.setCentralWidget(central)
        
        # Horizontal layout: Left Sidebar + Right Stacked Panel
        main_layout = QtWidgets.QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 1. Left Sidebar
        self._build_sidebar()
        main_layout.addWidget(self.sidebar_frame)
        
        # 2. Right Stacked Container
        self.stacked_widget = QtWidgets.QStackedWidget()
        main_layout.addWidget(self.stacked_widget, 1)
        
        # Add main pages (indices must match sidebar nav_items)
        self._create_dashboard_page()            # Index 0
        self._create_assembly_management_page()  # Index 1  ← NEW
        self._create_live_monitor_page()         # Index 2
        self._create_compliance_page()           # Index 3
        self._create_sop_config_page()           # Index 4
        self._create_operator_page()             # Index 5
        self._create_analytics_page()            # Index 6
        self._create_logs_page()                 # Index 7
        self._create_tool_page()                 # Index 8
        self._create_reference_page()            # Index 9
        self._create_settings_page()             # Index 10

        # Default start page (Live Monitor — now index 2)
        self.stacked_widget.setCurrentIndex(2)
        self.sidebar_buttons[2].setChecked(True)

    def _build_sidebar(self) -> None:
        self.sidebar_frame = QtWidgets.QFrame()
        self.sidebar_frame.setObjectName("sidebar")
        self.sidebar_frame.setFixedWidth(250)
        
        layout = QtWidgets.QVBoxLayout(self.sidebar_frame)
        layout.setContentsMargins(0, 24, 0, 24)
        layout.setSpacing(6)
        
        # Logo Icon & Title
        logo_layout = QtWidgets.QHBoxLayout()
        logo_layout.setContentsMargins(20, 0, 20, 10)
        logo_icon = QtWidgets.QLabel("")
        logo_icon.setStyleSheet("font-size: 24px; background-color: transparent;")
        title_label = QtWidgets.QLabel("Shravtek")
        title_label.setObjectName("sidebarTitle")
        logo_layout.addWidget(logo_icon)
        logo_layout.addWidget(title_label)
        logo_layout.addStretch(1)
        layout.addLayout(logo_layout)
        
        # Subtitle
        sub_label = QtWidgets.QLabel("HAGRID Industrial SOP System")
        sub_label.setStyleSheet("color: #64748b; font-size: 11px; font-weight: 700; margin-bottom: 20px; padding-left: 20px;")
        layout.addWidget(sub_label)
        
        # Navigation Buttons
        self.sidebar_buttons: List[QtWidgets.QPushButton] = []
        nav_items = [
            ("Dashboard", 0),
            ("Assembly Management", 1),   # NEW
            ("Live Monitor", 2),
            ("Compliance Monitor", 3),
            ("SOP Configuration", 4),
            ("Operator Management", 5),
            ("Production Analytics", 6),
            ("Detection Logs", 7),
            ("Tool Management", 8),
            ("Reference Images", 9),
            ("Settings", 10),
        ]
        
        self.btn_group = QtWidgets.QButtonGroup(self)
        self.btn_group.setExclusive(True)
        
        for text, index in nav_items:
            btn = QtWidgets.QPushButton(text)
            btn.setObjectName("sidebarBtn")
            btn.setCheckable(True)
            btn.setCursor(QtCore.Qt.PointingHandCursor)
            btn.clicked.connect(lambda checked, idx=index: self._on_sidebar_click(idx))
            self.sidebar_buttons.append(btn)
            self.btn_group.addButton(btn)
            layout.addWidget(btn)
            
        layout.addStretch(1)
        
        # Connected camera info
        self.conn_status = QtWidgets.QLabel("AI ENGINE ONLINE")
        self.conn_status.setStyleSheet("color: #16a34a; font-weight: 700; font-size: 11px; padding: 10px; margin: 10px; background-color: #f0fdf4; border-radius: 8px; border: 1px solid #bbf7d0;")
        self.conn_status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.conn_status)

    def _on_sidebar_click(self, index: int) -> None:
        nav_names = [
            "Dashboard", "Assembly Management", "Live Monitor", "Compliance Monitor",
            "SOP Configuration", "Operator Management", "Production Analytics",
            "Detection Logs", "Tool Management", "Reference Images", "Settings",
        ]
        page_name = nav_names[index] if index < len(nav_names) else f"Page {index}"
        self._activity_logger.log("NAVIGATION", "Sidebar", f"Navigate to {page_name}", f"Page index: {index}")
        self.stacked_widget.setCurrentIndex(index)
        if index == 0:
            self._update_dashboard_kpis()
        elif index == 1:
            self._refresh_assembly_management_page()
        elif index == 3 and hasattr(self, "compliance_page"):
            self.compliance_page.refresh_history()
        elif index == 6:
            self._update_analytics_charts()
            self._update_compliance_analytics()
        elif index == 7:
            self._refresh_logs_table()
            self._refresh_activity_logs_table()

    # -- PAGE 1: ASSEMBLY MANAGEMENT ----------------------------------------
    def _create_assembly_management_page(self) -> None:
        page = QtWidgets.QWidget()
        page.setObjectName("page")
        self.stacked_widget.addWidget(page)

        outer = QtWidgets.QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        outer.addWidget(scroll)

        content = QtWidgets.QWidget()
        scroll.setWidget(content)

        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(28, 28, 28, 28)
        layout.setSpacing(20)

        # ── Page header ──────────────────────────────────────────────────────
        hdr_row = QtWidgets.QHBoxLayout()
        title_col = QtWidgets.QVBoxLayout()
        am_title = QtWidgets.QLabel("Assembly Management")
        am_title.setStyleSheet("font-size: 22px; font-weight: 800; color: #0f172a;")
        am_sub = QtWidgets.QLabel(
            "Select the active assembly process. Live Monitor and SOP Configuration will "
            "automatically reconfigure for the chosen assembly."
        )
        am_sub.setStyleSheet("color: #64748b; font-size: 12px; font-weight: 500;")
        am_sub.setWordWrap(True)
        title_col.addWidget(am_title)
        title_col.addWidget(am_sub)
        hdr_row.addLayout(title_col, 1)

        # Active assembly badge (top-right)
        self.am_active_badge = QtWidgets.QLabel("— no assembly selected —")
        self.am_active_badge.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.am_active_badge.setStyleSheet(
            "background-color: #dcfce7; color: #15803d; border: 1px solid #86efac; "
            "border-radius: 10px; padding: 4px 14px; font-size: 11px; font-weight: 800;"
        )
        hdr_row.addWidget(self.am_active_badge)

        # Add New Assembly button (primary pill style, right-aligned)
        add_asm_btn = QtWidgets.QPushButton("+ Add New Assembly")
        add_asm_btn.setCursor(QtCore.Qt.PointingHandCursor)
        add_asm_btn.setStyleSheet("""
            QPushButton {
                background-color: #2563eb;
                color: #ffffff;
                border: none;
                border-radius: 8px;
                padding: 8px 20px;
                font-size: 13px;
                font-weight: 700;
                min-height: 36px;
            }
            QPushButton:hover { background-color: #1d4ed8; }
            QPushButton:pressed { background-color: #1e40af; }
        """)
        add_asm_btn.clicked.connect(self._show_add_assembly_dialog)
        hdr_row.addWidget(add_asm_btn)

        layout.addLayout(hdr_row)

        # ── Separator ────────────────────────────────────────────────────────
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.HLine)
        sep.setStyleSheet("background-color: #e2e8f0;")
        sep.setFixedHeight(1)
        layout.addWidget(sep)

        # ── Assembly cards grid ───────────────────────────────────────────────
        cards_label = QtWidgets.QLabel("Available Assembly Processes")
        cards_label.setStyleSheet("font-size: 15px; font-weight: 700; color: #0f172a;")
        layout.addWidget(cards_label)

        self.am_cards_layout = QtWidgets.QGridLayout()
        self.am_cards_layout.setSpacing(18)
        self.am_card_widgets: dict = {}  # assembly_id -> QFrame

        assemblies = self._assembly_manager.get_all_assemblies()
        active_id = self._assembly_manager.get_active_assembly_id()

        for col_idx, asm in enumerate(assemblies):
            card = self._build_assembly_card(asm, active=(asm["assembly_id"] == active_id))
            self.am_cards_layout.addWidget(card, 0, col_idx)
            self.am_card_widgets[asm["assembly_id"]] = card

        # Fill remaining columns with stretch spacers if fewer than 3 assemblies
        for c in range(len(assemblies), 3):
            self.am_cards_layout.setColumnStretch(c, 1)

        layout.addLayout(self.am_cards_layout)

        # ── Info banner for stub assemblies ───────────────────────────────────
        self.am_stub_banner = QtWidgets.QLabel(
            "NOTE: The selected assembly uses a stub (placeholder) pipeline. "
            "Live Monitor camera feed is active, but SOP monitoring logic is disabled "
            "until the detection pipeline for this assembly is implemented."
        )
        self.am_stub_banner.setWordWrap(True)
        self.am_stub_banner.setStyleSheet(
            "background-color: #fffbeb; color: #92400e; border: 1px solid #fde68a; "
            "border-radius: 8px; padding: 12px 16px; font-size: 12px; font-weight: 600;"
        )
        self.am_stub_banner.setVisible(active_id != "screw_tightening" and
                                       self._assembly_manager.get_detection_mode() == "stub")
        layout.addWidget(self.am_stub_banner)

        layout.addStretch(1)

        # Update top badge text
        self._refresh_assembly_management_page()

    def _build_assembly_card(self, asm: dict, active: bool) -> QtWidgets.QFrame:
        """Build a single assembly selection card."""
        assembly_id = asm["assembly_id"]
        display_name = asm.get("display_name", assembly_id)
        det_mode = asm.get("detection_mode", "stub")
        step_count = self._assembly_manager.get_step_count(assembly_id)

        card = QtWidgets.QFrame()
        card.setObjectName("card")
        card.setCursor(QtCore.Qt.PointingHandCursor)
        card.setMinimumHeight(200)
        card.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        self._style_assembly_card(card, active)

        card_layout = QtWidgets.QVBoxLayout(card)
        card_layout.setContentsMargins(22, 22, 22, 22)
        card_layout.setSpacing(12)

        # ── Top row: active badge (no icon — emoji renders as broken box) ────
        top_row = QtWidgets.QHBoxLayout()
        top_row.addStretch(1)

        if active:
            active_lbl = QtWidgets.QLabel("● ACTIVE")
            active_lbl.setStyleSheet(
                "color: #15803d; font-size: 11px; font-weight: 800; "
                "background-color: #dcfce7; border-radius: 8px; padding: 2px 10px; "
                "border: 1px solid #86efac;"
            )
            top_row.addWidget(active_lbl)
        card_layout.addLayout(top_row)

        # ── Assembly name ────────────────────────────────────────────────────
        name_lbl = QtWidgets.QLabel(display_name)
        name_lbl.setWordWrap(True)
        name_lbl.setStyleSheet(
            "font-size: 15px; font-weight: 800; color: #0f172a; background: transparent;"
        )
        card_layout.addWidget(name_lbl)

        # ── Metadata badges row ───────────────────────────────────────────────
        badge_row = QtWidgets.QHBoxLayout()
        badge_row.setSpacing(8)

        mode_badge = QtWidgets.QLabel(
            "Rotation Monitor" if det_mode == "screw_monitor" else "Pipeline: Stub"
        )
        mode_badge.setStyleSheet(
            f"background-color: {'#dbeafe' if det_mode == 'screw_monitor' else '#f3f4f6'}; "
            f"color: {'#1d4ed8' if det_mode == 'screw_monitor' else '#6b7280'}; "
            "border-radius: 6px; padding: 3px 10px; font-size: 11px; font-weight: 700;"
        )
        badge_row.addWidget(mode_badge)

        steps_badge = QtWidgets.QLabel(f"{step_count} SOP Steps")
        steps_badge.setStyleSheet(
            "background-color: #f0f9ff; color: #0369a1; border-radius: 6px; "
            "padding: 3px 10px; font-size: 11px; font-weight: 700;"
        )
        badge_row.addWidget(steps_badge)
        badge_row.addStretch(1)
        card_layout.addLayout(badge_row)

        # ── Activate button ───────────────────────────────────────────────────
        btn = QtWidgets.QPushButton("✓ Currently Active" if active else "Set as Active Assembly")
        btn.setObjectName("primaryBtn" if not active else "secondaryBtn")
        btn.setCursor(QtCore.Qt.PointingHandCursor)
        btn.setEnabled(not active)
        btn.clicked.connect(lambda _=False, aid=assembly_id: self._set_active_assembly(aid))
        card_layout.addWidget(btn)

        # Store the button reference for refresh
        card._activate_btn = btn
        card._name_lbl = name_lbl

        return card

    def _style_assembly_card(self, card: QtWidgets.QFrame, active: bool) -> None:
        if active:
            card.setStyleSheet(
                "QFrame#card { background-color: #eff6ff; border: 2px solid #2563eb; "
                "border-radius: 14px; }"
            )
        else:
            card.setStyleSheet(
                "QFrame#card { background-color: #ffffff; border: 1px solid #cbd5e1; "
                "border-radius: 14px; }"
                "QFrame#card:hover { border: 1px solid #2563eb; background-color: #f8fafc; }"
            )

    def _refresh_assembly_management_page(self) -> None:
        """Update assembly card highlights and the active badge label."""
        if not hasattr(self, "am_card_widgets"):
            return
        active_id = self._assembly_manager.get_active_assembly_id()
        active_name = self._assembly_manager.get_active_assembly().get(
            "display_name", active_id
        )
        if hasattr(self, "am_active_badge"):
            self.am_active_badge.setText(f"Active: {active_name}")

        for aid, card in self.am_card_widgets.items():
            is_active = (aid == active_id)
            self._style_assembly_card(card, is_active)
            if hasattr(card, "_activate_btn"):
                card._activate_btn.setEnabled(not is_active)
                card._activate_btn.setObjectName("secondaryBtn" if is_active else "primaryBtn")
                card._activate_btn.setText(
                    "✓ Currently Active" if is_active else "Set as Active Assembly"
                )

        det_mode = self._assembly_manager.get_detection_mode(active_id)
        if hasattr(self, "am_stub_banner"):
            self.am_stub_banner.setVisible(det_mode == "stub")

    def _show_add_assembly_dialog(self) -> None:
        """Open the 'Add New Assembly' form dialog."""
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Add New Assembly")
        dlg.setMinimumWidth(500)
        dlg.setModal(True)

        root = QtWidgets.QVBoxLayout(dlg)
        root.setSpacing(18)
        root.setContentsMargins(28, 28, 28, 28)

        # ── Title ────────────────────────────────────────────────────────────
        title_lbl = QtWidgets.QLabel("New Assembly Profile")
        title_lbl.setStyleSheet("font-size: 16px; font-weight: 800; color: #0f172a;")
        root.addWidget(title_lbl)

        sub_lbl = QtWidgets.QLabel(
            "Fill in the details below. The new assembly will appear in the card list "
            "in an inactive state — switch to it from the Assembly Management page."
        )
        sub_lbl.setWordWrap(True)
        sub_lbl.setStyleSheet("color: #64748b; font-size: 12px;")
        root.addWidget(sub_lbl)

        form = QtWidgets.QFormLayout()
        form.setSpacing(12)
        form.setLabelAlignment(Qt.AlignRight)

        # Assembly Display Name
        name_edit = QtWidgets.QLineEdit()
        name_edit.setPlaceholderText("e.g. Wire Harness Assembly")
        name_edit.setMinimumWidth(300)
        form.addRow("Assembly Name:", name_edit)

        # Assembly ID (auto-generated slug, editable)
        id_edit = QtWidgets.QLineEdit()
        id_edit.setPlaceholderText("e.g. wire_harness  (lowercase, underscores)")
        id_edit.setMinimumWidth(300)

        def _auto_slug(text: str) -> None:
            slug = text.strip().lower().replace(" ", "_")
            # Keep only alphanumeric and underscore
            slug = "".join(c if c.isalnum() or c == "_" else "" for c in slug)
            id_edit.blockSignals(True)
            id_edit.setText(slug)
            id_edit.blockSignals(False)

        name_edit.textChanged.connect(_auto_slug)
        form.addRow("Assembly ID:", id_edit)

        # Detection Mode
        mode_combo = QtWidgets.QComboBox()
        mode_combo.addItem("stub  (Placeholder — pipeline TBD)", userData="stub")
        mode_combo.addItem("screw_monitor  (Rotation-count pipeline)", userData="screw_monitor")
        form.addRow("Detection Mode:", mode_combo)

        # Turn Target (screw_monitor only)
        turn_spin = QtWidgets.QDoubleSpinBox()
        turn_spin.setRange(0.5, 20.0)
        turn_spin.setSingleStep(0.5)
        turn_spin.setValue(2.5)
        turn_spin.setEnabled(False)  # disabled until screw_monitor selected
        form.addRow("Rotation Turn Target:", turn_spin)

        def _on_mode_changed(_: int) -> None:
            turn_spin.setEnabled(mode_combo.currentData() == "screw_monitor")

        mode_combo.currentIndexChanged.connect(_on_mode_changed)

        # Error label
        err_lbl = QtWidgets.QLabel("")
        err_lbl.setStyleSheet("color: #dc2626; font-size: 12px; font-weight: 600;")
        err_lbl.setWordWrap(True)

        root.addLayout(form)
        root.addWidget(err_lbl)

        # ── Buttons ───────────────────────────────────────────────────────────
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)

        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.setStyleSheet(
            "QPushButton { background: #f1f5f9; color: #334155; border: 1px solid #cbd5e1; "
            "border-radius: 7px; padding: 7px 22px; font-size: 13px; font-weight: 600; }"
            "QPushButton:hover { background: #e2e8f0; }"
        )
        cancel_btn.clicked.connect(dlg.reject)
        btn_row.addWidget(cancel_btn)

        create_btn = QtWidgets.QPushButton("Create Assembly")
        create_btn.setStyleSheet(
            "QPushButton { background: #2563eb; color: #fff; border: none; "
            "border-radius: 7px; padding: 7px 22px; font-size: 13px; font-weight: 700; }"
            "QPushButton:hover { background: #1d4ed8; }"
            "QPushButton:pressed { background: #1e40af; }"
        )
        btn_row.addWidget(create_btn)
        root.addLayout(btn_row)

        def _on_create() -> None:
            d_name = name_edit.text().strip()
            asm_id = id_edit.text().strip()
            det_mode = mode_combo.currentData()

            # Validation
            if not d_name:
                err_lbl.setText("Assembly Name is required.")
                return
            if not asm_id:
                err_lbl.setText("Assembly ID is required (auto-generated from name).")
                return
            if not asm_id.replace("_", "").isalnum():
                err_lbl.setText("Assembly ID may only contain lowercase letters, digits, and underscores.")
                return
            if self._assembly_manager.assembly_id_exists(asm_id):
                err_lbl.setText(f"An assembly with ID '{asm_id}' already exists. Choose a different name.")
                return

            t_params: dict = {}
            if det_mode == "screw_monitor":
                t_params = {"turn_target": float(turn_spin.value()), "align_threshold_px": 50}

            ok = self._assembly_manager.add_assembly(
                assembly_id=asm_id,
                display_name=d_name,
                detection_mode=det_mode,
                target_params=t_params,
                sop_steps=[],
            )
            if not ok:
                err_lbl.setText("Failed to create assembly (duplicate ID).")
                return

            # Refresh SOP config combo so the new assembly appears there too
            if hasattr(self, "sop_asm_combo"):
                self.sop_asm_combo.addItem(d_name, userData=asm_id)

            self._activity_logger.log(
                "ASSEMBLY", "Assembly Management", "Add Assembly",
                f"Created: {d_name} ({asm_id})"
            )
            dlg.accept()
            self._rebuild_all_assembly_cards()
            QtWidgets.QMessageBox.information(
                self, "Assembly Created",
                f"'{d_name}' has been added successfully.\n\n"
                "You can now edit its SOP steps in the SOP Configuration page, "
                "then switch to it from the Assembly Management page."
            )

        create_btn.clicked.connect(_on_create)
        dlg.exec()

    def _rebuild_all_assembly_cards(self) -> None:
        """Tear down and repopulate the assembly cards grid in-place."""
        if not hasattr(self, "am_cards_layout") or not hasattr(self, "am_card_widgets"):
            return

        # Remove all existing widgets from the grid
        while self.am_cards_layout.count():
            item = self.am_cards_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.am_card_widgets.clear()

        # Re-add cards for all assemblies (includes newly created ones)
        assemblies = self._assembly_manager.get_all_assemblies()
        active_id = self._assembly_manager.get_active_assembly_id()

        cols = 3  # max columns before wrapping
        for idx, asm in enumerate(assemblies):
            row, col = divmod(idx, cols)
            card = self._build_assembly_card(asm, active=(asm["assembly_id"] == active_id))
            self.am_cards_layout.addWidget(card, row, col)
            self.am_card_widgets[asm["assembly_id"]] = card

        self._refresh_assembly_management_page()

    def _set_active_assembly(self, assembly_id: str) -> None:
        """Switch the active assembly — stop/restart monitor if it is running."""
        monitor_was_running = (

            self.worker is not None and self.worker.isRunning()
        )
        if monitor_was_running:
            reply = QtWidgets.QMessageBox.question(
                self, "Switch Assembly?",
                "The Live Monitor is currently running.\n\n"
                "Switching assemblies will stop the monitor, reset all counters, "
                "and reinitialise with the new assembly's pipeline and SOP steps.\n\n"
                "Continue?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if reply != QtWidgets.QMessageBox.Yes:
                return
            self._stop_worker()

        self._assembly_manager.set_active_assembly(assembly_id)
        _asm = self._assembly_manager.get_active_assembly()
        self.active_sop_name = _asm.get("display_name", assembly_id)
        self.active_turn_target = float(_asm.get("target_params", {}).get("turn_target", 2.5))
        self._active_detection_mode = _asm.get("detection_mode", "screw_monitor")
        self._active_align_threshold = int(_asm.get("target_params", {}).get("align_threshold_px", 50))

        self._reload_live_monitor_for_assembly()
        self._reload_sop_config_for_assembly()
        self._refresh_assembly_management_page()
        self._update_dashboard_kpis()

        self._activity_logger.log(
            "ASSEMBLY", "Assembly Management", "Set Active Assembly",
            f"Switched to: {self.active_sop_name}"
        )
        self.statusBar().showMessage(
            f"Active assembly set to: {self.active_sop_name}", 4000
        )

    def _reload_live_monitor_for_assembly(self) -> None:
        """Rebuild the Live Monitor header and SOP panel for the active assembly."""
        # Update subtitle label
        _asm = self._assembly_manager.get_active_assembly()
        det_mode = _asm.get("detection_mode", "screw_monitor")
        if hasattr(self, "sub_title_sop"):
            if det_mode == "screw_monitor":
                self.sub_title_sop.setText(
                    f"SOP: {self.active_sop_name} | Target Turns: {self.active_turn_target}"
                )
            else:
                self.sub_title_sop.setText(
                    f"SOP: {self.active_sop_name} | Pipeline: Under Development"
                )

        # Enable/disable the SOP Monitor Logic checkbox
        if hasattr(self, "chk_monitor"):
            if det_mode == "stub":
                self.chk_monitor.setChecked(False)
                self.chk_monitor.setEnabled(False)
                self.chk_monitor.setToolTip(
                    "SOP monitoring is disabled for this assembly (pipeline not yet implemented)."
                )
            else:
                self.chk_monitor.setEnabled(True)
                self.chk_monitor.setChecked(True)
                self.chk_monitor.setToolTip("")

        # Rebuild the SOP panel with steps for the active assembly
        new_steps = self._assembly_manager.get_sop_steps()
        if hasattr(self, "sop_panel") and self.sop_panel is not None:
            self.sop_panel.rebuild_steps(new_steps)
            self.sop_panel.set_theme(getattr(self.sop_panel, "is_dark", False))

    def _reload_sop_config_for_assembly(self, assembly_id: str | None = None) -> None:
        """Repopulate the SOP Configuration page for the given (or active) assembly."""
        if assembly_id is None:
            assembly_id = self._assembly_manager.get_active_assembly_id()
        _asm = self._assembly_manager._profile_by_id(assembly_id)
        if _asm is None:
            return

        det_mode = _asm.get("detection_mode", "screw_monitor")
        params = _asm.get("target_params", {})
        steps = self._assembly_manager.get_sop_steps(assembly_id)

        # Update form fields
        if hasattr(self, "sop_name_edit"):
            self.sop_name_edit.setText(_asm.get("display_name", ""))
        if hasattr(self, "turns_spin"):
            self.turns_spin.setValue(float(params.get("turn_target", 2.5)))
            self.turns_spin.setEnabled(det_mode == "screw_monitor")
        if hasattr(self, "align_thresh_spin"):
            self.align_thresh_spin.setValue(int(params.get("align_threshold_px", 50)))
            self.align_thresh_spin.setEnabled(det_mode == "screw_monitor")

        # Sync the SOP config assembly selector combo (if it exists)
        if hasattr(self, "sop_asm_combo"):
            idx = self.sop_asm_combo.findData(assembly_id)
            if idx >= 0:
                self.sop_asm_combo.blockSignals(True)
                self.sop_asm_combo.setCurrentIndex(idx)
                self.sop_asm_combo.blockSignals(False)

        # Repopulate steps table
        if not hasattr(self, "steps_table"):
            return
        self.steps_table.setRowCount(0)
        for step in steps:
            self._add_step_row_data(
                str(step.index),
                step.title,
                step.description,
                step.ai_validation,
                step.expected_result,
                step.timeout,
                step.criteria,
                step.warning_msg,
                step.next_step,
            )

    # -- PAGE 0: DASHBOARD --------------------------------------------------
    def _create_dashboard_page(self) -> None:

        page = QtWidgets.QWidget()
        page.setObjectName("page")
        self.stacked_widget.addWidget(page)
        
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)
        
        # Page Title / Header
        header = QtWidgets.QHBoxLayout()
        title_section = QtWidgets.QVBoxLayout()
        self.dash_title_lbl = QtWidgets.QLabel("Dashboard Summary")
        self.dash_title_lbl.setStyleSheet("font-size: 22px; font-weight: 800; color: #0f172a;")
        self.dash_sub_lbl = QtWidgets.QLabel("Real-time summary of AI inspections, operator metrics, and system status.")
        self.dash_sub_lbl.setStyleSheet("color: #64748b; font-size: 12px;")
        title_section.addWidget(self.dash_title_lbl)
        title_section.addWidget(self.dash_sub_lbl)
        header.addLayout(title_section)
        
        # Shift and Date info
        info_layout = QtWidgets.QVBoxLayout()
        self.dash_time_lbl = QtWidgets.QLabel(_now_short())
        self.dash_time_lbl.setStyleSheet("font-weight: 700; color: #2563eb; font-size: 13px;")
        self.dash_shift_lbl = QtWidgets.QLabel("Shift: Shift A (06:00 - 14:00)")
        self.dash_shift_lbl.setStyleSheet("color: #475569; font-size: 11px; font-weight: 600;")
        info_layout.addWidget(self.dash_time_lbl)
        info_layout.addWidget(self.dash_shift_lbl)
        header.addLayout(info_layout)
        layout.addLayout(header)
        
        # KPI Grid Layout (4 columns x 3 rows)
        kpi_grid = QtWidgets.QGridLayout()
        kpi_grid.setSpacing(14)
        
        self.dash_kpi_total = StatCard("Total Production", "142 units", "#2563eb")
        self.dash_kpi_ok = StatCard("Total OK", "138 units", "#16a34a")
        self.dash_kpi_ng = StatCard("Total NG", "4 units", "#dc2626")
        self.dash_kpi_yield = StatCard("Yield Rate", "97.2%", "#0284c7")
        
        self.dash_kpi_acc = StatCard("Detection Accuracy", "98.5%", "#8b5cf6")
        self.dash_kpi_workers = StatCard("Active Worker Count", "1 Operator", "#ec4899")
        self.dash_kpi_sop = StatCard("Running SOP", self.active_sop_name, "#3b82f6")
        self.dash_kpi_today = StatCard("Today's Count", "142 items", "#10b981")
        
        self.dash_kpi_status = StatCard("Machine Status", "RUNNING", "#16a34a")
        self.dash_kpi_fps = StatCard("Camera FPS", "30 FPS", "#f97316")
        self.dash_kpi_warnings = StatCard("Compliance Warnings", "0", "#dc2626")
        self.dash_kpi_compliance = StatCard("Compliance Rate", "100%", "#16a34a")
        
        kpi_grid.addWidget(self.dash_kpi_total, 0, 0)
        kpi_grid.addWidget(self.dash_kpi_ok, 0, 1)
        kpi_grid.addWidget(self.dash_kpi_ng, 0, 2)
        kpi_grid.addWidget(self.dash_kpi_yield, 0, 3)
        
        kpi_grid.addWidget(self.dash_kpi_acc, 1, 0)
        kpi_grid.addWidget(self.dash_kpi_workers, 1, 1)
        kpi_grid.addWidget(self.dash_kpi_sop, 1, 2)
        kpi_grid.addWidget(self.dash_kpi_today, 1, 3)
        
        kpi_grid.addWidget(self.dash_kpi_status, 2, 0)
        kpi_grid.addWidget(self.dash_kpi_fps, 2, 1)
        kpi_grid.addWidget(self.dash_kpi_warnings, 2, 2)
        kpi_grid.addWidget(self.dash_kpi_compliance, 2, 3)
        layout.addLayout(kpi_grid)
        
        # Bottom Grid: System Health Connection Statuses
        status_card = QtWidgets.QFrame()
        status_card.setObjectName("card")
        status_layout = QtWidgets.QVBoxLayout(status_card)
        status_layout.setContentsMargins(18, 18, 18, 18)
        status_layout.setSpacing(12)
        
        self.dash_health_title = QtWidgets.QLabel("System Health & Connectivity Indicators")
        self.dash_health_title.setStyleSheet("font-size: 15px; font-weight: 700; color: #0f172a;")
        status_layout.addWidget(self.dash_health_title)
        
        indicators_layout = QtWidgets.QHBoxLayout()
        self.ind_cam = self._make_badge("Camera Connected", "success")
        self.ind_ai = self._make_badge("AI Engine Active", "success")
        self.ind_db = self._make_badge("Database Connected", "success")
        self.ind_mes = self._make_badge("MES/ERP Connected", "success")
        self.ind_gpu = self._make_badge("GPU Available", "success")
        
        indicators_layout.addWidget(self.ind_cam)
        indicators_layout.addWidget(self.ind_ai)
        indicators_layout.addWidget(self.ind_db)
        indicators_layout.addWidget(self.ind_mes)
        indicators_layout.addWidget(self.ind_gpu)
        status_layout.addLayout(indicators_layout)
        
        # CPU/Mem Usage Info
        self.dash_sys_details = QtWidgets.QLabel("Hardware: CPU Usage: 12% | GPU Temp: 54°C | Memory Usage: 34% (5.4GB / 16.0GB) | SQLite Active")
        self.dash_sys_details.setStyleSheet("color: #64748b; font-size: 12px; font-weight: 500;")
        status_layout.addWidget(self.dash_sys_details)
        
        layout.addWidget(status_card)
        layout.addStretch(1)

    def _make_badge(self, text: str, status_type: str) -> QtWidgets.QLabel:
        lbl = QtWidgets.QLabel(text)
        lbl.setAlignment(Qt.AlignCenter)
        if status_type == "success":
            lbl.setStyleSheet("background-color: #dcfce7; color: #16a34a; border: 1px solid #bbf7d0; border-radius: 6px; padding: 6px 12px; font-weight: 700; font-size: 11px;")
        elif status_type == "danger":
            lbl.setStyleSheet("background-color: #fee2e2; color: #dc2626; border: 1px solid #fecaca; border-radius: 6px; padding: 6px 12px; font-weight: 700; font-size: 11px;")
        else: # warning
            lbl.setStyleSheet("background-color: #fffbeb; color: #d97706; border: 1px solid #fde68a; border-radius: 6px; padding: 6px 12px; font-weight: 700; font-size: 11px;")
        return lbl

    def _update_dashboard_kpis(self) -> None:
        self.dash_time_lbl.setText(_now_short())
        self.dash_kpi_sop.set_value(self.active_sop_name)
        if os.path.exists(LOG_CSV_PATH):
            try:
                ok_cnt = 0
                ng_cnt = 0
                with open(LOG_CSV_PATH, newline="") as fh:
                    reader = csv.reader(fh)
                    next(reader, None)  # header
                    for row in reader:
                        if len(row) > 9:
                            if row[9] == "Completed":
                                ok_cnt += 1
                            else:
                                ng_cnt += 1
                total = ok_cnt + ng_cnt
                if total > 0:
                    yield_rate = (ok_cnt / total) * 100.0
                    self.dash_kpi_total.set_value(f"{total} units")
                    self.dash_kpi_ok.set_value(f"{ok_cnt} units")
                    self.dash_kpi_ng.set_value(f"{ng_cnt} units")
                    self.dash_kpi_yield.set_value(f"{yield_rate:.1f}%")
                    self.dash_kpi_today.set_value(f"{total} items")
            except Exception:
                pass
        try:
            summary = self._compliance_logger.daily_summary(days=1)
            self.dash_kpi_warnings.set_value(str(summary.get("warnings", 0)))
            self.dash_kpi_compliance.set_value(f"{summary.get('compliance_pct', 100)}%")
        except Exception:
            pass

    # -- PAGE 1: LIVE MONITOR -----------------------------------------------
    def _create_live_monitor_page(self) -> None:
        page = QtWidgets.QWidget()
        page.setObjectName("page")
        self.stacked_widget.addWidget(page)
        
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)
        
        # Top Actionbar — title on its own line
        header = QtWidgets.QHBoxLayout()
        title_section = QtWidgets.QVBoxLayout()
        title = QtWidgets.QLabel("Operations Live Feed")
        title.setStyleSheet("font-size: 20px; font-weight: 800; color: #0f172a;")
        self.sub_title_sop = QtWidgets.QLabel(f"SOP: {self.active_sop_name} | Target Turns: {self.active_turn_target}")
        self.sub_title_sop.setStyleSheet("color: #64748b; font-size: 12px;")
        title_section.addWidget(title)
        title_section.addWidget(self.sub_title_sop)
        header.addLayout(title_section)
        header.addStretch(1)
        layout.addLayout(header)

        # Action row: overlay/detection checkboxes (left) + Start/Stop/Reset (right)
        action_row = QtWidgets.QHBoxLayout()
        action_row.setContentsMargins(0, 0, 0, 0)
        action_row.setSpacing(16)

        self.chk_landmarks = QtWidgets.QCheckBox("Show Landmarks Overlay")
        self.chk_landmarks.setChecked(True)
        self.chk_landmarks.toggled.connect(self._on_flags)
        
        self.chk_gesture = QtWidgets.QCheckBox("Gesture Detection")
        self.chk_gesture.setChecked(True)
        self.chk_gesture.toggled.connect(self._on_flags)
        
        self.chk_monitor = QtWidgets.QCheckBox("SOP Monitor Logic")
        self.chk_monitor.setChecked(True)
        self.chk_monitor.toggled.connect(self._on_flags)

        self.chk_compliance = QtWidgets.QCheckBox("Compliance Monitoring")
        self.chk_compliance.setChecked(bool(self._compliance_settings.get("enabled", True)))
        self.chk_compliance.toggled.connect(self._on_flags)

        action_row.addWidget(self.chk_landmarks)
        action_row.addWidget(self.chk_gesture)
        action_row.addWidget(self.chk_monitor)
        action_row.addWidget(self.chk_compliance)
        action_row.addStretch(1)

        # Top action buttons
        self.start_btn = QtWidgets.QPushButton("Start Monitor")
        self.start_btn.setObjectName("primaryBtn")
        self.start_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.start_btn.clicked.connect(self._start_worker)
        
        self.stop_btn = QtWidgets.QPushButton("Stop Feed")
        self.stop_btn.setObjectName("dangerBtn")
        self.stop_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.stop_btn.clicked.connect(self._stop_worker)
        
        self.reset_btn = QtWidgets.QPushButton("Reset Step")
        self.reset_btn.setObjectName("secondaryBtn")
        self.reset_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.reset_btn.clicked.connect(self._reset_monitor)
        
        action_row.addWidget(self.start_btn)
        action_row.addWidget(self.stop_btn)
        action_row.addWidget(self.reset_btn)
        layout.addLayout(action_row)
        
        # Main content area: split left and right
        split_layout = QtWidgets.QHBoxLayout()
        split_layout.setSpacing(14)
        
        # Left Side: Camera display + overlays controls
        left_layout = QtWidgets.QVBoxLayout()
        left_layout.setSpacing(10)
        
        self.camera_view = CameraView()
        cam_container = QtWidgets.QFrame()
        cam_container.setObjectName("card")
        cam_layout = QtWidgets.QVBoxLayout(cam_container)
        cam_layout.setContentsMargins(2, 2, 2, 2)
        cam_layout.addWidget(self.camera_view)
        left_layout.addWidget(cam_container, 1)
        
        # Controls bar: Operator + Camera picker (left) / Refresh + Test (right)
        controls_layout = QtWidgets.QHBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(16)

        controls_layout.addWidget(QtWidgets.QLabel("Operator:"))
        self.worker_id_input = QtWidgets.QLineEdit(self.default_worker_id)
        self.worker_id_input.setFixedWidth(100)
        controls_layout.addWidget(self.worker_id_input)
        
        controls_layout.addWidget(QtWidgets.QLabel("Camera:"))
        self.source_combo = QtWidgets.QComboBox()
        self.source_combo.setFixedWidth(220)
        self.source_combo.setToolTip(
            "Index 0 is the system's default/built-in camera. "
            "1, 2, ... are external USB/webcams, if connected."
        )
        controls_layout.addWidget(self.source_combo)

        self.select_cam_btn = QtWidgets.QPushButton("Select Camera")
        self.select_cam_btn.setObjectName("primaryBtn")
        self.select_cam_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.select_cam_btn.setToolTip(
            "Open and start monitoring with the camera currently chosen in the dropdown."
        )
        self.select_cam_btn.clicked.connect(self._select_camera)
        controls_layout.addWidget(self.select_cam_btn)

        controls_layout.addStretch(1)

        self.refresh_cam_btn = QtWidgets.QPushButton("Refresh Cameras")
        self.refresh_cam_btn.setObjectName("secondaryBtn")
        self.refresh_cam_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.refresh_cam_btn.clicked.connect(self._refresh_camera_list)
        controls_layout.addWidget(self.refresh_cam_btn)

        self.test_cam_btn = QtWidgets.QPushButton("Test Camera")
        self.test_cam_btn.setObjectName("secondaryBtn")
        self.test_cam_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.test_cam_btn.setToolTip(
            "Grabs a single preview frame from the selected camera index, "
            "without starting full monitoring — use this to confirm the "
            "dropdown selection actually opens the physical camera you expect."
        )
        self.test_cam_btn.clicked.connect(self._test_selected_camera)
        controls_layout.addWidget(self.test_cam_btn)

        self._refresh_camera_list(preferred_source=self.source)

        left_layout.addLayout(controls_layout)
        split_layout.addLayout(left_layout, 3)
        
        # Right Side: scrollable stack of status / SOP / compliance panels.
        # A dedicated vertical scrollbar keeps every section fully reachable
        # without overlapping when the window is shorter than content.
        self.right_scroll = QtWidgets.QScrollArea()
        self.right_scroll.setObjectName("liveRightScroll")
        self.right_scroll.setWidgetResizable(True)
        self.right_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.right_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.right_scroll.setMinimumWidth(280)
        self.right_scroll.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        self.right_scroll.setStyleSheet(
            "QScrollArea#liveRightScroll { background: transparent; border: none; }"
        )

        right_content = QtWidgets.QWidget()
        right_content.setObjectName("liveRightContent")
        right_content.setStyleSheet("QWidget#liveRightContent { background: transparent; }")
        # Minimum vertical policy prevents the scroll area from squashing children
        # below their sizeHint (which previously caused overlapping cards).
        right_content.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum
        )
        right_layout = QtWidgets.QVBoxLayout(right_content)
        right_layout.setContentsMargins(0, 0, 8, 0)
        right_layout.setSpacing(14)
        right_layout.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)

        self.status_panel = StatusPanel()
        right_layout.addWidget(self.status_panel, 0, Qt.AlignTop)

        self.sop_panel = SopStepPanel()
        self.sop_panel.refresh_callback = self.sync_steps_from_config
        right_layout.addWidget(self.sop_panel, 0, Qt.AlignTop)

        self.compliance_panel = CompliancePanel()
        right_layout.addWidget(self.compliance_panel, 0, Qt.AlignTop)

        right_layout.addStretch(1)
        self.right_scroll.setWidget(right_content)
        split_layout.addWidget(self.right_scroll, 2)
        layout.addLayout(split_layout, 1)
        
        # Bottom area: Log summary
        self.log_panel = LogPanel()
        layout.addWidget(self.log_panel)

    # -- PAGE 2: SOP CONFIGURATION ------------------------------------------
    def _create_sop_config_page(self) -> None:
        page = QtWidgets.QWidget()
        page.setObjectName("page")
        self.stacked_widget.addWidget(page)
        
        main_layout = QtWidgets.QVBoxLayout(page)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        main_layout.addWidget(scroll)
        
        content = QtWidgets.QWidget()
        scroll.setWidget(content)
        
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)
        
        title = QtWidgets.QLabel("SOP Configuration & Parameters")
        title.setStyleSheet("font-size: 20px; font-weight: 800; color: #0f172a;")
        layout.addWidget(title)

        # ── Assembly selector (per-assembly SOP editing) ──────────────────────
        asm_row = QtWidgets.QHBoxLayout()
        asm_lbl = QtWidgets.QLabel("Editing SOP for Assembly:")
        asm_lbl.setStyleSheet("font-size: 13px; font-weight: 700; color: #0f172a;")
        asm_row.addWidget(asm_lbl)

        self.sop_asm_combo = QtWidgets.QComboBox()
        self.sop_asm_combo.setMinimumWidth(280)
        for asm in self._assembly_manager.get_all_assemblies():
            self.sop_asm_combo.addItem(asm["display_name"], userData=asm["assembly_id"])
        # Select the currently active assembly
        _cur_idx = self.sop_asm_combo.findData(
            self._assembly_manager.get_active_assembly_id()
        )
        if _cur_idx >= 0:
            self.sop_asm_combo.setCurrentIndex(_cur_idx)
        self.sop_asm_combo.currentIndexChanged.connect(self._on_sop_asm_combo_changed)
        asm_row.addWidget(self.sop_asm_combo)
        asm_row.addStretch(1)
        layout.addLayout(asm_row)
        
        form_frame = QtWidgets.QFrame()
        form_frame.setObjectName("card")
        form_layout = QtWidgets.QFormLayout(form_frame)
        form_layout.setContentsMargins(20, 20, 20, 20)
        form_layout.setSpacing(14)
        
        self.sop_name_edit = QtWidgets.QLineEdit(self.active_sop_name)
        form_layout.addRow(QtWidgets.QLabel("SOP / Assembly Name:"), self.sop_name_edit)
        
        self.prod_name_edit = QtWidgets.QLineEdit(self.active_product_name)
        form_layout.addRow(QtWidgets.QLabel("Product Name:"), self.prod_name_edit)
        
        self.prod_variant_edit = QtWidgets.QLineEdit(self.active_variant)
        form_layout.addRow(QtWidgets.QLabel("Product Variant:"), self.prod_variant_edit)
        
        self.turns_spin = QtWidgets.QDoubleSpinBox()
        self.turns_spin.setRange(0.5, 10.0)
        self.turns_spin.setSingleStep(0.5)
        self.turns_spin.setValue(self.active_turn_target)
        form_layout.addRow(QtWidgets.QLabel("Rotation Turn Target:"), self.turns_spin)
        
        self.align_thresh_spin = QtWidgets.QSpinBox()
        self.align_thresh_spin.setRange(10, 200)
        self.align_thresh_spin.setValue(50)
        self.align_thresh_spin.setSuffix("px")
        form_layout.addRow(QtWidgets.QLabel("Alignment Distance Threshold:"), self.align_thresh_spin)
        
        layout.addWidget(form_frame)
        
        # SOP Steps checklist builder
        steps_card = QtWidgets.QFrame()
        steps_card.setObjectName("card")
        steps_layout = QtWidgets.QVBoxLayout(steps_card)
        steps_layout.setContentsMargins(16, 16, 16, 16)
        
        steps_header = QtWidgets.QHBoxLayout()
        steps_title = QtWidgets.QLabel("Sequence Steps Checklist Builder")
        steps_title.setStyleSheet("font-size: 16px; font-weight: 700; color: #0f172a;")
        steps_header.addWidget(steps_title)
        steps_header.addStretch(1)
        
        # Add Preset button dropdown
        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.addItems([
            "Select Preset Option Step...",
            "Step 9 – Weight Verification",
            "Step 10 – Barcode / QR Code Scan",
            "Step 11 – Label Verification",
            "Step 12 – Reference Image Comparison",
            "Step 13 – Work Area Cleanliness",
            "Step 14 – Final Supervisor Approval"
        ])
        self.preset_combo.currentIndexChanged.connect(self._add_preset_step)
        self.preset_combo.setMinimumWidth(220)
        self.preset_combo.setObjectName("secondaryBtn")
        self.preset_combo.setStyleSheet("font-size: 12px; height: 32px; padding: 0px 8px;")
        steps_header.addWidget(self.preset_combo)

        self.delete_step_btn = QtWidgets.QPushButton("Delete")
        self.delete_step_btn.setObjectName("dangerBtn")
        self.delete_step_btn.clicked.connect(self._delete_sop_step_row)
        steps_header.addWidget(self.delete_step_btn)

        self.add_step_btn = QtWidgets.QPushButton("+ Add Step")
        self.add_step_btn.setObjectName("secondaryBtn")
        self.add_step_btn.clicked.connect(self._add_sop_step_row)
        steps_header.addWidget(self.add_step_btn)
        steps_layout.addLayout(steps_header)
        
        self.steps_table = QtWidgets.QTableWidget(0, 9)
        self.steps_table.setHorizontalHeaderLabels([
            "Step Index", 
            "Title", 
            "Instruction Description", 
            "AI Validation Type",
            "Expected Result",
            "Timeout (sec)",
            "Pass/Fail Criteria",
            "Warning Message",
            "Next Step"
        ])
        self.steps_table.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.steps_table.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.steps_table.horizontalHeader().setStretchLastSection(False)
        self.steps_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Interactive)
        self.steps_table.verticalHeader().setVisible(False)
        self.steps_table.verticalHeader().setDefaultSectionSize(44)
        self.steps_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.steps_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.steps_table.setMinimumHeight(440)
        
        # Wide column widths so text is fully readable and slide bar (scrollbar) is active
        column_widths = [105, 180, 320, 260, 160, 110, 200, 240, 100]
        for col_idx, w in enumerate(column_widths):
            self.steps_table.setColumnWidth(col_idx, w)
        
        # Populate initial 8 steps
        for step in STEPS:
            self._add_step_row_data(
                str(step.index),
                step.title,
                step.description,
                step.ai_validation,
                step.expected_result,
                step.timeout,
                step.criteria,
                step.warning_msg,
                step.next_step
            )
            
        steps_layout.addWidget(self.steps_table)
        layout.addWidget(steps_card)

        # Compliance detector configuration
        comp_card = QtWidgets.QFrame()
        comp_card.setObjectName("card")
        comp_layout = QtWidgets.QVBoxLayout(comp_card)
        comp_layout.setContentsMargins(16, 16, 16, 16)
        comp_layout.setSpacing(10)
        comp_title = QtWidgets.QLabel("Operator Compliance Detectors")
        comp_title.setStyleSheet("font-size: 14px; font-weight: 700; color: #0f172a;")
        comp_layout.addWidget(comp_title)

        self.chk_comp_master = QtWidgets.QCheckBox("Enable Compliance Monitoring Module")
        self.chk_comp_master.setChecked(bool(self._compliance_settings.get("enabled", True)))
        comp_layout.addWidget(self.chk_comp_master)

        dets = self._compliance_settings.get("detectors", {})
        det_row = QtWidgets.QHBoxLayout()
        self.chk_det_phone = QtWidgets.QCheckBox("Mobile Phone")
        self.chk_det_phone.setChecked(bool(dets.get("mobile_phone", True)))
        self.chk_det_shirt = QtWidgets.QCheckBox("Shirt Button")
        self.chk_det_shirt.setChecked(bool(dets.get("shirt_button", True)))
        self.chk_det_buds = QtWidgets.QCheckBox("Bluetooth / Earbuds")
        self.chk_det_buds.setChecked(bool(dets.get("bluetooth_earbuds", True)))
        self.chk_det_specs = QtWidgets.QCheckBox("Spectacles")
        self.chk_det_specs.setChecked(bool(dets.get("spectacles", True)))
        self.chk_det_write = QtWidgets.QCheckBox("Writing Activity")
        self.chk_det_write.setChecked(bool(dets.get("writing", True)))
        for w in (self.chk_det_phone, self.chk_det_shirt, self.chk_det_buds, self.chk_det_specs, self.chk_det_write):
            det_row.addWidget(w)
        det_row.addStretch(1)
        comp_layout.addLayout(det_row)

        thresh_form = QtWidgets.QFormLayout()
        self.comp_conf_spin = QtWidgets.QDoubleSpinBox()
        self.comp_conf_spin.setRange(0.10, 0.95)
        self.comp_conf_spin.setSingleStep(0.05)
        self.comp_conf_spin.setValue(float(self._compliance_settings.get("confidence_threshold", 0.45)))
        thresh_form.addRow(QtWidgets.QLabel("Compliance Confidence Threshold:"), self.comp_conf_spin)

        self.comp_cooldown_spin = QtWidgets.QDoubleSpinBox()
        self.comp_cooldown_spin.setRange(1.0, 60.0)
        self.comp_cooldown_spin.setSuffix(" s")
        self.comp_cooldown_spin.setValue(float(self._compliance_settings.get("warning_cooldown_sec", 8.0)))
        thresh_form.addRow(QtWidgets.QLabel("Warning Cooldown:"), self.comp_cooldown_spin)

        self.chk_comp_evidence = QtWidgets.QCheckBox("Save evidence screenshots")
        self.chk_comp_evidence.setChecked(bool(self._compliance_settings.get("save_evidence_images", True)))
        thresh_form.addRow(self.chk_comp_evidence)

        self.chk_comp_notify_pass = QtWidgets.QCheckBox("Notify on Compliance Passed")
        self.chk_comp_notify_pass.setChecked(bool(self._compliance_settings.get("notify_on_pass", False)))
        thresh_form.addRow(self.chk_comp_notify_pass)
        comp_layout.addLayout(thresh_form)
        layout.addWidget(comp_card)
        
        save_btn = QtWidgets.QPushButton("Save SOP Settings & Restart Monitor")
        save_btn.setObjectName("primaryBtn")
        save_btn.clicked.connect(self._save_sop_settings)
        layout.addWidget(save_btn)
        
        layout.addStretch(1)

    def _create_compliance_page(self) -> None:
        self.compliance_page = CompliancePage(logger=self._compliance_logger)
        self.stacked_widget.addWidget(self.compliance_page)

    def _add_step_row_data(
        self,
        idx_str: str,
        title_str: str,
        desc_str: str,
        ai_val: str,
        expected_str: str,
        timeout_val: int,
        criteria_str: str,
        warning_str: str,
        next_step_str: str
    ) -> None:
        row = self.steps_table.rowCount()
        self.steps_table.insertRow(row)
        self.steps_table.setRowHeight(row, 44)

        # Step Index (Column 0): Bold, centered, read-only
        item0 = QtWidgets.QTableWidgetItem(str(idx_str))
        item0.setTextAlignment(QtCore.Qt.AlignCenter)
        font0 = item0.font()
        font0.setBold(True)
        item0.setFont(font0)
        item0.setFlags(item0.flags() & ~QtCore.Qt.ItemIsEditable)
        self.steps_table.setItem(row, 0, item0)
        
        self.steps_table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(title_str)))
        self.steps_table.setItem(row, 2, QtWidgets.QTableWidgetItem(str(desc_str)))
        
        # Combo box for AI Validation Type with clean, spacious styling
        combo = QtWidgets.QComboBox()
        combo.addItems([
            "Face Verification",
            "PPE Detection",
            "Mobile Phone Detection",
            "Bluetooth / Earbuds Detection",
            "Shirt Button Open/Close Detection",
            "Tool Detection",
            "Tool Alignment Detection",
            "Hand Position Detection",
            "Gesture Detection",
            "Rotation Direction Detection",
            "Rotation Count Verification",
            "Torque Verification",
            "Barcode / QR Verification",
            "OCR Verification",
            "Weight Verification",
            "Final Quality Inspection",
            "Database & Screenshot Logging",
            "None"
        ])
        combo.setCurrentText(ai_val if ai_val else "None")
        combo.setStyleSheet("""
            QComboBox {
                background-color: #ffffff;
                color: #0f172a;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                padding: 3px 8px;
                font-size: 12px;
                font-weight: 600;
                min-height: 28px;
            }
            QComboBox:hover {
                border-color: #2563eb;
                background-color: #f8fafc;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 22px;
                border-left: 1px solid #cbd5e1;
                border-top-right-radius: 6px;
                border-bottom-right-radius: 6px;
            }
            QComboBox QAbstractItemView {
                background-color: #ffffff;
                color: #0f172a;
                selection-background-color: #2563eb;
                selection-color: #ffffff;
                border: 1px solid #cbd5e1;
            }
        """)
        self.steps_table.setCellWidget(row, 3, combo)
        
        self.steps_table.setItem(row, 4, QtWidgets.QTableWidgetItem(str(expected_str)))
        
        item5 = QtWidgets.QTableWidgetItem(str(timeout_val))
        item5.setTextAlignment(QtCore.Qt.AlignCenter)
        self.steps_table.setItem(row, 5, item5)
        
        self.steps_table.setItem(row, 6, QtWidgets.QTableWidgetItem(str(criteria_str)))
        self.steps_table.setItem(row, 7, QtWidgets.QTableWidgetItem(str(warning_str)))
        
        item8 = QtWidgets.QTableWidgetItem(str(next_step_str))
        item8.setTextAlignment(QtCore.Qt.AlignCenter)
        self.steps_table.setItem(row, 8, item8)

    def sync_steps_from_config(self, assembly_id: str | None = None) -> None:
        """Read rows from the SOP Configuration table, update global STEPS, rebuild the SOP panel,
        and persist to assemblies.yaml for the specified (or active) assembly."""
        from sop_panel import SopStep, STEPS
        if assembly_id is None:
            assembly_id = self._assembly_manager.get_active_assembly_id()
        STEPS.clear()
        for row in range(self.steps_table.rowCount()):
            idx_item = self.steps_table.item(row, 0)
            title_item = self.steps_table.item(row, 1)
            desc_item = self.steps_table.item(row, 2)
            
            combo = self.steps_table.cellWidget(row, 3)
            ai_val = combo.currentText() if combo else "None"
            
            expected_item = self.steps_table.item(row, 4)
            timeout_item = self.steps_table.item(row, 5)
            criteria_item = self.steps_table.item(row, 6)
            warning_item = self.steps_table.item(row, 7)
            next_step_item = self.steps_table.item(row, 8)
            
            try:
                idx = int(idx_item.text()) if idx_item and idx_item.text().isdigit() else row
            except Exception:
                idx = row
                
            title = title_item.text() if title_item and title_item.text() else f"Step {row}"
            desc = desc_item.text() if desc_item else ""
            expected = expected_item.text() if expected_item else "Success"
            try:
                timeout = int(timeout_item.text()) if timeout_item else 60
            except ValueError:
                timeout = 60
            criteria = criteria_item.text() if criteria_item else "Match"
            warning = warning_item.text() if warning_item else "Step failed"
            next_step = next_step_item.text() if next_step_item else "Next"
            
            STEPS.append(SopStep(
                index=idx,
                title=title,
                description=desc,
                ai_validation=ai_val,
                expected_result=expected,
                timeout=timeout,
                criteria=criteria,
                warning_msg=warning,
                next_step=next_step
            ))

        # Persist steps to assemblies.yaml for the target assembly
        self._assembly_manager.save_sop_steps(assembly_id, STEPS)

        # Only rebuild the live SOP panel if we're syncing the ACTIVE assembly
        if assembly_id == self._assembly_manager.get_active_assembly_id():
            if hasattr(self, "sop_panel") and self.sop_panel is not None:
                self.sop_panel.rebuild_steps(STEPS)
                self.sop_panel.set_theme(self.sop_panel.is_dark)

    def _add_preset_step(self, idx: int) -> None:
        if idx <= 0:
            return
        
        presets = {
            1: ("Weight Verification", "Verify assembled product weight is within tolerance.", "Weight Verification", "Weight in Tolerance", 15, "Weight == Target +- 0.05", "Weight out of tolerance range", "Next"),
            2: ("Barcode / QR Code Scan", "Verify correct product and batch information.", "Barcode / QR Verification", "Scan Success", 20, "Valid Barcode Scanned", "Failed to scan barcode or wrong product", "Next"),
            3: ("Label Verification", "Ensure the correct label is applied.", "OCR Verification", "Label Verified", 15, "Label Match", "Missing or incorrect label", "Next"),
            4: ("Reference Image Comparison", "Compare the assembled product against the reference image.", "Reference Image Comparison", "Image Match", 20, "Similarity > 0.85", "Visual comparison match failed", "Next"),
            5: ("Work Area Cleanliness", "Ensure no extra tools or foreign objects remain in the workstation.", "Work Area Cleanliness", "Clean Station", 30, "No FOD Detected", "Foreign objects/tools detected in workspace", "Next"),
            6: ("Final Supervisor Approval", "Wait for operator or supervisor confirmation before completing the SOP.", "Final Supervisor Approval", "Approved", 60, "Signature Verified", "Supervisor approval pending", "End")
        }
        
        if idx in presets:
            title, desc, ai_val, expected, timeout, criteria, warning, next_step = presets[idx]
            row = self.steps_table.rowCount()
            self._add_step_row_data(str(row), title, desc, ai_val, expected, timeout, criteria, warning, next_step)
            self.sync_steps_from_config()
            
        # Reset selection
        self.preset_combo.setCurrentIndex(0)

    def _add_sop_step_row(self) -> None:
        self._activity_logger.log("ADD", "SOP Configuration", "Add SOP Step", "New step row added")
        row = self.steps_table.rowCount()
        self._add_step_row_data(
            str(row),
            "New Step",
            "Step details...",
            "None",
            "Success",
            60,
            "Criteria",
            "Warning Message",
            "Next"
        )
        self.sync_steps_from_config()

    def _delete_sop_step_row(self) -> None:
        self._activity_logger.log("DELETE", "SOP Configuration", "Delete SOP Step", "Delete button clicked")
        row = self.steps_table.currentRow()
        if row < 0:
            QtWidgets.QMessageBox.warning(self, "No Selection", "Please select a step row to delete.")
            return
        
        title_item = self.steps_table.item(row, 1)
        step_title = title_item.text() if title_item else f"Step {row}"
        
        reply = QtWidgets.QMessageBox.question(
            self, "Delete Reference",
            f"Delete reference entry:\n'{step_title}'?\n\n(The step entry will be removed from the list.)",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        if reply == QtWidgets.QMessageBox.Yes:
            self.steps_table.removeRow(row)
            # Re-index remaining rows so step index is sequential, centered, and bold
            for r in range(self.steps_table.rowCount()):
                idx_item = self.steps_table.item(r, 0)
                if not idx_item:
                    idx_item = QtWidgets.QTableWidgetItem(str(r))
                    self.steps_table.setItem(r, 0, idx_item)
                else:
                    idx_item.setText(str(r))
                idx_item.setTextAlignment(QtCore.Qt.AlignCenter)
                font_r = idx_item.font()
                font_r.setBold(True)
                idx_item.setFont(font_r)
                idx_item.setFlags(idx_item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.sync_steps_from_config()

    def _save_sop_settings(self) -> None:
        self._activity_logger.log("SAVE", "SOP Configuration", "Save SOP Settings", f"SOP: {self.sop_name_edit.text()}")

        # Determine which assembly we are editing (from the combo on SOP Config page)
        if hasattr(self, "sop_asm_combo"):
            asm_id = self.sop_asm_combo.currentData()
        else:
            asm_id = self._assembly_manager.get_active_assembly_id()

        # Update in-memory state only if we're editing the ACTIVE assembly
        if asm_id == self._assembly_manager.get_active_assembly_id():
            self.active_sop_name = self.sop_name_edit.text()
            self.active_product_name = self.prod_name_edit.text()
            self.active_variant = getattr(self, 'prod_variant_edit', None) and self.prod_variant_edit.text() or self.active_variant
            self.active_turn_target = self.turns_spin.value()

            self.sub_title_sop.setText(
                f"SOP: {self.active_sop_name} | Target Turns: {self.active_turn_target}"
            )

        # Re-apply turn target to the live ScrewMonitor (if active assembly)
        if (asm_id == self._assembly_manager.get_active_assembly_id()
                and self.worker is not None
                and self.worker._monitor is not None):
            self.worker._monitor.turn_target = self.turns_spin.value()
            self.worker._monitor.align_threshold = self.align_thresh_spin.value()

        # Sync steps from table and persist to YAML
        self.sync_steps_from_config(assembly_id=asm_id)

        # Persist target_params to YAML
        self._assembly_manager.save_target_params(asm_id, {
            "turn_target": float(self.turns_spin.value()),
            "align_threshold_px": int(self.align_thresh_spin.value()),
        })

        # Persist + apply compliance detector settings
        self._collect_compliance_settings_from_ui()
        self._persist_compliance_settings()
        if hasattr(self, "chk_compliance"):
            self.chk_compliance.setChecked(bool(self._compliance_settings.get("enabled", True)))

        # Only restart the worker if we changed the ACTIVE assembly
        if asm_id == self._assembly_manager.get_active_assembly_id():
            self._start_worker()
            QtWidgets.QMessageBox.information(
                self, "Success",
                "SOP configuration updated and monitor restarted."
            )
        else:
            QtWidgets.QMessageBox.information(
                self, "Saved",
                f"SOP steps saved for: {self.sop_name_edit.text()}\n"
                "(This is not the active assembly — monitor was not restarted.)"
            )

    def _on_sop_asm_combo_changed(self, _index: int) -> None:
        """Reload the SOP Configuration page for the assembly selected in the combo."""
        if not hasattr(self, "sop_asm_combo"):
            return
        asm_id = self.sop_asm_combo.currentData()
        if asm_id:
            self._reload_sop_config_for_assembly(asm_id)

    def _collect_compliance_settings_from_ui(self) -> None:
        if not hasattr(self, "chk_comp_master"):
            return
        self._compliance_settings = {
            "enabled": self.chk_comp_master.isChecked(),
            "detectors": {
                "mobile_phone": self.chk_det_phone.isChecked(),
                "shirt_button": self.chk_det_shirt.isChecked(),
                "bluetooth_earbuds": self.chk_det_buds.isChecked(),
                "spectacles": self.chk_det_specs.isChecked(),
                "writing": self.chk_det_write.isChecked(),
            },
            "confidence_threshold": float(self.comp_conf_spin.value()),
            "warning_cooldown_sec": float(self.comp_cooldown_spin.value()),
            "save_evidence_images": self.chk_comp_evidence.isChecked(),
            "notify_on_pass": self.chk_comp_notify_pass.isChecked(),
        }

    def _persist_compliance_settings(self) -> None:
        path = os.path.join(_GUI_DIR, "app_settings.json")
        settings = {}
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    settings = json.load(f)
            except Exception:
                settings = {}
        settings["compliance"] = self._compliance_settings
        try:
            with open(path, "w") as f:
                json.dump(settings, f, indent=2)
        except Exception as e:
            print(f"[settings] failed to save compliance: {e}")
        if self.worker is not None:
            self.worker.apply_compliance_settings(self._compliance_settings)

    # -- PAGE 3: OPERATOR MANAGEMENT -----------------------------------------
    def _load_operators(self) -> None:
        path = os.path.join(_GUI_DIR, "operators.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self._operators = json.load(f)
            except Exception as e:
                print(f"[MainWindow] Error loading operators.json: {e}")
                self._operators = []
        if not self._operators:
            self._operators = [
                {
                    "id": "EMP001",
                    "name": "Dharanidharan",
                    "rfid": "RFID-88392-X",
                    "shift": "Shift A (06:00 - 14:00)",
                    "role": "Primary Operator",
                    "supervisor": "Giri",
                    "line": "Assembly Line A",
                    "attendance": "Present",
                    "is_active": True,
                    "login_time": "06:01:22",
                    "working_duration": "04:18:23",
                    "break_duration": "00:15:00",
                    "total_processed": 142,
                    "sop_compliance": "97.2%",
                    "avg_cycle_time": "12.5 s"
                },
                {
                    "id": "EMP002",
                    "name": "Sarah Smith",
                    "rfid": "RFID-44120-B",
                    "shift": "Shift A (06:00 - 14:00)",
                    "role": "Assembly Lead",
                    "supervisor": "Giri",
                    "line": "Assembly Line B",
                    "attendance": "Present",
                    "is_active": False,
                    "login_time": "06:15:00",
                    "working_duration": "03:45:10",
                    "break_duration": "00:10:00",
                    "total_processed": 128,
                    "sop_compliance": "98.5%",
                    "avg_cycle_time": "11.8 s"
                },
                {
                    "id": "EMP003",
                    "name": "Rajesh Kumar",
                    "rfid": "RFID-99231-C",
                    "shift": "Shift B (14:00 - 22:00)",
                    "role": "Quality Inspector",
                    "supervisor": "Giri",
                    "line": "Testing Line 1",
                    "attendance": "Absent",
                    "is_active": False,
                    "login_time": "--:--:--",
                    "working_duration": "00:00:00",
                    "break_duration": "00:00:00",
                    "total_processed": 0,
                    "sop_compliance": "100.0%",
                    "avg_cycle_time": "0.0 s"
                }
            ]
            self._save_operators()

    def _save_operators(self) -> None:
        path = os.path.join(_GUI_DIR, "operators.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._operators, f, indent=2)
        except Exception as e:
            print(f"[MainWindow] Error saving operators.json: {e}")

    def _create_operator_page(self) -> None:
        page = QtWidgets.QWidget()
        page.setObjectName("page")
        self.stacked_widget.addWidget(page)

        # Outer layout holds only the scroll area (no margins — scroll area fills page)
        outer_layout = QtWidgets.QVBoxLayout(page)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        # ── Scroll Area (vertical + horizontal) ─────────────────────────────
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            /* Vertical scrollbar (right side) */
            QScrollBar:vertical {
                background: #f1f5f9;
                width: 10px;
                margin: 0px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: #94a3b8;
                min-height: 32px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover {
                background: #64748b;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0px;
            }
            /* Horizontal scrollbar (bottom) */
            QScrollBar:horizontal {
                background: #f1f5f9;
                height: 10px;
                margin: 0px;
                border-radius: 5px;
            }
            QScrollBar::handle:horizontal {
                background: #94a3b8;
                min-width: 32px;
                border-radius: 5px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #64748b;
            }
            QScrollBar::add-line:horizontal,
            QScrollBar::sub-line:horizontal {
                width: 0px;
            }
        """)

        # Inner scroll-content widget  (min-width forces horizontal scrollbar when window is narrow)
        scroll_content = QtWidgets.QWidget()
        scroll_content.setObjectName("page")
        scroll_content.setMinimumWidth(1050)
        scroll_area.setWidget(scroll_content)
        outer_layout.addWidget(scroll_area)

        main_layout = QtWidgets.QVBoxLayout(scroll_content)
        main_layout.setContentsMargins(24, 20, 24, 20)
        main_layout.setSpacing(16)
        
        # 1. Header & Summary Metrics Bar
        header_row = QtWidgets.QHBoxLayout()
        header_row.setSpacing(12)
        
        title_box = QtWidgets.QVBoxLayout()
        title = QtWidgets.QLabel("Operator Management & Shift Authentications")
        title.setStyleSheet("font-size: 22px; font-weight: 800; color: #0f172a;")
        sub = QtWidgets.QLabel("Add, update, track attendance (Present/Absent), and set active working operators for live shift monitoring.")
        sub.setStyleSheet("color: #64748b; font-size: 12px;")
        title_box.addWidget(title)
        title_box.addWidget(sub)
        header_row.addLayout(title_box, 1)
        
        # Summary Metric Badges
        self.op_badge_total = QtWidgets.QLabel("Total: 0")
        self.op_badge_present = QtWidgets.QLabel("Present: 0")
        self.op_badge_absent = QtWidgets.QLabel("Absent: 0")
        self.op_badge_active = QtWidgets.QLabel("Active: --")
        
        for badge, bg, fg in [
            (self.op_badge_total, "#f1f5f9", "#334155"),
            (self.op_badge_present, "#dcfce7", "#15803d"),
            (self.op_badge_absent, "#fee2e2", "#b91c1c"),
            (self.op_badge_active, "#dbeafe", "#1d4ed8")
        ]:
            badge.setFixedHeight(26)
            badge.setStyleSheet(f"""
                background-color: {bg};
                color: {fg};
                font-weight: 700;
                font-size: 11px;
                padding: 0px 10px;
                border-radius: 13px;
                border: 1px solid {fg}40;
            """)
            header_row.addWidget(badge)
            
        main_layout.addLayout(header_row)
        
        # 2. Form & Active Card Split View
        split = QtWidgets.QHBoxLayout()
        split.setSpacing(16)
        
        # Profile Details / Form Card
        form_card = QtWidgets.QFrame()
        form_card.setObjectName("card")
        form_card.setStyleSheet("""
            QFrame#card {
                background-color: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 10px;
            }
        """)
        form_layout = QtWidgets.QFormLayout(form_card)
        form_layout.setContentsMargins(20, 18, 20, 18)
        form_layout.setSpacing(10)
        
        form_title = QtWidgets.QLabel("Add / Edit Employee Details")
        form_title.setStyleSheet("font-size: 15px; font-weight: 700; color: #0f172a; margin-bottom: 6px;")
        form_layout.addRow(form_title)
        
        self.op_id_input = QtWidgets.QLineEdit(self.default_worker_id)
        self.op_id_input.setPlaceholderText("e.g. EMP001")
        form_layout.addRow(QtWidgets.QLabel("Employee ID:"), self.op_id_input)
        
        self.op_name_input = QtWidgets.QLineEdit("Dharanidharan")
        self.op_name_input.setPlaceholderText("e.g. Dharanidharan")
        form_layout.addRow(QtWidgets.QLabel("Operator Name:"), self.op_name_input)
        
        self.op_rfid_input = QtWidgets.QLineEdit("RFID-88392-X")
        self.op_rfid_input.setPlaceholderText("e.g. RFID-88392-X")
        form_layout.addRow(QtWidgets.QLabel("RFID / Barcode ID:"), self.op_rfid_input)
        
        self.op_shift_comb = QtWidgets.QComboBox()
        self.op_shift_comb.addItems(["Shift A (06:00 - 14:00)", "Shift B (14:00 - 22:00)", "Shift C (22:00 - 06:00)"])
        form_layout.addRow(QtWidgets.QLabel("Current Shift:"), self.op_shift_comb)
        
        self.op_role_comb = QtWidgets.QComboBox()
        self.op_role_comb.addItems(["Primary Operator", "Assembly Lead", "Quality Inspector", "Supervisor"])
        form_layout.addRow(QtWidgets.QLabel("Shift Role:"), self.op_role_comb)
        
        self.op_super_input = QtWidgets.QLineEdit("Giri")
        self.op_super_input.setPlaceholderText("e.g. Sarah Smith / Giri")
        form_layout.addRow(QtWidgets.QLabel("Supervisor:"), self.op_super_input)
        
        self.op_line_input = QtWidgets.QLineEdit("Assembly Line A")
        self.op_line_input.setPlaceholderText("e.g. Assembly Line A")
        form_layout.addRow(QtWidgets.QLabel("Workstation / Line:"), self.op_line_input)
        
        self.op_attend_comb = QtWidgets.QComboBox()
        self.op_attend_comb.addItems(["Present", "Absent"])
        form_layout.addRow(QtWidgets.QLabel("Attendance Status:"), self.op_attend_comb)
        
        # Form Buttons
        btn_box = QtWidgets.QHBoxLayout()
        btn_box.setSpacing(10)
        
        save_op_btn = QtWidgets.QPushButton("Save / Add Employee")
        save_op_btn.setCursor(QtCore.Qt.PointingHandCursor)
        save_op_btn.setStyleSheet("""
            QPushButton {
                background-color: #2563eb; color: #ffffff; font-weight: 700;
                padding: 8px 16px; border-radius: 6px; border: none;
            }
            QPushButton:hover { background-color: #1d4ed8; }
        """)
        save_op_btn.clicked.connect(self._save_operator_details)
        btn_box.addWidget(save_op_btn)
        
        clear_op_btn = QtWidgets.QPushButton("Reset Form")
        clear_op_btn.setCursor(QtCore.Qt.PointingHandCursor)
        clear_op_btn.setStyleSheet("""
            QPushButton {
                background-color: #f1f5f9; color: #475569; font-weight: 600;
                padding: 8px 14px; border-radius: 6px; border: 1px solid #cbd5e1;
            }
            QPushButton:hover { background-color: #e2e8f0; }
        """)
        clear_op_btn.clicked.connect(self._clear_operator_fields)
        btn_box.addWidget(clear_op_btn)
        
        set_active_form_btn = QtWidgets.QPushButton("Set Active Shift Worker")
        set_active_form_btn.setCursor(QtCore.Qt.PointingHandCursor)
        set_active_form_btn.setStyleSheet("""
            QPushButton {
                background-color: #16a34a; color: #ffffff; font-weight: 700;
                padding: 8px 14px; border-radius: 6px; border: none;
            }
            QPushButton:hover { background-color: #15803d; }
        """)
        set_active_form_btn.clicked.connect(self._set_form_operator_active)
        btn_box.addWidget(set_active_form_btn)
        
        form_layout.addRow(btn_box)
        split.addWidget(form_card, 3)
        
        # Active Operator Profile Card (Right)
        perf_card = QtWidgets.QFrame()
        perf_card.setObjectName("card")
        perf_card.setStyleSheet("""
            QFrame#card {
                background-color: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 10px;
            }
        """)
        perf_layout = QtWidgets.QVBoxLayout(perf_card)
        perf_layout.setContentsMargins(20, 18, 20, 18)
        perf_layout.setSpacing(8)
        
        card_header = QtWidgets.QLabel("Active Working Operator (Current Shift)")
        card_header.setStyleSheet("font-size: 14px; font-weight: 700; color: #2563eb;")
        perf_layout.addWidget(card_header)
        
        self.perf_name_lbl = QtWidgets.QLabel("Dharanidharan (Active)")
        self.perf_name_lbl.setStyleSheet("font-weight: 800; font-size: 16px; color: #0f172a;")
        self.perf_name_lbl.setAlignment(QtCore.Qt.AlignCenter)
        perf_layout.addWidget(self.perf_name_lbl)
        
        # Details list
        self.op_login_time = QtWidgets.QLabel("Login Time: 06:01:22")
        self.op_duration = QtWidgets.QLabel("Working Duration: 04:18:23")
        self.op_break_lbl = QtWidgets.QLabel("Break Duration: 00:15:00")
        
        for lbl in (self.op_login_time, self.op_duration, self.op_break_lbl):
            lbl.setStyleSheet("color: #475569; font-size: 12px; font-weight: 500;")
            perf_layout.addWidget(lbl)
            
        perf_layout.addSpacing(6)
        perf_title = QtWidgets.QLabel("Operator KPIs Summary")
        perf_title.setStyleSheet("font-size: 13px; font-weight: 700; color: #0f172a; border-top: 1px solid #e2e8f0; padding-top: 8px;")
        perf_layout.addWidget(perf_title)
        
        self.op_stat_prod = QtWidgets.QLabel("Total Processed: 142 units")
        self.op_stat_acc = QtWidgets.QLabel("SOP Compliance: 97.2%")
        self.op_stat_cycle = QtWidgets.QLabel("Avg Cycle Time: 12.5 s")
        
        for lbl in (self.op_stat_prod, self.op_stat_acc, self.op_stat_cycle):
            lbl.setStyleSheet("color: #475569; font-size: 12px; font-weight: 500;")
            perf_layout.addWidget(lbl)
            
        perf_layout.addStretch(1)
        split.addWidget(perf_card, 2)
        main_layout.addLayout(split)
        
        # 3. Bottom Section: Registered Employee List CRUD Table
        table_container = QtWidgets.QFrame()
        table_container.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 10px;
            }
        """)
        tbl_layout = QtWidgets.QVBoxLayout(table_container)
        tbl_layout.setContentsMargins(16, 14, 16, 14)
        tbl_layout.setSpacing(10)
        
        tbl_header_row = QtWidgets.QHBoxLayout()
        tbl_title = QtWidgets.QLabel("Registered Employee List & Shift Assignment (CRUD)")
        tbl_title.setStyleSheet("font-size: 15px; font-weight: 700; color: #0f172a;")
        tbl_header_row.addWidget(tbl_title)
        tbl_header_row.addStretch(1)
        
        # Search input
        self.op_search_input = QtWidgets.QLineEdit()
        self.op_search_input.setPlaceholderText("Search by Employee ID or Name...")
        self.op_search_input.setFixedWidth(260)
        self.op_search_input.setStyleSheet("padding: 6px 10px; border: 1px solid #cbd5e1; border-radius: 6px;")
        self.op_search_input.textChanged.connect(self._refresh_operator_table)
        tbl_header_row.addWidget(self.op_search_input)
        
        tbl_layout.addLayout(tbl_header_row)
        
        # Table Widget
        self.op_table = QtWidgets.QTableWidget()
        headers = ["Emp ID", "Operator Name", "RFID ID", "Shift", "Role", "Workstation", "Attendance", "Shift Work Status", "Actions"]
        self.op_table.setColumnCount(len(headers))
        self.op_table.setHorizontalHeaderLabels(headers)
        # Stretch ALL columns proportionally so they fill the full table width
        hdr = self.op_table.horizontalHeader()
        hdr.setStretchLastSection(False)
        hdr.setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        # Override fixed-size columns that should stay compact
        hdr.setSectionResizeMode(0, QtWidgets.QHeaderView.Fixed)   # Emp ID
        self.op_table.setColumnWidth(0, 90)
        hdr.setSectionResizeMode(6, QtWidgets.QHeaderView.Fixed)   # Attendance
        self.op_table.setColumnWidth(6, 110)
        hdr.setSectionResizeMode(7, QtWidgets.QHeaderView.Fixed)   # Shift Work Status
        self.op_table.setColumnWidth(7, 155)
        hdr.setSectionResizeMode(8, QtWidgets.QHeaderView.Fixed)   # Actions
        self.op_table.setColumnWidth(8, 155)
        self.op_table.verticalHeader().setVisible(False)
        # Table gets its own horizontal scrollbar when columns overflow
        self.op_table.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.op_table.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.op_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.op_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.op_table.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding
        )
        self.op_table.setMinimumHeight(200)
        self.op_table.setStyleSheet("""
            QTableWidget {
                background-color: #ffffff;
                gridline-color: #f1f5f9;
                border: none;
                font-size: 12px;
                outline: none;
            }
            QTableWidget::item {
                padding: 4px 8px;
                border-bottom: 1px solid #f1f5f9;
            }
            QTableWidget::item:selected {
                background-color: #eff6ff;
                color: #1e40af;
            }
            QHeaderView::section {
                background-color: #f8fafc;
                color: #334155;
                font-weight: 700;
                border: none;
                border-right: 1px solid #e2e8f0;
                border-bottom: 2px solid #2563eb;
                padding: 8px 6px;
                font-size: 12px;
            }
            /* Table's own horizontal scrollbar */
            QScrollBar:horizontal {
                background: #f1f5f9;
                height: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:horizontal {
                background: #cbd5e1;
                border-radius: 4px;
                min-width: 20px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #94a3b8;
            }
            QScrollBar::add-line:horizontal,
            QScrollBar::sub-line:horizontal { width: 0px; }
        """)
        tbl_layout.addWidget(self.op_table)
        main_layout.addWidget(table_container, 1)

        # Push content to top so scroll area doesn't stretch gaps
        main_layout.addStretch(1)

        # Initial populate
        self._refresh_operator_table()
        self._update_operator_badges()
        self._sync_active_operator_card()

    def _update_operator_badges(self) -> None:
        if not hasattr(self, "_operators"):
            return
        total = len(self._operators)
        present = sum(1 for op in self._operators if op.get("attendance") == "Present")
        absent = sum(1 for op in self._operators if op.get("attendance") == "Absent")
        active_op = next((op for op in self._operators if op.get("is_active")), None)
        active_name = active_op.get("name", "--") if active_op else "--"
        
        self.op_badge_total.setText(f"Total: {total}")
        self.op_badge_present.setText(f"Present: {present}")
        self.op_badge_absent.setText(f"Absent: {absent}")
        self.op_badge_active.setText(f"Active Operator: {active_name}")

    def _sync_active_operator_card(self) -> None:
        active_op = next((op for op in self._operators if op.get("is_active")), None)
        if active_op:
            name = active_op.get("name", "Unknown")
            id_val = active_op.get("id", "EMP001")
            login = active_op.get("login_time", "06:01:22")
            dur = active_op.get("working_duration", "04:18:23")
            brk = active_op.get("break_duration", "00:15:00")
            proc = active_op.get("total_processed", 142)
            comp = active_op.get("sop_compliance", "97.2%")
            cyc = active_op.get("avg_cycle_time", "12.5 s")
            
            self.perf_name_lbl.setText(f"{name} (Active)")
            self.op_login_time.setText(f"Login Time: {login}")
            self.op_duration.setText(f"Working Duration: {dur}")
            self.op_break_lbl.setText(f"Break Duration: {brk}")
            self.op_stat_prod.setText(f"Total Processed: {proc} units")
            self.op_stat_acc.setText(f"SOP Compliance: {comp}")
            self.op_stat_cycle.setText(f"Avg Cycle Time: {cyc}")
            
            if hasattr(self, "worker_id_input"):
                self.worker_id_input.setText(id_val)
            if hasattr(self, "worker") and self.worker is not None:
                self.worker.set_operator_info(id_val, name)

    def _refresh_operator_table(self) -> None:
        if not hasattr(self, "op_table") or not hasattr(self, "_operators"):
            return
        
        filter_text = self.op_search_input.text().strip().lower() if hasattr(self, "op_search_input") else ""
        
        filtered = []
        for op in self._operators:
            if not filter_text or filter_text in op.get("id", "").lower() or filter_text in op.get("name", "").lower():
                filtered.append(op)
                
        self.op_table.setRowCount(len(filtered))
        
        for row, op in enumerate(filtered):
            self.op_table.setRowHeight(row, 42)
            emp_id = op.get("id", "")
            emp_name = op.get("name", "")
            
            # 0: Emp ID
            item_id = QtWidgets.QTableWidgetItem(emp_id)
            item_id.setTextAlignment(QtCore.Qt.AlignCenter)
            item_id.setFlags(item_id.flags() & ~QtCore.Qt.ItemIsEditable)
            self.op_table.setItem(row, 0, item_id)
            
            # 1: Operator Name
            item_name = QtWidgets.QTableWidgetItem(emp_name)
            font_n = item_name.font()
            font_n.setBold(True)
            item_name.setFont(font_n)
            item_name.setFlags(item_name.flags() & ~QtCore.Qt.ItemIsEditable)
            self.op_table.setItem(row, 1, item_name)
            
            # 2: RFID
            item_rfid = QtWidgets.QTableWidgetItem(op.get("rfid", ""))
            item_rfid.setFlags(item_rfid.flags() & ~QtCore.Qt.ItemIsEditable)
            self.op_table.setItem(row, 2, item_rfid)
            
            # 3: Shift
            item_shift = QtWidgets.QTableWidgetItem(op.get("shift", ""))
            item_shift.setFlags(item_shift.flags() & ~QtCore.Qt.ItemIsEditable)
            self.op_table.setItem(row, 3, item_shift)
            
            # 4: Role
            item_role = QtWidgets.QTableWidgetItem(op.get("role", ""))
            item_role.setFlags(item_role.flags() & ~QtCore.Qt.ItemIsEditable)
            self.op_table.setItem(row, 4, item_role)
            
            # 5: Workstation
            item_line = QtWidgets.QTableWidgetItem(op.get("line", ""))
            item_line.setFlags(item_line.flags() & ~QtCore.Qt.ItemIsEditable)
            self.op_table.setItem(row, 5, item_line)
            
            # 6: Attendance Status (Toggle Button / Badge)
            attend_val = op.get("attendance", "Present")
            attend_btn = QtWidgets.QPushButton(f"● {attend_val}")
            if attend_val == "Present":
                attend_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #dcfce7; color: #15803d; font-weight: 700;
                        border: 1px solid #86efac; border-radius: 12px; padding: 4px 10px;
                    }
                    QPushButton:hover { background-color: #bbf7d0; }
                """)
            else:
                attend_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #fee2e2; color: #b91c1c; font-weight: 700;
                        border: 1px solid #fca5a5; border-radius: 12px; padding: 4px 10px;
                    }
                    QPushButton:hover { background-color: #fecaca; }
                """)
            attend_btn.clicked.connect(lambda _, id_v=emp_id: self._toggle_operator_attendance(id_v))
            self.op_table.setCellWidget(row, 6, attend_btn)
            
            # 7: Shift Work Status (Active / Make Active Button)
            is_act = op.get("is_active", False)
            if is_act:
                act_lbl = QtWidgets.QLabel("🟢 Active (Working)")
                act_lbl.setAlignment(QtCore.Qt.AlignCenter)
                act_lbl.setStyleSheet("color: #15803d; font-weight: 800; font-size: 11px;")
                self.op_table.setCellWidget(row, 7, act_lbl)
            else:
                act_btn = QtWidgets.QPushButton("⚡ Activate for Shift")
                act_btn.setCursor(QtCore.Qt.PointingHandCursor)
                act_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #eff6ff; color: #2563eb; font-weight: 700;
                        border: 1px solid #bfdbfe; border-radius: 6px; padding: 3px 8px;
                    }
                    QPushButton:hover { background-color: #2563eb; color: #ffffff; }
                """)
                act_btn.clicked.connect(lambda _, id_v=emp_id: self._set_active_operator(id_v))
                self.op_table.setCellWidget(row, 7, act_btn)
                
            # 8: Actions (Edit & Delete Buttons)
            actions_widget = QtWidgets.QWidget()
            actions_lay = QtWidgets.QHBoxLayout(actions_widget)
            actions_lay.setContentsMargins(2, 2, 2, 2)
            actions_lay.setSpacing(6)
            
            edit_btn = QtWidgets.QPushButton("Edit")
            edit_btn.setCursor(QtCore.Qt.PointingHandCursor)
            edit_btn.setStyleSheet("""
                QPushButton {
                    background-color: #f8fafc; color: #0284c7; font-weight: 700;
                    border: 1px solid #cbd5e1; border-radius: 4px; padding: 2px 8px;
                }
                QPushButton:hover { background-color: #0284c7; color: #ffffff; }
            """)
            edit_btn.clicked.connect(lambda _, id_v=emp_id: self._edit_operator_row(id_v))
            
            del_btn = QtWidgets.QPushButton("Delete")
            del_btn.setCursor(QtCore.Qt.PointingHandCursor)
            del_btn.setStyleSheet("""
                QPushButton {
                    background-color: #fff1f2; color: #e11d48; font-weight: 700;
                    border: 1px solid #fecdd3; border-radius: 4px; padding: 2px 8px;
                }
                QPushButton:hover { background-color: #e11d48; color: #ffffff; }
            """)
            del_btn.clicked.connect(lambda _, id_v=emp_id: self._delete_operator_row(id_v))
            
            actions_lay.addWidget(edit_btn)
            actions_lay.addWidget(del_btn)
            self.op_table.setCellWidget(row, 8, actions_widget)

        # ── Auto-resize table height so ALL rows are visible (outer scroll handles vertical) ──
        row_count = self.op_table.rowCount()
        header_h = self.op_table.horizontalHeader().height()
        row_h = 44  # matches setRowHeight above + borders
        total_h = header_h + (row_count * row_h) + 10   # +10 for scrollbar track area
        self.op_table.setMinimumHeight(max(200, total_h))
        self.op_table.setMaximumHeight(max(200, total_h))

    def _save_operator_details(self) -> None:
        emp_id = self.op_id_input.text().strip()
        emp_name = self.op_name_input.text().strip()
        if not emp_id or not emp_name:
            QtWidgets.QMessageBox.warning(self, "Validation Error", "Employee ID and Operator Name are required.")
            return
            
        existing = next((op for op in self._operators if op["id"] == emp_id), None)
        if existing:
            existing["name"] = emp_name
            existing["rfid"] = self.op_rfid_input.text().strip()
            existing["shift"] = self.op_shift_comb.currentText()
            existing["role"] = self.op_role_comb.currentText()
            existing["supervisor"] = self.op_super_input.text().strip()
            existing["line"] = self.op_line_input.text().strip()
            existing["attendance"] = self.op_attend_comb.currentText()
            msg = f"Operator {emp_name} ({emp_id}) updated successfully."
        else:
            new_op = {
                "id": emp_id,
                "name": emp_name,
                "rfid": self.op_rfid_input.text().strip() or f"RFID-{emp_id}",
                "shift": self.op_shift_comb.currentText(),
                "role": self.op_role_comb.currentText(),
                "supervisor": self.op_super_input.text().strip() or "Supervisor",
                "line": self.op_line_input.text().strip() or "Line A",
                "attendance": self.op_attend_comb.currentText(),
                "is_active": False,
                "login_time": "06:00:00",
                "working_duration": "00:00:00",
                "break_duration": "00:00:00",
                "total_processed": 0,
                "sop_compliance": "100.0%",
                "avg_cycle_time": "0.0 s"
            }
            self._operators.append(new_op)
            msg = f"New operator {emp_name} ({emp_id}) added successfully."
            
        self._save_operators()
        self._refresh_operator_table()
        self._update_operator_badges()
        self._sync_active_operator_card()
        
        self._activity_logger.log("SAVE", "Operator Management", "Save Operator", f"Name: {emp_name}, ID: {emp_id}")
        QtWidgets.QMessageBox.information(self, "Success", msg)

    def _clear_operator_fields(self) -> None:
        self.op_id_input.clear()
        self.op_name_input.clear()
        self.op_rfid_input.clear()
        self.op_super_input.clear()
        self.op_line_input.clear()

    def _set_form_operator_active(self) -> None:
        emp_id = self.op_id_input.text().strip()
        if not emp_id:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please enter an Employee ID first.")
            return
        self._save_operator_details()
        self._set_active_operator(emp_id)

    def _edit_operator_row(self, emp_id: str) -> None:
        op = next((o for o in self._operators if o["id"] == emp_id), None)
        if not op:
            return
        self.op_id_input.setText(op.get("id", ""))
        self.op_name_input.setText(op.get("name", ""))
        self.op_rfid_input.setText(op.get("rfid", ""))
        self.op_super_input.setText(op.get("supervisor", ""))
        self.op_line_input.setText(op.get("line", ""))
        
        idx_shift = self.op_shift_comb.findText(op.get("shift", ""))
        if idx_shift >= 0:
            self.op_shift_comb.setCurrentIndex(idx_shift)
            
        idx_role = self.op_role_comb.findText(op.get("role", ""))
        if idx_role >= 0:
            self.op_role_comb.setCurrentIndex(idx_role)
            
        idx_att = self.op_attend_comb.findText(op.get("attendance", "Present"))
        if idx_att >= 0:
            self.op_attend_comb.setCurrentIndex(idx_att)

    def _delete_operator_row(self, emp_id: str) -> None:
        op = next((o for o in self._operators if o["id"] == emp_id), None)
        if not op:
            return
        
        reply = QtWidgets.QMessageBox.question(
            self, "Confirm Delete",
            f"Are you sure you want to delete employee record:\n{op.get('name')} ({emp_id})?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        if reply == QtWidgets.QMessageBox.Yes:
            self._operators = [o for o in self._operators if o["id"] != emp_id]
            self._save_operators()
            self._refresh_operator_table()
            self._update_operator_badges()
            self._sync_active_operator_card()
            self._activity_logger.log("DELETE", "Operator Management", "Delete Operator", f"ID: {emp_id}")

    def _toggle_operator_attendance(self, emp_id: str) -> None:
        op = next((o for o in self._operators if o["id"] == emp_id), None)
        if not op:
            return
        curr = op.get("attendance", "Present")
        new_att = "Absent" if curr == "Present" else "Present"
        op["attendance"] = new_att
        
        if new_att == "Absent" and op.get("is_active"):
            op["is_active"] = False
            QtWidgets.QMessageBox.information(
                self, "Attendance Changed",
                f"Operator {op.get('name')} marked as Absent and deactivated from shift."
            )
            
        self._save_operators()
        self._refresh_operator_table()
        self._update_operator_badges()
        self._sync_active_operator_card()

    def _set_active_operator(self, emp_id: str) -> None:
        op_target = next((o for o in self._operators if o["id"] == emp_id), None)
        if not op_target:
            return
            
        for op in self._operators:
            if op["id"] == emp_id:
                op["is_active"] = True
                op["attendance"] = "Present"
            else:
                op["is_active"] = False
                
        self._save_operators()
        self._refresh_operator_table()
        self._update_operator_badges()
        self._sync_active_operator_card()
        
        self._activity_logger.log("ACTIVATE", "Operator Management", "Set Active Worker", f"Name: {op_target.get('name')}, ID: {emp_id}")
        QtWidgets.QMessageBox.information(
            self, "Active Operator Updated",
            f"⚡ {op_target.get('name')} ({emp_id}) is now set as the ACTIVE working operator for {op_target.get('shift')}!"
        )

    # -- PAGE 4: PRODUCTION ANALYTICS ---------------------------------------
    def _create_analytics_page(self) -> None:
        page = QtWidgets.QWidget()
        page.setObjectName("page")
        self.stacked_widget.addWidget(page)
        
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)
        
        title_row = QtWidgets.QHBoxLayout()
        title_section = QtWidgets.QVBoxLayout()
        title = QtWidgets.QLabel("Production Analytics & AI Reports")
        title.setStyleSheet("font-size: 22px; font-weight: 800; color: #0f172a;")
        sub = QtWidgets.QLabel("Visual reports on assembly performance, sensor thresholds, and accuracy trends.")
        sub.setStyleSheet("color: #64748b; font-size: 12px;")
        title_section.addWidget(title)
        title_section.addWidget(sub)
        title_row.addLayout(title_section)
        layout.addLayout(title_row)
        
        charts_layout = QtWidgets.QHBoxLayout()
        charts_layout.setSpacing(16)
        
        self.chart_temp = TrendChart("Temperature Trend", "Compare with Yesterday | Select Time Range", "°C", 200, 250)
        self.chart_pressure = TrendChart("Injection Pressure Trend", "Compare with Yesterday | Select Time Range", "bar", 120, 170)
        self.chart_moisture = TrendChart("Moisture Level Trend", "Compare with Yesterday | Select Time Range", "%", 0.0, 0.15)
        
        self.chart_temp.set_points([222, 224, 221, 226, 223, 225, 222, 227, 224, 228, 225, 223, 226, 222, 224])
        self.chart_pressure.set_points([142, 145, 148, 141, 149, 143, 146, 142, 151, 144, 147, 143, 146, 141, 148])
        self.chart_moisture.set_points([0.06, 0.07, 0.05, 0.08, 0.06, 0.09, 0.07, 0.05, 0.08, 0.06, 0.09, 0.07, 0.06, 0.08, 0.07])
        
        charts_layout.addWidget(self.chart_temp)
        charts_layout.addWidget(self.chart_pressure)
        charts_layout.addWidget(self.chart_moisture)
        layout.addLayout(charts_layout)
        
        reports_card = QtWidgets.QFrame()
        reports_card.setObjectName("card")
        rep_layout = QtWidgets.QVBoxLayout(reports_card)
        rep_layout.setContentsMargins(20, 20, 20, 20)
        rep_layout.setSpacing(14)
        
        rep_title = QtWidgets.QLabel("AI Detection Accuracy Reports (Confidence & Precision)")
        rep_title.setStyleSheet("font-size: 15px; font-weight: 700; color: #0f172a;")
        rep_layout.addWidget(rep_title)
        
        grid_pb = QtWidgets.QGridLayout()
        grid_pb.setSpacing(14)
        
        hand_lbl = QtWidgets.QLabel("Hand Detection Accuracy (MediaPipe)")
        hand_lbl.setStyleSheet("font-weight: 600; font-size: 12px; color: #475569;")
        self.hand_pb = QtWidgets.QProgressBar()
        self.hand_pb.setValue(99)
        self.hand_pb.setStyleSheet("QProgressBar { border: 1px solid #cbd5e1; border-radius: 6px; background-color: #f1f5f9; text-align: center; font-weight: bold; } QProgressBar::chunk { background-color: #10b981; border-radius: 6px; }")
        grid_pb.addWidget(hand_lbl, 0, 0)
        grid_pb.addWidget(self.hand_pb, 0, 1)
        
        gest_lbl = QtWidgets.QLabel("Gesture Classification (Neural Network)")
        gest_lbl.setStyleSheet("font-weight: 600; font-size: 12px; color: #475569;")
        self.gest_pb = QtWidgets.QProgressBar()
        self.gest_pb.setValue(98)
        self.gest_pb.setStyleSheet("QProgressBar { border: 1px solid #cbd5e1; border-radius: 6px; background-color: #f1f5f9; text-align: center; font-weight: bold; } QProgressBar::chunk { background-color: #8b5cf6; border-radius: 6px; }")
        grid_pb.addWidget(gest_lbl, 1, 0)
        grid_pb.addWidget(self.gest_pb, 1, 1)
        
        tool_lbl = QtWidgets.QLabel("Tool Verification (SIFT + FLANN)")
        tool_lbl.setStyleSheet("font-weight: 600; font-size: 12px; color: #475569;")
        self.tool_pb = QtWidgets.QProgressBar()
        self.tool_pb.setValue(97)
        self.tool_pb.setStyleSheet("QProgressBar { border: 1px solid #cbd5e1; border-radius: 6px; background-color: #f1f5f9; text-align: center; font-weight: bold; } QProgressBar::chunk { background-color: #2563eb; border-radius: 6px; }")
        grid_pb.addWidget(tool_lbl, 2, 0)
        grid_pb.addWidget(self.tool_pb, 2, 1)
        
        obj_lbl = QtWidgets.QLabel("Object / Part Detection (YOLO)")
        obj_lbl.setStyleSheet("font-weight: 600; font-size: 12px; color: #475569;")
        self.obj_pb = QtWidgets.QProgressBar()
        self.obj_pb.setValue(96)
        self.obj_pb.setStyleSheet("QProgressBar { border: 1px solid #cbd5e1; border-radius: 6px; background-color: #f1f5f9; text-align: center; font-weight: bold; } QProgressBar::chunk { background-color: #06b6d4; border-radius: 6px; }")
        grid_pb.addWidget(obj_lbl, 3, 0)
        grid_pb.addWidget(self.obj_pb, 3, 1)
        
        rep_layout.addLayout(grid_pb)
        layout.addWidget(reports_card)

        # Compliance analytics summary
        comp_card = QtWidgets.QFrame()
        comp_card.setObjectName("card")
        c_lay = QtWidgets.QVBoxLayout(comp_card)
        c_lay.setContentsMargins(20, 20, 20, 20)
        c_lay.setSpacing(10)
        c_title = QtWidgets.QLabel("Operator Compliance Analytics (7-day)")
        c_title.setStyleSheet("font-size: 15px; font-weight: 700; color: #0f172a;")
        c_lay.addWidget(c_title)
        self.comp_analytics_lbl = QtWidgets.QLabel("Loading…")
        self.comp_analytics_lbl.setStyleSheet("color: #475569; font-size: 12px; font-weight: 500;")
        self.comp_analytics_lbl.setWordWrap(True)
        c_lay.addWidget(self.comp_analytics_lbl)
        self.comp_type_pb = QtWidgets.QProgressBar()
        self.comp_type_pb.setValue(100)
        self.comp_type_pb.setFormat("Compliance %: %p%")
        self.comp_type_pb.setStyleSheet(
            "QProgressBar { border: 1px solid #cbd5e1; border-radius: 6px; background-color: #f1f5f9; "
            "text-align: center; font-weight: bold; } "
            "QProgressBar::chunk { background-color: #16a34a; border-radius: 6px; }"
        )
        c_lay.addWidget(self.comp_type_pb)
        layout.addWidget(comp_card)
        layout.addStretch(1)

    def _update_compliance_analytics(self) -> None:
        if not hasattr(self, "comp_analytics_lbl"):
            return
        summary = self._compliance_logger.daily_summary(days=7)
        by = summary.get("by_type") or {}
        lines = [
            f"Total events: {summary.get('total', 0)} | Warnings: {summary.get('warnings', 0)} | "
            f"Passed: {summary.get('passed', 0)}",
            "By type: " + (", ".join(f"{k}={v}" for k, v in sorted(by.items())) or "none"),
        ]
        ops = summary.get("by_operator") or {}
        if ops:
            lines.append("Top operators (warnings): " + ", ".join(f"{k}:{v}" for k, v in list(ops.items())[:5]))
        self.comp_analytics_lbl.setText("\n".join(lines))
        pct = int(round(float(summary.get("compliance_pct", 100))))
        self.comp_type_pb.setValue(max(0, min(100, pct)))

    def _update_analytics_charts(self) -> None:
        import random
        self.chart_temp.set_points([220 + random.randint(0, 15) for _ in range(15)])
        self.chart_pressure.set_points([135 + random.randint(0, 20) for _ in range(15)])
        self.chart_moisture.set_points([0.04 + random.random() * 0.07 for _ in range(15)])

    # -- PAGE 5: DETECTION LOGS + ACTIVITY LOGS --------------------------------
    def _create_logs_page(self) -> None:
        page = QtWidgets.QWidget()
        page.setObjectName("page")
        self.stacked_widget.addWidget(page)

        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        title = QtWidgets.QLabel("Logs")
        title.setStyleSheet("font-size: 22px; font-weight: 800; color: #0f172a;")
        layout.addWidget(title)

        # -- Tab Widget with two tabs --
        self.logs_tab_widget = QtWidgets.QTabWidget()
        self.logs_tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                background-color: #ffffff;
            }
            QTabBar::tab {
                background-color: #f8fafc;
                color: #64748b;
                border: 1px solid #e2e8f0;
                border-bottom: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                padding: 10px 32px;
                font-weight: 700;
                font-size: 13px;
                margin-right: 4px;
                min-width: 160px;
            }
            QTabBar::tab:selected {
                background-color: #ffffff;
                color: #2563eb;
                border-bottom: 2px solid #2563eb;
            }
            QTabBar::tab:hover:!selected {
                background-color: #eff6ff;
                color: #1e293b;
            }
        """)
        layout.addWidget(self.logs_tab_widget, 1)

        #  TAB 1: Detection Logs 
        detection_tab = QtWidgets.QWidget()
        det_layout = QtWidgets.QVBoxLayout(detection_tab)
        det_layout.setContentsMargins(16, 16, 16, 16)
        det_layout.setSpacing(12)

        filters_bar = QtWidgets.QFrame()
        filters_bar.setObjectName("card")
        filt_layout = QtWidgets.QGridLayout(filters_bar)
        filt_layout.setContentsMargins(14, 14, 14, 14)
        filt_layout.setSpacing(10)

        filt_layout.addWidget(QtWidgets.QLabel("Operator ID:"), 0, 0)
        self.filter_logs_op = QtWidgets.QLineEdit()
        self.filter_logs_op.setPlaceholderText("e.g. EMP001")
        self.filter_logs_op.textChanged.connect(self._refresh_logs_table)
        filt_layout.addWidget(self.filter_logs_op, 0, 1)

        filt_layout.addWidget(QtWidgets.QLabel("SOP Selected:"), 0, 2)
        self.filter_logs_sop = QtWidgets.QComboBox()
        self.filter_logs_sop.addItems(["All SOPs", "Industrial Screw-Tightening Assembly", "Safety Inspection Checklist", "Calibration Routine"])
        self.filter_logs_sop.currentTextChanged.connect(self._refresh_logs_table)
        filt_layout.addWidget(self.filter_logs_sop, 0, 3)

        filt_layout.addWidget(QtWidgets.QLabel("Tool Used:"), 0, 4)
        self.filter_logs_tool = QtWidgets.QComboBox()
        self.filter_logs_tool.addItems(["All Tools", "Screwdriver", "Spanner", "Pen", "Weight Block", "Allen Key"])
        self.filter_logs_tool.currentTextChanged.connect(self._refresh_logs_table)
        filt_layout.addWidget(self.filter_logs_tool, 0, 5)

        filt_layout.addWidget(QtWidgets.QLabel("Gesture:"), 1, 0)
        self.filter_logs_gest = QtWidgets.QComboBox()
        self.filter_logs_gest.addItems(["All Gestures", "ok", "like", "dislike", "fist", "palm", "one", "peace", "no_gesture"])
        self.filter_logs_gest.currentTextChanged.connect(self._refresh_logs_table)
        filt_layout.addWidget(self.filter_logs_gest, 1, 1)

        filt_layout.addWidget(QtWidgets.QLabel("Inspection Status:"), 1, 2)
        self.filter_logs_status = QtWidgets.QComboBox()
        self.filter_logs_status.addItems(["All Detections", "Completed", "Pending", "Error"])
        self.filter_logs_status.currentTextChanged.connect(self._refresh_logs_table)
        filt_layout.addWidget(self.filter_logs_status, 1, 3)

        self.btn_logs_refresh = QtWidgets.QPushButton("Apply Filter")
        self.btn_logs_refresh.setObjectName("secondaryBtn")
        self.btn_logs_refresh.clicked.connect(self._refresh_logs_table)
        filt_layout.addWidget(self.btn_logs_refresh, 1, 4)

        export_btn = QtWidgets.QPushButton("Export CSV")
        export_btn.setObjectName("primaryBtn")
        export_btn.clicked.connect(self._export_logs_csv)
        filt_layout.addWidget(export_btn, 1, 5)

        det_layout.addWidget(filters_bar)

        split_layout = QtWidgets.QHBoxLayout()
        split_layout.setSpacing(16)

        self.logs_table = QtWidgets.QTableWidget()
        self.logs_table.setColumnCount(len(CSV_COLUMNS))
        self.logs_table.setHorizontalHeaderLabels(CSV_COLUMNS)
        self.logs_table.horizontalHeader().setStretchLastSection(True)
        self.logs_table.verticalHeader().setVisible(False)
        self.logs_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.logs_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.logs_table.itemSelectionChanged.connect(self._on_logs_row_selected)
        split_layout.addWidget(self.logs_table, 3)

        self.logs_preview_card = QtWidgets.QFrame()
        self.logs_preview_card.setObjectName("card")
        self.logs_preview_card.setFixedWidth(380)
        prev_layout = QtWidgets.QVBoxLayout(self.logs_preview_card)
        prev_layout.setContentsMargins(14, 14, 14, 14)
        prev_layout.setSpacing(8)

        prev_title = QtWidgets.QLabel("Quality Inspection Screenshot")
        prev_title.setStyleSheet("font-size: 14px; font-weight: 700; color: #2563eb;")
        prev_layout.addWidget(prev_title)

        self.logs_img_lbl = QtWidgets.QLabel("Select a row to preview quality check screenshot")
        self.logs_img_lbl.setAlignment(Qt.AlignCenter)
        self.logs_img_lbl.setWordWrap(True)
        prev_layout.addWidget(self.logs_img_lbl)

        self.logs_details_lbl = QtWidgets.QLabel("")
        self.logs_details_lbl.setWordWrap(True)
        prev_layout.addWidget(self.logs_details_lbl)

        prev_layout.addStretch(1)
        split_layout.addWidget(self.logs_preview_card, 1)

        det_layout.addLayout(split_layout, 1)

        self.logs_tab_widget.addTab(detection_tab, "Detection Logs")

        #  TAB 2: Activity Logs 
        activity_tab = QtWidgets.QWidget()
        act_layout = QtWidgets.QVBoxLayout(activity_tab)
        act_layout.setContentsMargins(16, 16, 16, 16)
        act_layout.setSpacing(12)

        #  Row 1: Date selector, refresh, export, event count 
        act_top = QtWidgets.QHBoxLayout()
        act_top.setSpacing(10)

        self.activity_date_lbl = QtWidgets.QLabel("Date Filter:")
        act_top.addWidget(self.activity_date_lbl)

        self.activity_file_combo = QtWidgets.QComboBox()
        self.activity_file_combo.setMinimumWidth(220)
        self.activity_file_combo.currentIndexChanged.connect(self._on_activity_file_changed)
        act_top.addWidget(self.activity_file_combo)

        act_refresh_btn = QtWidgets.QPushButton("Refresh")
        act_refresh_btn.setObjectName("secondaryBtn")
        act_refresh_btn.clicked.connect(self._refresh_activity_logs_full)
        act_top.addWidget(act_refresh_btn)

        act_export_btn = QtWidgets.QPushButton("Export Activity CSV")
        act_export_btn.setObjectName("primaryBtn")
        act_export_btn.clicked.connect(self._export_activity_csv)
        act_top.addWidget(act_export_btn)

        act_top.addStretch(1)

        # Event count badge
        self.activity_count_lbl = QtWidgets.QLabel("0 events")
        act_top.addWidget(self.activity_count_lbl)

        act_layout.addLayout(act_top)

        #  Row 2: Search bar 
        self.search_frame = QtWidgets.QFrame()
        search_row = QtWidgets.QHBoxLayout(self.search_frame)
        search_row.setContentsMargins(12, 8, 12, 8)
        search_row.setSpacing(10)

        search_icon_lbl = QtWidgets.QLabel("")
        search_icon_lbl.setStyleSheet("font-size: 16px; border: none; background: transparent;")
        search_row.addWidget(search_icon_lbl)

        self.activity_search_input = QtWidgets.QLineEdit()
        self.activity_search_input.setPlaceholderText("Search logs by keyword (timestamp, event type, component, action, details)…")
        self.activity_search_input.returnPressed.connect(self._refresh_activity_logs_table)
        search_row.addWidget(self.activity_search_input, 1)

        self.activity_search_btn = QtWidgets.QPushButton("Search")
        self.activity_search_btn.setObjectName("primaryBtn")
        self.activity_search_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.activity_search_btn.setStyleSheet("""
            QPushButton {
                background-color: #2563eb;
                color: #ffffff;
                border: none;
                border-radius: 6px;
                padding: 8px 20px;
                font-weight: 700;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #1d4ed8;
            }
            QPushButton:pressed {
                background-color: #1e40af;
            }
        """)
        self.activity_search_btn.clicked.connect(self._refresh_activity_logs_table)
        search_row.addWidget(self.activity_search_btn)

        self.activity_clear_search_btn = QtWidgets.QPushButton("Clear")
        self.activity_clear_search_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.activity_clear_search_btn.setStyleSheet("""
            QPushButton {
                background-color: #fee2e2;
                color: #dc2626;
                border: 1px solid #fca5a5;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 700;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #fecaca;
                border-color: #f87171;
            }
        """)
        self.activity_clear_search_btn.clicked.connect(self._clear_activity_search)
        search_row.addWidget(self.activity_clear_search_btn)

        act_layout.addWidget(self.search_frame)

        # CSV path display
        self.activity_path_lbl = QtWidgets.QLabel("")
        self.activity_path_lbl.setStyleSheet("color: #94a3b8; font-size: 11px; font-style: italic;")
        act_layout.addWidget(self.activity_path_lbl)

        # Activity logs table
        self.activity_logs_table = QtWidgets.QTableWidget()
        self.activity_logs_table.setColumnCount(len(ACTIVITY_COLUMNS))
        self.activity_logs_table.setHorizontalHeaderLabels(ACTIVITY_COLUMNS)
        self.activity_logs_table.horizontalHeader().setStretchLastSection(True)
        self.activity_logs_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Interactive)
        self.activity_logs_table.verticalHeader().setVisible(False)
        self.activity_logs_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.activity_logs_table.setAlternatingRowColors(True)
        act_layout.addWidget(self.activity_logs_table, 1)

        self.logs_tab_widget.addTab(activity_tab, "Activity Logs")

        # Initial data load
        self._refresh_logs_table()
        self._populate_activity_file_combo()
        self._refresh_activity_logs_table()

    def _refresh_logs_table(self) -> None:
        self.logs_table.setRowCount(0)
        if not os.path.exists(LOG_CSV_PATH):
            return
            
        rows: List[list] = []
        try:
            with open(LOG_CSV_PATH, newline="") as fh:
                reader = csv.reader(fh)
                header = next(reader, None)
                rows = list(reader)
        except Exception as exc:
            print(f"[logs] failed to read logs: {exc}")
            return
            
        filtered_rows = []
        op_filter = self.filter_logs_op.text().strip().lower()
        sop_filter = self.filter_logs_sop.currentText()
        tool_filter = self.filter_logs_tool.currentText()
        gest_filter = self.filter_logs_gest.currentText()
        status_filter = self.filter_logs_status.currentText()
        
        for r in rows:
            if not r or len(r) < len(CSV_COLUMNS):
                continue
            
            worker_id = r[1].lower()
            action = r[2]
            tool = r[4].lower()
            gesture = r[5].lower()
            status = r[9]
            
            if op_filter and op_filter not in worker_id:
                continue
            if sop_filter != "All SOPs"and action != sop_filter:
                pass
            if tool_filter != "All Tools"and tool_filter.lower() not in tool:
                continue
            if gest_filter != "All Gestures"and gest_filter.lower() not in gesture:
                continue
            if status_filter != "All Detections"and status_filter != status:
                continue
                
            filtered_rows.append(r)
            
        self.logs_table.setRowCount(len(filtered_rows))
        for row_idx, r in enumerate(reversed(filtered_rows)):
            for col_idx, val in enumerate(r):
                item = QtWidgets.QTableWidgetItem(val)
                if col_idx == 9:
                    if val == "Completed":
                        item.setForeground(QtGui.QColor("#16a34a"))
                    else:
                        item.setForeground(QtGui.QColor("#dc2626"))
                self.logs_table.setItem(row_idx, col_idx, item)

    def _on_logs_row_selected(self) -> None:
        selected = self.logs_table.selectedItems()
        if not selected:
            return
        row = selected[0].row()
        
        timestamp = self.logs_table.item(row, 0).text()
        worker = self.logs_table.item(row, 1).text()
        action = self.logs_table.item(row, 2).text()
        turns = self.logs_table.item(row, 7).text()
        status = self.logs_table.item(row, 9).text()
        screenshot_path = self.logs_table.item(row, 10).text()
        
        self.logs_details_lbl.setText(
            f"<b>Timestamp:</b> {timestamp}<br>"
            f"<b>Operator:</b> {worker}<br>"
            f"<b>Action:</b> {action}<br>"
            f"<b>Turns:</b> {turns}<br>"
            f"<b>Status:</b> {status}<br>"
            f"<b>Path:</b> {screenshot_path}"
        )
        
        if screenshot_path and os.path.exists(screenshot_path):
            pix = QtGui.QPixmap(screenshot_path)
            if not pix.isNull():
                scaled = pix.scaled(self.logs_img_lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.logs_img_lbl.setPixmap(scaled)
            else:
                self.logs_img_lbl.setText("Invalid image file")
                self.logs_img_lbl.setPixmap(QtGui.QPixmap())
        else:
            self.logs_img_lbl.setText(f"Image not found at path:\n{screenshot_path}")
            self.logs_img_lbl.setPixmap(QtGui.QPixmap())

    def _export_logs_csv(self) -> None:
        self._activity_logger.log("EXPORT", "Detection Logs", "Export CSV", "Export button clicked")
        if os.path.exists(LOG_CSV_PATH):
            import shutil
            save_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Logs CSV", "", "CSV Files (*.csv)")
            if save_path:
                try:
                    shutil.copy(LOG_CSV_PATH, save_path)
                    QtWidgets.QMessageBox.information(self, "Export Complete", f"Successfully exported logs to:\n{save_path}")
                except Exception as exc:
                    QtWidgets.QMessageBox.critical(self, "Export Failed", f"Could not save CSV file:\n{exc}")
        else:
            QtWidgets.QMessageBox.warning(self, "No Logs", "There are no detection logs to export yet.")

    def _refresh_history_table(self) -> None:
        self._refresh_logs_table()

    def _on_history_row_selected(self) -> None:
        self._on_logs_row_selected()

    # -- Activity Logs helpers -----------------------------------------------
    def _navigate_to_logs(self) -> None:
        """Navigate to the Logs page (index 7) and switch to Activity Logs tab."""
        self._activity_logger.log("NAVIGATION", "Menu", "Navigate to Logs", "Via View > Logs menu")
        self.stacked_widget.setCurrentIndex(7)
        self.sidebar_buttons[7].setChecked(True)
        self._refresh_logs_table()
        self._refresh_activity_logs_table()
        # Switch to Activity Logs tab (index 1)
        if hasattr(self, 'logs_tab_widget'):
            self.logs_tab_widget.setCurrentIndex(1)

    def _populate_activity_file_combo(self) -> None:
        """Fill the date file selector combo with 'All Dates'+ available log CSV files."""
        if not hasattr(self, 'activity_file_combo'):
            return
        prev_data = self.activity_file_combo.currentData()
        self.activity_file_combo.blockSignals(True)
        self.activity_file_combo.clear()

        # First entry: All Dates
        self.activity_file_combo.addItem("All Dates", userData="__ALL__")

        # Individual date files (newest first)
        files = self._activity_logger.get_all_log_files()
        import os as _os
        import datetime as _dt
        _MONTH_ABBR = ["","Jan","Feb","Mar","Apr","May","Jun",
                       "Jul","Aug","Sep","Oct","Nov","Dec"]
        today_str = _dt.datetime.now().strftime("%m%d%y")
        yesterday_str = (_dt.datetime.now() - _dt.timedelta(days=1)).strftime("%m%d%y")

        for f in files:
            basename = _os.path.basename(f)           # e.g. "072126.csv"
            name_part = basename.replace(".csv", "")  # e.g. "072126"
            # Parse mmddyy → readable date WITHOUT strptime (avoids locale RecursionError)
            try:
                if len(name_part) == 6 and name_part.isdigit():
                    mm = int(name_part[0:2])
                    dd = int(name_part[2:4])
                    yy = int(name_part[4:6])
                    yyyy = 2000 + yy
                    mon = _MONTH_ABBR[mm] if 1 <= mm <= 12 else name_part
                    readable = f"{dd:02d} {mon} {yyyy}"
                    if name_part == today_str:
                        readable += "(Today)"
                    elif name_part == yesterday_str:
                        readable += "(Yesterday)"
                    label = f"{readable}"
                else:
                    label = f"{basename}"
            except (ValueError, IndexError):
                label = f"{basename}"
            self.activity_file_combo.addItem(label, userData=f)

        self.activity_file_combo.blockSignals(False)

        # Restore previous selection or default to "All Dates"
        if prev_data:
            for i in range(self.activity_file_combo.count()):
                if self.activity_file_combo.itemData(i) == prev_data:
                    self.activity_file_combo.setCurrentIndex(i)
                    return
        # Default: select "All Dates"(index 0)
        self.activity_file_combo.setCurrentIndex(0)

    def _refresh_activity_logs_full(self) -> None:
        """Called by the Refresh button: repopulate file combo then refresh table."""
        self._populate_activity_file_combo()
        self._refresh_activity_logs_table()

    def _refresh_activity_logs_table(self) -> None:
        """Reload the Activity Logs table from the selected log file (or all files), with search filtering."""
        if not hasattr(self, 'activity_logs_table'):
            return
        # NOTE: Do NOT call _populate_activity_file_combo() here — it triggers
        # currentIndexChanged → _on_activity_file_changed → _refresh → infinite recursion.
        # Call _populate_activity_file_combo() only from explicit Refresh button clicks.

        # Determine which file(s) to load
        idx = self.activity_file_combo.currentIndex()
        selected_data = self.activity_file_combo.itemData(idx) if idx >= 0 else "__ALL__"

        if selected_data == "__ALL__":
            # Load ALL log files combined
            rows = []
            files = self._activity_logger.get_all_log_files()
            for f in files:
                if f == self._activity_logger.get_csv_path():
                    rows.extend(self._activity_logger.get_rows())
                else:
                    rows.extend(self._activity_logger.load_file(f))
            csv_path = "All log files"
        else:
            csv_path = selected_data
            if csv_path == self._activity_logger.get_csv_path():
                rows = self._activity_logger.get_rows()
            else:
                rows = self._activity_logger.load_file(csv_path)

        # Apply search filter
        search_term = ""
        if hasattr(self, 'activity_search_input'):
            search_term = self.activity_search_input.text().strip().lower()
        if search_term:
            filtered = []
            for r in rows:
                row_text = "".join(str(val).lower() for val in r)
                if search_term in row_text:
                    filtered.append(r)
            rows = filtered

        # Update path label
        if hasattr(self, 'activity_path_lbl'):
            if selected_data == "__ALL__":
                file_count = len(self._activity_logger.get_all_log_files())
                self.activity_path_lbl.setText(f"Showing all {file_count} log file(s) combined")
            else:
                self.activity_path_lbl.setText(f"{csv_path}")

        # Populate table
        self.activity_logs_table.setRowCount(len(rows))
        # Event-type colour map
        type_colors = {
            "NAVIGATION": "#2563eb",
            "CLICK":      "#0284c7",
            "ADD":        "#16a34a",
            "DELETE":     "#dc2626",
            "SAVE":       "#8b5cf6",
            "EXPORT":     "#f97316",
            "SETTING":    "#64748b",
            "SYSTEM":     "#94a3b8",
        }
        for row_idx, r in enumerate(rows):
            for col_idx, val in enumerate(r[:len(ACTIVITY_COLUMNS)]):
                item = QtWidgets.QTableWidgetItem(str(val))
                item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                if col_idx == 1:  # Event_Type column — colour by type
                    color = type_colors.get(str(val), "#475569")
                    item.setForeground(QtGui.QColor(color))
                self.activity_logs_table.setItem(row_idx, col_idx, item)

        # Update count badge
        if hasattr(self, 'activity_count_lbl'):
            suffix = ""
            if search_term:
                suffix = f"(filtered)"
            self.activity_count_lbl.setText(f"{len(rows)} events{suffix}")

    def _clear_activity_search(self) -> None:
        """Clear the search input and refresh the activity logs table."""
        if hasattr(self, 'activity_search_input'):
            self.activity_search_input.clear()
        self._refresh_activity_logs_table()

    def _on_activity_file_changed(self, index: int) -> None:
        """Called when the user selects a different date log file."""
        self._refresh_activity_logs_table()

    def _export_activity_csv(self) -> None:
        """Export the currently displayed activity log CSV to a user-chosen path."""
        self._activity_logger.log("EXPORT", "Activity Logs", "Export Activity CSV", "Export button clicked")
        import shutil as _shutil
        idx = self.activity_file_combo.currentIndex()
        selected_data = self.activity_file_combo.itemData(idx) if idx >= 0 else "__ALL__"

        if selected_data == "__ALL__":
            # Export all: merge all files into one temp CSV
            import tempfile as _tf
            import csv as _csv
            rows = []
            files = self._activity_logger.get_all_log_files()
            for f in files:
                if f == self._activity_logger.get_csv_path():
                    rows.extend(self._activity_logger.get_rows())
                else:
                    rows.extend(self._activity_logger.load_file(f))
            save_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save All Activity Logs CSV", "all_activity_logs.csv", "CSV Files (*.csv)"
            )
            if save_path:
                try:
                    with open(save_path, "w", newline="") as fh:
                        writer = _csv.writer(fh)
                        writer.writerow(ACTIVITY_COLUMNS)
                        for r in rows:
                            writer.writerow(r)
                    QtWidgets.QMessageBox.information(
                        self, "Export Complete", f"All activity logs exported to:\n{save_path}"
                    )
                except Exception as exc:
                    QtWidgets.QMessageBox.critical(self, "Export Failed", f"Could not save file:\n{exc}")
        else:
            src = selected_data
            if not src or not os.path.exists(src):
                QtWidgets.QMessageBox.warning(self, "No Logs", "No activity log file found to export.")
                return
            save_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save Activity Logs CSV", os.path.basename(src), "CSV Files (*.csv)"
            )
            if save_path:
                try:
                    _shutil.copy(src, save_path)
                    QtWidgets.QMessageBox.information(
                        self, "Export Complete", f"Activity log exported to:\n{save_path}"
                    )
                except Exception as exc:
                    QtWidgets.QMessageBox.critical(self, "Export Failed", f"Could not save file:\n{exc}")

    # -- PAGE 6: TOOL & PARTS MANAGEMENT -------------------------------------
    def _create_tool_page(self) -> None:
        page = QtWidgets.QWidget()
        page.setObjectName("page")
        self.stacked_widget.addWidget(page)
        
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)
        
        title = QtWidgets.QLabel("Tool & Parts Calibration Management")
        title.setStyleSheet("font-size: 22px; font-weight: 800; color: #0f172a;")
        layout.addWidget(title)
        
        tool_card = QtWidgets.QFrame()
        tool_card.setObjectName("card")
        t_layout = QtWidgets.QVBoxLayout(tool_card)
        t_layout.setContentsMargins(16, 16, 16, 16)
        
        t_header = QtWidgets.QHBoxLayout()
        t_header.addWidget(QtWidgets.QLabel("Configured Industrial AI Tools"))
        t_header.addStretch(1)

        self.add_tool_btn = QtWidgets.QPushButton("+ Add New Tool")
        self.add_tool_btn.setObjectName("secondaryBtn")
        self.add_tool_btn.clicked.connect(self._add_tool_row)
        t_header.addWidget(self.add_tool_btn)

        self.delete_tool_btn = QtWidgets.QPushButton("Delete")
        self.delete_tool_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.delete_tool_btn.setStyleSheet("""
            QPushButton {
                background-color: #fee2e2;
                color: #dc2626;
                border: 1px solid #fca5a5;
                border-radius: 8px;
                padding: 10px 20px;
                font-weight: 700;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #fecaca;
                border-color: #f87171;
            }
            QPushButton:pressed {
                background-color: #fca5a5;
            }
        """)
        self.delete_tool_btn.clicked.connect(self._delete_tool_row)
        t_header.addWidget(self.delete_tool_btn)

        t_layout.addLayout(t_header)
        
        self.tools_table = QtWidgets.QTableWidget(6, 6)
        self.tools_table.setHorizontalHeaderLabels([
            "Tool ID", "Tool Name", "Category", "Calibration Status", "Last Calibrated", "Availability"
        ])
        self.tools_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.tools_table.verticalHeader().setVisible(False)
        self.tools_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tools_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        
        tools_data = [
            ("T-SCREW-01", "Torque Screwdriver Alpha", "Screwdriver", "Calibrated", _now_short(), "Available"),
            ("T-SPAN-01", "12mm Industrial Spanner", "Spanner", "Calibrated", _now_short(), "Available"),
            ("T-PEN-01", "Ballpoint Verification Pen", "Pen", "Calibrated", _now_short(), "Available"),
            ("T-SCALE-01", "Precision Digital Scale", "Scale / Weight", "Requires Calibration", "2026-07-01", "In Use"),
            ("T-KEY-01", "Hex Allen Key Set", "Allen Key", "Calibrated", _now_short(), "Available"),
            ("T-WREN-01", "Pneumatic Torque Wrench", "Wrench", "Out of Calibration", "2026-05-12", "Maintenance"),
        ]
        
        for row_idx, (tid, tname, cat, cal, last, avail) in enumerate(tools_data):
            self.tools_table.setItem(row_idx, 0, QtWidgets.QTableWidgetItem(tid))
            self.tools_table.setItem(row_idx, 1, QtWidgets.QTableWidgetItem(tname))
            self.tools_table.setItem(row_idx, 2, QtWidgets.QTableWidgetItem(cat))
            
            cal_item = QtWidgets.QTableWidgetItem(cal)
            if cal == "Calibrated":
                cal_item.setForeground(QtGui.QColor("#16a34a"))
            else:
                cal_item.setForeground(QtGui.QColor("#d97706"))
            self.tools_table.setItem(row_idx, 3, cal_item)
            
            self.tools_table.setItem(row_idx, 4, QtWidgets.QTableWidgetItem(last))
            self.tools_table.setItem(row_idx, 5, QtWidgets.QTableWidgetItem(avail))
            
        t_layout.addWidget(self.tools_table)
        
        t_btn_layout = QtWidgets.QHBoxLayout()
        cal_btn = QtWidgets.QPushButton("Run Tool Calibration")
        cal_btn.setObjectName("primaryBtn")
        cal_btn.clicked.connect(self._run_tool_calibration)
        t_btn_layout.addWidget(cal_btn)
        t_btn_layout.addStretch(1)
        t_layout.addLayout(t_btn_layout)
        
        layout.addWidget(tool_card)
        
        checklist_card = QtWidgets.QFrame()
        checklist_card.setObjectName("card")
        chk_layout = QtWidgets.QVBoxLayout(checklist_card)
        chk_layout.setContentsMargins(16, 16, 16, 16)
        
        chk_title = QtWidgets.QLabel("Bill of Materials (BOM) & Parts Quality Verification")
        chk_title.setStyleSheet("font-size: 15px; font-weight: 700; color: #0f172a;")
        chk_layout.addWidget(chk_title)
        
        stats_layout = QtWidgets.QHBoxLayout()
        stats_layout.addWidget(StatCard("Required Parts", "4 Items", "#2563eb"))
        stats_layout.addWidget(StatCard("Verified Safe", "2 Items", "#16a34a"))
        stats_layout.addWidget(StatCard("Errors Flagged", "0 Issues", "#dc2626"))
        chk_layout.addLayout(stats_layout)
        
        self.checklist_table = QtWidgets.QTableWidget(4, 5)
        self.checklist_table.setHorizontalHeaderLabels([
            "Part Reference ID", "Part Description", "Assembly SOP Requirement", "Verification Status", "Action Timestamp"
        ])
        self.checklist_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.checklist_table.verticalHeader().setVisible(False)
        
        parts_data = [
            ("P-SCREW-01", "Primary Case Fastener Screw", "Aligned & rotate clockwise 2.50 turns", "Correct", _now_short()),
            ("P-GLOVE-01", "Operator ESD Protection Glove", "Safety glove worn on active hand", "Correct", _now_short()),
            ("P-BOARD-01", "Control PCB Panel Module", "Aligned and locked into chassis slots", "Pending", "--"),
            ("P-CABLE-01", "Auxiliary 12V Power Connection", "Verify insertion and locking clips", "Pending", "--"),
        ]
        
        for r_idx, (p_id, desc, req, status, t_stamp) in enumerate(parts_data):
            self.checklist_table.setItem(r_idx, 0, QtWidgets.QTableWidgetItem(p_id))
            self.checklist_table.setItem(r_idx, 1, QtWidgets.QTableWidgetItem(desc))
            self.checklist_table.setItem(r_idx, 2, QtWidgets.QTableWidgetItem(req))
            
            status_item = QtWidgets.QTableWidgetItem(status)
            if status == "Correct":
                status_item.setForeground(QtGui.QColor("#16a34a"))
            elif status == "Not Correct":
                status_item.setForeground(QtGui.QColor("#dc2626"))
            else:
                status_item.setForeground(QtGui.QColor("#64748b"))
                
            self.checklist_table.setItem(r_idx, 3, status_item)
            self.checklist_table.setItem(r_idx, 4, QtWidgets.QTableWidgetItem(t_stamp))
            
        chk_layout.addWidget(self.checklist_table)
        
        verify_bom_btn = QtWidgets.QPushButton("Re-verify Bill of Materials Checklist")
        verify_bom_btn.setObjectName("primaryBtn")
        chk_layout.addWidget(verify_bom_btn)
        
        layout.addWidget(checklist_card)
        layout.addStretch(1)

    def _add_tool_row(self) -> None:
        self._activity_logger.log("ADD", "Tool Management", "Add New Tool", "New tool row added")
        row = self.tools_table.rowCount()
        self.tools_table.insertRow(row)
        self.tools_table.setItem(row, 0, QtWidgets.QTableWidgetItem(f"T-NEW-0{row}"))
        self.tools_table.setItem(row, 1, QtWidgets.QTableWidgetItem("New Torque Tool"))
        self.tools_table.setItem(row, 2, QtWidgets.QTableWidgetItem("Custom"))
        self.tools_table.setItem(row, 3, QtWidgets.QTableWidgetItem("Pending"))
        self.tools_table.setItem(row, 4, QtWidgets.QTableWidgetItem("--"))
        self.tools_table.setItem(row, 5, QtWidgets.QTableWidgetItem("Available"))

    def _delete_tool_row(self) -> None:
        """Delete the selected row from the tools table after confirmation."""
        selected = self.tools_table.selectedItems()
        if not selected:
            QtWidgets.QMessageBox.warning(
                self, "No Selection", "Please select a tool row to delete."
            )
            return
        row = selected[0].row()
        tool_name = self.tools_table.item(row, 1)
        name_text = tool_name.text() if tool_name else f"Row {row + 1}"
        reply = QtWidgets.QMessageBox.question(
            self,
            "Confirm Delete",
            f"Are you sure you want to delete the tool:\n\n  {name_text}\n\nThis action cannot be undone.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if reply == QtWidgets.QMessageBox.Yes:
            self._activity_logger.log(
                "DELETE", "Tool Management", "Delete Tool", f"Deleted: {name_text}"
            )
            self.tools_table.removeRow(row)

    def _run_tool_calibration(self) -> None:
        self._activity_logger.log("CLICK", "Tool Management", "Run Tool Calibration", "Calibration initiated")
        selected = self.tools_table.selectedItems()
        if selected:
            row = selected[0].row()
            tool_name = self.tools_table.item(row, 1).text()
            self.tools_table.item(row, 3).setText("Calibrated")
            self.tools_table.item(row, 3).setForeground(QtGui.QColor("#16a34a"))
            self.tools_table.item(row, 4).setText(_now_short())
            QtWidgets.QMessageBox.information(self, "Calibration Routine", f"Successfully completed calibration routine for:\n{tool_name}")
        else:
            QtWidgets.QMessageBox.warning(self, "Selection Required", "Please select a tool from the table to run calibration.")

    # -- PAGE 7: REFERENCE IMAGES DATABASE -----------------------------------
    # -- PAGE 7: REFERENCE IMAGES DATABASE -----------------------------------
    _REF_IMG_DIR  = os.path.join(_GUI_DIR, "reference_images")
    _REF_METADATA = os.path.join(_REF_IMG_DIR, "metadata.json")

    def _create_reference_page(self) -> None:
        page = QtWidgets.QWidget()
        page.setObjectName("page")
        self.stacked_widget.addWidget(page)

        # Outer horizontal split: left add-panel + right grid
        outer = QtWidgets.QHBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        #  LEFT ADD PANEL 
        self.ref_add_panel = QtWidgets.QFrame()
        self.ref_add_panel.setObjectName("sidebar")
        self.ref_add_panel.setFixedWidth(280)
        add_layout = QtWidgets.QVBoxLayout(self.ref_add_panel)
        add_layout.setContentsMargins(20, 24, 20, 24)
        add_layout.setSpacing(14)

        self.ref_panel_title = QtWidgets.QLabel("Add Reference Image")
        self.ref_panel_title.setStyleSheet("font-size: 15px; font-weight: 800;")
        add_layout.addWidget(self.ref_panel_title)

        self.ref_sep = QtWidgets.QFrame()
        self.ref_sep.setFrameShape(QtWidgets.QFrame.HLine)
        self.ref_sep.setFixedHeight(1)
        add_layout.addWidget(self.ref_sep)

        # Image preview
        self.ref_add_preview = QtWidgets.QLabel("\nNo image selected")
        self.ref_add_preview.setAlignment(Qt.AlignCenter)
        self.ref_add_preview.setWordWrap(True)
        add_layout.addWidget(self.ref_add_preview)

        # Browse button
        browse_btn = QtWidgets.QPushButton("Browse Image File…")
        browse_btn.setObjectName("secondaryBtn")
        browse_btn.setCursor(QtCore.Qt.PointingHandCursor)
        browse_btn.clicked.connect(self._ref_browse_image)
        add_layout.addWidget(browse_btn)

        # Form fields
        self.ref_form_labels = []

        lbl_t = QtWidgets.QLabel("Title:")
        self.ref_form_labels.append(lbl_t)
        self.ref_title_edit = QtWidgets.QLineEdit()
        self.ref_title_edit.setPlaceholderText("Reference Title  (e.g. Correct Hand Position)")
        add_layout.addWidget(lbl_t)
        add_layout.addWidget(self.ref_title_edit)

        lbl_d = QtWidgets.QLabel("Description:")
        self.ref_form_labels.append(lbl_d)
        self.ref_desc_edit = QtWidgets.QLineEdit()
        self.ref_desc_edit.setPlaceholderText("Short description of this reference")
        add_layout.addWidget(lbl_d)
        add_layout.addWidget(self.ref_desc_edit)

        lbl_c = QtWidgets.QLabel("Category:")
        self.ref_form_labels.append(lbl_c)
        self.ref_cat_edit = QtWidgets.QComboBox()
        self.ref_cat_edit.addItems([
            "Hand Detection", "Tool Alignment", "Scale Verification",
            "Pen Detection", "Glove Check", "Object Detection", "Custom"
        ])
        add_layout.addWidget(lbl_c)
        add_layout.addWidget(self.ref_cat_edit)

        lbl_v = QtWidgets.QLabel("Version:")
        self.ref_form_labels.append(lbl_v)
        self.ref_ver_edit = QtWidgets.QLineEdit("V1.0")
        self.ref_ver_edit.setPlaceholderText("Version (e.g. V1.0)")
        add_layout.addWidget(lbl_v)
        add_layout.addWidget(self.ref_ver_edit)

        add_layout.addSpacing(8)

        save_ref_btn = QtWidgets.QPushButton("Save Reference Image")
        save_ref_btn.setObjectName("primaryBtn")
        save_ref_btn.setCursor(QtCore.Qt.PointingHandCursor)
        save_ref_btn.clicked.connect(self._ref_save_entry)
        add_layout.addWidget(save_ref_btn)

        # Storage info
        self.ref_store_info = QtWidgets.QLabel(
            f"Saved to:\nguide/reference_images/"
        )
        self.ref_store_info.setWordWrap(True)
        add_layout.addWidget(self.ref_store_info)

        add_layout.addStretch(1)
        outer.addWidget(self.ref_add_panel)

        #  RIGHT CONTENT AREA 
        right_widget = QtWidgets.QWidget()
        right_widget.setObjectName("page")
        right_layout = QtWidgets.QVBoxLayout(right_widget)
        right_layout.setContentsMargins(24, 24, 24, 24)
        right_layout.setSpacing(16)

        # Header row
        hdr = QtWidgets.QHBoxLayout()
        self.ref_header_title = QtWidgets.QLabel("SOP Reference Image Database")
        self.ref_header_title.setObjectName("pageTitle")
        hdr.addWidget(self.ref_header_title)
        hdr.addStretch(1)

        self.ref_count_lbl = QtWidgets.QLabel("0 references")
        self.ref_count_lbl.setStyleSheet(
            "background-color: #eff6ff; color: #2563eb; border: 1px solid #bfdbfe;"
            "border-radius: 12px; padding: 4px 12px; font-size: 12px; font-weight: 700;"
        )
        hdr.addWidget(self.ref_count_lbl)
        right_layout.addLayout(hdr)

        # Scrollable grid
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("background-color: transparent;")

        self.ref_grid_widget = QtWidgets.QWidget()
        self.ref_grid_widget.setStyleSheet("background-color: transparent;")
        self.ref_grid_layout = QtWidgets.QGridLayout(self.ref_grid_widget)
        self.ref_grid_layout.setSpacing(16)
        self.ref_grid_layout.setAlignment(Qt.AlignTop)

        scroll.setWidget(self.ref_grid_widget)
        right_layout.addWidget(scroll, 1)

        outer.addWidget(right_widget, 1)

        # Internal state
        self._ref_selected_path: str = ""
        self._ref_entries: list = []

        # Load saved metadata
        self._ref_load_all()

    #  REFERENCE IMAGE HELPERS 
    def _ref_load_all(self) -> None:
        """Load all saved reference entries from JSON and rebuild the grid."""
        import json, shutil

        os.makedirs(self._REF_IMG_DIR, exist_ok=True)
        self._ref_entries = []

        if os.path.exists(self._REF_METADATA):
            try:
                with open(self._REF_METADATA, "r") as f:
                    self._ref_entries = json.load(f)
            except Exception as e:
                print(f"[ref] failed to load metadata: {e}")

        self._ref_rebuild_grid()

    def _ref_rebuild_grid(self) -> None:
        """Clear and repopulate the reference image grid."""
        # Remove existing widgets
        while self.ref_grid_layout.count():
            item = self.ref_grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        count = len(self._ref_entries)
        self.ref_count_lbl.setText(f"{count} reference{'s'if count != 1 else ''}")

        if count == 0:
            empty_lbl = QtWidgets.QLabel(
                "No reference images yet.\n\nUse the  panel on the left\nto add your first reference image."
            )
            empty_lbl.setAlignment(Qt.AlignCenter)
            empty_lbl.setStyleSheet("color: #94a3b8; font-size: 14px; padding: 60px;")
            self.ref_grid_layout.addWidget(empty_lbl, 0, 0, 1, 2)
            return

        for idx, entry in enumerate(self._ref_entries):
            card = self._ref_make_card(entry, idx)
            row = idx // 2
            col = idx % 2
            self.ref_grid_layout.addWidget(card, row, col)

    def _ref_make_card(self, entry: dict, idx: int) -> QtWidgets.QFrame:
        """Build one reference image card widget."""
        is_dark = getattr(self, "is_dark_theme", False)
        card = QtWidgets.QFrame()
        card.setObjectName("card")
        card_layout = QtWidgets.QVBoxLayout(card)
        card_layout.setContentsMargins(14, 14, 14, 14)
        card_layout.setSpacing(8)

        # Title row
        title_row = QtWidgets.QHBoxLayout()
        c_title = QtWidgets.QLabel(entry.get("title", "Untitled"))
        t_col = "#60a5fa" if is_dark else "#1d4ed8"
        c_title.setStyleSheet(f"font-weight: 800; font-size: 14px; color: {t_col};")
        title_row.addWidget(c_title)
        title_row.addStretch(1)

        cat_badge = QtWidgets.QLabel(entry.get("category", ""))
        b_bg = "#172554" if is_dark else "#eff6ff"
        b_col = "#60a5fa" if is_dark else "#1d4ed8"
        b_bdr = "#1e3a5f" if is_dark else "#bfdbfe"
        cat_badge.setStyleSheet(
            f"background-color: {b_bg}; color: {b_col}; border: 1px solid {b_bdr};"
            "border-radius: 8px; padding: 2px 8px; font-size: 10px; font-weight: 700;"
        )
        title_row.addWidget(cat_badge)
        card_layout.addLayout(title_row)

        # Image display
        img_lbl = QtWidgets.QLabel()
        img_lbl.setAlignment(Qt.AlignCenter)
        img_lbl.setFixedHeight(150)
        box_bg = "#0f172a" if is_dark else "#f8fafc"
        box_bdr = "#334155" if is_dark else "#e2e8f0"
        img_lbl.setStyleSheet(
            f"border: 1px solid {box_bdr}; border-radius: 8px;"
            f"background-color: {box_bg};"
        )

        img_path = entry.get("image_path", "")
        if img_path and os.path.exists(img_path):
            pix = QtGui.QPixmap(img_path)
            if not pix.isNull():
                scaled = pix.scaled(
                    img_lbl.size().width() if img_lbl.size().width() > 0 else 400,
                    148, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                img_lbl.setPixmap(scaled)
            else:
                img_lbl.setText("Cannot load image")
                err_bg = "#450a0a" if is_dark else "#fef2f2"
                err_col = "#f87171" if is_dark else "#dc2626"
                img_lbl.setStyleSheet(
                    f"border: 1px dashed #fca5a5; border-radius: 8px;"
                    f"background-color: {err_bg}; color: {err_col}; font-size: 11px;"
                )
        else:
            img_lbl.setText("No image assigned\nClick 'Set Image' to attach one")
            dash_bdr = "#334155" if is_dark else "#cbd5e1"
            dash_txt = "#cbd5e1" if is_dark else "#475569"
            img_lbl.setStyleSheet(
                f"border: 2px dashed {dash_bdr}; border-radius: 8px;"
                f"background-color: {box_bg}; color: {dash_txt}; font-size: 11px;"
            )
            img_lbl.setWordWrap(True)

        card_layout.addWidget(img_lbl)

        # Description & version
        c_desc = QtWidgets.QLabel(entry.get("description", ""))
        desc_col = "#cbd5e1" if is_dark else "#334155"
        c_desc.setStyleSheet(f"color: {desc_col}; font-size: 11px;")
        c_desc.setWordWrap(True)
        card_layout.addWidget(c_desc)

        ver_row = QtWidgets.QHBoxLayout()
        c_ver = QtWidgets.QLabel(f"Version: {entry.get('version', 'V1.0')}")
        ver_col = "#94a3b8" if is_dark else "#64748b"
        c_ver.setStyleSheet(f"color: {ver_col}; font-size: 10px; font-weight: 700;")
        ver_row.addWidget(c_ver)
        ver_row.addStretch(1)
        card_layout.addLayout(ver_row)

        # Action buttons
        actions_h = QtWidgets.QHBoxLayout()
        actions_h.setSpacing(8)

        set_img_btn = QtWidgets.QPushButton("Set Image")
        set_img_btn.setObjectName("secondaryBtn")
        set_img_btn.setCursor(QtCore.Qt.PointingHandCursor)
        set_img_btn.clicked.connect(lambda checked, i=idx: self._ref_set_image_for(i))

        zoom_btn = QtWidgets.QPushButton("Zoom")
        zoom_btn.setObjectName("secondaryBtn")
        zoom_btn.setCursor(QtCore.Qt.PointingHandCursor)
        zoom_btn.clicked.connect(lambda checked, e=entry: self._zoom_reference_img(e))

        del_btn = QtWidgets.QPushButton("Delete")
        del_btn.setStyleSheet("""
            QPushButton {
                background-color: #ef4444;
                color: #ffffff;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-weight: 700;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #dc2626;
            }
            QPushButton:pressed {
                background-color: #b91c1c;
            }
        """)
        del_btn.setCursor(QtCore.Qt.PointingHandCursor)
        del_btn.setToolTip("Delete this reference entry")
        del_btn.clicked.connect(lambda checked, i=idx: self._ref_delete_entry(i))

        actions_h.addWidget(set_img_btn)
        actions_h.addWidget(zoom_btn)
        actions_h.addWidget(del_btn)
        card_layout.addLayout(actions_h)


        return card

    def _ref_browse_image(self) -> None:
        """Open file dialog to pick an image for the add-panel preview."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Reference Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp *.tiff *.tif)"
        )
        if path:
            self._ref_selected_path = path
            pix = QtGui.QPixmap(path)
            if not pix.isNull():
                scaled = pix.scaled(240, 130, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.ref_add_preview.setPixmap(scaled)
                self.ref_add_preview.setStyleSheet(
                    "border: 2px solid #2563eb; border-radius: 10px;"
                    "background-color: #eff6ff; min-height: 140px;"
                )
            else:
                self.ref_add_preview.setText("Cannot preview\nthis file")

    def _ref_save_entry(self) -> None:
        """Save the new reference entry (copy image + update JSON)."""
        import json, shutil

        title = self.ref_title_edit.text().strip()
        if not title:
            QtWidgets.QMessageBox.warning(self, "Missing Title", "Please enter a title for this reference image.")
            return

        os.makedirs(self._REF_IMG_DIR, exist_ok=True)

        # Copy image to reference_images folder
        saved_path = ""
        if self._ref_selected_path and os.path.exists(self._ref_selected_path):
            ext = os.path.splitext(self._ref_selected_path)[1]
            safe_name = title.lower().replace("", "_").replace("/", "_") + ext
            dest = os.path.join(self._REF_IMG_DIR, safe_name)
            try:
                shutil.copy2(self._ref_selected_path, dest)
                saved_path = dest
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Copy Failed", f"Could not copy image:\n{e}")
                return

        entry = {
            "title": title,
            "description": self.ref_desc_edit.text().strip(),
            "category": self.ref_cat_edit.currentText(),
            "version": self.ref_ver_edit.text().strip() or "V1.0",
            "image_path": saved_path,
        }

        self._ref_entries.append(entry)
        self._ref_save_metadata()
        self._ref_rebuild_grid()

        # Reset form
        self.ref_title_edit.clear()
        self.ref_desc_edit.clear()
        self.ref_ver_edit.setText("V1.0")
        self.ref_add_preview.setPixmap(QtGui.QPixmap())
        self.ref_add_preview.setText("\nNo image selected")
        self.ref_add_preview.setStyleSheet(
            "border: 2px dashed #cbd5e1; border-radius: 10px;"
            "background-color: #f8fafc; color: #94a3b8; font-size: 12px; min-height: 140px;"
        )
        self._ref_selected_path = ""

        QtWidgets.QMessageBox.information(
            self, "Saved",
            f"Reference image '{title}'saved successfully!\n Path: {saved_path or '(no image file attached)'}"
        )

    def _ref_set_image_for(self, idx: int) -> None:
        """Open file dialog to attach / replace an image for an existing entry."""
        import json, shutil

        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Image for Reference Entry", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp *.tiff *.tif)"
        )
        if not path:
            return

        entry = self._ref_entries[idx]
        ext = os.path.splitext(path)[1]
        safe_name = entry.get("title", f"ref_{idx}").lower().replace("", "_") + ext
        dest = os.path.join(self._REF_IMG_DIR, safe_name)
        try:
            shutil.copy2(path, dest)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Copy Failed", f"Could not copy image:\n{e}")
            return

        self._ref_entries[idx]["image_path"] = dest
        self._ref_save_metadata()
        self._ref_rebuild_grid()

    def _ref_delete_entry(self, idx: int) -> None:
        """Remove an entry from the reference database."""
        import json

        entry = self._ref_entries[idx]
        reply = QtWidgets.QMessageBox.question(
            self, "Delete Reference",
            f"Delete reference entry:\n'{entry.get('title','')}'?\n\n(The image file on disk will NOT be deleted.)",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        if reply == QtWidgets.QMessageBox.Yes:
            self._ref_entries.pop(idx)
            self._ref_save_metadata()
            self._ref_rebuild_grid()

    def _ref_save_metadata(self) -> None:
        """Persist the reference entries list to JSON."""
        import json

        os.makedirs(self._REF_IMG_DIR, exist_ok=True)
        try:
            with open(self._REF_METADATA, "w") as f:
                json.dump(self._ref_entries, f, indent=2)
        except Exception as e:
            print(f"[ref] failed to save metadata: {e}")

    def _zoom_reference_img(self, entry) -> None:
        """Show full-size image in a modal dialog."""
        if isinstance(entry, str):
            # Legacy call with just a title string
            QtWidgets.QMessageBox.information(self, "Reference Zoom", f"Reference: {entry}")
            return

        img_path = entry.get("image_path", "")
        is_dark = getattr(self, "is_dark_theme", False)
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(f"{entry.get('title', 'Reference Image')}")
        dialog.setMinimumSize(800, 600)
        
        dlg_bg = "#0f172a" if is_dark else "#f8fafc"
        txt_head = "#f8fafc" if is_dark else "#0f172a"
        box_bg = "#1e293b" if is_dark else "#ffffff"
        box_bdr = "#334155" if is_dark else "#e2e8f0"
        txt_sub = "#cbd5e1" if is_dark else "#475569"

        dialog.setStyleSheet(f"background-color: {dlg_bg};")

        dlg_layout = QtWidgets.QVBoxLayout(dialog)
        dlg_layout.setContentsMargins(20, 20, 20, 20)
        dlg_layout.setSpacing(12)

        title_lbl = QtWidgets.QLabel(entry.get("title", ""))
        title_lbl.setStyleSheet(f"font-size: 18px; font-weight: 800; color: {txt_head};")
        dlg_layout.addWidget(title_lbl)

        img_lbl = QtWidgets.QLabel()
        img_lbl.setAlignment(Qt.AlignCenter)
        img_lbl.setStyleSheet(
            f"border: 1px solid {box_bdr}; border-radius: 12px; background-color: {box_bg}; color: {txt_sub};"
        )

        if img_path and os.path.exists(img_path):
            pix = QtGui.QPixmap(img_path)
            if not pix.isNull():
                scaled = pix.scaled(760, 480, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                img_lbl.setPixmap(scaled)
            else:
                img_lbl.setText("Cannot load image file.")
        else:
            img_lbl.setText(
                "No image assigned to this reference entry.\n\n"
                "Use 'Set Image' on the card to attach a photo."
            )
            dash_bdr = "#334155" if is_dark else "#cbd5e1"
            dash_bg = "#0f172a" if is_dark else "#f8fafc"
            dash_txt = "#cbd5e1" if is_dark else "#64748b"
            img_lbl.setStyleSheet(
                f"border: 2px dashed {dash_bdr}; border-radius: 12px;"
                f"background-color: {dash_bg}; color: {dash_txt}; font-size: 14px; padding: 60px;"
            )

        dlg_layout.addWidget(img_lbl, 1)

        desc_lbl = QtWidgets.QLabel(
            f"{entry.get('description','')}   |   "
            f"Category: {entry.get('category','')}   |   "
            f"Version: {entry.get('version','')}"
        )
        desc_lbl.setStyleSheet(f"color: {txt_sub}; font-size: 12px; font-weight: 500;")
        dlg_layout.addWidget(desc_lbl)

        if img_path:
            path_lbl = QtWidgets.QLabel(f"{img_path}")
            path_lbl.setStyleSheet("color: #94a3b8; font-size: 10px;")
            path_lbl.setWordWrap(True)
            dlg_layout.addWidget(path_lbl)

        close_btn = QtWidgets.QPushButton("Close")
        close_btn.setObjectName("primaryBtn")
        close_btn.clicked.connect(dialog.accept)
        dlg_layout.addWidget(close_btn, 0, Qt.AlignRight)

        dialog.exec()

    def _compare_reference_img(self, title: str) -> None:
        QtWidgets.QMessageBox.information(
            self, "Reference Comparison",
            f"Comparing operator real-time camera frame against reference:\n{title}"
        )


    # -- PAGE 8: SETTINGS CONFIG --------------------------------------------
    # Settings JSON file path
    _SETTINGS_PATH = os.path.join(_GUI_DIR, "app_settings.json")

    # Map combo index → (width, height)
    _RES_MAP = {
        0: (1280, 720),   # HD
        1: (1920, 1080),  # Full HD
        2: (640, 480),    # VGA
        3: (3840, 2160),  # 4K
    }
    # Map combo index → rotation degrees
    _ROT_MAP = {0: 0, 1: 90, 2: 180, 3: 270}

    def _create_settings_page(self) -> None:
        page = QtWidgets.QWidget()
        page.setObjectName("page")
        self.stacked_widget.addWidget(page)

        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        # Header
        hdr = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("System Configuration & Camera Settings")
        title.setObjectName("pageTitle")
        hdr.addWidget(title)
        hdr.addStretch(1)

        self.theme_toggle = ThemeToggleSwitch(is_dark=self.is_dark_theme)
        self.theme_toggle.toggled.connect(self._on_theme_toggled)
        hdr.addWidget(self.theme_toggle)

        hdr.addSpacing(16)

        self.settings_status_lbl = QtWidgets.QLabel("Settings not saved")
        self.settings_status_lbl.setStyleSheet(
            "background-color: #fff7ed; color: #ea580c; border: 1px solid #fed7aa;"
            "border-radius: 10px; padding: 4px 12px; font-size: 11px; font-weight: 700;"
        )
        hdr.addWidget(self.settings_status_lbl)
        layout.addLayout(hdr)

        form_frame = QtWidgets.QFrame()
        form_frame.setObjectName("card")
        form_layout = QtWidgets.QFormLayout(form_frame)
        form_layout.setContentsMargins(20, 20, 20, 20)
        form_layout.setSpacing(14)
        form_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)

        #  Camera Config 
        cam_hdr = QtWidgets.QLabel("Camera Configuration")
        cam_hdr.setStyleSheet(
            "font-weight: 700; color: #2563eb; font-size: 13px;"
            "background-color: #eff6ff; border-radius: 6px; padding: 6px 10px; margin-top: 6px;"
        )
        form_layout.addRow(cam_hdr)

        self.set_cam_res = QtWidgets.QComboBox()
        self.set_cam_res.addItems([
            "HD  (1280 × 720)",
            "Full HD  (1920 × 1080)",
            "VGA  (640 × 480)",
            "4K  (3840 × 2160)",
        ])
        self.set_cam_res.setFixedHeight(36)
        form_layout.addRow(QtWidgets.QLabel("Camera Resolution:  "), self.set_cam_res)

        self.set_cam_rotation = QtWidgets.QComboBox()
        self.set_cam_rotation.addItems([
            "0°  — No Rotation",
            "90°  — Rotate Clockwise",
            "180°  — Flip Upside Down",
            "270°  — Rotate Counter-Clockwise",
        ])
        self.set_cam_rotation.setFixedHeight(36)
        form_layout.addRow(QtWidgets.QLabel("Feed Rotation:  "), self.set_cam_rotation)

        #  AI Engine 
        ai_hdr = QtWidgets.QLabel("AI Engine Parameters")
        ai_hdr.setStyleSheet(
            "font-weight: 700; color: #2563eb; font-size: 13px;"
            "background-color: #eff6ff; border-radius: 6px; padding: 6px 10px; margin-top: 6px;"
        )
        form_layout.addRow(ai_hdr)

        self.set_ai_conf = QtWidgets.QDoubleSpinBox()
        self.set_ai_conf.setRange(0.10, 1.00)
        self.set_ai_conf.setSingleStep(0.05)
        self.set_ai_conf.setDecimals(2)
        self.set_ai_conf.setValue(0.50)
        self.set_ai_conf.setSuffix("(0.10 – 1.00)")
        self.set_ai_conf.setFixedHeight(36)
        form_layout.addRow(QtWidgets.QLabel("Confidence Threshold:  "), self.set_ai_conf)

        self.set_ai_rot_sens = QtWidgets.QDoubleSpinBox()
        self.set_ai_rot_sens.setRange(1.0, 10.0)
        self.set_ai_rot_sens.setSingleStep(0.5)
        self.set_ai_rot_sens.setDecimals(1)
        self.set_ai_rot_sens.setValue(5.0)
        self.set_ai_rot_sens.setSuffix("(1.0 – 10.0)")
        self.set_ai_rot_sens.setFixedHeight(36)
        form_layout.addRow(QtWidgets.QLabel("Rotation Sensitivity:  "), self.set_ai_rot_sens)

        #  Database & Backup 
        db_hdr = QtWidgets.QLabel("Database & Backup Settings")
        db_hdr.setStyleSheet(
            "font-weight: 700; color: #2563eb; font-size: 13px;"
            "background-color: #eff6ff; border-radius: 6px; padding: 6px 10px; margin-top: 6px;"
        )
        form_layout.addRow(db_hdr)

        self.set_db_type = QtWidgets.QComboBox()
        self.set_db_type.addItems([
            "SQLite  (Local File)",
            "MySQL  (Remote Server)",
            "PostgreSQL  (Cloud Database)",
        ])
        self.set_db_type.setFixedHeight(36)
        form_layout.addRow(QtWidgets.QLabel("Database Provider:  "), self.set_db_type)

        self.backup_sched = QtWidgets.QComboBox()
        self.backup_sched.addItems([
            "Daily at 00:00",
            "Weekly on Sundays",
            "Hourly Logs Backup",
        ])
        self.backup_sched.setFixedHeight(36)
        form_layout.addRow(QtWidgets.QLabel("Backup Schedule:  "), self.backup_sched)

        db_actions = QtWidgets.QHBoxLayout()
        db_actions.setSpacing(10)
        self.btn_backup_now = QtWidgets.QPushButton("Backup Database Now")
        self.btn_backup_now.setObjectName("secondaryBtn")
        self.btn_backup_now.clicked.connect(self._backup_db_routine)

        self.btn_restore_db = QtWidgets.QPushButton("Restore from Backup")
        self.btn_restore_db.setObjectName("secondaryBtn")
        self.btn_restore_db.clicked.connect(self._restore_db_routine)

        db_actions.addWidget(self.btn_backup_now)
        db_actions.addWidget(self.btn_restore_db)
        db_actions.addStretch(1)
        form_layout.addRow(QtWidgets.QLabel("Database Tools:  "), db_actions)

        layout.addWidget(form_frame)

        # Active settings summary card
        self.active_settings_card = QtWidgets.QFrame()
        self.active_settings_card.setObjectName("card")
        self.active_settings_card.setStyleSheet(
            "QFrame#card { background-color: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 10px; }"
        )
        acs_layout = QtWidgets.QHBoxLayout(self.active_settings_card)
        acs_layout.setContentsMargins(16, 12, 16, 12)
        acs_layout.setSpacing(24)

        self.acs_res_lbl  = QtWidgets.QLabel("Resolution: HD (1280×720)")
        self.acs_rot_lbl  = QtWidgets.QLabel("Rotation: 0°")
        self.acs_conf_lbl = QtWidgets.QLabel("Confidence: 0.50")
        self.acs_sens_lbl = QtWidgets.QLabel("Sensitivity: 5.0")

        for lbl in (self.acs_res_lbl, self.acs_rot_lbl, self.acs_conf_lbl, self.acs_sens_lbl):
            lbl.setStyleSheet("color: #15803d; font-size: 12px; font-weight: 700; background: transparent;")
            acs_layout.addWidget(lbl)
        acs_layout.addStretch(1)

        layout.addWidget(self.active_settings_card)

        # Save button
        save_layout = QtWidgets.QHBoxLayout()
        btn_save_config = QtWidgets.QPushButton("Save & Apply All Settings")
        btn_save_config.setObjectName("primaryBtn")
        btn_save_config.setFixedHeight(42)
        btn_save_config.clicked.connect(self._save_global_settings)
        save_layout.addWidget(btn_save_config)
        save_layout.addStretch(1)
        layout.addLayout(save_layout)

        layout.addStretch(1)

        # Load persisted settings on page creation
        self._load_settings_from_disk()

    #  SETTINGS HELPERS 
    def _load_settings_from_disk(self) -> None:
        """Read app_settings.json and populate the form fields."""
        import json
        if not os.path.exists(self._SETTINGS_PATH):
            return
        try:
            with open(self._SETTINGS_PATH, "r") as f:
                s = json.load(f)
            self.set_cam_res.setCurrentIndex(s.get("res_index", 0))
            self.set_cam_rotation.setCurrentIndex(s.get("rot_index", 0))
            self.set_ai_conf.setValue(s.get("confidence", 0.50))
            self.set_ai_rot_sens.setValue(s.get("sensitivity", 5.0))
            self.set_db_type.setCurrentIndex(s.get("db_index", 0))
            self.backup_sched.setCurrentIndex(s.get("backup_index", 0))
            if hasattr(self, "theme_toggle"):
                self.theme_toggle.set_checked(s.get("theme", "light") == "dark")
            self._apply_settings_to_summary()
            self.settings_status_lbl.setText("Settings loaded from disk")
            self.settings_status_lbl.setStyleSheet(
                "background-color: #f0fdf4; color: #16a34a; border: 1px solid #bbf7d0;"
                "border-radius: 10px; padding: 4px 12px; font-size: 11px; font-weight: 700;"
            )
        except Exception as e:
            print(f"[settings] load error: {e}")

    def _apply_settings_to_summary(self) -> None:
        """Update the green active-settings summary row."""
        res_idx = self.set_cam_res.currentIndex()
        rot_idx = self.set_cam_rotation.currentIndex()
        w, h = self._RES_MAP.get(res_idx, (1280, 720))
        rot   = self._ROT_MAP.get(rot_idx, 0)
        conf  = self.set_ai_conf.value()
        sens  = self.set_ai_rot_sens.value()
        self.acs_res_lbl.setText(f"Resolution: {w}×{h}")
        self.acs_rot_lbl.setText(f"Rotation: {rot}°")
        self.acs_conf_lbl.setText(f"Confidence: {conf:.2f}")
        self.acs_sens_lbl.setText(f"Sensitivity: {sens:.1f}")

    def _save_global_settings(self) -> None:
        self._activity_logger.log("SAVE", "Settings", "Save Global Settings", "Settings save initiated")
        """Save all settings to disk, apply them to the worker, and restart monitoring."""
        import json

        res_idx = self.set_cam_res.currentIndex()
        rot_idx = self.set_cam_rotation.currentIndex()
        w, h  = self._RES_MAP.get(res_idx, (1280, 720))
        rot   = self._ROT_MAP.get(rot_idx, 0)
        conf  = self.set_ai_conf.value()
        sens  = self.set_ai_rot_sens.value()

        # -- Persist to JSON -----------------------------------------------
        settings = {}
        if os.path.exists(self._SETTINGS_PATH):
            try:
                with open(self._SETTINGS_PATH, "r") as f:
                    settings = json.load(f)
            except Exception:
                settings = {}
        settings.update({
            "res_index":    res_idx,
            "rot_index":    rot_idx,
            "cam_width":    w,
            "cam_height":   h,
            "rotation":     rot,
            "confidence":   conf,
            "sensitivity":  sens,
            "db_index":     self.set_db_type.currentIndex(),
            "backup_index": self.backup_sched.currentIndex(),
            "theme":        "dark" if self.is_dark_theme else "light",
            "compliance":   self._compliance_settings,
        })
        try:
            with open(self._SETTINGS_PATH, "w") as f:
                json.dump(settings, f, indent=2)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save Error", f"Could not write settings:\n{e}")
            return

        # -- Store on self for _start_worker to use -------------------------
        self._active_cam_width   = w
        self._active_cam_height  = h
        self._active_rotation    = rot
        self._active_confidence  = conf
        self._active_sensitivity = sens

        # -- Apply to live worker without full restart if possible ----------
        if self.worker is not None:
            self.worker.cam_width          = w
            self.worker.cam_height         = h
            self.worker.rotation           = rot
            self.worker.confidence_threshold = conf
            self.worker.rot_sensitivity    = sens
            if self.worker._monitor is not None:
                self.worker._monitor.turn_target = self.active_turn_target

        # -- Update summary row --------------------------------------------
        self._apply_settings_to_summary()

        # -- Update status badge -------------------------------------------
        self.settings_status_lbl.setText("Settings saved & applied")
        self.settings_status_lbl.setStyleSheet(
            "background-color: #f0fdf4; color: #16a34a; border: 1px solid #bbf7d0;"
            "border-radius: 10px; padding: 4px 12px; font-size: 11px; font-weight: 700;"
        )

        # -- Restart worker so new res/rotation/confidence take effect -----
        reply = QtWidgets.QMessageBox.question(
            self,
            "Restart Camera Monitor",
            f"Settings saved:\n"
            f"• Resolution: {w}×{h}\n"
            f"• Rotation: {rot}°\n"
            f"• Confidence: {conf:.2f}\n"
            f"• Sensitivity: {sens:.1f}\n\n"
            "Restart the camera monitor now to apply all changes?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.Yes,
        )
        if reply == QtWidgets.QMessageBox.Yes:
            self._start_worker()
            self.statusBar().showMessage(
                f"Settings applied — {w}×{h} @ {rot}° | Conf: {conf:.2f} | Sens: {sens:.1f}",
                8000
            )

    def _backup_db_routine(self) -> None:
        """Copy the CSV log file to a timestamped backup."""
        self._activity_logger.log("CLICK", "Settings", "Backup Database", "Backup initiated")
        import shutil
        if not os.path.exists(LOG_CSV_PATH):
            QtWidgets.QMessageBox.warning(
                self, "No Log Data",
                "No detection log file found yet.\nRun some inspections first."
            )
            return
        backup_dir = os.path.join(_PROJECT_ROOT, "backups")
        os.makedirs(backup_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        dest = os.path.join(backup_dir, f"logs_backup_{stamp}.csv")
        try:
            shutil.copy2(LOG_CSV_PATH, dest)
            QtWidgets.QMessageBox.information(
                self, "Backup Successful",
                f"Database backup completed!\n\n Saved to:\n{dest}"
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Backup Failed", f"Could not create backup:\n{e}")

    def _restore_db_routine(self) -> None:
        """Restore a backup CSV by picking from backups/ folder."""
        self._activity_logger.log("CLICK", "Settings", "Restore Database", "Restore initiated")
        import shutil
        backup_dir = os.path.join(_PROJECT_ROOT, "backups")
        if not os.path.isdir(backup_dir):
            QtWidgets.QMessageBox.warning(
                self, "No Backups Found",
                f"No backups directory found at:\n{backup_dir}\n\nCreate a backup first."
            )
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Backup File to Restore",
            backup_dir, "CSV Backup Files (*.csv)"
        )
        if not path:
            return
        try:
            shutil.copy2(path, LOG_CSV_PATH)
            QtWidgets.QMessageBox.information(
                self, "Restore Successful",
                f"Database restored from:\n{path}\n\nLog view will refresh now."
            )
            self._refresh_logs_table()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Restore Failed", f"Could not restore backup:\n{e}")


    # -- WORKER INTERFACE ---------------------------------------------------
    def _refresh_camera_list(self, preferred_source: Optional[int] = None) -> None:
        """Rescan for connected cameras (built-in + hot-plugged USB/webcams)
        and repopulate the camera combo box — without blocking the GUI.

        The actual probing runs on a background CameraScanWorker; this method
        only kicks it off and returns immediately. Results are applied in
        _on_camera_scan_done() once the scan finishes.
        """
        from camera_worker import CameraScanWorker

        if preferred_source is None:
            preferred_source = self._current_source()

        # If monitoring is currently running, its camera device is held open
        # exclusively by the worker thread. Probing that same index here would
        # fail (device busy) and wrongly make it look "disconnected" in the
        # dropdown even though it's actively streaming fine. So we skip
        # re-probing it and just keep it in the list as-is.
        active_source = None
        if self.worker is not None and self.worker.isRunning():
            active_source = self.source

        # Avoid piling up scans if the user mashes "Refresh Cameras" —
        # let an in-flight scan finish rather than starting another.
        if getattr(self, "_cam_scan_worker", None) is not None and self._cam_scan_worker.isRunning():
            return

        self.statusBar().showMessage("Scanning for cameras...", 4000)
        self._cam_scan_preferred = preferred_source
        self._cam_scan_active_source = active_source

        self._cam_scan_worker = CameraScanWorker(skip_index=active_source, parent=self)
        self._cam_scan_worker.scan_done.connect(self._on_camera_scan_done)
        self._cam_scan_worker.start()

    def _on_camera_scan_done(self, available: list) -> None:
        preferred_source = getattr(self, "_cam_scan_preferred", self.source)
        active_source = getattr(self, "_cam_scan_active_source", None)

        if active_source is not None and active_source not in available:
            available = sorted(available + [active_source])

        self.source_combo.blockSignals(True)
        self.source_combo.clear()
        if not available:
            self.source_combo.addItem("No cameras detected", -1)
        else:
            for idx in available:
                label = f"Camera {idx} (Built-in / Default)" if idx == 0 else f"Camera {idx} (External USB)"
                if idx == active_source:
                    label += " — In Use"
                self.source_combo.addItem(label, idx)
            # Restore the previously selected camera if it's still present.
            match_idx = self.source_combo.findData(preferred_source)
            self.source_combo.setCurrentIndex(match_idx if match_idx >= 0 else 0)
        self.source_combo.blockSignals(False)

    def _current_source(self) -> int:
        data = self.source_combo.currentData()
        return int(data) if data is not None and data != -1 else self.source

    def _select_camera(self) -> None:
        """Open and start monitoring with whatever camera is currently
        chosen in the dropdown — switches live if monitoring is already
        running on a different camera."""
        source = self._current_source()
        if source == -1:
            QtWidgets.QMessageBox.warning(
                self, "No Camera Available",
                "No cameras were detected. Connect a built-in or USB camera, "
                "then click 'Refresh Cameras' and try again."
            )
            return
        self._start_worker()

    def _test_selected_camera(self) -> None:
        """Grab one frame from the currently selected camera index and show
        it in a popup — a fast way to confirm which physical camera an index
        actually maps to, independent of the full detection pipeline. If two
        different indices show the identical picture, that's the OS/driver
        exposing the same physical device under multiple indices, not a bug
        in this app's selection logic."""
        from camera_worker import grab_test_frame
        import cv2

        source = self._current_source()
        if source == -1:
            QtWidgets.QMessageBox.warning(self, "No Camera", "No camera selected.")
            return

        if self.worker is not None and self.worker.isRunning() and source == self.source:
            QtWidgets.QMessageBox.information(
                self, "Camera Preview",
                f"Camera {source} is already live — check the main view above."
            )
            return

        self.statusBar().showMessage(f"Testing camera {source}...", 2000)
        QtWidgets.QApplication.processEvents()
        frame = grab_test_frame(source)
        if frame is None:
            QtWidgets.QMessageBox.warning(
                self, "Camera Test Failed",
                f"Could not grab a frame from camera {source}.\n\n"
                "It may be in use by another application, or disconnected."
            )
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, w * 3, QtGui.QImage.Format_RGB888).copy()
        pix = QtGui.QPixmap.fromImage(qimg).scaledToWidth(480, QtCore.Qt.SmoothTransformation)

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(f"Camera {source} Preview")
        lay = QtWidgets.QVBoxLayout(dlg)
        lbl = QtWidgets.QLabel()
        lbl.setPixmap(pix)
        lay.addWidget(lbl)
        note = QtWidgets.QLabel(
            f"This is a single frame grabbed directly from camera index {source}.\n"
            "If this looks like the wrong physical camera, the OS/driver is "
            "mapping this index to that device — try a different index or "
            "check your OS camera device list."
        )
        note.setWordWrap(True)
        lay.addWidget(note)
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        lay.addWidget(close_btn)
        dlg.exec()

    def _start_worker(self) -> None:
        was_running = self.worker is not None and self.worker.isRunning()
        self._stop_worker()
        if was_running:
            # Give the OS/driver a moment to fully release the previous
            # camera's exclusive handle before opening a (possibly
            # different) device. Reopening immediately is a common cause of
            # the new selection silently falling back to whatever camera
            # the OS still has cached as active.
            QtWidgets.QApplication.processEvents()
            time.sleep(0.4)

        if hasattr(self, '_activity_logger'):
            self._activity_logger.log("CLICK", "Live Monitor", "Start Monitor", "Camera worker started")
        worker_id = self.worker_id_input.text().strip() or "EMP001"
        source = self._current_source()

        if source == -1:
            QtWidgets.QMessageBox.warning(
                self, "No Camera Available",
                "No cameras were detected. Connect a built-in or USB camera, "
                "then click 'Refresh Cameras' and try again."
            )
            self.camera_view.show_placeholder("No camera available.")
            return

        self.source = source
        op_name = ""
        if hasattr(self, "op_name_input"):
            op_name = self.op_name_input.text().strip()
        # Resolve active assembly params for this session
        _asm = self._assembly_manager.get_active_assembly()
        _det_mode = _asm.get("detection_mode", "screw_monitor")
        _turn_target = float(_asm.get("target_params", {}).get("turn_target", 2.5))
        _align_px = int(_asm.get("target_params", {}).get("align_threshold_px", 50))

        self.worker = CameraWorker(
            self.config_path,
            source=source,
            worker_id=worker_id,
            compliance_settings=self._compliance_settings,
            compliance_logger=self._compliance_logger,
            operator_name=op_name,
            compliance_mock=self._compliance_mock,
            assembly_id=self._assembly_manager.get_active_assembly_id(),
            detection_mode=_det_mode,
            turn_target=_turn_target,
            align_threshold_px=_align_px,
        )
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.state_ready.connect(self._on_state)
        self.worker.gesture_ready.connect(self._on_gesture)
        self.worker.compliance_ready.connect(self._on_compliance)
        self.worker.error.connect(self._on_error)
        self.worker.camera_unavailable.connect(self._on_camera_unavailable)

        self.worker.show_landmarks = self.chk_landmarks.isChecked()
        self.worker.gesture_enabled = self.chk_gesture.isChecked()
        self.worker.monitor_enabled = self.chk_monitor.isChecked()
        self.worker.compliance_enabled = (
            self.chk_compliance.isChecked()
            if hasattr(self, "chk_compliance")
            else bool(self._compliance_settings.get("enabled", True))
        )

        # Push saved settings from Settings page into the new worker
        self.worker.cam_width          = getattr(self, "_active_cam_width",   640)
        self.worker.cam_height         = getattr(self, "_active_cam_height",  480)
        self.worker.rotation           = getattr(self, "_active_rotation",    0)
        self.worker.confidence_threshold = getattr(self, "_active_confidence", 0.50)
        self.worker.rot_sensitivity    = getattr(self, "_active_sensitivity",  5.0)

        self.worker.start()
        self.statusBar().showMessage(
            f"Camera {source} started — operator: {worker_id} | "
            f"{self.worker.cam_width}×{self.worker.cam_height} "
            f"rot:{self.worker.rotation}° conf:{self.worker.confidence_threshold:.2f}"
        )


    def _stop_worker(self) -> None:
        if hasattr(self, '_activity_logger'):
            self._activity_logger.log("CLICK", "Live Monitor", "Stop Monitor", "Camera worker stopped")
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
        self.camera_view.show_placeholder("Monitoring offline. Click 'Start Monitor'to activate.")

    def _reset_monitor(self) -> None:
        self._activity_logger.log("CLICK", "Live Monitor", "Reset Monitor", "Monitor stats and rotation reset")
        if self.worker is not None:
            self.worker.reset_monitor()
            self.statusBar().showMessage("Monitor stats and rotation count reset.", 3000)

    def _on_flags(self) -> None:
        if self.worker is not None:
            self.worker.show_landmarks = self.chk_landmarks.isChecked()
            self.worker.gesture_enabled = self.chk_gesture.isChecked()
            self.worker.monitor_enabled = self.chk_monitor.isChecked()
            if hasattr(self, "chk_compliance"):
                self.worker.compliance_enabled = self.chk_compliance.isChecked()
                self._compliance_settings["enabled"] = self.chk_compliance.isChecked()
                if self.worker._compliance is not None:
                    self.worker._compliance.enabled = self.chk_compliance.isChecked()

    # -- WORKER SLOTS -------------------------------------------------------
    def _on_frame(self, frame) -> None:
        self.camera_view.update_frame(frame)
        # Tell the worker this frame is done with so it emits the next one
        # instead of piling more frames into the queue while we're behind.
        if self.worker is not None:
            self.worker.mark_frame_consumed()

    def _on_compliance(self, state: dict) -> None:
        self.last_compliance_state = state
        if hasattr(self, "compliance_panel"):
            self.compliance_panel.update_state(state)
        if hasattr(self, "compliance_page"):
            self.compliance_page.update_live_state(state)
        if state.get("new_events"):
            try:
                summary = self._compliance_logger.daily_summary(days=1)
                if hasattr(self, "dash_kpi_warnings"):
                    self.dash_kpi_warnings.set_value(str(summary.get("warnings", 0)))
                    self.dash_kpi_compliance.set_value(f"{summary.get('compliance_pct', 100)}%")
            except Exception:
                pass

    def _on_state(self, state: dict) -> None:
        # Merge compliance info if available
        if hasattr(self, "last_compliance_state") and self.last_compliance_state:
            state["compliance_details"] = self.last_compliance_state
            
        # Update live status and SOP sequences
        self.status_panel.update_state(state)
        is_screw_active = self.sop_panel.update_state(state)
        if hasattr(self, "worker") and self.worker is not None and getattr(self.worker, "_monitor", None) is not None:
            self.worker._monitor.show_screw_target = is_screw_active
        
        # Auto-update Parts Checklist based on live status
        glove = state.get("glove", "Normal")
        turns = state.get("rotation_count", 0.0)
        target = state.get("turn_target", 2.5)
        
        # update ESD glove status in part 1
        if glove == "Glove":
            self.checklist_table.item(1, 3).setText("Correct")
            self.checklist_table.item(1, 3).setForeground(QtGui.QColor("#16a34a"))
            self.checklist_table.item(1, 4).setText(_now_short())
        else:
            self.checklist_table.item(1, 3).setText("Not Correct")
            self.checklist_table.item(1, 3).setForeground(QtGui.QColor("#dc2626"))
            
        # update Screw Fastener status in part 0
        if turns >= target:
            self.checklist_table.item(0, 3).setText("Correct")
            self.checklist_table.item(0, 3).setForeground(QtGui.QColor("#16a34a"))
            self.checklist_table.item(0, 4).setText(_now_short())
        elif turns > 0:
            self.checklist_table.item(0, 3).setText(f"In-Progress ({turns:.1f} turns)")
            self.checklist_table.item(0, 3).setForeground(QtGui.QColor("#d97706"))
            
        if state.get("card_visible") and not getattr(self, "_card_notified", False):
            self._card_notified = True
            self.statusBar().showMessage("Assembly step completed. screenshot taken and logged.", 8000)
            self.log_panel.refresh()
            self._refresh_history_table()
        elif not state.get("card_visible"):
            self._card_notified = False

        # Tell the worker we're done with this state snapshot so it can
        # emit the next one instead of queuing more while we're behind.
        if self.worker is not None:
            self.worker.mark_state_consumed()

    def _on_gesture(self, gesture: str, conf: float, handed: str) -> None:
        pass

    def _on_error(self, message: str) -> None:
        self.statusBar().showMessage(f"Critical Error: {message}", 8000)
        self.camera_view.show_placeholder(message)

    def _on_camera_unavailable(self, requested_source: int, available: list) -> None:
        """The worker couldn't open/keep open the requested camera. Stop it,
        refresh the picker with whatever cameras ARE usable, and let the
        operator choose one instead of leaving them stuck."""
        self._stop_worker()
        self._refresh_camera_list()

        if available:
            options = ", ".join(str(i) for i in available)
            QtWidgets.QMessageBox.warning(
                self, "Camera Unavailable",
                f"Camera index {requested_source} is unavailable.\n\n"
                f"Detected working cameras: {options}.\n\n"
                "Pick another camera from the dropdown and click 'Start Monitor'."
            )
        else:
            QtWidgets.QMessageBox.critical(
                self, "No Camera Available",
                f"Camera index {requested_source} is unavailable, and no other "
                "cameras were detected.\n\nCheck that a built-in or USB camera "
                "is connected, then click 'Refresh Cameras'."
            )

    # -- MISC & ACTIONS -----------------------------------------------------
    def _build_menu(self) -> None:
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        
        act_refresh = QAction("Refresh Log Files", self)
        act_refresh.triggered.connect(self.log_panel.refresh)
        file_menu.addAction(act_refresh)
        
        file_menu.addSeparator()
        act_quit = QAction("Close App", self)
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        view_menu = menubar.addMenu("View")
        self.act_fullscreen = QAction("Fullscreen Toggle", self)
        self.act_fullscreen.setShortcut("F11")
        self.act_fullscreen.setCheckable(True)
        self.act_fullscreen.triggered.connect(self._toggle_fullscreen)
        view_menu.addAction(self.act_fullscreen)

        view_menu.addSeparator()
        act_logs = QAction("Logs", self)
        act_logs.setShortcut("Ctrl+L")
        act_logs.triggered.connect(self._navigate_to_logs)
        view_menu.addAction(act_logs)

        help_menu = menubar.addMenu("Help")
        act_about = QAction("System Info", self)
        act_about.triggered.connect(self._show_about)
        help_menu.addAction(act_about)

    def _toggle_fullscreen(self, checked: bool) -> None:
        if checked:
            self.showFullScreen()
        else:
            self.showNormal()

    def _show_about(self) -> None:
        QtWidgets.QMessageBox.about(
            self,
            "About HAGRID SOP Monitor",
            f"<h3>{APP_NAME}</h3>"
            f"<p>Version {APP_VERSION}</p>"
            "<p>Industrial SOP checking and computer vision validation tool using "
            "MediaPipe tracking, PyTorch classification networks, and FLANN pattern matches.</p>"
        )

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if hasattr(self, '_activity_logger'):
            self._activity_logger.log("SYSTEM", "Application", "APP_CLOSE", "Application closed")
        self._stop_worker()
        super().closeEvent(event)


# Helper classes to prevent GUI desync
class StatCard(QtWidgets.QFrame):
    def __init__(self, label: str, value: str = "--", accent: str = "#2563eb", parent=None):
        super().__init__(parent)
        self._accent = accent
        self._is_dark = False
        self.setObjectName("statCard")
        self.setStyleSheet(f"""
            QFrame#statCard {{
                background-color: #ffffff;
                border: 1px solid #e2e8f0;
                border-left: 4px solid {accent};
                border-radius: 8px;
            }}
        """)
        self.label = QtWidgets.QLabel(label.upper())
        self.label.setStyleSheet("color: #64748b; font-size: 10px; font-weight: 700; letter-spacing: 0.5px;")
        
        self.value = QtWidgets.QLabel(value)
        self.value.setStyleSheet(f"color: {accent}; font-size: 18px; font-weight: 700;")
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(2)
        layout.addWidget(self.label)
        layout.addWidget(self.value)

    def set_value(self, value: str) -> None:
        self.value.setText(value)

    def set_theme(self, is_dark: bool) -> None:
        self._is_dark = is_dark
        if is_dark:
            bg = "#1e293b"
            border = "#334155"
            lbl_col = "#94a3b8"
        else:
            bg = "#ffffff"
            border = "#e2e8f0"
            lbl_col = "#64748b"
        self.setStyleSheet(f"""
            QFrame#statCard {{
                background-color: {bg};
                border: 1px solid {border};
                border-left: 4px solid {self._accent};
                border-radius: 8px;
            }}
        """)
        self.label.setStyleSheet(f"color: {lbl_col}; font-size: 10px; font-weight: 700; letter-spacing: 0.5px; background: transparent;")
        self.value.setStyleSheet(f"color: {self._accent}; font-size: 18px; font-weight: 700; background: transparent;")


class TrendChart(QtWidgets.QWidget):
    def __init__(self, title: str, subtitle: str, unit: str, min_val: float, max_val: float, parent=None):
        super().__init__(parent)
        self.title = title
        self.subtitle = subtitle
        self.unit = unit
        self.min_val = min_val
        self.max_val = max_val
        self.points = []
        self.setFixedHeight(190)
        self.setMinimumWidth(200)
        self.setStyleSheet("background-color: transparent;")

    def set_points(self, points: list) -> None:
        self.points = points
        self.update()

    def paintEvent(self, event) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        rect = self.rect()

        # -- Card background --
        painter.setBrush(QtGui.QColor("#ffffff"))
        painter.setPen(QtGui.QPen(QtGui.QColor("#e2e8f0"), 1))
        painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), 10, 10)

        # -- Title (left-aligned, clipped to avoid arrow overlap) --
        painter.setPen(QtGui.QColor("#0f172a"))
        font = painter.font()
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)
        title_rect = QtCore.QRect(14, 10, rect.width() - 90, 20)
        painter.drawText(title_rect, Qt.AlignLeft | Qt.AlignVCenter, self.title)
        # -- Subtitle --
        font.setPointSize(7)
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(QtGui.QColor("#94a3b8"))
        sub_rect = QtCore.QRect(14, 30, rect.width() - 28, 16)
        painter.drawText(sub_rect, Qt.AlignLeft | Qt.AlignVCenter, self.subtitle)

        # -- Current value (top right) --
        last_val = self.points[-1] if self.points else None
        if last_val is not None:
            if isinstance(last_val, float):
                val_str = f"{last_val:.2f}{self.unit}"
            else:
                val_str = f"{last_val}{self.unit}"

            font.setPointSize(11)
            font.setBold(True)
            painter.setFont(font)
            painter.setPen(QtGui.QColor("#2563eb"))
            metrics = painter.fontMetrics()
            tw = metrics.horizontalAdvance(val_str)
            painter.drawText(rect.right() - 14 - tw - 22, 26, val_str)

            # -- Trend Arrow:  up (blue) or  down (red) --
            if len(self.points) >= 2:
                first_half_avg = sum(self.points[:len(self.points)//2]) / max(1, len(self.points)//2)
                second_half_avg = sum(self.points[len(self.points)//2:]) / max(1, len(self.points) - len(self.points)//2)
                is_up = second_half_avg >= first_half_avg

                arrow_x = rect.right() - 24
                arrow_y_center = 16
                arrow_size = 10

                if is_up:
                    # Blue upward triangle 
                    arrow_color = QtGui.QColor("#2563eb")
                    pts_arrow = [
                        QtCore.QPointF(arrow_x, arrow_y_center - arrow_size // 2),
                        QtCore.QPointF(arrow_x - arrow_size // 2, arrow_y_center + arrow_size // 2),
                        QtCore.QPointF(arrow_x + arrow_size // 2, arrow_y_center + arrow_size // 2),
                    ]
                else:
                    # Red downward triangle 
                    arrow_color = QtGui.QColor("#dc2626")
                    pts_arrow = [
                        QtCore.QPointF(arrow_x, arrow_y_center + arrow_size // 2),
                        QtCore.QPointF(arrow_x - arrow_size // 2, arrow_y_center - arrow_size // 2),
                        QtCore.QPointF(arrow_x + arrow_size // 2, arrow_y_center - arrow_size // 2),
                    ]

                # Draw arrow badge background
                badge_rect = QtCore.QRectF(arrow_x - 14, arrow_y_center - 14, 28, 28)
                bg = QtGui.QColor(37, 99, 235, 18) if is_up else QtGui.QColor(220, 38, 38, 18)
                painter.setBrush(bg)
                painter.setPen(QtCore.Qt.NoPen)
                painter.drawRoundedRect(badge_rect, 6, 6)

                # Draw the triangle arrow
                polygon = QtGui.QPolygonF(pts_arrow)
                painter.setBrush(arrow_color)
                painter.setPen(QtCore.Qt.NoPen)
                painter.drawPolygon(polygon)

        if not self.points:
            return

        # -- Graph area --
        graph_rect = rect.adjusted(14, 54, -14, -14)
        w = graph_rect.width()
        h = graph_rect.height()

        # Draw subtle grid lines
        painter.setPen(QtGui.QPen(QtGui.QColor("#f1f5f9"), 1))
        for i in range(1, 4):
            gy = graph_rect.top() + (h / 4) * i
            painter.drawLine(graph_rect.left(), int(gy), graph_rect.right(), int(gy))

        n = len(self.points)
        dx = w / (n - 1) if n > 1 else w

        path = QtGui.QPainterPath()
        fill_path = QtGui.QPainterPath()

        pts = []
        for i, val in enumerate(self.points):
            x = graph_rect.left() + i * dx
            norm = (val - self.min_val) / (self.max_val - self.min_val) if self.max_val != self.min_val else 0.5
            norm = max(0.0, min(1.0, norm))
            y = graph_rect.bottom() - norm * h
            pts.append((x, y))
            if i == 0:
                path.moveTo(x, y)
                fill_path.moveTo(x, graph_rect.bottom())
                fill_path.lineTo(x, y)
            else:
                path.lineTo(x, y)
                fill_path.lineTo(x, y)

        if pts:
            fill_path.lineTo(pts[-1][0], graph_rect.bottom())
            fill_path.closeSubpath()

        # Gradient fill
        grad = QtGui.QLinearGradient(0, graph_rect.top(), 0, graph_rect.bottom())
        grad.setColorAt(0.0, QtGui.QColor(37, 99, 235, 55))
        grad.setColorAt(1.0, QtGui.QColor(37, 99, 235, 0))
        painter.setBrush(grad)
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawPath(fill_path)

        # Line
        pen = QtGui.QPen(QtGui.QColor("#2563eb"), 2)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawPath(path)

        # Data points (dots)
        painter.setBrush(QtGui.QColor("#2563eb"))
        painter.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), 1.5))
        for x, y in pts:
            painter.drawEllipse(QtCore.QPointF(x, y), 3.0, 3.0)



# End of helper classes


def _now_short() -> str:
    return time.strftime("%H:%M:%S")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=APP_NAME)
    parser.add_argument("-p", "--config", default="configs/gesture.yaml", help="Path to gesture YAML config")
    parser.add_argument("--source", type=int, default=0, help="Camera index")
    args = parser.parse_args()

    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")

    window = MainWindow(config_path=args.config, source=args.source)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()