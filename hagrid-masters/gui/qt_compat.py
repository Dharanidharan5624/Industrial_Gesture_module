"""PyQt5 / PySide6 compatibility shim.

Imports the Qt framework once and re-exports the common symbols so the rest
of the GUI code stays toolkit-agnostic. PyQt5 is preferred (matches the
spec) but PySide6 is used as a fallback when PyQt5 is not installed.
"""

from __future__ import annotations

import importlib

_QT = None

for _name in ("PyQt5", "PySide6"):
    try:
        mod = importlib.import_module(_name)
        _QT = _name
        if _name == "PyQt5":
            from PyQt5 import QtCore, QtGui, QtWidgets  # noqa: F401
            from PyQt5.QtCore import pyqtSignal as Signal, pyqtSlot as Slot, Qt
            from PyQt5.QtWidgets import QAction
        else:
            from PySide6 import QtCore, QtGui, QtWidgets  # noqa: F401
            from PySide6.QtCore import Signal, Slot, Qt
            from PySide6.QtGui import QAction
        break
    except Exception:
        continue

if _QT is None:  # pragma: no cover - sandbox path
    raise ImportError(
        "Neither PyQt5 nor PySide6 is installed. Install one with:\n"
        "  pip install PyQt5   # or\n  pip install PySide6"
    )

QT_BINDING = _QT

__all__ = [
    "QtCore",
    "QtGui",
    "QtWidgets",
    "Signal",
    "Slot",
    "Qt",
    "QT_BINDING",
    "QAction",
]
