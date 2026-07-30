"""Activity Logger – records every user interaction in the application.

Logs are stored as CSV files under PROJECT_ROOT/logs/ with the naming
convention ``mmddyy.csv`` (e.g. ``072026.csv`` for 20-Jul-2026).

Each row captures:
    Timestamp, Event_Type, Component, Action, Details

The logger is a singleton that gets imported once at app startup and hooks
into all button clicks, sidebar navigations, and custom events.
"""

from __future__ import annotations

import csv
import os
import datetime as _dt
from typing import List, Optional

_GUI_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_GUI_DIR)
_LOG_DIR = os.path.join(_PROJECT_ROOT, "logs")

ACTIVITY_COLUMNS = [
    "Timestamp",
    "Event_Type",
    "Component",
    "Action",
    "Details",
]


def _today_filename() -> str:
    """Return the CSV filename for today in ``mmddyy.csv`` format."""
    return _dt.datetime.now().strftime("%m%d%y") + ".csv"


def _now_iso() -> str:
    """Return current datetime as a human-readable ISO-like string."""
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


class ActivityLogger:
    """Singleton-style activity logger that appends to daily CSV files."""

    _instance: Optional["ActivityLogger"] = None

    def __new__(cls) -> "ActivityLogger":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialised = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialised:
            return
        self._initialised = True
        self._log_dir = _LOG_DIR
        os.makedirs(self._log_dir, exist_ok=True)
        self._cache: List[list] = []  # in-memory copy for UI table
        self._current_file: str = ""
        self._ensure_file()
        # Record app-open event
        self.log("SYSTEM", "Application", "APP_START", "Application opened")

    # -- file management ---------------------------------------------------
    def _ensure_file(self) -> None:
        """Create today's CSV with a header if it doesn't exist yet."""
        fname = _today_filename()
        path = os.path.join(self._log_dir, fname)
        self._current_file = path
        if not os.path.exists(path):
            with open(path, "w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(ACTIVITY_COLUMNS)
        else:
            # Load existing rows into cache
            try:
                with open(path, newline="") as fh:
                    reader = csv.reader(fh)
                    next(reader, None)  # skip header
                    self._cache = list(reader)
            except Exception:
                pass

    def _rotate_if_needed(self) -> None:
        """If the date has changed since last write, switch to a new file."""
        expected = os.path.join(self._log_dir, _today_filename())
        if expected != self._current_file:
            self._cache.clear()
            self._ensure_file()

    # -- public API --------------------------------------------------------
    def log(
        self,
        event_type: str,
        component: str,
        action: str,
        details: str = "",
    ) -> None:
        """Append one activity row to today's CSV and in-memory cache.

        Parameters
        ----------
        event_type : str
            Category such as ``CLICK``, ``NAVIGATION``, ``SYSTEM``,
            ``ADD``, ``DELETE``, ``SETTING``, ``EXPORT``, etc.
        component : str
            The UI component involved (e.g. ``Sidebar``, ``SOP Config``,
            ``Tool Management``).
        action : str
            What actually happened (e.g. ``Navigate to Dashboard``,
            ``Start Monitor``, ``Add Tool Row``).
        details : str, optional
            Any extra context.
        """
        self._rotate_if_needed()
        row = [_now_iso(), event_type, component, action, details]
        self._cache.append(row)
        try:
            with open(self._current_file, "a", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(row)
        except Exception as exc:
            print(f"[ActivityLogger] write error: {exc}")

    def get_rows(self) -> List[list]:
        """Return all cached rows for the current day (newest first)."""
        return list(reversed(self._cache))

    def get_csv_path(self) -> str:
        """Return the absolute path of today's CSV file."""
        self._rotate_if_needed()
        return self._current_file

    def get_all_log_files(self) -> List[str]:
        """Return a sorted list of all log CSV files in the logs dir."""
        files = []
        try:
            for f in os.listdir(self._log_dir):
                if f.endswith(".csv"):
                    files.append(os.path.join(self._log_dir, f))
        except Exception:
            pass
        files.sort(reverse=True)
        return files

    def load_file(self, path: str) -> List[list]:
        """Load rows from a specific log CSV file."""
        rows: List[list] = []
        try:
            with open(path, newline="") as fh:
                reader = csv.reader(fh)
                next(reader, None)
                rows = list(reader)
        except Exception:
            pass
        return list(reversed(rows))
