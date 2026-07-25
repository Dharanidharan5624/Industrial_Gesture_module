"""CSV + SQLite persistence for compliance events."""

from __future__ import annotations

import csv
import os
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from constants import (
    COMPLIANCE_CSV_COLUMNS,
    COMPLIANCE_CSV_PATH,
    COMPLIANCE_DB_PATH,
    COMPLIANCE_SCREENSHOT_DIR,
    COMPLIANCE_SCREENSHOT_PREFIX,
    LOG_DIR,
)
from compliance.models import ComplianceEvent


class ComplianceLogger:
    """Thread-safe compliance event writer (CSV + SQLite) with in-memory cache."""

    def __init__(
        self,
        csv_path: str = COMPLIANCE_CSV_PATH,
        db_path: str = COMPLIANCE_DB_PATH,
        screenshot_dir: str = COMPLIANCE_SCREENSHOT_DIR,
    ):
        self.csv_path = csv_path
        self.db_path = db_path
        self.screenshot_dir = screenshot_dir
        self._lock = threading.Lock()
        self._cache: List[Dict[str, Any]] = []
        self._warning_count = 0
        os.makedirs(LOG_DIR, exist_ok=True)
        os.makedirs(self.screenshot_dir, exist_ok=True)
        self._ensure_csv()
        self._ensure_db()
        self._load_cache_from_csv()

    def _ensure_csv(self) -> None:
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=COMPLIANCE_CSV_COLUMNS)
                writer.writeheader()

    def _ensure_db(self) -> None:
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    operator_id TEXT,
                    operator_name TEXT,
                    event_type TEXT,
                    detection_result TEXT,
                    camera_id TEXT,
                    confidence REAL,
                    screenshot_path TEXT,
                    status TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_ts ON events(timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_op ON events(operator_id)"
            )
            conn.commit()

    def _load_cache_from_csv(self, limit: int = 500) -> None:
        if not os.path.exists(self.csv_path):
            return
        try:
            with open(self.csv_path, newline="") as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)
            self._cache = rows[-limit:]
            self._warning_count = sum(1 for r in rows if r.get("Status") == "Warning")
        except Exception as exc:
            print(f"[compliance] cache load failed: {exc}")

    @property
    def warning_count(self) -> int:
        return self._warning_count

    def log_event(self, event: ComplianceEvent) -> Dict[str, Any]:
        row = event.to_csv_row()
        with self._lock:
            with open(self.csv_path, "a", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=COMPLIANCE_CSV_COLUMNS)
                writer.writerow(row)
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        """
                        INSERT INTO events (
                            timestamp, operator_id, operator_name, event_type,
                            detection_result, camera_id, confidence,
                            screenshot_path, status
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            event.timestamp,
                            event.operator_id,
                            event.operator_name,
                            event.event_type,
                            event.detection_result,
                            event.camera_id,
                            float(event.confidence),
                            event.screenshot_path,
                            event.status,
                        ),
                    )
                    conn.commit()
            except Exception as exc:
                print(f"[compliance] sqlite write failed: {exc}")
            self._cache.append(row)
            if len(self._cache) > 500:
                self._cache = self._cache[-500:]
            if event.status == "Warning":
                self._warning_count += 1
        return row

    def save_evidence(self, frame, event_type: str) -> str:
        """Save BGR frame evidence; return relative path or empty string."""
        try:
            import cv2

            os.makedirs(self.screenshot_dir, exist_ok=True)
            name = (
                f"{COMPLIANCE_SCREENSHOT_PREFIX}_{event_type}_"
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
            path = os.path.join(self.screenshot_dir, name)
            cv2.imwrite(path, frame)
            return path
        except Exception as exc:
            print(f"[compliance] evidence save failed: {exc}")
            return ""

    def recent_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._cache[-limit:])

    def query_events(
        self,
        operator_id: Optional[str] = None,
        event_type: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        clauses = []
        params: List[Any] = []
        if operator_id and operator_id not in ("", "All"):
            clauses.append("operator_id = ?")
            params.append(operator_id)
        if event_type and event_type not in ("", "All"):
            clauses.append("event_type = ?")
            params.append(event_type)
        if date_from:
            clauses.append("timestamp >= ?")
            params.append(date_from)
        if date_to:
            clauses.append("timestamp <= ?")
            params.append(date_to + " 23:59:59" if len(date_to) == 10 else date_to)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT timestamp, operator_id, operator_name, event_type, detection_result, camera_id, confidence, screenshot_path, status FROM events{where} ORDER BY id DESC LIMIT ?"
        params.append(limit)
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.execute(sql, params)
                return [dict(r) for r in cur.fetchall()]
        except Exception as exc:
            print(f"[compliance] query failed: {exc}")
            return self.recent_events(limit)

    def daily_summary(self, days: int = 7) -> Dict[str, Any]:
        """Aggregate counts for analytics charts."""
        since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d 00:00:00")
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute(
                    """
                    SELECT event_type, status, COUNT(*) as cnt
                    FROM events WHERE timestamp >= ?
                    GROUP BY event_type, status
                    """,
                    (since,),
                )
                by_type: Dict[str, int] = {}
                warnings = 0
                total = 0
                for event_type, status, cnt in cur.fetchall():
                    by_type[event_type] = by_type.get(event_type, 0) + cnt
                    total += cnt
                    if status == "Warning":
                        warnings += cnt
                passed = by_type.get("COMPLIANCE_PASSED", 0)
                compliance_pct = (
                    round(100.0 * (total - warnings) / total, 1) if total else 100.0
                )
                cur2 = conn.execute(
                    """
                    SELECT operator_id, COUNT(*) as cnt
                    FROM events WHERE timestamp >= ? AND status = 'Warning'
                    GROUP BY operator_id ORDER BY cnt DESC LIMIT 10
                    """,
                    (since,),
                )
                by_operator = {row[0]: row[1] for row in cur2.fetchall()}
                return {
                    "total": total,
                    "warnings": warnings,
                    "passed": passed,
                    "compliance_pct": compliance_pct,
                    "by_type": by_type,
                    "by_operator": by_operator,
                }
        except Exception as exc:
            print(f"[compliance] summary failed: {exc}")
            return {
                "total": 0,
                "warnings": self._warning_count,
                "passed": 0,
                "compliance_pct": 100.0,
                "by_type": {},
                "by_operator": {},
            }

    def export_csv(self, dest_path: str, rows: Optional[List[Dict[str, Any]]] = None) -> str:
        data = rows if rows is not None else self.query_events(limit=10000)
        with open(dest_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=COMPLIANCE_CSV_COLUMNS)
            writer.writeheader()
            for r in data:
                # Normalize sqlite keys to CSV headers
                out = {
                    "Timestamp": r.get("Timestamp") or r.get("timestamp", ""),
                    "Operator_ID": r.get("Operator_ID") or r.get("operator_id", ""),
                    "Operator_Name": r.get("Operator_Name") or r.get("operator_name", ""),
                    "Event_Type": r.get("Event_Type") or r.get("event_type", ""),
                    "Detection_Result": r.get("Detection_Result") or r.get("detection_result", ""),
                    "Camera_ID": r.get("Camera_ID") or r.get("camera_id", ""),
                    "Confidence": r.get("Confidence") or r.get("confidence", ""),
                    "Screenshot_Path": r.get("Screenshot_Path") or r.get("screenshot_path", ""),
                    "Status": r.get("Status") or r.get("status", ""),
                }
                writer.writerow(out)
        return dest_path

    def clear_all_logs(self) -> None:
        """Clears SQLite events table, rewrites empty CSV file, and resets cache."""
        with self._lock:
            self._cache.clear()
            self._warning_count = 0
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("DELETE FROM events")
                    conn.commit()
            except Exception as exc:
                print(f"[compliance] clear db failed: {exc}")

            try:
                with open(self.csv_path, "w", newline="") as fh:
                    writer = csv.writer(fh)
                    writer.writerow(COMPLIANCE_CSV_COLUMNS)
            except Exception as exc:
                print(f"[compliance] clear csv failed: {exc}")
