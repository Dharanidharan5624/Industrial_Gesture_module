"""Data contracts for Operator Compliance & Safety Monitoring."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DetectorResult:
    detected: bool = False
    confidence: float = 0.0
    detail: str = ""
    bbox: Optional[tuple] = None  # (x1, y1, x2, y2) when available

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detected": self.detected,
            "confidence": round(float(self.confidence), 3),
            "detail": self.detail,
            "bbox": self.bbox,
        }


@dataclass
class ComplianceState:
    phone: DetectorResult = field(default_factory=DetectorResult)
    shirt_button_open: DetectorResult = field(default_factory=DetectorResult)
    earbuds: DetectorResult = field(default_factory=DetectorResult)
    bluetooth: DetectorResult = field(default_factory=DetectorResult)
    spectacles: DetectorResult = field(default_factory=DetectorResult)
    writing: DetectorResult = field(default_factory=DetectorResult)
    active_alerts: List[str] = field(default_factory=list)
    warning_count: int = 0
    new_events: List["ComplianceEvent"] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phone": self.phone.to_dict(),
            "shirt_button_open": self.shirt_button_open.to_dict(),
            "earbuds": self.earbuds.to_dict(),
            "bluetooth": self.bluetooth.to_dict(),
            "spectacles": self.spectacles.to_dict(),
            "writing": self.writing.to_dict(),
            "active_alerts": list(self.active_alerts),
            "warning_count": self.warning_count,
            "new_events": [e.to_dict() for e in self.new_events],
        }


@dataclass
class ComplianceEvent:
    timestamp: str
    operator_id: str
    operator_name: str
    event_type: str
    detection_result: str
    camera_id: str
    confidence: float
    screenshot_path: str = ""
    status: str = "Warning"  # Warning | Passed | Info

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_csv_row(self) -> Dict[str, Any]:
        return {
            "Timestamp": self.timestamp,
            "Operator_ID": self.operator_id,
            "Operator_Name": self.operator_name,
            "Event_Type": self.event_type,
            "Detection_Result": self.detection_result,
            "Camera_ID": self.camera_id,
            "Confidence": f"{self.confidence:.3f}",
            "Screenshot_Path": self.screenshot_path,
            "Status": self.status,
        }
