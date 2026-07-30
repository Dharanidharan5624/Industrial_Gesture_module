"""Map compliance event types to operator-facing notification text."""

from __future__ import annotations

from constants import (
    COMPLIANCE_EVENT_BLUETOOTH,
    COMPLIANCE_EVENT_EARBUDS,
    COMPLIANCE_EVENT_MOBILE_PHONE,
    COMPLIANCE_EVENT_PASSED,
    COMPLIANCE_EVENT_SHIRT_BUTTON,
    COMPLIANCE_EVENT_SPECTACLES,
    COMPLIANCE_EVENT_WRITING,
)

NOTIFICATION_MESSAGES = {
    COMPLIANCE_EVENT_MOBILE_PHONE: "Mobile Phone Detected",
    COMPLIANCE_EVENT_SHIRT_BUTTON: "Shirt Button Open",
    COMPLIANCE_EVENT_BLUETOOTH: "Bluetooth Device Detected",
    COMPLIANCE_EVENT_EARBUDS: "Earbuds Detected",
    COMPLIANCE_EVENT_SPECTACLES: "Spectacles Detected",
    COMPLIANCE_EVENT_WRITING: "Writing Activity Detected",
    COMPLIANCE_EVENT_PASSED: "Compliance Passed",
}


def message_for(event_type: str, detail: str = "") -> str:
    base = NOTIFICATION_MESSAGES.get(event_type, event_type.replace("_", " ").title())
    ignore = {
        "",
        base,
        "disabled",
        "model_unavailable",
        "no_phone",
        "no_face",
        "no_earbuds",
        "no_writing",
        "idle",
    }
    if detail and detail not in ignore:
        return detail
    return base


def is_warning(event_type: str) -> bool:
    return event_type in {
        COMPLIANCE_EVENT_MOBILE_PHONE,
        COMPLIANCE_EVENT_SHIRT_BUTTON,
        COMPLIANCE_EVENT_BLUETOOTH,
        COMPLIANCE_EVENT_EARBUDS,
    }
