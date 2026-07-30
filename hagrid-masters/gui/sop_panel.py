"""SOP workflow step indicator widget styled for the dark theme."""

from __future__ import annotations

from typing import List, Optional
from qt_compat import QtCore, QtGui, QtWidgets, Qt


class SopStep:
    def __init__(
        self,
        index: int,
        title: str,
        description: str,
        ai_validation: str = "None",
        expected_result: str = "Success",
        timeout: int = 60,
        criteria: str = "Match",
        warning_msg: str = "Step verification failed",
        next_step: str = "Next",
    ):
        self.index = index
        self.title = title
        self.description = description
        self.ai_validation = ai_validation
        self.expected_result = expected_result
        self.timeout = timeout
        self.criteria = criteria
        self.warning_msg = warning_msg
        self.next_step = next_step
        self.status = "pending"  # pending | active | done | error


# Initial default steps as specified by the user
STEPS: List[SopStep] = [
    SopStep(
        0,
        "Operator Verification",
        "Verify authorized operator identity and workstation readiness.",
        ai_validation="Face Verification",
        expected_result="Authorized Operator",
        timeout=15,
        criteria="Face Matched",
        warning_msg="Unauthorized operator or face not detected",
        next_step="1",
    ),
    SopStep(
        1,
        "Safety & PPE Check",
        "Ensure gloves, uniform, safety goggles, helmet, and other required PPE are worn correctly.",
        ai_validation="PPE Detection",
        expected_result="PPE Worn",
        timeout=30,
        criteria="Glove and PPE Detected",
        warning_msg="Please wear hand glove and safety equipment",
        next_step="2",
    ),
    SopStep(
        2,
        "Component Verification",
        "Confirm the correct product, component, and screw are present before operation begins.",
        ai_validation="Object Detection",
        expected_result="Components Present",
        timeout=20,
        criteria="Correct Screw & Part Verified",
        warning_msg="Component missing or incorrect target",
        next_step="3",
    ),
    SopStep(
        3,
        "Tool Verification",
        "Verify the correct tool is selected and ready for operation.",
        ai_validation="Tool Detection",
        expected_result="Screwdriver Ready",
        timeout=20,
        criteria="Tool Detected",
        warning_msg="Tool not found or incorrect tool selected",
        next_step="4",
    ),
    SopStep(
        4,
        "Tool Alignment",
        "Align the screwdriver tip accurately with the target screw before rotation.",
        ai_validation="Tool Alignment Detection",
        expected_result="Aligned",
        timeout=30,
        criteria="Aligned == True",
        warning_msg="Align screwdriver tip to target screw",
        next_step="5",
    ),
    SopStep(
        5,
        "Rotation & Torque Monitoring",
        "Detect clockwise/counter-clockwise rotation, count turns, verify torque target, and validate SOP requirements.",
        ai_validation="Rotation Count Verification",
        expected_result="Target Turns Reached",
        timeout=45,
        criteria="Turns >= 2.5",
        warning_msg="Rotate screwdriver to target turns (2.5)",
        next_step="6",
    ),
    SopStep(
        6,
        "Quality Verification",
        "Confirm the screw is fully tightened/loosened, tool removed safely, and the final assembly passes inspection.",
        ai_validation="Final Quality Inspection",
        expected_result="Pass",
        timeout=30,
        criteria="Tightening Complete & Validated",
        warning_msg="Quality inspection failed",
        next_step="7",
    ),
    SopStep(
        7,
        "Operation Completion & Logging",
        "Capture the final image, save logs, update production records, and mark the SOP as completed.",
        ai_validation="Database & Screenshot Logging",
        expected_result="Logged",
        timeout=10,
        criteria="Screenshot saved and DB entry written",
        warning_msg="Failed to write log records",
        next_step="End",
    ),
]


class SopStepCard(QtWidgets.QFrame):
    STATUS_STYLES = {
        "pending": """
            QFrame#stepCard { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; }
            QLabel#title { color: #475569; font-weight: 600; }
            QLabel#desc  { color: #64748b; }
        """,
        "active": """
            QFrame#stepCard { background: #eff6ff; border: 1px solid #2563eb; border-radius: 8px; }
            QLabel#title { color: #1e3a8a; font-weight: 700; }
            QLabel#desc  { color: #3b82f6; }
        """,
        "done": """
            QFrame#stepCard { background: #f0fdf4; border: 1px solid #22c55e; border-radius: 8px; }
            QLabel#title { color: #14532d; font-weight: 600; }
            QLabel#desc  { color: #16a34a; }
        """,
        "error": """
            QFrame#stepCard { background: #fef2f2; border: 1px solid #ef4444; border-radius: 8px; }
            QLabel#title { color: #7f1d1d; font-weight: 700; }
            QLabel#desc  { color: #dc2626; }
        """,
    }
    ICONS = {"pending": "○", "active": "◑", "done": "✓", "error": "!"}

    def __init__(self, step: SopStep, parent=None):
        super().__init__(parent)
        self.step = step
        self.is_dark = False
        self.setObjectName("stepCard")
        self.setMinimumHeight(68)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        self.title = QtWidgets.QLabel("")
        self.title.setObjectName("title")
        self.title.setStyleSheet("font-size: 13px; background-color: transparent;")
        self.title.setWordWrap(True)

        self.desc = QtWidgets.QLabel("")
        self.desc.setObjectName("desc")
        self.desc.setStyleSheet("font-size: 11px; background-color: transparent;")
        self.desc.setWordWrap(True)

        self.icon_label = QtWidgets.QLabel(self.ICONS["pending"])
        self.icon_label.setFixedSize(32, 32)
        self.icon_label.setAlignment(Qt.AlignCenter)
        self.icon_label.setStyleSheet(
            "font-size: 14px; font-weight: 800; color: #94a3b8;"
            "background-color: #f1f5f9; border-radius: 16px; border: 2px solid #cbd5e1;"
        )

        text_layout = QtWidgets.QVBoxLayout()
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(1)
        text_layout.addWidget(self.title)
        text_layout.addWidget(self.desc)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(12, 7, 12, 7)
        layout.setSpacing(10)
        layout.addWidget(self.icon_label)
        layout.addLayout(text_layout, 1)
        self.setStyleSheet(self.STATUS_STYLES["pending"])

    def set_theme(self, is_dark: bool) -> None:
        self.is_dark = is_dark
        if is_dark:
            self.STATUS_STYLES = {
                "pending": "QFrame#stepCard { background: #1e293b; border: 1px solid #334155; border-radius: 8px; } QLabel#title { color: #e2e8f0; font-weight: 600; } QLabel#desc { color: #94a3b8; }",
                "active":  "QFrame#stepCard { background: #1e3a5f; border: 1px solid #2563eb; border-radius: 8px; } QLabel#title { color: #60a5fa; font-weight: 700; } QLabel#desc { color: #93c5fd; }",
                "done":    "QFrame#stepCard { background: #14532d; border: 1px solid #166534; border-radius: 8px; } QLabel#title { color: #4ade80; font-weight: 600; } QLabel#desc { color: #86efac; }",
                "error":   "QFrame#stepCard { background: #450a0a; border: 1px solid #5c1414; border-radius: 8px; } QLabel#title { color: #f87171; font-weight: 700; } QLabel#desc { color: #fca5a5; }",
            }
        else:
            self.STATUS_STYLES = {
                "pending": "QFrame#stepCard { background: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px; } QLabel#title { color: #0f172a; font-weight: 700; } QLabel#desc { color: #334155; }",
                "active":  "QFrame#stepCard { background: #eff6ff; border: 1px solid #1d4ed8; border-radius: 8px; } QLabel#title { color: #1e3a8a; font-weight: 800; } QLabel#desc { color: #1d4ed8; font-weight: 600; }",
                "done":    "QFrame#stepCard { background: #f0fdf4; border: 1px solid #16a34a; border-radius: 8px; } QLabel#title { color: #14532d; font-weight: 700; } QLabel#desc { color: #15803d; font-weight: 600; }",
                "error":   "QFrame#stepCard { background: #fef2f2; border: 1px solid #dc2626; border-radius: 8px; } QLabel#title { color: #7f1d1d; font-weight: 800; } QLabel#desc { color: #b91c1c; font-weight: 600; }",
            }
        self.set_status(self.step.status, force_style=True)

    def set_status(self, status: str, force_style: bool = False) -> None:
        if not force_style and self.step.status == status:
            return

        self.icon_label.setText(self.ICONS[status])
        if self.is_dark:
            colors   = {"pending": "#94a3b8", "active": "#60a5fa", "done": "#4ade80", "error": "#f87171"}
            bg_colors = {"pending": "#334155", "active": "#1e3a5f", "done": "#14532d",  "error": "#450a0a"}
            border_color = colors[status] if status != 'pending' else '#475569'
        else:
            colors   = {"pending": "#94a3b8", "active": "#2563eb", "done": "#16a34a", "error": "#dc2626"}
            bg_colors = {"pending": "#f1f5f9", "active": "#eff6ff", "done": "#dcfce7",  "error": "#fee2e2"}
            border_color = colors[status] if status != 'pending' else '#cbd5e1'

        self.icon_label.setStyleSheet(
            f"font-size: 14px; font-weight: 800; color: {colors[status]};"
            f"background-color: {bg_colors[status]}; border-radius: 16px; border: 2px solid {border_color};"
        )
        self.setStyleSheet(self.STATUS_STYLES[status])
        self.step.status = status


class SopStepPanel(QtWidgets.QWidget):
    """Vertical list of SOP step cards."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_dark = False
        self.cards: List[SopStepCard] = []
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # Header row: title on the left, Refresh button on the right
        header_row = QtWidgets.QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)

        self.header = QtWidgets.QLabel("SOP Workflow Sequence")
        self.header.setStyleSheet("font-size: 15px; font-weight: 700; color: #0f172a; padding: 2px 0px;")
        header_row.addWidget(self.header)
        header_row.addStretch(1)

        self.refresh_btn = QtWidgets.QPushButton("⟳  Refresh")
        self.refresh_btn.setToolTip("Reset all SOP steps back to Pending")
        self.refresh_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #2563eb;
                color: #ffffff;
                border: none;
                border-radius: 6px;
                padding: 4px 12px;
                font-size: 12px;
                font-weight: 700;
                min-height: 28px;
            }
            QPushButton:hover {
                background-color: #1d4ed8;
            }
            QPushButton:pressed {
                background-color: #1e40af;
            }
        """)
        self.refresh_btn.clicked.connect(self.reset_all_steps)
        header_row.addWidget(self.refresh_btn)

        # "Next Step" button — manually advances the current active manual step
        self.next_step_btn = QtWidgets.QPushButton("Next Step  >")
        self.next_step_btn.setToolTip(
            "Mark the current active step as complete and advance to the next one.\n"
            "Use this for manual steps that do not have automated AI detection."
        )
        self.next_step_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.next_step_btn.setEnabled(False)  # enabled only when a manual step is active
        self.next_step_btn.setStyleSheet("""
            QPushButton {
                background-color: #16a34a;
                color: #ffffff;
                border: none;
                border-radius: 6px;
                padding: 4px 14px;
                font-size: 12px;
                font-weight: 700;
                min-height: 28px;
            }
            QPushButton:hover  { background-color: #15803d; }
            QPushButton:pressed { background-color: #166534; }
            QPushButton:disabled {
                background-color: #d1fae5;
                color: #6ee7b7;
            }
        """)
        self.next_step_btn.clicked.connect(self.advance_manual_step)
        header_row.addWidget(self.next_step_btn)

        layout.addLayout(header_row)

        self.scroll_content = QtWidgets.QWidget()
        self.scroll_content.setObjectName("scrollContent")
        self.scroll_content.setStyleSheet("background-color: transparent;")
        self.scroll_content.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum
        )

        self.scroll_layout = QtWidgets.QVBoxLayout(self.scroll_content)
        self.scroll_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_layout.setSpacing(10)

        self.refresh_callback = None

        for step in STEPS:
            card = SopStepCard(step, self.scroll_content)
            card.title.setText(step.title)
            card.desc.setText(step.description)
            self.cards.append(card)
            self.scroll_layout.addWidget(card)

        layout.addWidget(self.scroll_content)

    def reset_all_steps(self) -> None:
        """Reset all step statuses to pending and fully rebuild the panel from the current STEPS list or callback."""
        if callable(getattr(self, "refresh_callback", None)):
            self.refresh_callback()
            return

        import sop_panel as _sp
        # Mark every step in the live STEPS list as pending
        for step in _sp.STEPS:
            step.status = "pending"
        # Rebuild all cards from current STEPS (restores any steps removed since last save)
        self.rebuild_steps(_sp.STEPS)
        # Re-apply current light/dark theme to the freshly created cards
        self.set_theme(self.is_dark)


    def set_theme(self, is_dark: bool) -> None:
        self.is_dark = is_dark
        hdr_col = "#f8fafc" if is_dark else "#0f172a"
        self.header.setStyleSheet(f"font-size: 15px; font-weight: 700; color: {hdr_col}; padding: 2px 0px;")
        # Keep action buttons consistent across themes
        self.refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #2563eb; color: #ffffff; border: none;
                border-radius: 6px; padding: 4px 12px;
                font-size: 12px; font-weight: 700; min-height: 28px;
            }
            QPushButton:hover  { background-color: #1d4ed8; }
            QPushButton:pressed { background-color: #1e40af; }
        """)
        self.next_step_btn.setStyleSheet("""
            QPushButton {
                background-color: #16a34a; color: #ffffff; border: none;
                border-radius: 6px; padding: 4px 14px;
                font-size: 12px; font-weight: 700; min-height: 28px;
            }
            QPushButton:hover  { background-color: #15803d; }
            QPushButton:pressed { background-color: #166534; }
            QPushButton:disabled { background-color: #d1fae5; color: #6ee7b7; }
        """)
        for card in self.cards:
            card.set_theme(is_dark)

    def advance_manual_step(self) -> None:
        """Manually mark the currently active step as done and advance to the next step.

        Called by the 'Next Step' button. Works for both manual steps and AI steps during operation.
        """
        for card in self.cards:
            if card.step.status == "active":
                card.set_status("done")
                card.desc.setText("Step confirmed / completed")
                break
        self._sync_next_step_btn()

    def _sync_next_step_btn(self) -> None:
        """Enable the Next Step button whenever any step is actively pending completion."""
        if not hasattr(self, "next_step_btn"):
            return
        has_active = any(c.step.status == "active" for c in self.cards)
        self.next_step_btn.setEnabled(has_active)

    def rebuild_steps(self, steps: List[SopStep]) -> None:
        self.cards.clear()

        # Clear existing layout items/widgets
        while self.scroll_layout.count():
            child = self.scroll_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        for step in steps:
            # Reset step status to pending when rebuilding
            step.status = "pending"
            card = SopStepCard(step, self.scroll_content)
            card.title.setText(step.title)
            card.desc.setText(step.description)
            card.set_theme(self.is_dark)
            self.cards.append(card)
            self.scroll_layout.addWidget(card)

        # Set first step as active if steps exist
        if self.cards:
            self.cards[0].set_status("active")
        self._sync_next_step_btn()

    def update_state(self, state: dict) -> None:
        """Progress steps dynamically based on validation type.

        Key rule: once a step reaches 'done' it NEVER goes back.
        Steps only progress forward: pending -> active -> done.
        """
        n_steps = len(self.cards)
        if n_steps == 0:
            return False

        default_desc = [card.step.description for card in self.cards]

        for i in range(n_steps):
            step = self.cards[i].step
            card = self.cards[i]
            val_type = step.ai_validation
            current = step.status

            if current == "done":
                continue

            prev_done = (i == 0) or (self.cards[i - 1].step.status == "done")

            if not prev_done:
                if current != "pending":
                    card.set_status("pending", force_style=True)
                continue

            new_status = current
            new_desc = default_desc[i]

            if val_type in ("Face Verification", "Operator Verification"):
                new_status = "done"

            elif val_type in ("PPE Check", "PPE Detection", "Safety & PPE Check"):
                handedness = state.get("handedness", "")
                glove = state.get("glove", "Normal")
                if current == "error":
                    if handedness and glove == "Glove":
                        new_status = "done"
                        new_desc = "Glove worn and hand detected"
                elif handedness and glove == "Glove":
                    new_status = "done"
                    new_desc = "Glove worn and hand detected"
                elif handedness:
                    new_status = "error"
                    new_desc = step.warning_msg or "Please wear hand glove"
                else:
                    new_status = "active"

            elif val_type in (
                "Object Detection", "Component Verification", "Object Detection & Tracking",
                "Object & Feature Detection", "Alignment & Object Detection", "Pose & Object Rotation Detection",
                "Motion & Pose Detection", "Action / Pressing Detection", "Object Absence Detection"
            ):
                if current == "pending":
                    new_status = "active"

            elif val_type in ("Tool Detection", "Tool Verification"):
                if state.get("aligned") or state.get("handedness", ""):
                    new_status = "done"
                elif current == "pending":
                    new_status = "active"

            elif val_type in ("Tool Alignment Detection", "Tool Alignment"):
                if state.get("aligned"):
                    new_status = "done"
                elif current == "pending":
                    new_status = "active"

            elif val_type in ("Rotation Count Verification", "Rotation"):
                turns = state.get("rotation_count", 0.0)
                target = state.get("turn_target", 2.5)
                if turns >= target:
                    new_status = "done"
                else:
                    if current == "pending":
                        new_status = "active"
                    if turns > 0:
                        new_desc = f"Rotating: {turns:.1f} / {target:.1f} turns"

            elif val_type in ("Final Quality Inspection", "Quality Verification", "Quality Inspection Engine", "Quality Inspection"):
                prev_rotation_done = any(
                    self.cards[j].step.ai_validation in ("Rotation Count Verification", "Rotation")
                    and self.cards[j].step.status == "done"
                    for j in range(i)
                )
                if prev_rotation_done or state.get("completed"):
                    new_status = "done"
                elif current == "pending":
                    new_status = "active"

            elif val_type in ("Database & Screenshot Logging", "Operation Completion & Logging"):
                if state.get("completed") or state.get("rotation_count", 0.0) >= state.get("turn_target", 2.5):
                    new_status = "done"
                elif current == "pending":
                    new_status = "active"

            elif val_type == "Mobile Phone Detection":
                phone_det = state.get("compliance_details", {}).get("phone", False)
                if phone_det:
                    new_status = "error"
                    new_desc = step.warning_msg or "Mobile phone detected in hand!"
                else:
                    new_status = "done"

            elif val_type == "Bluetooth / Earbuds Detection":
                buds_det = state.get("compliance_details", {}).get("earbuds", False)
                if buds_det:
                    new_status = "error"
                    new_desc = step.warning_msg or "Earbuds detected!"
                else:
                    new_status = "done"

            elif val_type == "Shirt Button Open/Close Detection":
                shirt_det = state.get("compliance_details", {}).get("shirt_button_open", False)
                if shirt_det:
                    new_status = "error"
                    new_desc = step.warning_msg or "Shirt button open!"
                else:
                    new_status = "done"

            elif val_type in ("Weight Verification", "Barcode / QR Verification", "OCR Verification",
                              "Final Supervisor Approval", "Work Area Cleanliness",
                              "Label Verification", "Reference Image Comparison"):
                new_status = "done"

            else:
                # Manual steps or default active step progression
                if current == "pending":
                    new_status = "active"

            # Apply changes via set_status
            if new_status != current:
                card.set_status(new_status)
            if new_desc != card.desc.text():
                card.desc.setText(new_desc)

        self._sync_next_step_btn()

        return any(
            c.step.ai_validation in ("Rotation Count Verification", "Tool Alignment Detection", "Tool Alignment", "Rotation")
            and c.step.status == "active"
            for c in self.cards
        )
