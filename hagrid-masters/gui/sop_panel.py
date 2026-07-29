"""SOP workflow step indicator widget styled for the dark theme."""

from __future__ import annotations

from typing import List
from qt_compat import QtCore, QtGui, QtWidgets, Qt


class SopStep:
    def __init__(self, index: int, title: str, description: str):
        self.index = index
        self.title = title
        self.description = description
        self.status = "pending"  # pending | active | done


STEPS: List[SopStep] = [
    SopStep(0, "Safety Check", "Glove worn and hand detected"),
    SopStep(1, "Tool Alignment", "Screwdriver tip aligned to screw"),
    SopStep(2, "Rotation", "Turn screw clockwise until target reached"),
    SopStep(3, "Completion", "Operation logged and screenshot saved"),
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
        self.setFixedHeight(64)

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
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        
        self.header = QtWidgets.QLabel("SOP Workflow Sequence")
        self.header.setStyleSheet("font-size: 15px; font-weight: 700; color: #0f172a; padding: 2px 0px;")
        layout.addWidget(self.header)
        
        # Vertical scroll area for SOP steps
        self.scroll_area = QtWidgets.QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setStyleSheet("background-color: transparent;")
        
        self.scroll_content = QtWidgets.QWidget()
        self.scroll_content.setObjectName("scrollContent")
        self.scroll_content.setStyleSheet("background-color: transparent;")
        
        self.scroll_layout = QtWidgets.QVBoxLayout(self.scroll_content)
        self.scroll_layout.setContentsMargins(0, 0, 6, 0)
        self.scroll_layout.setSpacing(10)
        
        for step in STEPS:
            card = SopStepCard(step, self.scroll_content)
            card.title.setText(step.title)
            card.desc.setText(step.description)
            self.cards.append(card)
            self.scroll_layout.addWidget(card)
        
        self.scroll_layout.addStretch(1)
        self.scroll_area.setWidget(self.scroll_content)
        layout.addWidget(self.scroll_area, 1)

    def set_theme(self, is_dark: bool) -> None:
        self.is_dark = is_dark
        hdr_col = "#f8fafc" if is_dark else "#0f172a"
        self.header.setStyleSheet(f"font-size: 15px; font-weight: 700; color: {hdr_col}; padding: 2px 0px;")
        for card in self.cards:
            card.set_theme(is_dark)

    def rebuild_steps(self, steps: List[SopStep]) -> None:
        self.cards.clear()
        
        # Clear existing layout items/widgets
        while self.scroll_layout.count():
            child = self.scroll_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
                
        for step in steps:
            card = SopStepCard(step, self.scroll_content)
            card.title.setText(step.title)
            card.desc.setText(step.description)
            self.cards.append(card)
            self.scroll_layout.addWidget(card)
            
        self.scroll_layout.addStretch(1)

    def update_state(self, state: dict) -> None:
        """Progress steps from the monitor state snapshot."""
        n_steps = len(self.cards)
        statuses = ["pending"] * n_steps
        default_desc = [card.step.description for card in self.cards]
        
        # Step 0: safety - hand detected and glove worn
        if n_steps > 0:
            handedness = state.get("handedness", "")
            glove = state.get("glove", "Normal")
            if handedness and glove == "Glove":
                statuses[0] = "done"
                default_desc[0] = "Glove worn and hand detected"
            elif handedness:
                statuses[0] = "error"
                default_desc[0] = "Please wear hand glove"
                
        # Step 1: alignment
        if n_steps > 1:
            if state.get("aligned"):
                statuses[1] = "done"
            elif n_steps > 0 and statuses[0] == "done":
                statuses[1] = "active"
                
        # Step 2: rotation
        if n_steps > 2:
            turns = state.get("rotation_count", 0.0)
            target = state.get("turn_target", 2.5)
            if turns > 0 and n_steps > 1 and statuses[1] == "done":
                statuses[2] = "active"
            if turns >= target:
                statuses[2] = "done"
                
        # Step 3: completion
        if n_steps > 3:
            if state.get("completed"):
                statuses[3] = "done"
            elif n_steps > 2 and statuses[2] == "done":
                statuses[3] = "active"
                
        # For any additional steps (index >= 4), if previous is done, make it active/pending
        for i in range(4, n_steps):
            if statuses[i-1] == "done":
                statuses[i] = "active"
                
        for card, status, desc in zip(self.cards, statuses, default_desc):
            if card.desc.text() != desc:
                card.desc.setText(desc)
            if card.step.status != status:
                card.set_status(status)
