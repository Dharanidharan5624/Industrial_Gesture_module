"""Live status panel redesigned for a premium dark-theme interface."""

from __future__ import annotations

from qt_compat import QtCore, QtGui, QtWidgets, Qt

GESTURE_ICONS = {
    "ok": "OK", "like": "LIKE", "dislike": "DOWN", "fist": "FIST",
    "palm": "PALM", "one": "1", "peace": "V", "three": "3",
    "four": "4", "rock": "ROCK", "call": "CALL", "middle_finger": "MID",
    "little_finger": "PINKY", "thumb_index": "GUN", "no_gesture": "--",
    "thumbs_up": "LIKE", "thumbs_down": "DOWN", "two": "2", "five": "5",
    "stop": "STOP", "mute": "MUTE", "two_up": "2", "peace_inv": "V",
}


class StatCard(QtWidgets.QFrame):
    """Metric card with White-Blue theme styling and optional trend arrow."""

    def __init__(self, label: str, value: str = "--", accent: str = "#2563eb", parent=None):
        super().__init__(parent)
        self.accent = accent
        self.setObjectName("statCard")
        self.setStyleSheet(f"""
            QFrame#statCard {{
                background-color: #ffffff;
                border: 1px solid #e2e8f0;
                border-left: 4px solid {accent};
                border-radius: 8px;
            }}
        """)
        self.setMinimumHeight(52)
        self.label = QtWidgets.QLabel(label.upper())
        self.label.setStyleSheet("color: #64748b; font-size: 10px; font-weight: 700; letter-spacing: 0.8px;")

        # Value + trend arrow row
        val_row = QtWidgets.QHBoxLayout()
        val_row.setContentsMargins(0, 0, 0, 0)
        val_row.setSpacing(6)

        self.value = QtWidgets.QLabel(value)
        self.value.setStyleSheet(f"color: {accent}; font-size: 16px; font-weight: 700; background: transparent;")
        val_row.addWidget(self.value)

        self.trend_lbl = QtWidgets.QLabel("")
        self.trend_lbl.setStyleSheet("font-size: 14px; font-weight: 800; background: transparent;")
        val_row.addWidget(self.trend_lbl)
        val_row.addStretch(1)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(2)
        layout.addWidget(self.label)
        layout.addLayout(val_row)

    def set_theme(self, is_dark: bool) -> None:
        self.is_dark = is_dark
        bg = "#1e293b" if is_dark else "#ffffff"
        border = "#334155" if is_dark else "#cbd5e1"
        lbl_col = "#cbd5e1" if is_dark else "#334155"
        self.setStyleSheet(f"""
            QFrame#statCard {{
                background-color: {bg};
                border: 1px solid {border};
                border-left: 4px solid {self.accent};
                border-radius: 8px;
            }}
        """)
        self.label.setStyleSheet(f"color: {lbl_col}; font-size: 10px; font-weight: 800; letter-spacing: 0.8px;")

    def set_value(self, value: str) -> None:
        self.value.setText(value)

    def set_trend(self, direction: str) -> None:
        """Set trend arrow. direction: 'up' | 'down' | 'flat' | ''"""
        if direction == "up":
            self.trend_lbl.setText("▲")
            self.trend_lbl.setStyleSheet("font-size: 13px; font-weight: 800; color: #2563eb; background: transparent;")
        elif direction == "down":
            self.trend_lbl.setText("▼")
            self.trend_lbl.setStyleSheet("font-size: 13px; font-weight: 800; color: #dc2626; background: transparent;")
        else:
            self.trend_lbl.setText("")


class StatusPanel(QtWidgets.QWidget):
    """Live gesture and tool status panel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_dark = False
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        self.title = QtWidgets.QLabel("Live Detection Status")
        self.title.setStyleSheet("font-size: 15px; font-weight: 700; color: #0f172a; padding: 2px 0px;")
        layout.addWidget(self.title)

        # Gesture Card with gradient
        self.gesture_card = QtWidgets.QFrame()
        self.gesture_card.setObjectName("gestureCard")
        self.gesture_card.setStyleSheet("""
            QFrame#gestureCard {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #ffffff, stop:1 #f8fafc);
                border: 1px solid #e2e8f0;
                border-radius: 10px;
            }
        """)
        self.gesture_card.setFixedHeight(92)
        self.gesture_icon = QtWidgets.QLabel("--")
        self.gesture_icon.setAlignment(Qt.AlignCenter)
        self.gesture_icon.setStyleSheet(
            "color: #1e3a8a; font-size: 24px; font-weight: 800; letter-spacing: 1px; background-color: transparent;"
        )
        
        self.gesture_name = QtWidgets.QLabel("No Gesture")
        self.gesture_name.setAlignment(Qt.AlignCenter)
        self.gesture_name.setStyleSheet("color: #0f172a; font-size: 16px; font-weight: 700; background-color: transparent;")
        
        self.gesture_conf = QtWidgets.QLabel("0.0%")
        self.gesture_conf.setAlignment(Qt.AlignCenter)
        self.gesture_conf.setStyleSheet("color: #2563eb; font-size: 12px; font-weight: 600; background-color: transparent;")
        
        gc_layout = QtWidgets.QVBoxLayout(self.gesture_card)
        gc_layout.setContentsMargins(8, 8, 8, 8)
        gc_layout.setSpacing(2)
        gc_layout.addWidget(self.gesture_icon)
        gc_layout.addWidget(self.gesture_name)
        gc_layout.addWidget(self.gesture_conf)
        layout.addWidget(self.gesture_card)

        # Metrics grid
        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        self.card_hand = StatCard("Hand Side", "--", "#38bdf8")
        self.card_glove = StatCard("ESD Glove", "--", "#e0a800")
        self.card_align = StatCard("Alignment", "--", "#16a34a")
        self.card_dir = StatCard("Direction", "--", "#8b5cf6")
        
        grid.addWidget(self.card_hand, 0, 0)
        grid.addWidget(self.card_glove, 0, 1)
        grid.addWidget(self.card_align, 1, 0)
        grid.addWidget(self.card_dir, 1, 1)
        layout.addLayout(grid)

        self.card_turns = StatCard("Rotation Turns", "0.00 / 2.50", "#f97316")
        self.card_turns.setMinimumHeight(58)
        layout.addWidget(self.card_turns)

    def set_theme(self, is_dark: bool) -> None:
        self.is_dark = is_dark
        if is_dark:
            self.title.setStyleSheet("font-size: 15px; font-weight: 700; color: #f8fafc; padding: 2px 0px;")
            self.gesture_card.setStyleSheet("QFrame#gestureCard { background: #1e293b; border: 1px solid #334155; border-radius: 10px; }")
            self.gesture_icon.setStyleSheet("color: #60a5fa; font-size: 24px; font-weight: 800; letter-spacing: 1px; background-color: transparent;")
            self.gesture_name.setStyleSheet("color: #f8fafc; font-size: 16px; font-weight: 700; background-color: transparent;")
        else:
            self.title.setStyleSheet("font-size: 15px; font-weight: 700; color: #0f172a; padding: 2px 0px;")
            self.gesture_card.setStyleSheet("QFrame#gestureCard { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #ffffff, stop:1 #f8fafc); border: 1px solid #e2e8f0; border-radius: 10px; }")
            self.gesture_icon.setStyleSheet("color: #1e3a8a; font-size: 24px; font-weight: 800; letter-spacing: 1px; background-color: transparent;")
            self.gesture_name.setStyleSheet("color: #0f172a; font-size: 16px; font-weight: 700; background-color: transparent;")

        for card in (self.card_hand, self.card_glove, self.card_align, self.card_dir, self.card_turns):
            card.set_theme(is_dark)

    def update_state(self, state: dict) -> None:
        hand_detected = state.get("hand_detected", False)
        gesture = state.get("gesture", "no_gesture")

        # ── Gesture card ──────────────────────────────────────────────────
        if gesture == "no_gesture" or not hand_detected:
            self.gesture_icon.setText("--")
            self.gesture_name.setText("No Gesture")
            self.gesture_conf.setText("0.0%")
        else:
            self.gesture_icon.setText(GESTURE_ICONS.get(gesture, gesture[:4].upper()))
            self.gesture_name.setText(gesture.replace("_", " ").title())
            self.gesture_conf.setText(f"{state.get('gesture_confidence', 0.0):.1f}%")

        # ── Hand Side card ────────────────────────────────────────────────
        handedness = state.get("handedness", "--")
        self.card_hand.set_value(handedness if handedness else "--")

        # Dynamic card background/border colors based on theme
        bg = "#1e293b" if self.is_dark else "#ffffff"
        border = "#334155" if self.is_dark else "#e2e8f0"

        # ── ESD Glove card ────────────────────────────────────────────────
        glove = state.get("glove", "--")
        self.card_glove.set_value(glove)
        if glove == "Glove":
            glove_color = "#16a34a"   # green  – glove is present (safe)
        elif glove == "Normal":
            glove_color = "#dc2626"   # red    – bare hand (unsafe, no glove)
        else:
            glove_color = "#94a3b8" if self.is_dark else "#64748b"   # slate  – no hand detected
        self.card_glove.setStyleSheet(f"QFrame#statCard {{ background-color: {bg}; border: 1px solid {border}; border-left: 4px solid {glove_color}; border-radius: 8px; }}")
        self.card_glove.value.setStyleSheet(f"color: {glove_color}; font-size: 16px; font-weight: 700; background: transparent;")
        self.card_glove.set_trend("")

        # ── Alignment card ────────────────────────────────────────────────
        if hand_detected:
            aligned = state.get("aligned", False)
            self.card_align.set_value("ALIGNED" if aligned else "NOT ALIGNED")
            align_color = "#16a34a" if aligned else "#dc2626"
        else:
            self.card_align.set_value("--")
            align_color = "#94a3b8" if self.is_dark else "#64748b"
        self.card_align.setStyleSheet(f"QFrame#statCard {{ background-color: {bg}; border: 1px solid {border}; border-left: 4px solid {align_color}; border-radius: 8px; }}")
        self.card_align.value.setStyleSheet(f"color: {align_color}; font-size: 16px; font-weight: 700; background: transparent;")
        self.card_align.set_trend("")

        # ── Direction card ────────────────────────────────────────────────
        direction = state.get("direction", "Idle")
        self.card_dir.set_value(direction if hand_detected else "--")
        self.card_dir.set_trend("")

        # ── Rotation Turns card ───────────────────────────────────────────
        turns = state.get("rotation_count", 0.0)
        target = state.get("turn_target", 2.5)
        self.card_turns.set_value(f"{turns:.2f} / {target:.2f}")
        self.card_turns.set_trend("")

