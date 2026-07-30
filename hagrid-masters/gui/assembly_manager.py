"""Assembly Manager — singleton data layer for the HAGRID SOP Monitor.

Loads assembly profiles from ``configs/assemblies.yaml`` and exposes a clean
API that the UI can call to:

* query the list of configured assemblies,
* get / set the currently active assembly (persisted to app_settings.json),
* convert YAML step dicts into ``SopStep`` objects ready for ``SopStepPanel``,
* save per-assembly SOP step edits back to the YAML file.

This module intentionally has **no PyQt5 dependency** so it can be imported
freely from both GUI and non-GUI code paths.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

# Resolve project paths relative to this file's location.
_GUI_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_GUI_DIR)
_ASSEMBLIES_YAML = os.path.join(_PROJECT_ROOT, "configs", "assemblies.yaml")
_SETTINGS_JSON = os.path.join(_GUI_DIR, "app_settings.json")

# Default active assembly if nothing is persisted yet.
_DEFAULT_ACTIVE_ID = "screw_tightening"


def _load_yaml(path: str) -> dict:
    """Load a YAML file, returning an empty dict on any error."""
    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception as exc:
        print(f"[AssemblyManager] YAML load error ({path}): {exc}")
        return {}


def _save_yaml(path: str, data: dict) -> None:
    """Persist *data* to a YAML file."""
    try:
        import yaml  # type: ignore

        with open(path, "w", encoding="utf-8") as fh:
            yaml.dump(data, fh, allow_unicode=True, default_flow_style=False, sort_keys=False)
    except Exception as exc:
        print(f"[AssemblyManager] YAML save error ({path}): {exc}")


class AssemblyManager:
    """Singleton that manages assembly profiles from ``assemblies.yaml``."""

    _instance: Optional["AssemblyManager"] = None

    def __new__(cls) -> "AssemblyManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._profiles: List[Dict[str, Any]] = []
        self._active_id: str = _DEFAULT_ACTIVE_ID
        self._load()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load profiles from YAML and resolve the active assembly ID."""
        raw = _load_yaml(_ASSEMBLIES_YAML)
        self._profiles = raw.get("assemblies", [])

        if not self._profiles:
            # Fallback: synthesize a minimal screw-tightening entry so the
            # app still starts even if the YAML is missing / corrupted.
            self._profiles = [
                {
                    "assembly_id": "screw_tightening",
                    "display_name": "Industrial Screw-Tightening Assembly",
                    "detection_mode": "screw_monitor",
                    "is_active": True,
                    "target_params": {"turn_target": 2.5, "align_threshold_px": 50},
                    "sop_steps": [],
                }
            ]

        # Priority: app_settings.json > YAML is_active flag > first profile
        persisted_id = self._read_settings_active_id()
        if persisted_id and any(p["assembly_id"] == persisted_id for p in self._profiles):
            self._active_id = persisted_id
        else:
            # Fall back to whichever profile has is_active: true in the YAML
            for p in self._profiles:
                if p.get("is_active", False):
                    self._active_id = p["assembly_id"]
                    break
            else:
                self._active_id = self._profiles[0]["assembly_id"]

    def _read_settings_active_id(self) -> str:
        """Read ``active_assembly_id`` from app_settings.json."""
        try:
            if os.path.exists(_SETTINGS_JSON):
                with open(_SETTINGS_JSON, "r", encoding="utf-8") as fh:
                    s = json.load(fh)
                return s.get("active_assembly_id", "")
        except Exception:
            pass
        return ""

    def _write_settings_active_id(self, assembly_id: str) -> None:
        """Persist ``active_assembly_id`` to app_settings.json."""
        settings: dict = {}
        try:
            if os.path.exists(_SETTINGS_JSON):
                with open(_SETTINGS_JSON, "r", encoding="utf-8") as fh:
                    settings = json.load(fh)
        except Exception:
            settings = {}
        settings["active_assembly_id"] = assembly_id
        try:
            with open(_SETTINGS_JSON, "w", encoding="utf-8") as fh:
                json.dump(settings, fh, indent=2)
        except Exception as exc:
            print(f"[AssemblyManager] settings write error: {exc}")

    def _profile_by_id(self, assembly_id: str) -> Optional[Dict[str, Any]]:
        for p in self._profiles:
            if p.get("assembly_id") == assembly_id:
                return p
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_all_assemblies(self) -> List[Dict[str, Any]]:
        """Return the list of all assembly profile dicts (shallow copies)."""
        return list(self._profiles)

    def get_active_assembly(self) -> Dict[str, Any]:
        """Return the active assembly profile dict.

        Falls back to the first profile if the stored ID is no longer valid.
        """
        profile = self._profile_by_id(self._active_id)
        if profile is None and self._profiles:
            profile = self._profiles[0]
        return dict(profile) if profile else {}

    def get_active_assembly_id(self) -> str:
        return self._active_id

    def set_active_assembly(self, assembly_id: str) -> bool:
        """Set the active assembly and persist the choice.

        Sets is_active=True for the specified assembly and is_active=False for all others,
        persisting both app_settings.json and assemblies.yaml.
        """
        if not self._profile_by_id(assembly_id):
            print(f"[AssemblyManager] Unknown assembly_id: {assembly_id!r}")
            return False
        self._active_id = assembly_id
        for p in self._profiles:
            p["is_active"] = (p.get("assembly_id") == assembly_id)
        _save_yaml(_ASSEMBLIES_YAML, {"assemblies": self._profiles})
        self._write_settings_active_id(assembly_id)
        return True

    def is_active_assembly(self, assembly_id: Optional[str] = None) -> bool:
        """Return True if the specified (or currently active) assembly is marked is_active."""
        target_id = assembly_id or self._active_id
        profile = self._profile_by_id(target_id)
        if not profile:
            return False
        return bool(profile.get("is_active", True))

    def deactivate_assembly(self, assembly_id: str) -> bool:
        """Deactivate an assembly, marking its is_active status to False."""
        profile = self._profile_by_id(assembly_id)
        if not profile:
            return False
        profile["is_active"] = False
        _save_yaml(_ASSEMBLIES_YAML, {"assemblies": self._profiles})
        return True

    def get_sop_steps(self, assembly_id: Optional[str] = None) -> list:
        """Convert YAML step dicts to ``SopStep`` objects.

        Importing SopStep here (inside the function) avoids a circular import
        between assembly_manager and sop_panel at module load time.
        """
        # Import here to avoid circular dependency at module level
        try:
            from sop_panel import SopStep  # type: ignore
        except ImportError:
            # Non-GUI context — return raw dicts
            profile = self._profile_by_id(assembly_id or self._active_id)
            return (profile or {}).get("sop_steps", [])

        profile = self._profile_by_id(assembly_id or self._active_id)
        if not profile:
            return []

        steps = []
        for s in profile.get("sop_steps", []):
            steps.append(
                SopStep(
                    index=int(s.get("index", 0)),
                    title=str(s.get("title", "Step")),
                    description=str(s.get("description", "")),
                    ai_validation=str(s.get("ai_validation", "None")),
                    expected_result=str(s.get("expected_result", "Success")),
                    timeout=int(s.get("timeout", 60)),
                    criteria=str(s.get("criteria", "Match")),
                    warning_msg=str(s.get("warning_msg", "Step failed")),
                    next_step=str(s.get("next_step", "Next")),
                )
            )
        return steps

    def save_sop_steps(self, assembly_id: str, sop_steps: list) -> None:
        """Persist an updated SOP step list for a given assembly to the YAML.

        *sop_steps* should be a list of ``SopStep`` objects or plain dicts.
        Only the target assembly's steps are modified; other assemblies are
        left untouched.
        """
        profile = self._profile_by_id(assembly_id)
        if profile is None:
            print(f"[AssemblyManager] save_sop_steps: unknown id {assembly_id!r}")
            return

        serialised = []
        for s in sop_steps:
            if isinstance(s, dict):
                serialised.append(s)
            else:
                # SopStep object
                serialised.append(
                    {
                        "index": s.index,
                        "title": s.title,
                        "description": s.description,
                        "ai_validation": s.ai_validation,
                        "expected_result": s.expected_result,
                        "timeout": s.timeout,
                        "criteria": s.criteria,
                        "warning_msg": s.warning_msg,
                        "next_step": s.next_step,
                    }
                )
        profile["sop_steps"] = serialised

        # Write the whole profiles list back to YAML
        _save_yaml(_ASSEMBLIES_YAML, {"assemblies": self._profiles})

    def save_target_params(self, assembly_id: str, params: dict) -> None:
        """Update target_params for a given assembly and persist to YAML."""
        profile = self._profile_by_id(assembly_id)
        if profile is None:
            return
        profile.setdefault("target_params", {}).update(params)
        _save_yaml(_ASSEMBLIES_YAML, {"assemblies": self._profiles})

    def get_detection_mode(self, assembly_id: Optional[str] = None) -> str:
        """Return the detection_mode string for an assembly ('screw_monitor' or 'stub')."""
        profile = self._profile_by_id(assembly_id or self._active_id)
        return (profile or {}).get("detection_mode", "stub")

    def get_step_count(self, assembly_id: str) -> int:
        profile = self._profile_by_id(assembly_id)
        return len((profile or {}).get("sop_steps", []))

    def assembly_id_exists(self, assembly_id: str) -> bool:
        """Return True if an assembly with this id already exists."""
        return self._profile_by_id(assembly_id) is not None

    def add_assembly(
        self,
        assembly_id: str,
        display_name: str,
        detection_mode: str = "stub",
        target_params: Optional[Dict[str, Any]] = None,
        sop_steps: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """Add a new assembly profile and persist to YAML.

        Returns True on success, False if the assembly_id already exists.
        The new assembly is added in non-active state.
        """
        if self.assembly_id_exists(assembly_id):
            print(f"[AssemblyManager] add_assembly: id {assembly_id!r} already exists")
            return False

        new_profile: Dict[str, Any] = {
            "assembly_id": assembly_id,
            "display_name": display_name,
            "detection_mode": detection_mode,
            "is_active": False,
            "target_params": target_params or {},
            "sop_steps": sop_steps or [],
        }
        self._profiles.append(new_profile)
        _save_yaml(_ASSEMBLIES_YAML, {"assemblies": self._profiles})
        return True

    def delete_assembly(self, assembly_id: str) -> bool:
        """Remove a non-active assembly and persist to YAML.

        Returns False if the id is the currently active assembly (cannot delete active),
        or if the id is not found.
        """
        if assembly_id == self._active_id:
            print(f"[AssemblyManager] delete_assembly: cannot delete the active assembly")
            return False
        before = len(self._profiles)
        self._profiles = [p for p in self._profiles if p.get("assembly_id") != assembly_id]
        if len(self._profiles) == before:
            return False
        _save_yaml(_ASSEMBLIES_YAML, {"assemblies": self._profiles})
        return True

