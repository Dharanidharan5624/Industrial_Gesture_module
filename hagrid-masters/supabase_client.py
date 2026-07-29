"""Thin Supabase client helper for cloud log sync.

Reads credentials from environment variables (``VITE_SUPABASE_URL`` and
``VITE_SUPABASE_ANON_KEY``) that are pre-populated in ``.env``. The helper
lazily creates the client so the module imports cleanly even when the
``supabase`` package is absent.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

try:
    from supabase import create_client, Client  # type: ignore
    _SUPABASE_OK = True
except Exception:  # pragma: no cover - optional dependency
    create_client = None
    Client = None  # type: ignore
    _SUPABASE_OK = False


def _load_env() -> None:
    """Load .env if present without requiring python-dotenv."""
    env_path = os.path.join(os.getcwd(), ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


@lru_cache(maxsize=1)
def get_supabase() -> Optional["Client"]:
    if not _SUPABASE_OK:
        return None
    _load_env()
    url = os.environ.get("VITE_SUPABASE_URL") or os.environ.get("SUPABASE_URL")
    key = os.environ.get("VITE_SUPABASE_ANON_KEY") or os.environ.get("SUPABASE_ANON_KEY")
    if not url or not key:
        return None
    try:
        return create_client(url, key)
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"[supabase] client init failed: {exc}")
        return None


def fetch_logs(table: str = "screw_monitoring_log", limit: int = 100):
    """Return recent log rows for the dashboard."""
    client = get_supabase()
    if client is None:
        return []
    try:
        resp = client.table(table).select("*").order("id", desc=True).limit(limit).execute()
        return resp.data or []
    except Exception as exc:  # pragma: no cover
        print(f"[supabase] fetch failed: {exc}")
        return []
