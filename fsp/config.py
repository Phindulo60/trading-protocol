"""Centralised path config — honours FSP_DATA_DIR env var for container deploys."""
from __future__ import annotations

import os
from pathlib import Path

def data_dir() -> Path:
    """Return FSP data directory. Defaults to ~/.fsp, override with FSP_DATA_DIR."""
    d = Path(os.environ.get("FSP_DATA_DIR", Path.home() / ".fsp"))
    d.mkdir(parents=True, exist_ok=True)
    return d
