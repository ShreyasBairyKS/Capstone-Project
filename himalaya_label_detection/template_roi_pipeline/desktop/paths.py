"""
paths.py
─────────────────────────────────────────────────────────────────────────────
Where the desktop app finds its files, from source or as a packaged exe.

    home      the folder that holds `models/` and `config/`
                packaged:  the folder Himalaya.exe sits in (so models and limits can be swapped without a rebuild)
                from source: the repository root
    config    <home>/config when it has the ROI templates, else the package's own config/
    data      per-user, writable: saved results, logs, the window's own storage
                %LOCALAPPDATA%\\Himalaya  (HIMALAYA_DATA_DIR overrides it)

HIMALAYA_HOME overrides `home`, which is how the packaged layout can be tried from source.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

FROZEN = bool(getattr(sys, "frozen", False))
PKG_ROOT = Path(__file__).resolve().parents[1]          # .../template_roi_pipeline (source layout)
REPO_ROOT = PKG_ROOT.parents[1]


def app_home() -> Path:
    if os.environ.get("HIMALAYA_HOME"):
        return Path(os.environ["HIMALAYA_HOME"]).expanduser().resolve()
    return Path(sys.executable).resolve().parent if FROZEN else REPO_ROOT


def config_dir(home: Path) -> Path:
    shipped = Path(home) / "config"
    return shipped if (shipped / "roi_templates").is_dir() else PKG_ROOT / "config"


def data_dir(override: str | os.PathLike | None = None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    if os.environ.get("HIMALAYA_DATA_DIR"):
        return Path(os.environ["HIMALAYA_DATA_DIR"]).expanduser().resolve()
    base = os.environ.get("LOCALAPPDATA")
    return (Path(base) if base else Path.home() / ".local" / "share") / "Himalaya"
