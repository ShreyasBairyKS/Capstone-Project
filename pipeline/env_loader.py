"""
Central .env loading and path resolution for VisionQAI.

- Loads <repo>/.env into os.environ at import time (real environment variables always win,
  so Docker/compose values are never overridden).
- `env_path()` resolves path settings: absolute paths are used as-is, relative paths are resolved
  against the repository root (never the current working directory), so the system behaves the same
  on every PC and regardless of where uvicorn is launched from.
"""
import os
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_env_file(env_file: Optional[Path] = None) -> None:
    """Parses KEY=VALUE lines from .env (ignores comments/blank lines, strips optional quotes)."""
    env_file = env_file or (REPO_ROOT / ".env")
    if not env_file.is_file():
        return
    for raw in env_file.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


def env_path(name: str, default: Path) -> Path:
    """Returns the path in env var `name` (or `default`), resolved against the repo root if relative."""
    raw = os.getenv(name, "").strip()
    p = Path(os.path.expandvars(os.path.expanduser(raw))) if raw else Path(default)
    return p if p.is_absolute() else (REPO_ROOT / p)


load_env_file()
