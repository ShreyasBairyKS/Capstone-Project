"""
tests/conftest.py — Session-level fixture that sets environment variables
BEFORE any test module imports the FastAPI app.

pytest imports conftest.py files from the outermost directory inward, so this
file runs before any test_*.py module-level code. This guarantees that
core.config.settings sees the correct values when it is first instantiated.
"""

from __future__ import annotations

import os

# Must be set before `from api.main import app` in any test module.
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["API_KEY"] = "test-key"
