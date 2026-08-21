"""Shared pytest fixtures for the wtools test suite."""

import sys
from pathlib import Path

import pytest

# Ensure the project root is on sys.path so that `wtools` and `tools` are
# importable when running pytest from the repo root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
