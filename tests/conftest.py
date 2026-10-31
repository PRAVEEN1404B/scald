"""Shared pytest configuration.

Adds the repository root to ``sys.path`` so the tests can import the project
modules even when the package has not been installed with ``pip install -e .``.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
