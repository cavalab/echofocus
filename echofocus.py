"""Compatibility wrapper for legacy `python echofocus.py ...` usage."""

from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from echofocus.app import EchoFocus
from echofocus.cli import main

__all__ = ["EchoFocus", "main"]


if __name__ == "__main__":
    main()
