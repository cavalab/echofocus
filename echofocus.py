"""Compatibility wrapper for legacy `python echofocus.py ...` usage."""

from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


_SRC_PKG = Path(__file__).resolve().parent / "src" / "echofocus"
_PKG_NAME = "_echofocus_src"


def _ensure_loaded_package():
    if _PKG_NAME in sys.modules:
        return sys.modules[_PKG_NAME]

    init_path = _SRC_PKG / "__init__.py"
    spec = spec_from_file_location(
        _PKG_NAME,
        init_path,
        submodule_search_locations=[str(_SRC_PKG)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load package from {init_path}")
    module = module_from_spec(spec)
    sys.modules[_PKG_NAME] = module
    spec.loader.exec_module(module)
    return module


_ensure_loaded_package()
EchoFocus = import_module(f"{_PKG_NAME}.echofocus").EchoFocus
main = import_module(f"{_PKG_NAME}.cli").main

__all__ = ["EchoFocus", "main"]


if __name__ == "__main__":
    main()
