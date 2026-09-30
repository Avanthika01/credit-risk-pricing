"""Import helpers from scripts/ (not a package) for testing."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "download_data", Path(__file__).resolve().parents[1] / "scripts" / "download_data.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
pick_accepted_file = _mod.pick_accepted_file
