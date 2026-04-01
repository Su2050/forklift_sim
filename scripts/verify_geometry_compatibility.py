#!/usr/bin/env python3
from pathlib import Path
import runpy


runpy.run_path(
    str(Path(__file__).resolve().parent / "validation" / "assets" / "verify_geometry_compatibility.py"),
    run_name="__main__",
)
