#!/usr/bin/env python3
"""Compatibility wrapper — see mpi/plots.py."""

from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parent / "mpi" / "plots.py"), run_name="__main__")
