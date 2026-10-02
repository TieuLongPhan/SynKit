#!/usr/bin/env python3
"""Compatibility entry point for the installed optional Synister kernel builder."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synkit.Chem.Mapper.exact.native_build import build_native, main  # noqa: E402,F401

if __name__ == "__main__":
    main()
