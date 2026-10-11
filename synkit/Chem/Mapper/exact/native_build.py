"""Compatibility entry point for :mod:`synkit.Chem.Mapper.exact.native.build`."""

import sys

from .native import build as _build


if __name__ == "__main__":
    _build.main()
else:
    # Keep import identity while preserving the historical ``python -m`` CLI.
    sys.modules[__name__] = _build
