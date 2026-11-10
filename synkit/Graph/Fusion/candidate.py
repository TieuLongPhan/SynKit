"""Compatibility import for :mod:`synkit.Rule.Fusion.candidate`."""

from importlib import import_module as _import_module
import sys as _sys

_sys.modules[__name__] = _import_module("synkit.Rule.Fusion.candidate")
