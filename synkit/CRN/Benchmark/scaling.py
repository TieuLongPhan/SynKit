"""Deprecated checkout-only alias for :mod:`Experiment.CRN.benchmarks.scaling`."""

import importlib
import sys
import warnings

warnings.warn(
    "synkit.CRN.Benchmark.scaling moved to Experiment.CRN.benchmarks.scaling; "
    "research helpers require a source checkout",
    DeprecationWarning,
    stacklevel=2,
)
sys.modules[__name__] = importlib.import_module("Experiment.CRN.benchmarks.scaling")
