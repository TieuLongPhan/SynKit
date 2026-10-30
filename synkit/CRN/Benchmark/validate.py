"""Deprecated checkout-only alias for :mod:`Experiment.CRN.benchmarks.validate`."""

import importlib
import sys
import warnings

warnings.warn(
    "synkit.CRN.Benchmark.validate moved to Experiment.CRN.benchmarks.validate; research helpers require a source checkout",
    DeprecationWarning,
    stacklevel=2,
)
sys.modules[__name__] = importlib.import_module("Experiment.CRN.benchmarks.validate")
