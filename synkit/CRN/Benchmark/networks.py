"""Deprecated checkout-only alias for :mod:`Experiment.CRN.benchmarks.networks`."""

import importlib
import sys
import warnings

warnings.warn(
    "synkit.CRN.Benchmark.networks moved to Experiment.CRN.benchmarks.networks; research helpers require a source checkout",
    DeprecationWarning,
    stacklevel=2,
)
sys.modules[__name__] = importlib.import_module("Experiment.CRN.benchmarks.networks")
