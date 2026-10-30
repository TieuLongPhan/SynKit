"""Deprecated checkout-only alias for :mod:`Experiment.CRN.benchmarks.crosschecks`."""

import importlib
import sys
import warnings

warnings.warn(
    "synkit.CRN.Benchmark.crosschecks moved to Experiment.CRN.benchmarks.crosschecks; "
    "research helpers require a source checkout",
    DeprecationWarning,
    stacklevel=2,
)
sys.modules[__name__] = importlib.import_module("Experiment.CRN.benchmarks.crosschecks")
