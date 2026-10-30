"""Deprecated checkout-only alias for :mod:`Experiment.CRN.benchmarks.kegg`."""

import importlib
import sys
import warnings

warnings.warn(
    "synkit.CRN.Benchmark.kegg moved to Experiment.CRN.benchmarks.kegg; research helpers require a source checkout",
    DeprecationWarning,
    stacklevel=2,
)
sys.modules[__name__] = importlib.import_module("Experiment.CRN.benchmarks.kegg")
