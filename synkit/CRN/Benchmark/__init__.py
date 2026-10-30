"""Deprecated checkout-only research imports.

Implementations live in Experiment.CRN.benchmarks.
This namespace remains temporarily for existing notebooks and manuscript code.
Runtime CRN analysis never imports it. New code should use the experiment paths.
"""

from importlib import import_module
import warnings

__all__ = [
    "BENCHMARK_NETWORKS",
    "BenchmarkNetwork",
    "get_network",
    "network_names",
    "brute_force_siphons",
    "check_conservation_laws",
    "check_semiflows",
    "deficiency_by_rank_identity",
    "persistence_by_brute_force",
    "rank_by_svd",
    "NETWORK_FAMILIES",
    "ScalingRecord",
    "TASKS",
    "generate_network",
    "run_scaling_benchmark",
    "scaling_table",
    "CASE_STUDY_MODULES",
    "GLYCOLYSIS_FLUX",
    "KeggModuleAnalysis",
    "analyze_kegg_module",
    "build_kegg_crn",
    "check_glycolysis_flux",
    "kegg_case_study_report",
    "load_kegg_cache",
    "module_ids",
    "NetworkValidation",
    "ValidationReport",
    "run_validation",
    "validate_network",
    "validation_table",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    warnings.warn(
        "synkit.CRN.Benchmark moved to Experiment.CRN.benchmarks; research helpers require a source checkout",
        DeprecationWarning,
        stacklevel=2,
    )
    return getattr(import_module("Experiment.CRN.benchmarks"), name)
