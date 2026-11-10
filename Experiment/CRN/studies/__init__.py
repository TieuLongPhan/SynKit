"""Shared inputs, reference checks, and experiment drivers for chemical reaction networks."""

from __future__ import annotations

from .networks import (
    BENCHMARK_NETWORKS,
    BenchmarkNetwork,
    get_network,
    network_names,
)
from .crosschecks import (
    brute_force_siphons,
    check_conservation_laws,
    check_semiflows,
    deficiency_by_rank_identity,
    persistence_by_brute_force,
    rank_by_svd,
)
from .scaling import (
    NETWORK_FAMILIES,
    ScalingRecord,
    TASKS,
    generate_network,
    run_scaling_benchmark,
    scaling_table,
)
from .kegg import (
    CASE_STUDY_MODULES,
    GLYCOLYSIS_FLUX,
    KeggModuleAnalysis,
    analyze_kegg_module,
    build_kegg_crn,
    check_glycolysis_flux,
    kegg_case_study_report,
    load_kegg_cache,
    module_ids,
)
from .validation import (
    NetworkValidation,
    ValidationReport,
    run_validation,
    validate_network,
    validation_table,
)

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
