"""Benchmark, validation and case-study material for :mod:`synkit.CRN`.

The package includes:

- **A validation set** (:mod:`~Experiment.CRN.benchmarks.networks`,
  :mod:`~Experiment.CRN.benchmarks.validate`) — literature and regression networks
  with the structural verdicts they must reproduce, each quantity additionally
  recomputed by an independent route from
  :mod:`~Experiment.CRN.benchmarks.crosschecks`.
- **A scaling benchmark** (:mod:`~Experiment.CRN.benchmarks.scaling`) — wall-clock
  timings of the analysis stack against network size.
- **A metabolic case study** (:mod:`~Experiment.CRN.benchmarks.kegg`) — cached KEGG
  modules processed from pathway retrieval through
  conservation laws, persistence and flux realizability.

.. rubric:: Example

.. code-block:: python

    from Experiment.CRN.benchmarks import run_validation, validation_table

    print(validation_table(run_validation()))
"""

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
from .validate import (
    NetworkValidation,
    ValidationReport,
    run_validation,
    validate_network,
    validation_table,
)

__all__ = [
    # networks
    "BENCHMARK_NETWORKS",
    "BenchmarkNetwork",
    "get_network",
    "network_names",
    # independent cross-checks
    "brute_force_siphons",
    "check_conservation_laws",
    "check_semiflows",
    "deficiency_by_rank_identity",
    "persistence_by_brute_force",
    "rank_by_svd",
    # scaling benchmark
    "NETWORK_FAMILIES",
    "ScalingRecord",
    "TASKS",
    "generate_network",
    "run_scaling_benchmark",
    "scaling_table",
    # KEGG case study
    "CASE_STUDY_MODULES",
    "GLYCOLYSIS_FLUX",
    "KeggModuleAnalysis",
    "analyze_kegg_module",
    "build_kegg_crn",
    "check_glycolysis_flux",
    "kegg_case_study_report",
    "load_kegg_cache",
    "module_ids",
    # validation
    "NetworkValidation",
    "ValidationReport",
    "run_validation",
    "validate_network",
    "validation_table",
]
