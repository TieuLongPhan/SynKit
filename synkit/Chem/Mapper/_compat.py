"""Historical module names for the reorganized mapper implementation.

Aliases share module objects, so monkeypatching and classes retain their
identity across old and new imports. Internal code uses canonical paths.
"""

from importlib import import_module
import sys


MAPPER_MODULE_ALIASES = {
    "analysis": "reporting.analysis",
    "native_analysis": "reporting.native_analysis",
    "spectrum": "reporting.spectrum",
    "alternatives": "reporting.alternatives",
    "prediction_adapter": "adapters.prediction",
    "template_adapter": "adapters.template",
    "aam_validator": "validation.aam_validator",
    "identifiability": "validation.identifiability",
    "evaluation": "validation.bonds",
    "annotation_evaluation": "validation.annotations",
    "orbit_evaluation": "validation.orbits",
    "cohort_evaluation": "validation.cohorts",
    "numerical_scope": "validation.numerical_scope",
}

EXACT_MODULE_ALIASES = {
    "distance": "distance_search.search",
    "distance_bounds": "distance_search.bounds",
    "distance_inputs": "distance_search.inputs",
    "distance_records": "distance_search.records",
    "distance_verify": "distance_search.verify",
    "assignment_certificate": "assignment.assignment_certificate",
    "assignment_domains": "assignment.assignment_domains",
    "assignment_verification": "assignment.assignment_verification",
    "incremental_assignment": "assignment.incremental_assignment",
    "kernel": "kernel_solvers.kernel",
    "branching": "kernel_solvers.branching",
    "exhaustive": "distance_search.exhaustive",
    "milp": "kernel_solvers.milp",
    "enumerate": "kernel_solvers.enumerate",
    "enumeration_result": "kernel_solvers.enumeration_result",
    "propagation": "pabs.propagation",
    "propagation_search": "pabs.propagation_search",
    "propagation_bounds": "pabs.propagation_bounds",
    "propagation_limits": "pabs.propagation_limits",
    "propagation_symmetry": "pabs.propagation_symmetry",
    "multi_shell": "pabs.multi_shell",
    "suffix_spectrum": "pabs.suffix_spectrum",
    "factor_spectrum": "pabs.factor_spectrum",
    "separator_spectrum": "pabs.separator_spectrum",
    "separator_bound": "pabs.separator_bound",
    "cost_lattice": "pabs.cost_lattice",
    "residual_typed_bonds": "pabs.residual_typed_bonds",
    "reward_frontier": "pabs.reward_frontier",
    "sparse_reward_bound": "pabs.sparse_reward_bound",
    "seed": "seeds.seed",
    "seed_fragments": "seeds.seed_fragments",
    "seed_relaxation": "seeds.seed_relaxation",
    "native_candidates": "native.candidates",
    "native_canonical": "native.canonical",
    "native_frontier": "native.frontier",
    "native_its": "native.its",
    "native_parallel": "native.parallel",
    "native_search": "native.search",
    "orbit_aggregation": "native.orbit_aggregation",
    "disk_frontier": "native.disk_frontier",
}


def register_module_aliases(package_name, aliases):
    """Bind historical children to canonical modules on an existing package."""
    package = sys.modules[package_name]
    for old_name, canonical_name in aliases.items():
        module = import_module(f".{canonical_name}", package_name)
        sys.modules[f"{package_name}.{old_name}"] = module
        setattr(package, old_name, module)
