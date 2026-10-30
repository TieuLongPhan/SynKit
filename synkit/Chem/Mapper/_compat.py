"""Historical module names for the reorganized mapper implementation.

Aliases share module objects, so monkeypatching and classes retain their
identity across old and new imports. Internal code uses canonical paths.
"""

from importlib import import_module
from importlib.abc import Loader, MetaPathFinder
from importlib.util import find_spec, spec_from_loader
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


_SOLVER_PREFIXES = {
    f"{namespace}.{old}": f"synkit.Chem.Mapper.solvers.{new}"
    for namespace in ("synkit.Chem.Mapper", "synkit.Chem.Reaction.Mapper")
    for old, new in (("exact", "exact"), ("slap", "slap"), ("_native", "native"))
}


class _SolverAliasLoader(Loader):
    """Resolve a historical solver module without creating a second implementation."""

    def __init__(self, canonical_name):
        self.canonical_name = canonical_name

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        sys.modules[module.__spec__.name] = import_module(self.canonical_name)

    def get_code(self, fullname):
        # runpy requests code without importing the alias. Call the canonical
        # builder's main directly so its relative imports keep their new base.
        if self.canonical_name.endswith((".build", ".native_build", ".reaction_build")):
            source = f"from importlib import import_module\nimport_module({self.canonical_name!r}).main()\n"
        else:
            source = f"import runpy\nrunpy.run_module({self.canonical_name!r}, run_name='__main__')\n"
        return compile(source, f"<compatibility {fullname}>", "exec")


class _SolverAliasFinder(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        for old, canonical in _SOLVER_PREFIXES.items():
            if fullname == old or fullname.startswith(old + "."):
                name = canonical + fullname[len(old):]
                original = find_spec(name)
                if original is None:
                    return None
                return spec_from_loader(
                    fullname, _SolverAliasLoader(name),
                    origin=original.origin,
                    is_package=original.submodule_search_locations is not None,
                )
        return None


def install_solver_aliases():
    """Keep old solver imports and build commands working after the source move."""
    if not any(isinstance(finder, _SolverAliasFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _SolverAliasFinder())


def register_module_aliases(package_name, aliases):
    """Bind historical children to canonical modules on an existing package."""
    package = sys.modules[package_name]
    for old_name, canonical_name in aliases.items():
        module = import_module(f".{canonical_name}", package_name)
        sys.modules[f"{package_name}.{old_name}"] = module
        setattr(package, old_name, module)
