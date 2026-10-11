"""
Sequential linear-assignment (SLAP) matching and LAP-based utilities.

LAP means linear assignment problem: match reactant atoms to product atoms
using a matrix of pair costs. SLAP combines successive LAP decisions with
WL graph-label refinement to construct heuristic graph mappings. It is the
default solver used by ``AAMapper.map_smiles``. The utility functions here
also serve exact solvers; importing a LAP helper does not select the SLAP
matching workflow.

Submodules
----------
sequential
    :class:`GraphMatcher`: iterative WL-label refinement with LAP branching.
lap
    LAP utilities, :func:`chemical_distance`, :func:`dual_lap_lower_bound`,
    and :func:`recover_mapping`.
"""

from .sequential import GraphMatcher
from .lap import solve_lap, recover_mapping, chemical_distance, dual_lap_lower_bound

__all__ = [
    "GraphMatcher",
    "solve_lap",
    "recover_mapping",
    "chemical_distance",
    "dual_lap_lower_bound",
]
