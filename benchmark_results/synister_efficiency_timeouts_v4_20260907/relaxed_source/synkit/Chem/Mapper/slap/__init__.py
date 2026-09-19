"""
mapper.slap — sequential LAP engine and LAP-based utilities.

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
