# from __future__ import annotations

# """
# High-level structural and dynamical properties for chemical reaction networks.

# This subpackage provides a CRN-agnostic API that works with either
# :class:`CRNNetwork` (legacy core representation) or
# :class:`CRNHyperGraph` (new hypergraph-based representation).

# Most public functions accept ``crn`` of type :data:`CRNLike` and internally
# convert to :class:`CRNNetwork` using
# :func:`synkit.CRN.Hypergraph.adapters.hypergraph_to_crnnetwork`.
# """

# from typing import Union

# from ..core import CRNNetwork
# from ..Hypergraph.hypergraph import CRNHyperGraph

# CRNLike = Union[CRNNetwork, CRNHyperGraph]

# # Re-export commonly used helpers
# from .stoich import (
#     stoichiometric_matrix,
#     stoichiometric_rank,
#     left_nullspace,
# )

# from .deficiency import (
#     DeficiencySummary,
#     compute_deficiency_summary,
#     deficiency_zero_theorem_applicable,
# )

# from .petri import (
#     find_p_semiflows,
#     find_t_semiflows,
#     find_siphons,
#     find_traps,
#     siphon_persistence_condition,
# )

# from .injectivity import (
#     build_species_reaction_graph,
#     count_sr_cycles,
#     is_sr_graph_acyclic,
# )

# from .thermo import (
#     ThermoSummary,
#     compute_thermo_summary,
# )

# from .dynamics import (
#     DynamicTheoremSummary,
#     summarize_dynamics,
# )

# __all__ = [
#     "CRNLike",
#     # stoich
#     "stoichiometric_matrix",
#     "stoichiometric_rank",
#     "left_nullspace",
#     # deficiency
#     "DeficiencySummary",
#     "compute_deficiency_summary",
#     "deficiency_zero_theorem_applicable",
#     # petri
#     "find_p_semiflows",
#     "find_t_semiflows",
#     "find_siphons",
#     "find_traps",
#     "siphon_persistence_condition",
#     # injectivity
#     "build_species_reaction_graph",
#     "count_sr_cycles",
#     "is_sr_graph_acyclic",
#     # thermo
#     "ThermoSummary",
#     "compute_thermo_summary",
#     # dynamics
#     "DynamicTheoremSummary",
#     "summarize_dynamics",
# ]
