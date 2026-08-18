"""Structural, thermodynamic and dynamical properties of reaction networks.

Three layers live here:

- **stoichiometry** — the matrix ``S``, its rank and kernels, and integer
  conservation laws;
- **CRNT** (:mod:`~synkit.CRN.Props.deficiency`) — complexes, linkage classes,
  weak reversibility, deficiency, and the Deficiency Zero and Deficiency One
  theorems;
- **thermodynamics and dynamics** — conservativity, consistency, symbolic
  Jacobians and structural singularity.

Every function here accepts a *CRN-like* input: a
:class:`~synkit.CRN.Structure.syncrn.SynCRN`, a species-reaction bipartite
NetworkX graph, or any object exposing ``to_digraph()``. Reaction nodes may be
spelled ``kind="reaction"`` or ``kind="rule"``; see :mod:`synkit.CRN.kinds`.

.. rubric:: Example

.. code-block:: python

    from synkit.CRN.Structure import SynCRN
    from synkit.CRN.Props import integer_conservation_laws, summary

    crn = SynCRN.from_reaction_strings(["A>>B", "B>>A"])
    print(summary(crn))
    print(integer_conservation_laws(crn))
"""

from __future__ import annotations

from .stoich import (
    StoichSummary,
    build_S,
    build_S_minus_plus,
    conserved_moieties,
    integer_conservation_laws,
    left_nullspace,
    left_right_kernels,
    right_nullspace,
    stoichiometric_matrix,
    stoichiometric_rank,
    summary,
)
from .deficiency import (
    Complex,
    CRNTSummary,
    complex_graph,
    complexes,
    crnt_summary,
    deficiency,
    deficiency_one_verdict,
    deficiency_zero_verdict,
    is_deficiency_one_applicable,
    is_deficiency_zero_applicable,
    is_reversible,
    is_weakly_reversible,
    linkage_class_deficiencies,
    linkage_classes,
    strong_linkage_classes,
    terminal_strong_linkage_classes,
)
from .thermo import (
    ThermoSummary,
    compute_conservativity,
    compute_thermo_summary,
    has_irreversible_futile_cycles,
    is_conservative,
    is_consistent,
)
from .dynamics import (
    StructuralSingularitySummary,
    jacobian_sign_pattern,
    jacobian_sparsity,
    species_influence_graph,
    structural_singularity_summary,
    symbolic_jacobian,
    symbolic_reactivity_matrix,
)

__all__ = [
    # stoichiometry
    "StoichSummary",
    "build_S",
    "build_S_minus_plus",
    "conserved_moieties",
    "integer_conservation_laws",
    "left_nullspace",
    "left_right_kernels",
    "right_nullspace",
    "stoichiometric_matrix",
    "stoichiometric_rank",
    "summary",
    # CRNT: complexes, linkage classes, deficiency
    "Complex",
    "CRNTSummary",
    "complex_graph",
    "complexes",
    "crnt_summary",
    "deficiency",
    "deficiency_one_verdict",
    "deficiency_zero_verdict",
    "is_deficiency_one_applicable",
    "is_deficiency_zero_applicable",
    "is_reversible",
    "is_weakly_reversible",
    "linkage_class_deficiencies",
    "linkage_classes",
    "strong_linkage_classes",
    "terminal_strong_linkage_classes",
    # thermodynamics
    "ThermoSummary",
    "compute_conservativity",
    "compute_thermo_summary",
    "has_irreversible_futile_cycles",
    "is_conservative",
    "is_consistent",
    # dynamics
    "StructuralSingularitySummary",
    "jacobian_sign_pattern",
    "jacobian_sparsity",
    "species_influence_graph",
    "structural_singularity_summary",
    "symbolic_jacobian",
    "symbolic_reactivity_matrix",
]
