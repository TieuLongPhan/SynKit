"""Petri-net view of a reaction network: incidence, invariants, persistence.

A CRN maps onto a Petri net by reading species as places and reactions as
transitions. This subpackage exposes that view together with the standard
structural diagnostics computed on it.

Semiflows here are genuine semiflows: minimal, non-negative and integer-valued
(:mod:`synkit.CRN.Petrinet.minimal_semiflows`). If you want a plain kernel
basis instead, ask for :func:`left_kernel_basis` or :func:`right_kernel_basis`
explicitly.

.. rubric:: Example

.. code-block:: python

    from synkit.CRN.Structure import SynCRN
    from synkit.CRN.Petrinet import PetriAnalyzer

    crn = SynCRN.from_reaction_strings(["A>>B", "B>>A"])
    print(PetriAnalyzer(crn, max_siphon_size=4).compute_all().explain())
"""

from __future__ import annotations

from .net import (
    Marking,
    Multiset,
    PetriNet,
    Place,
    SynCRNIncidence,
    Transition,
    TransitionId,
    extract_syncrn_incidence,
)
from .minimal_semiflows import (
    is_semiflow,
    minimal_semiflow_supports,
    minimal_semiflows,
)
from .semiflows import (
    find_p_semiflows,
    find_t_semiflows,
    left_kernel_basis,
    right_kernel_basis,
    semiflow_supports,
    stoichiometric_matrix,
)
from .structure import (
    find_siphons,
    find_traps,
    species_transition_neighborhoods,
)
from .persistence import (
    PersistenceCheckResult,
    siphon_persistence_condition,
    siphon_persistence_details,
)
from .analyzer import PetriAnalyzer, PetriSummary

__all__ = [
    # net
    "Marking",
    "Multiset",
    "PetriNet",
    "Place",
    "SynCRNIncidence",
    "Transition",
    "TransitionId",
    "extract_syncrn_incidence",
    # invariants
    "find_p_semiflows",
    "find_t_semiflows",
    "semiflow_supports",
    "minimal_semiflows",
    "minimal_semiflow_supports",
    "is_semiflow",
    "left_kernel_basis",
    "right_kernel_basis",
    "stoichiometric_matrix",
    # structure
    "find_siphons",
    "find_traps",
    "species_transition_neighborhoods",
    # persistence
    "PersistenceCheckResult",
    "siphon_persistence_condition",
    "siphon_persistence_details",
    # analysis
    "PetriAnalyzer",
    "PetriSummary",
]
