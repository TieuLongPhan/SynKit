# synkit/CRN/petri/__init__.py
"""
Petri net style structural properties for CRNs.

This subpackage provides small, focused helpers to compute classic Petri net
objects for chemical reaction networks:

* PetriNet container (:mod:`synkit.CRN.petri.net`),
* P-semiflows and T-semiflows (:mod:`synkit.CRN.petri.semiflows`),
* siphons and traps (:mod:`synkit.CRN.petri.structure`),
* a siphon-based persistence condition
  (:mod:`synkit.CRN.petri.persistence`),
* an OOP wrapper :class:`PetriAnalyzer`
  (:mod:`synkit.CRN.petri.analyzer`).
"""

from .net import PetriNet, Transition, Place, TransitionId, Marking, Multiset
from .semiflows import find_p_semiflows, find_t_semiflows
from .structure import find_siphons, find_traps
from .persistence import siphon_persistence_condition
from .analyzer import PetriAnalyzer, PetriSummary

__all__ = [
    # net
    "PetriNet",
    "Transition",
    "Place",
    "TransitionId",
    "Marking",
    "Multiset",
    # semiflows
    "find_p_semiflows",
    "find_t_semiflows",
    # structure
    "find_siphons",
    "find_traps",
    # persistence
    "siphon_persistence_condition",
    # analyzer
    "PetriAnalyzer",
    "PetriSummary",
]
