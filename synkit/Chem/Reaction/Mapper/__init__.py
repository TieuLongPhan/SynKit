"""
synkit.Chem.Reaction.Mapper
===========================

An atom-to-atom mapping (AAM) toolkit built on Weisfeiler-Lehman label
refinement coupled with sequential linear assignment problems (SLAP). The
approximate SLAP result can be refined into a *certified* optimum on the
ambiguous reaction-center kernel using the
:mod:`synkit.Chem.Reaction.Mapper.exact` solvers.

Research basis
--------------
This enhanced mapper develops the SLAP mapping idea from:
Shin-ichi Koda and Shinji Saito, "General and scalable atom-to-atom mapping
via Weisfeiler-Lehman-like approximate graph matching", ChemRxiv (2025).
DOI: 10.26434/chemrxiv-2025-hthwn

Quick start
-----------
>>> from synkit.Chem.Reaction.Mapper import AAMapper, AAMValidator
>>> m = AAMapper(binary=True)
>>> m.map_smiles("CC(=O)O.OC>>CC(=O)OC")
>>> print(m.results[0]["smiles"])
>>> validator = AAMValidator()

Package layout
--------------
:mod:`synkit.Chem.Reaction.Mapper.graph`
    Labeled-graph data structure, WL/2-WL refinement, automorphisms,
    block-cut-tree decomposition.
:mod:`synkit.Chem.Reaction.Mapper.slap`
    Sequential-LAP engine (:class:`GraphMatcher`) and LAP utilities
    (chemical distance, Gilmore-Lawler lower bound).
:mod:`synkit.Chem.Reaction.Mapper.exact`
    Kernelization, MILP/QAP solver, orbital-branching solver, exhaustive
    DFS mapper, symmetry-distinct enumeration, and optimality certificates.
:mod:`synkit.Chem.Reaction.Mapper.chem`
    RDKit SMILES I/O and ITS-based deduplication / electron-balance checks.
:mod:`synkit.Chem.Reaction.Mapper.io`
    Index-mapping-string helpers.
:mod:`synkit.Chem.Reaction.Mapper.aam_validator`
    AAM validation against ground-truth mapped reaction SMILES.

Public API
----------
"""

from .graph.labeled_graph import LabeledGraph
from .slap.sequential import GraphMatcher
from .chem.aam import AAMapper
from .aam_validator import AAMValidator

RESEARCH_BASIS_TITLE = (
    "General and scalable atom-to-atom mapping via "
    "Weisfeiler-Lehman-like approximate graph matching"
)
RESEARCH_BASIS_DOI = "10.26434/chemrxiv-2025-hthwn"
RESEARCH_BASIS_URL = f"https://doi.org/{RESEARCH_BASIS_DOI}"
RESEARCH_BASIS_AUTHORS = "Shin-ichi Koda and Shinji Saito"

__all__ = [
    "LabeledGraph",
    "GraphMatcher",
    "AAMapper",
    "AAMValidator",
    "RESEARCH_BASIS_TITLE",
    "RESEARCH_BASIS_DOI",
    "RESEARCH_BASIS_URL",
    "RESEARCH_BASIS_AUTHORS",
]
