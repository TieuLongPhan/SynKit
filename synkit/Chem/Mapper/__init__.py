"""
synkit.Chem.Mapper
==================

An atom-to-atom mapping (AAM) toolkit built on Weisfeiler-Lehman label
refinement coupled with sequential linear assignment problems (SLAP). The
approximate SLAP result can be refined into a *certified* optimum on the
ambiguous reaction-center kernel using the
:mod:`synkit.Chem.Mapper.exact` solvers.

Research basis
--------------
This enhanced mapper develops the SLAP mapping idea from:
Shin-ichi Koda and Shinji Saito, "General and scalable atom-to-atom mapping
via Weisfeiler-Lehman-like approximate graph matching", ChemRxiv (2025).
DOI: 10.26434/chemrxiv-2025-hthwn

Quick start
-----------
>>> from synkit.Chem.Mapper import AAMapper, AAMValidator
>>> m = AAMapper(binary=True)
>>> m.map_smiles("CC(=O)O.OC>>CC(=O)OC")
>>> print(m.results[0]["smiles"])
>>> validator = AAMValidator()

Package layout
--------------
:mod:`synkit.Chem.Mapper.graph`
    Labeled-graph data structure, WL/2-WL refinement, automorphisms,
    block-cut-tree decomposition.
:mod:`synkit.Chem.Mapper.slap`
    Sequential-LAP engine (:class:`GraphMatcher`) and LAP utilities
    (chemical distance, Gilmore-Lawler lower bound).
:mod:`synkit.Chem.Mapper.exact`
    Kernelization, MILP/QAP solver, orbital-branching solver, exhaustive
    DFS mapper, symmetry-distinct enumeration, and optimality certificates.
:mod:`synkit.Chem.Mapper.chem`
    RDKit SMILES I/O and ITS-based deduplication / electron-balance checks.
:mod:`synkit.Chem.Mapper.io`
    Index-mapping-string helpers.
:mod:`synkit.Chem.Mapper.aam_validator`
    AAM validation against ground-truth mapped reaction SMILES.

Public API
----------
"""

from .graph.labeled_graph import LabeledGraph
from .slap.sequential import GraphMatcher
from .chem.aam import AAMapper
from .chem.blind import BlindedReactionProblem, blinded_mapped_reaction_problem
from .aam_validator import AAMValidator
from .analysis import (
    GlobalShellAnalysisResult,
    GlobalShellConfig,
    ReactionCenterSpectrum,
    analyze_reference_blinded_global_shell,
)
from .spectrum import (
    ExactStructureSpectrum,
    ExactStructureSpectrumAccumulator,
    exact_its_and_template_codes,
)
from .alternatives import (
    AlternativeTarget,
    ExactITSAlternative,
    ExactITSAlternativeResult,
    MappedReactionITSAlternativeResult,
    SeedMode,
    enumerate_exact_its_alternatives,
    enumerate_mapped_reaction_its_alternatives,
)
from .exact.distance import (
    CertificateVerificationError,
    DistanceEnumerationCertificate,
    DistanceEnumerationResult,
    ExactEnumerationLimitError,
    enumerate_distance_mappings,
    verify_distance_enumeration_certificate,
)
from .exact.core import (
    ITSComponentReduction,
    reduce_to_reference_its_components,
)
from .exact.hydrogen import (
    HydrogenEnumerationResult,
    HydrogenTransferPlan,
    enumerate_lgp_hydrogen_transfers,
    enumerate_minimal_hydrogen_transfers,
)
from .exact.edit_support import (
    BinaryEditBudget,
    BinaryEditSupportResult,
    binary_edit_budget,
    enumerate_binary_edit_support_mappings,
)
from .exact.hybrid import (
    HybridBackend,
    HybridBackendDecision,
    enumerate_hybrid_distance_mappings,
)

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
    "BlindedReactionProblem",
    "blinded_mapped_reaction_problem",
    "GlobalShellAnalysisResult",
    "GlobalShellConfig",
    "ReactionCenterSpectrum",
    "analyze_reference_blinded_global_shell",
    "ExactStructureSpectrum",
    "ExactStructureSpectrumAccumulator",
    "exact_its_and_template_codes",
    "AlternativeTarget",
    "ExactITSAlternative",
    "ExactITSAlternativeResult",
    "MappedReactionITSAlternativeResult",
    "SeedMode",
    "enumerate_exact_its_alternatives",
    "enumerate_mapped_reaction_its_alternatives",
    "CertificateVerificationError",
    "DistanceEnumerationCertificate",
    "DistanceEnumerationResult",
    "ExactEnumerationLimitError",
    "enumerate_distance_mappings",
    "verify_distance_enumeration_certificate",
    "ITSComponentReduction",
    "reduce_to_reference_its_components",
    "HydrogenEnumerationResult",
    "HydrogenTransferPlan",
    "enumerate_lgp_hydrogen_transfers",
    "enumerate_minimal_hydrogen_transfers",
    "BinaryEditBudget",
    "BinaryEditSupportResult",
    "binary_edit_budget",
    "enumerate_binary_edit_support_mappings",
    "HybridBackend",
    "HybridBackendDecision",
    "enumerate_hybrid_distance_mappings",
    "RESEARCH_BASIS_TITLE",
    "RESEARCH_BASIS_DOI",
    "RESEARCH_BASIS_URL",
    "RESEARCH_BASIS_AUTHORS",
]
