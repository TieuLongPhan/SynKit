"""
synkit.Chem.Mapper
==================

Atom-to-atom mapping (AAM) with exact minimum or selected chemical distance.
The reaction-SMILES entry point is :func:`map_reaction`; graph inputs use
:func:`enumerate_pabs_mappings`. PABS (Propagation and Assignment-Bounded
Search) offers explicit Python and C++ backends. The Python implementation
uses reversible graph-assignment search; the optional C++ kernel performs
integer shell search. Neither PABS backend invokes MILP.

The approximate Weisfeiler-Lehman/SLAP mapper :class:`AAMapper` and the
kernel MILP solver remain available through their established interfaces.

Research basis
--------------
The approximate SLAP mapper develops the mapping idea from:
Shin-ichi Koda and Shinji Saito, "General and scalable atom-to-atom mapping
via Weisfeiler-Lehman-like approximate graph matching", ChemRxiv (2025).
DOI: 10.26434/chemrxiv-2025-hthwn

Quick start
-----------
>>> from synkit.Chem.Mapper import map_reaction
>>> result = map_reaction("CO>>CO", CD="minimal", time_limit_seconds=10)
>>> result.complete, result.minimum_cost
(True, 0.0)

Several inequivalent symmetry classes can share a minimum. Inspect
``result.mapped_reactions`` and ``result.complete`` separately. For the native
backend, explicitly build a library with ``exact.native_build.build_native``
and pass ``backend="cpp", library_path=library`` to the same interface.

Package layout
--------------
:mod:`synkit.Chem.Mapper.graph`
    Labeled-graph data structure, WL/2-WL refinement, automorphisms,
    block-cut-tree decomposition.
:mod:`synkit.Chem.Mapper.slap`
    Sequential-LAP engine (:class:`GraphMatcher`) and LAP utilities
    (chemical distance, Gilmore-Lawler lower bound).
:mod:`synkit.Chem.Mapper.exact`
    PABS graph search, kernelization, MILP/QAP solver, orbital branching,
    exhaustive DFS, symmetry-distinct enumeration and certificates.
    Optional C++ source lives in ``exact/native_distance.cpp``; explicit
    compilation is provided by ``exact.native_build`` and Python bindings
    by ``exact.native_candidates`` and the other ``exact.native_*`` modules.
    ``enumerate_pabs_mappings`` selects Python or C++ through one interface.
:mod:`synkit.Chem.Mapper.reaction_search`
    Unmapped reaction SMILES, exact symmetry class representatives,
    explicit or compressed hydrogen, and optional missing-atom padding.
:mod:`synkit.Chem.Mapper.chem`
    RDKit SMILES I/O and ITS-based deduplication / electron-balance checks.
:mod:`synkit.Chem.Mapper.io`
    Index-mapping-string helpers.
:mod:`synkit.Chem.Mapper.aam_validator`
    AAM validation against ground-truth mapped reaction SMILES.
:mod:`synkit.Chem.Mapper.analysis`, :mod:`synkit.Chem.Mapper.alternatives`
    Reference-blinded shells and exact ITS class exports.
:mod:`synkit.Chem.Mapper.native_analysis`
    Explicit native search with two-sided orbit aggregation.
:mod:`synkit.Chem.Mapper.prediction_adapter`, :mod:`synkit.Chem.Mapper.template_adapter`
    Prediction seeds, validated class correspondences, and executable rules.

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
from .template_adapter import (
    ClassCorrespondence,
    class_correspondence_from_export,
    SourceReplayResult,
    ProductRecoveryScore,
    correspondence_from_export,
    executable_rule_from_correspondence,
    mapped_reaction_from_correspondence,
    prospective_products,
    replay_executable_rule_on_source,
    score_product_recovery,
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
from .exact.propagation import PropagationConfig, enumerate_synister_cp_mappings
from .exact.search import enumerate_pabs_mappings
from .reaction_search import ReactionMapping, ReactionMappingResult, map_reaction

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
    "SourceReplayResult",
    "ClassCorrespondence",
    "class_correspondence_from_export",
    "ProductRecoveryScore",
    "correspondence_from_export",
    "executable_rule_from_correspondence",
    "mapped_reaction_from_correspondence",
    "prospective_products",
    "score_product_recovery",
    "replay_executable_rule_on_source",
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
    "PropagationConfig",
    "enumerate_synister_cp_mappings",
    "enumerate_pabs_mappings",
    "ReactionMapping",
    "ReactionMappingResult",
    "map_reaction",
    "RESEARCH_BASIS_TITLE",
    "RESEARCH_BASIS_DOI",
    "RESEARCH_BASIS_URL",
    "RESEARCH_BASIS_AUTHORS",
]
