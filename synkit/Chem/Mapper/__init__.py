"""
synkit.Chem.Mapper
==================

Atom-to-atom mapping offers several explicit workflows:

* ``Mapper(method='propagate').map(...)`` provides one interface and result
  type for eight named methods. ``Mapper.methods()`` lists their names.
* ``map_reaction(...)`` defaults to full-space Python exact propagation search.
  ``backend='cpp'`` selects the optional compiled integer-shell search.
* ``AAMapper().map_smiles(...)`` defaults to the approximate WL/SLAP mapper.
* ``AAMapper().map_smiles(..., enumerate_exact=True)`` runs SLAP followed by
  exact refinement of a reduced uncertainty kernel.
* ``AAMapper().map_smiles(..., certify='exact')`` requests optional kernel MILP
  certification of the SLAP results.
* ``AAMapper().enumerate_smiles(...)`` or ``map_smiles(..., CD=...)`` runs the
  established full-distance search; SLAP may provide a seed for a minimum query.

PABS means Propagation and Assignment-Bounded Search. It is the package's
descriptive name for the exact interface used by ``map_reaction`` and
``enumerate_pabs_mappings``. Its Python solver combines one-to-one domain
propagation with LAP bounds; its C++ backend uses a separate traversal.
LAP means linear assignment problem. SLAP uses sequential LAP decisions for
heuristic matching. A kernel is the reduced subproblem extracted from SLAP
results for optional refinement; optimality within it is scoped to its fixed
assignments and candidate domains. The PABS routes search without extracting
this uncertainty kernel or invoking its MILP solver.

Research basis
--------------
The approximate SLAP mapper develops the mapping idea from:
Shin-ichi Koda and Shinji Saito, "General and scalable atom-to-atom mapping
via Weisfeiler-Lehman-like approximate graph matching", ChemRxiv (2025).
DOI: 10.26434/chemrxiv-2025-hthwn

Quick start
-----------
>>> from synkit.Chem.Mapper import Mapper
>>> result = Mapper(method="propagate").map("CO>>CO")
>>> result.complete, result.minimum_cost
(True, 0.0)

Several inequivalent symmetry classes can share a minimum. Inspect
``result.mapped_reactions`` and ``result.complete`` separately. For the native
backend, explicitly build a library with ``exact.native.build.build_native``
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
    Distinct exact methods live in ``distance_search/``, ``kernel_solvers/``,
    ``pabs/``, and ``native/``. Shared LAP proofs live in ``assignment/`` and
    feasible-incumbent heuristics in ``seeds/``. ``enumerate_pabs_mappings``
    selects Python or C++ through one interface. The C++ source and explicit
    builder are adjacent in ``exact/native/``.
:mod:`synkit.Chem.Mapper.reaction_search`
    Unmapped reaction SMILES, exact symmetry class representatives,
    explicit or compressed hydrogen, and optional missing-atom padding.
:mod:`synkit.Chem.Mapper.chem`
    RDKit SMILES I/O and ITS-based deduplication / electron-balance checks.
:mod:`synkit.Chem.Mapper.io`
    Index-mapping-string helpers.
:mod:`synkit.Chem.Mapper.validation`
    AAM validation, identifiability, and bond/annotation/cohort evaluation.
:mod:`synkit.Chem.Mapper.reporting`
    Reference-blinded shell analysis, ITS spectra, structural alternatives,
    and native two-sided orbit reports.
:mod:`synkit.Chem.Mapper.adapters`
    Prediction seeds, validated class correspondences, and executable rules.

Historical flat module imports remain aliases to the same implementation
objects. See ``README.rst`` beside this module for method scopes and examples.

Public API
----------
"""

from ._compat import (
    MAPPER_MODULE_ALIASES as _MAPPER_MODULE_ALIASES,
    register_module_aliases as _register_module_aliases,
)
from .graph.labeled_graph import LabeledGraph
from .slap.sequential import GraphMatcher
from .chem.aam import AAMapper
from .chem.blind import BlindedReactionProblem, blinded_mapped_reaction_problem
from .validation.aam_validator import AAMValidator
from .reporting.analysis import (
    GlobalShellAnalysisResult,
    GlobalShellConfig,
    ReactionCenterSpectrum,
    analyze_reference_blinded_global_shell,
)
from .reporting.spectrum import (
    ExactStructureSpectrum,
    ExactStructureSpectrumAccumulator,
    exact_its_and_template_codes,
)
from .reporting.alternatives import (
    AlternativeTarget,
    ExactITSAlternative,
    ExactITSAlternativeResult,
    MappedReactionITSAlternativeResult,
    SeedMode,
    enumerate_exact_its_alternatives,
    enumerate_mapped_reaction_its_alternatives,
)
from .adapters.template import (
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
from .exact.distance_search.search import (
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
from .exact.pabs.propagation import PropagationConfig, enumerate_synister_cp_mappings
from .exact.search import enumerate_pabs_mappings
from .reaction_search import ReactionMapping, ReactionMappingResult, map_reaction
from .mapper import Mapper, MappingResult

RESEARCH_BASIS_TITLE = (
    "General and scalable atom-to-atom mapping via "
    "Weisfeiler-Lehman-like approximate graph matching"
)
RESEARCH_BASIS_DOI = "10.26434/chemrxiv-2025-hthwn"
RESEARCH_BASIS_URL = f"https://doi.org/{RESEARCH_BASIS_DOI}"
RESEARCH_BASIS_AUTHORS = "Shin-ichi Koda and Shinji Saito"

__all__ = [
    "Mapper",
    "MappingResult",
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

# Preserve flat imports while keeping implementation files in focused packages.
_register_module_aliases(__name__, _MAPPER_MODULE_ALIASES)
