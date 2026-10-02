"""Chemical reaction networks: construction, representation and analysis.

``synkit.CRN`` covers the whole chain from chemistry to structural verdict:

- **Construct** — expand a rule set over a seed pool into a reaction network.
- **Query** — retrieve and curate KEGG-derived pathway data.
- **Structure** — normalize any of the above into a :class:`SynCRN`, the
  canonical object every other subpackage consumes.
- **Props** — stoichiometry, conservation laws, thermodynamic and dynamical
  summaries.
- **Petrinet** — Petri-net view: minimal semiflows, siphons, traps, persistence.
- **Pathway** — reachability, path finding, flow realizability.
- **Symmetry** — canonical forms, automorphisms and isomorphism of networks.
- **Visualize** — layout and drawing.
- **IO** — SBML import and export for interoperability with the wider CRN
  ecosystem.

The names re-exported here are the stable public API. Anything reached by a
deeper import path may move between releases.

.. rubric:: Example

.. code-block:: python

    from synkit.CRN import SynCRN, PetriAnalyzer, integer_conservation_laws

    crn = SynCRN.from_reaction_strings(["A>>B", "B>>A", "C>>D", "D>>C"])

    print(crn)
    print(integer_conservation_laws(crn))
    print(PetriAnalyzer(crn, max_siphon_size=4).compute_all().summary)
"""

from __future__ import annotations

from .kinds import (
    ALL_KINDS,
    REACTION_KINDS,
    SPECIES_KINDS,
    is_reaction_node,
    is_species_node,
    node_kind,
)
from .Structure import Reaction, RXNSide, Rule, Species, SynCRN
from .Construct import (
    ConstructionStrategy,
    CRNExpand,
    DerivationLog,
    DerivationRecord,
    DerivationState,
    FrontierStrategy,
    ReactionDeltaFlattener,
    build_crn_from_smarts,
)
from .Props import (
    Complex,
    CRNTSummary,
    StoichSummary,
    ThermoSummary,
    complex_graph,
    complexes,
    compute_thermo_summary,
    conserved_moieties,
    crnt_summary,
    deficiency,
    deficiency_one_verdict,
    deficiency_zero_verdict,
    integer_conservation_laws,
    is_deficiency_one_applicable,
    is_deficiency_zero_applicable,
    is_reversible,
    is_weakly_reversible,
    left_nullspace,
    linkage_class_deficiencies,
    linkage_classes,
    right_nullspace,
    stoichiometric_matrix,
    stoichiometric_rank,
    strong_linkage_classes,
    summary,
    terminal_strong_linkage_classes,
)
from .Petrinet import (
    PersistenceCheckResult,
    PetriAnalyzer,
    PetriNet,
    PetriSummary,
    find_p_semiflows,
    find_siphons,
    find_t_semiflows,
    find_traps,
    minimal_semiflows,
    semiflow_supports,
    siphon_persistence_condition,
    siphon_persistence_details,
)
from .Pathway import (
    PathwayFinder,
    PathwayReachability,
    PathwayRealizability,
)
from .Symmetry import (
    CRNCanonicalizer,
    CRNIsomorphism,
    CRNSymmetry,
    SymmetryConfig,
    are_isomorphic,
    canonical,
)
from .Visualize import CRNStyle, CRNVis, draw_crn
from .IO import crn_from_sbml, crn_to_sbml, read_sbml, write_sbml

__all__ = [
    # node-kind vocabulary
    "ALL_KINDS",
    "REACTION_KINDS",
    "SPECIES_KINDS",
    "is_reaction_node",
    "is_species_node",
    "node_kind",
    # structure
    "SynCRN",
    "Species",
    "Reaction",
    "RXNSide",
    "Rule",
    # construction
    "CRNExpand",
    "build_crn_from_smarts",
    "ConstructionStrategy",
    "FrontierStrategy",
    "DerivationState",
    "DerivationRecord",
    "DerivationLog",
    "ReactionDeltaFlattener",
    # properties
    "StoichSummary",
    "ThermoSummary",
    "compute_thermo_summary",
    "conserved_moieties",
    "integer_conservation_laws",
    "left_nullspace",
    "right_nullspace",
    "stoichiometric_matrix",
    "stoichiometric_rank",
    "summary",
    # CRNT
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
    # petri net
    "PetriNet",
    "PetriAnalyzer",
    "PetriSummary",
    "PersistenceCheckResult",
    "find_p_semiflows",
    "find_t_semiflows",
    "minimal_semiflows",
    "semiflow_supports",
    "find_siphons",
    "find_traps",
    "siphon_persistence_condition",
    "siphon_persistence_details",
    # pathway
    "PathwayReachability",
    "PathwayRealizability",
    "PathwayFinder",
    # symmetry
    "CRNCanonicalizer",
    "CRNIsomorphism",
    "CRNSymmetry",
    "SymmetryConfig",
    "are_isomorphic",
    "canonical",
    # visualization
    "CRNStyle",
    "CRNVis",
    "draw_crn",
    # interchange
    "crn_from_sbml",
    "crn_to_sbml",
    "read_sbml",
    "write_sbml",
]
