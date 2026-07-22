"""Typed validation contracts for wildcard-based reaction fusion.

The validator in this module is deliberately independent of a particular
fusion search strategy.  It checks invariants that every RBL exit path must
obey and returns structured issues instead of relying on logs or exceptions.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

import networkx as nx
from rdkit import Chem

from synkit.Graph.Morphism.constraints import (
    NodeStateKind,
    WildcardRole,
    adapt_legacy_node_state,
)
from synkit.Graph.Stereo import (
    mapped_stereo_subgraph_registries_match,
    stereo_registry_layers,
)
from synkit.IO.mol_to_graph import MolToGraph
from synkit.Synthesis.Reactor.strategy import Strategy

_ENDPOINT_NODE_ATTRS = (
    "element",
    "isotope",
    "charge",
    "aromatic",
    "hcount",
    "lone_pairs",
    "radical",
)
_ENDPOINT_EDGE_ATTRS = ("order", "sigma_order", "pi_order", "aromatic")


class FusionIssueCode(str, Enum):
    """Stable machine-readable issue codes emitted by fusion validation."""

    INVALID_REACTION = "FUSION_INVALID_REACTION"
    PARSE_FAILURE = "FUSION_PARSE_FAILURE"
    DANGLING_WILDCARD = "FUSION_DANGLING_WILDCARD"
    SIDE_ONLY_STANDALONE_HYDROGEN = "FUSION_SIDE_ONLY_STANDALONE_HYDROGEN"
    DUPLICATE_ATOM_MAP = "FUSION_DUPLICATE_ATOM_MAP"
    ELEMENT_MAP_CONFLICT = "FUSION_ELEMENT_MAP_CONFLICT"
    HYDROGEN_MAP_IMBALANCE = "FUSION_HYDROGEN_MAP_IMBALANCE"
    WILDCARD_ROLE_CONFLICT = "FUSION_WILDCARD_ROLE_CONFLICT"
    INTERFACE_INVALID = "FUSION_INTERFACE_INVALID"
    CONSTRUCTION_INVALID = "FUSION_CONSTRUCTION_INVALID"
    PROOF_FAILED = "FUSION_PROOF_FAILED"
    OPERATION_FAILED = "FUSION_OPERATION_FAILED"
    SERIALIZATION_FAILED = "FUSION_SERIALIZATION_FAILED"
    POSTPROCESS_FAILED = "FUSION_POSTPROCESS_FAILED"
    REACTANT_ENDPOINT_NOT_PRESERVED = "FUSION_REACTANT_ENDPOINT_NOT_PRESERVED"
    PRODUCT_ENDPOINT_NOT_PRESERVED = "FUSION_PRODUCT_ENDPOINT_NOT_PRESERVED"


@dataclass(frozen=True)
class FusionIssue:
    """One failed fusion invariant."""

    code: FusionIssueCode
    stage: str
    message: str
    context: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "code": self.code.value,
            "stage": self.stage,
            "message": self.message,
            "context": dict(self.context),
        }


@dataclass(frozen=True)
class FusionValidation:
    """Validation outcome shared by every RBL execution mode."""

    valid: bool
    stage: str = "fusion"
    issues: tuple[FusionIssue, ...] = ()
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "valid": self.valid,
            "stage": self.stage,
            "issues": [issue.to_dict() for issue in self.issues],
            "evidence": dict(self.evidence),
        }


def _issue(
    code: FusionIssueCode,
    message: str,
    **context: Any,
) -> FusionIssue:
    return FusionIssue(
        code=code,
        stage="fusion",
        message=message,
        context=context,
    )


def _mapped_elements(mol: Chem.Mol) -> tuple[dict[int, str], list[int]]:
    elements: dict[int, str] = {}
    maps: list[int] = []
    for atom in mol.GetAtoms():
        atom_map = int(atom.GetAtomMapNum())
        if atom_map <= 0:
            continue
        maps.append(atom_map)
        elements[atom_map] = atom.GetSymbol()
    return elements, maps


def _standalone_hydrogens(mol: Chem.Mol) -> Counter[int]:
    """Count disconnected explicit H atoms, keyed by map number (or zero)."""
    return Counter(
        int(atom.GetAtomMapNum())
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() == 1 and atom.GetDegree() == 0
    )


def validate_fusion_rsmi(
    rsmi: str,
    *,
    allow_wildcards: bool = False,
) -> FusionValidation:
    """Validate endpoint invariants of a fused reaction SMILES.

    This is not a reaction-balancing oracle.  It guards the representation
    boundary most vulnerable during RBL fusion: parseability, unique atom-map
    identities, element preservation, symmetric mapped hydrogen presence,
    no side-only isolated H, and no unresolved wildcard unless explicitly
    requested by the caller.
    """
    if not isinstance(rsmi, str) or rsmi.count(">>") != 1:
        issue = _issue(
            FusionIssueCode.INVALID_REACTION,
            "Expected exactly one '>>' reaction separator.",
        )
        return FusionValidation(valid=False, issues=(issue,))

    reactants, products = rsmi.split(">>", 1)
    mols: list[Chem.Mol] = []
    issues: list[FusionIssue] = []
    for side, smiles in (("reactants", reactants), ("products", products)):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            issues.append(
                _issue(
                    FusionIssueCode.PARSE_FAILURE,
                    f"RDKit could not parse the {side} endpoint.",
                    side=side,
                )
            )
        else:
            mols.append(mol)

    if issues:
        return FusionValidation(valid=False, issues=tuple(issues))

    reactant_mol, product_mol = mols
    if not allow_wildcards:
        for side, mol in (
            ("reactants", reactant_mol),
            ("products", product_mol),
        ):
            wildcard_maps = [
                int(atom.GetAtomMapNum())
                for atom in mol.GetAtoms()
                if atom.GetAtomicNum() == 0
            ]
            if wildcard_maps:
                issues.append(
                    _issue(
                        FusionIssueCode.DANGLING_WILDCARD,
                        "An unresolved wildcard crossed the final fusion boundary.",
                        side=side,
                        atom_maps=wildcard_maps,
                    )
                )

    endpoint_elements: list[dict[int, str]] = []
    endpoint_maps: list[list[int]] = []
    for side, mol in (
        ("reactants", reactant_mol),
        ("products", product_mol),
    ):
        elements, maps = _mapped_elements(mol)
        endpoint_elements.append(elements)
        endpoint_maps.append(maps)
        duplicates = sorted(
            atom_map for atom_map, count in Counter(maps).items() if count > 1
        )
        if duplicates:
            issues.append(
                _issue(
                    FusionIssueCode.DUPLICATE_ATOM_MAP,
                    "An atom-map identity occurs more than once on one endpoint.",
                    side=side,
                    atom_maps=duplicates,
                )
            )

    reactant_elements, product_elements = endpoint_elements
    conflicts = {
        atom_map: (reactant_elements[atom_map], product_elements[atom_map])
        for atom_map in sorted(reactant_elements.keys() & product_elements.keys())
        if reactant_elements[atom_map] != product_elements[atom_map]
    }
    if conflicts:
        issues.append(
            _issue(
                FusionIssueCode.ELEMENT_MAP_CONFLICT,
                "Mapped atom identities change element across the reaction.",
                conflicts=conflicts,
            )
        )

    reactant_h_maps = {
        atom_map for atom_map, element in reactant_elements.items() if element == "H"
    }
    product_h_maps = {
        atom_map for atom_map, element in product_elements.items() if element == "H"
    }
    if reactant_h_maps != product_h_maps:
        issues.append(
            _issue(
                FusionIssueCode.HYDROGEN_MAP_IMBALANCE,
                "Mapped explicit hydrogen identities are not side-symmetric.",
                reactants=sorted(reactant_h_maps),
                products=sorted(product_h_maps),
            )
        )

    reactant_standalone = _standalone_hydrogens(reactant_mol)
    product_standalone = _standalone_hydrogens(product_mol)
    if reactant_standalone != product_standalone:
        issues.append(
            _issue(
                FusionIssueCode.SIDE_ONLY_STANDALONE_HYDROGEN,
                "Standalone explicit hydrogen fragments differ between endpoints.",
                reactants=dict(reactant_standalone),
                products=dict(product_standalone),
            )
        )

    return FusionValidation(valid=not issues, issues=tuple(issues))


def _parse_unmapped_endpoint_graph(side: str) -> nx.Graph | None:
    """Parse one endpoint into SynKit's map-independent molecular graph."""
    if not side:
        graph = nx.Graph()
        graph.graph["stereo_descriptors"] = {}
        return graph
    mol = Chem.MolFromSmiles(side)
    if mol is None:
        return None
    mol = Chem.Mol(mol)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return MolToGraph(
        node_attrs=list(_ENDPOINT_NODE_ATTRS),
        edge_attrs=list(_ENDPOINT_EDGE_ATTRS),
        attr_profile="minimal",
    ).transform(mol)


def _endpoint_embedding_proof(
    original_side: str,
    candidate_side: str,
) -> list[dict[str, Any]] | None:
    """Find a component-injective exact subgraph embedding.

    Every original molecule must embed into a distinct candidate molecule.
    This prevents two observed species from being collapsed into one newly
    connected component while still allowing RBL to add missing components.
    Atom and bond queries are exact RDKit graph queries with specified
    chirality enabled; atom-map numbers are representation labels and are
    therefore excluded from the match relation.
    """
    # Keep this import local: ``subgraph_matcher`` imports Reactor.strategy,
    # while Reactor's public package imports this validation module.
    from synkit.Graph.Matcher.subgraph_matcher import SubgraphSearchEngine

    original = _parse_unmapped_endpoint_graph(original_side)
    candidate = _parse_unmapped_endpoint_graph(candidate_side)
    if original is None or candidate is None:
        return None

    query_has_stereo = bool(stereo_registry_layers(original))
    mappings = SubgraphSearchEngine.find_subgraph_mappings(
        candidate,
        original,
        node_attrs=list(_ENDPOINT_NODE_ATTRS),
        edge_attrs=list(_ENDPOINT_EDGE_ATTRS),
        strategy=Strategy.COMPONENT,
        max_results=None if query_has_stereo else 1,
        strict_cc_count=False,
        threshold=50_000,
    )
    mapping = next(
        (
            structural_mapping
            for structural_mapping in mappings
            if mapped_stereo_subgraph_registries_match(
                original,
                candidate,
                structural_mapping,
            )
        ),
        None,
    )
    if mapping is None:
        return None

    original_components = [
        set(component) for component in nx.connected_components(original)
    ]
    candidate_components = [
        set(component) for component in nx.connected_components(candidate)
    ]
    candidate_component_by_node = {
        node: component_index
        for component_index, component in enumerate(candidate_components)
        for node in component
    }
    proof: list[dict[str, Any]] = []
    for original_index, component in enumerate(original_components):
        ordered_nodes = sorted(component, key=repr)
        atom_image = [mapping[node] for node in ordered_nodes]
        candidate_component = candidate_component_by_node[atom_image[0]]
        proof.append(
            {
                "original_component": original_index,
                "candidate_component": candidate_component,
                "atom_mapping": [
                    [query_node, host_node]
                    for query_node, host_node in zip(ordered_nodes, atom_image)
                ],
            }
        )
    return proof


def validate_endpoint_preservation(
    original_rsmi: str,
    candidate_rsmi: str,
    *,
    required_sides: Sequence[str] = ("reactants", "products"),
) -> FusionValidation:
    """Prove component-injective endpoint embeddings on selected sides."""
    if original_rsmi.count(">>") != 1 or candidate_rsmi.count(">>") != 1:
        issue = _issue(
            FusionIssueCode.INVALID_REACTION,
            "Endpoint preservation requires two valid reaction separators.",
        )
        return FusionValidation(valid=False, issues=(issue,))

    original_reactants, original_products = original_rsmi.split(">>", 1)
    candidate_reactants, candidate_products = candidate_rsmi.split(">>", 1)
    required = set(required_sides)
    unknown = required - {"reactants", "products"}
    if unknown:
        raise ValueError(f"Unknown endpoint side(s): {sorted(unknown)!r}")

    reactant_proof = (
        _endpoint_embedding_proof(original_reactants, candidate_reactants)
        if "reactants" in required
        else []
    )
    product_proof = (
        _endpoint_embedding_proof(original_products, candidate_products)
        if "products" in required
        else []
    )
    issues: list[FusionIssue] = []
    if "reactants" in required and reactant_proof is None:
        issues.append(
            _issue(
                FusionIssueCode.REACTANT_ENDPOINT_NOT_PRESERVED,
                "The original reactant graph does not embed component-wise "
                "in the candidate reactant graph.",
            )
        )
    if "products" in required and product_proof is None:
        issues.append(
            _issue(
                FusionIssueCode.PRODUCT_ENDPOINT_NOT_PRESERVED,
                "The original product graph does not embed component-wise "
                "in the candidate product graph.",
            )
        )

    evidence: dict[str, Any] = {}
    evidence["matcher"] = "synkit.SubgraphSearchEngine"
    evidence["stereo_policy"] = "synkit.relative_stereo_subgraph"
    if "reactants" in required and reactant_proof is not None:
        evidence["reactant_embeddings"] = reactant_proof
    if "products" in required and product_proof is not None:
        evidence["product_embeddings"] = product_proof
    return FusionValidation(
        valid=not issues,
        issues=tuple(issues),
        evidence=evidence,
    )


def validate_rbl_candidate(
    original_rsmi: str,
    candidate_rsmi: str,
    *,
    allow_wildcards: bool = False,
    preserve_sides: Sequence[str] = ("products",),
) -> FusionValidation:
    """Apply the complete, mode-independent RBL acceptance predicate.

    RBL reconstructs missing reactants and coproducts, so its default target
    invariant preserves the observed product endpoint.  Callers that require
    a conservative balancing-only transformation may additionally request the
    reactant side through ``preserve_sides``.
    """
    fusion = validate_fusion_rsmi(
        candidate_rsmi,
        allow_wildcards=allow_wildcards,
    )
    preservation = validate_endpoint_preservation(
        original_rsmi,
        candidate_rsmi,
        required_sides=preserve_sides,
    )
    issues = fusion.issues + preservation.issues
    return FusionValidation(
        valid=not issues,
        issues=issues,
        evidence={
            "endpoint_preservation": dict(preservation.evidence),
        },
    )


def validate_wildcard_mapping_roles(
    graph1: nx.Graph,
    graph2: nx.Graph,
    mapping: Mapping[Any, Any],
    *,
    element_key: str = "element",
    wildcard_element: Any = ("*", "*"),
    role_key: str = "wildcard_role",
) -> FusionValidation:
    """Reject an explicit mapping that conflates wildcard semantics.

    Wildcards pruned before matching never enter this contract.  If a matcher
    does map wildcard nodes, both ends must be wildcards and must carry the
    same declared :class:`WildcardRole`.  Missing roles fail closed because
    accepting them would recreate the untyped ``* == *`` ambiguity that this
    boundary is intended to expose.
    """
    scalar_wildcard = (
        wildcard_element[0] if isinstance(wildcard_element, tuple) else wildcard_element
    )
    wildcard_values = (wildcard_element, scalar_wildcard)
    issues: list[FusionIssue] = []

    for node1, node2 in mapping.items():
        if node1 not in graph1 or node2 not in graph2:
            continue
        attrs1 = graph1.nodes[node1]
        attrs2 = graph2.nodes[node2]
        wildcard1 = attrs1.get(element_key) in wildcard_values
        wildcard2 = attrs2.get(element_key) in wildcard_values
        if not wildcard1 and not wildcard2:
            continue

        raw_role1 = attrs1.get(role_key)
        raw_role2 = attrs2.get(role_key)
        try:
            state1 = adapt_legacy_node_state(
                attrs1,
                element_key=element_key,
                role_key=role_key,
                wildcard_values=wildcard_values,
            )
            constraint1 = (
                state1.constraint if state1.kind is NodeStateKind.WILDCARD else None
            )
        except (TypeError, ValueError):
            constraint1 = None
        try:
            state2 = adapt_legacy_node_state(
                attrs2,
                element_key=element_key,
                role_key=role_key,
                wildcard_values=wildcard_values,
            )
            constraint2 = (
                state2.constraint if state2.kind is NodeStateKind.WILDCARD else None
            )
        except (TypeError, ValueError):
            constraint2 = None

        compatibility = None
        if constraint1 is not None and constraint2 is not None:
            compatibility = constraint1.relabel_owner(mapping).intersect(constraint2)

        if (
            not wildcard1
            or not wildcard2
            or compatibility is None
            or not compatibility.valid
        ):
            issues.append(
                _issue(
                    FusionIssueCode.WILDCARD_ROLE_CONFLICT,
                    "Mapped wildcard nodes require identical declared roles.",
                    graph1_node=node1,
                    graph2_node=node2,
                    graph1_role=(
                        constraint1.role.value if constraint1 is not None else raw_role1
                    ),
                    graph2_role=(
                        constraint2.role.value if constraint2 is not None else raw_role2
                    ),
                    constraint_issues=(
                        [item.to_dict() for item in compatibility.issues]
                        if compatibility is not None
                        else []
                    ),
                )
            )

    return FusionValidation(valid=not issues, issues=tuple(issues))


__all__: Sequence[str] = (
    "FusionIssue",
    "FusionIssueCode",
    "FusionValidation",
    "WildcardRole",
    "validate_fusion_rsmi",
    "validate_endpoint_preservation",
    "validate_rbl_candidate",
    "validate_wildcard_mapping_roles",
)
