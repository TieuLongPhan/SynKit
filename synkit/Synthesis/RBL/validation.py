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
from synkit.Graph.Fusion.identity import (
    graph_identity_digest,
    graphs_exactly_equivalent,
)
from synkit.Graph.Stereo import (
    mapped_stereo_subgraph_registries_match,
    stereo_registry_layers,
)
from synkit.IO.mol_to_graph import MolToGraph
from synkit.Synthesis.Reactor import Strategy

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


def _mapped_smiles_mol(smiles: str) -> Chem.Mol | None:
    """Parse mapped SMILES without discarding heavy-atom-bound H identities."""
    parameters = Chem.SmilesParserParams()
    parameters.removeHs = False
    return Chem.MolFromSmiles(smiles, parameters)


class FusionIssueCode(str, Enum):
    """Stable machine-readable issue codes emitted by fusion validation."""

    INVALID_REACTION = "FUSION_INVALID_REACTION"
    PARSE_FAILURE = "FUSION_PARSE_FAILURE"
    DANGLING_WILDCARD = "FUSION_DANGLING_WILDCARD"
    SIDE_ONLY_STANDALONE_HYDROGEN = "FUSION_SIDE_ONLY_STANDALONE_HYDROGEN"
    DUPLICATE_ATOM_MAP = "FUSION_DUPLICATE_ATOM_MAP"
    ATOM_MAP_IMBALANCE = "FUSION_ATOM_MAP_IMBALANCE"
    ELEMENT_MAP_CONFLICT = "FUSION_ELEMENT_MAP_CONFLICT"
    ISOTOPE_MAP_CONFLICT = "FUSION_ISOTOPE_MAP_CONFLICT"
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
    REACTANT_COMPONENT_NOT_PRESERVED = (
        "FUSION_REACTANT_COMPONENT_NOT_PRESERVED"
    )
    PRODUCT_COMPONENT_NOT_PRESERVED = "FUSION_PRODUCT_COMPONENT_NOT_PRESERVED"
    ELEMENT_ISOTOPE_IMBALANCE = "FUSION_ELEMENT_ISOTOPE_IMBALANCE"
    CHARGE_IMBALANCE = "FUSION_CHARGE_IMBALANCE"
    UNMAPPED_MATERIAL_ATOM = "FUSION_UNMAPPED_MATERIAL_ATOM"
    ENVIRONMENT_DELTA_MISMATCH = "FUSION_ENVIRONMENT_DELTA_MISMATCH"


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


def _mapped_isotopes(mol: Chem.Mol) -> dict[int, int]:
    return {
        int(atom.GetAtomMapNum()): int(atom.GetIsotope())
        for atom in mol.GetAtoms()
        if atom.GetAtomMapNum() > 0
    }


def _unmapped_standalone_hydrogens(mol: Chem.Mol) -> Counter[str]:
    """Count untraceable isolated H atoms by isotope/charge/radical state.

    A mapped H may legitimately move from a standalone proton or hydride into
    a bond; its conserved atom-map identity proves that it was transferred.
    An unmapped isolated H has no such identity and must remain endpoint
    symmetric to cross the verified representation boundary.
    """
    return Counter(
        f"isotope={atom.GetIsotope()},charge={atom.GetFormalCharge()},"
        f"radical={atom.GetNumRadicalElectrons()}"
        for atom in mol.GetAtoms()
        if (
            atom.GetAtomicNum() == 1
            and atom.GetDegree() == 0
            and atom.GetAtomMapNum() <= 0
        )
    )


def _mapped_identity_issues(
    endpoints: Sequence[tuple[str, Chem.Mol]],
) -> list[FusionIssue]:
    """Validate uniqueness, side symmetry, and element-stable mapped atoms."""
    issues: list[FusionIssue] = []
    endpoint_elements: list[dict[int, str]] = []
    for side, mol in endpoints:
        elements, maps = _mapped_elements(mol)
        endpoint_elements.append(elements)
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
    reactant_maps = set(reactant_elements)
    product_maps = set(product_elements)
    if reactant_maps != product_maps:
        issues.append(
            _issue(
                FusionIssueCode.ATOM_MAP_IMBALANCE,
                "Mapped atom identities are not side-symmetric.",
                reactants_only=sorted(reactant_maps - product_maps),
                products_only=sorted(product_maps - reactant_maps),
            )
        )
    conflicts = {
        atom_map: (reactant_elements[atom_map], product_elements[atom_map])
        for atom_map in sorted(reactant_maps & product_maps)
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

    endpoint_isotopes = [_mapped_isotopes(mol) for _, mol in endpoints]
    isotope_conflicts = {
        atom_map: (endpoint_isotopes[0][atom_map], endpoint_isotopes[1][atom_map])
        for atom_map in sorted(reactant_maps & product_maps)
        if endpoint_isotopes[0][atom_map] != endpoint_isotopes[1][atom_map]
    }
    if isotope_conflicts:
        issues.append(
            _issue(
                FusionIssueCode.ISOTOPE_MAP_CONFLICT,
                "Mapped atom identities change isotope across the reaction.",
                conflicts=isotope_conflicts,
            )
        )

    hydrogen_maps = [
        {atom_map for atom_map, element in elements.items() if element == "H"}
        for elements in endpoint_elements
    ]
    if hydrogen_maps[0] != hydrogen_maps[1]:
        issues.append(
            _issue(
                FusionIssueCode.HYDROGEN_MAP_IMBALANCE,
                "Mapped explicit hydrogen identities are not side-symmetric.",
                reactants=sorted(hydrogen_maps[0]),
                products=sorted(hydrogen_maps[1]),
            )
        )
    return issues


def validate_fusion_rsmi(
    rsmi: str,
    *,
    allow_wildcards: bool = False,
) -> FusionValidation:
    """Validate endpoint invariants of a fused reaction SMILES.

    This is not a reaction-balancing oracle.  It guards the representation
    boundary most vulnerable during RBL fusion: parseability, unique and
    side-symmetric atom-map identities, element preservation, no side-only
    *unmapped* isolated H, and no unresolved wildcard unless explicitly
    requested by the caller. A mapped isolated H may become bound because its
    symmetric map identity already certifies atom conservation.
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
        mol = _mapped_smiles_mol(smiles)
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

    issues.extend(
        _mapped_identity_issues(
            (("reactants", reactant_mol), ("products", product_mol))
        )
    )

    reactant_standalone = _unmapped_standalone_hydrogens(reactant_mol)
    product_standalone = _unmapped_standalone_hydrogens(product_mol)
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
    # Endpoint preservation is a molecular-graph claim, independent of
    # whether a bound hydrogen was serialized explicitly to retain its AAM
    # identity.  Clear AAM first, then fold only RDKit-removable bound H; H2
    # and standalone hydrogen components remain explicit.
    mol = Chem.RemoveHs(mol)
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
    # Keep this import local so validation that never reaches the endpoint
    # proof does not initialize the general subgraph-matching stack.
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


def _component_inventory(side: str) -> Counter[str] | None:
    """Return an atom-map-independent exact molecular-component multiset."""
    molecule = _mapped_smiles_mol(side)
    if molecule is None:
        return None
    inventory: Counter[str] = Counter()
    for fragment in Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=True):
        fragment = Chem.Mol(fragment)
        for atom in fragment.GetAtoms():
            atom.SetAtomMapNum(0)
        try:
            normalized = Chem.RemoveHs(fragment)
            signature = Chem.MolToSmiles(
                normalized,
                canonical=True,
                isomericSmiles=True,
            )
        except (RuntimeError, ValueError):
            return None
        inventory[signature] += 1
    return inventory


def _material_balance(
    rsmi: str,
) -> tuple[Counter[tuple[str, int]], Counter[tuple[str, int]], int, int] | None:
    if rsmi.count(">>") != 1:
        return None
    inventories: list[Counter[tuple[str, int]]] = []
    charges: list[int] = []
    for side in rsmi.split(">>"):
        molecule = _mapped_smiles_mol(side)
        if molecule is None:
            return None
        try:
            expanded = Chem.AddHs(molecule)
        except RuntimeError:
            return None
        inventories.append(
            Counter(
                (atom.GetSymbol(), int(atom.GetIsotope()))
                for atom in expanded.GetAtoms()
            )
        )
        charges.append(int(Chem.GetFormalCharge(molecule)))
    return inventories[0], inventories[1], charges[0], charges[1]


def _resource_delta(
    reactants: Counter[tuple[str, int]],
    products: Counter[tuple[str, int]],
    reactant_charge: int,
    product_charge: int,
) -> dict[str, int]:
    delta = {
        f"element:{element}:{isotope}": count
        for (element, isotope), count in products.items()
    }
    for (element, isotope), count in reactants.items():
        key = f"element:{element}:{isotope}"
        delta[key] = delta.get(key, 0) - count
    delta = {key: value for key, value in delta.items() if value}
    charge_delta = product_charge - reactant_charge
    if charge_delta:
        delta["formal_charge"] = charge_delta
    return dict(sorted(delta.items()))


def validate_strict_rbl_candidate(  # noqa: C901
    original_rsmi: str,
    candidate_rsmi: str,
    *,
    allow_wildcards: bool = False,
    boundary: str = "closed",
    environment_delta: Mapping[str, int] | None = None,
    require_mapped_material: bool = True,
) -> FusionValidation:
    """Validate strict component-completion and conservation semantics.

    Both observed endpoints must occur as exact molecular-component multisets;
    fragment embeddings are deliberately insufficient. A closed boundary
    requires isotope/element and net-formal-charge conservation. An open
    boundary must declare the exact material/charge delta supplied by its
    environment.
    """
    if boundary not in {"closed", "open"}:
        raise ValueError("boundary must be 'closed' or 'open'.")
    base = validate_fusion_rsmi(candidate_rsmi, allow_wildcards=allow_wildcards)
    issues = list(base.issues)
    evidence: dict[str, Any] = {
        "observation_relation": "exact_component_multiset_inclusion",
        "boundary": boundary,
    }
    if original_rsmi.count(">>") != 1 or candidate_rsmi.count(">>") != 1:
        return FusionValidation(valid=False, issues=tuple(issues), evidence=evidence)

    original_sides = original_rsmi.split(">>")
    candidate_sides = candidate_rsmi.split(">>")
    component_evidence: dict[str, Any] = {}
    for index, side_name in enumerate(("reactants", "products")):
        observed = _component_inventory(original_sides[index])
        completed = _component_inventory(candidate_sides[index])
        if observed is None or completed is None:
            issues.append(
                _issue(
                    FusionIssueCode.PARSE_FAILURE,
                    f"Could not normalize {side_name} component inventory.",
                    side=side_name,
                )
            )
            continue
        missing = observed - completed
        component_evidence[side_name] = {
            "observed": dict(sorted(observed.items())),
            "candidate": dict(sorted(completed.items())),
            "missing": dict(sorted(missing.items())),
        }
        if missing:
            code = (
                FusionIssueCode.REACTANT_COMPONENT_NOT_PRESERVED
                if side_name == "reactants"
                else FusionIssueCode.PRODUCT_COMPONENT_NOT_PRESERVED
            )
            issues.append(
                _issue(
                    code,
                    f"Observed {side_name} components are not an exact multiset "
                    "subset of the candidate endpoint.",
                    missing=dict(sorted(missing.items())),
                )
            )
    evidence["component_inventory"] = component_evidence

    balance = _material_balance(candidate_rsmi)
    if balance is None:
        issues.append(
            _issue(
                FusionIssueCode.PARSE_FAILURE,
                "Could not compute candidate material balance.",
            )
        )
    else:
        reactants, products, reactant_charge, product_charge = balance
        delta = _resource_delta(
            reactants,
            products,
            reactant_charge,
            product_charge,
        )
        evidence["resource_delta"] = delta
        if boundary == "closed":
            element_delta = {
                key: value for key, value in delta.items() if key != "formal_charge"
            }
            if element_delta:
                issues.append(
                    _issue(
                        FusionIssueCode.ELEMENT_ISOTOPE_IMBALANCE,
                        "Closed reconstruction does not conserve element/isotope inventory.",
                        delta=element_delta,
                    )
                )
            if delta.get("formal_charge", 0):
                issues.append(
                    _issue(
                        FusionIssueCode.CHARGE_IMBALANCE,
                        "Closed reconstruction does not conserve net formal charge.",
                        delta=delta["formal_charge"],
                    )
                )
        else:
            declared = dict(sorted((environment_delta or {}).items()))
            evidence["environment_delta"] = declared
            if delta != declared:
                issues.append(
                    _issue(
                        FusionIssueCode.ENVIRONMENT_DELTA_MISMATCH,
                        "Open reconstruction delta differs from its environment token.",
                        observed=delta,
                        declared=declared,
                    )
                )

    if require_mapped_material:
        unmapped: dict[str, list[int]] = {}
        for side_name, side in zip(
            ("reactants", "products"), candidate_sides, strict=True
        ):
            molecule = _mapped_smiles_mol(side)
            if molecule is None:
                continue
            missing_maps = [
                atom.GetIdx()
                for atom in molecule.GetAtoms()
                if atom.GetAtomicNum() > 1 and atom.GetAtomMapNum() <= 0
            ]
            if missing_maps:
                unmapped[side_name] = missing_maps
        if unmapped:
            issues.append(
                _issue(
                    FusionIssueCode.UNMAPPED_MATERIAL_ATOM,
                    "Every material atom in strict reconstruction requires provenance.",
                    atoms=unmapped,
                )
            )
        evidence["unmapped_material_atoms"] = unmapped

    return FusionValidation(
        valid=not issues,
        issues=tuple(issues),
        evidence=evidence,
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

    Wildcards pruned before matching never enter this contract.  A
    wildcard-to-wildcard mapping requires compatible declared roles.  A typed
    wildcard may instead map to a concrete node admitted by its declared
    domain; the verified interface subsequently proves owner incidence and
    substitution.  Missing roles fail closed.
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

        concrete_constraint = constraint1 if wildcard1 else constraint2
        concrete_attrs = attrs2 if wildcard1 else attrs1
        concrete_element = concrete_attrs.get(element_key)
        if isinstance(concrete_element, (tuple, list)) and len(concrete_element) == 2:
            concrete_element = (
                concrete_element[0]
                if concrete_element[0] == concrete_element[1]
                else None
            )
        concrete_charge = concrete_attrs.get("charge", 0)
        concrete_radical = concrete_attrs.get("radical", 0)
        if isinstance(concrete_charge, (tuple, list)) and len(concrete_charge) == 2:
            concrete_charge = (
                concrete_charge[0] if concrete_charge[0] == concrete_charge[1] else None
            )
        if isinstance(concrete_radical, (tuple, list)) and len(concrete_radical) == 2:
            concrete_radical = (
                concrete_radical[0]
                if concrete_radical[0] == concrete_radical[1]
                else None
            )
        concrete_ok = (
            concrete_constraint is not None
            and concrete_constraint.virtual_kind is None
            and (
                concrete_constraint.elements is None
                or concrete_element in concrete_constraint.elements
            )
            and (
                concrete_constraint.charges is None
                or concrete_charge in concrete_constraint.charges
            )
            and (
                concrete_constraint.radicals is None
                or concrete_radical in concrete_constraint.radicals
            )
        )
        valid = (
            compatibility is not None and compatibility.valid
            if wildcard1 and wildcard2
            else concrete_ok
        )

        if not valid:
            issues.append(
                _issue(
                    FusionIssueCode.WILDCARD_ROLE_CONFLICT,
                    "Mapped wildcards require compatible typed constraints.",
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


def _normalize_completion_materialization(
    source: nx.Graph,
    *,
    element_key: str,
    wildcard_element: Any,
    role_key: str,
    allowed_roles: set[str],
) -> tuple[nx.Graph, list[dict[str, str]], list[FusionIssue]]:
    """Materialize typed completion ports on a copy of the source graph."""
    normalized = source.copy()
    scalar_wildcard = (
        wildcard_element[0] if isinstance(wildcard_element, tuple) else wildcard_element
    )
    wildcard_values = (wildcard_element, scalar_wildcard)
    materialized: list[dict[str, str]] = []
    issues: list[FusionIssue] = []
    for node, attributes in normalized.nodes(data=True):
        if attributes.get(element_key) not in wildcard_values:
            continue
        raw_role = attributes.get(role_key)
        role = raw_role.value if isinstance(raw_role, WildcardRole) else raw_role
        if role not in allowed_roles:
            issues.append(
                _issue(
                    FusionIssueCode.PROOF_FAILED,
                    "Only typed completion wildcards may materialize as hydrogen.",
                    node=repr(node),
                    wildcard_role=role,
                )
            )
            continue
        attributes[element_key] = (
            ("H", "H") if isinstance(attributes.get(element_key), tuple) else "H"
        )
        types_gh = attributes.get("typesGH")
        if isinstance(types_gh, tuple) and len(types_gh) == 2:
            attributes["typesGH"] = tuple(
                (
                    (("H",) + tuple(endpoint[1:]))
                    if isinstance(endpoint, tuple) and endpoint
                    else endpoint
                )
                for endpoint in types_gh
            )
        attributes.pop(role_key, None)
        materialized.append({"node": repr(node), "role": str(role)})

    for _, attributes in normalized.nodes(data=True):
        neighbors = attributes.get("neighbors")
        if isinstance(neighbors, tuple) and len(neighbors) == 2:
            attributes["neighbors"] = tuple(
                ["H" if item == scalar_wildcard else item for item in endpoint or ()]
                for endpoint in neighbors
            )
        elif isinstance(neighbors, list):
            attributes["neighbors"] = [
                "H" if item == scalar_wildcard else item for item in neighbors
            ]
    return normalized, materialized, issues


def certify_fusion_postprocessing(
    source: nx.Graph,
    target: nx.Graph,
    *,
    materialize_hydrogen: bool,
    element_key: str = "element",
    wildcard_element: Any = ("*", "*"),
    role_key: str = "wildcard_role",
) -> FusionValidation:
    """Certify the narrow representation change allowed after a pushout.

    An exact graph identity is always accepted.  When hydrogen
    materialisation is requested, a wildcard may additionally become H only
    when it has an explicit completion role.  The consumed role annotation
    and wildcard tokens in cached neighbour/type fields are updated before an
    exact map-invariant graph comparison.  No nodes or edges may be added,
    removed, or otherwise changed.

    This is deliberately narrower than general reaction standardisation: a
    query atom, attachment port, side-presence placeholder, or untyped legacy
    wildcard is never evidence for a hydrogen atom.
    """
    source_digest = graph_identity_digest(source)
    target_digest = graph_identity_digest(target)
    if graphs_exactly_equivalent(source, target):
        return FusionValidation(
            valid=True,
            evidence={
                "postprocess_proof": {
                    "kind": "identity",
                    "source_digest": source_digest,
                    "normalized_digest": source_digest,
                    "target_digest": target_digest,
                    "materialized_nodes": [],
                }
            },
        )

    allowed_roles = {
        WildcardRole.RADICAL_COMPLETION.value,
        WildcardRole.HYDROGEN_COMPLETION.value,
    }
    if not materialize_hydrogen:
        return FusionValidation(
            valid=False,
            issues=(
                _issue(
                    FusionIssueCode.PROOF_FAILED,
                    "Post-processing changed the certified pushout graph.",
                    source_digest=source_digest,
                    target_digest=target_digest,
                ),
            ),
        )

    normalized, materialized, issues = _normalize_completion_materialization(
        source,
        element_key=element_key,
        wildcard_element=wildcard_element,
        role_key=role_key,
        allowed_roles=allowed_roles,
    )

    if issues:
        return FusionValidation(valid=False, issues=tuple(issues))
    if not materialized:
        return FusionValidation(
            valid=False,
            issues=(
                _issue(
                    FusionIssueCode.PROOF_FAILED,
                    "The graph changed without a typed wildcard materialisation.",
                    source_digest=source_digest,
                    target_digest=target_digest,
                ),
            ),
        )

    normalized_digest = graph_identity_digest(normalized)
    if not graphs_exactly_equivalent(normalized, target):
        return FusionValidation(
            valid=False,
            issues=(
                _issue(
                    FusionIssueCode.PROOF_FAILED,
                    "Post-processing exceeds typed wildcard-to-hydrogen materialisation.",
                    source_digest=source_digest,
                    normalized_digest=normalized_digest,
                    target_digest=target_digest,
                    materialized_nodes=materialized,
                ),
            ),
        )

    return FusionValidation(
        valid=True,
        evidence={
            "postprocess_proof": {
                "kind": "typed_wildcard_hydrogen_materialization",
                "source_digest": source_digest,
                "normalized_digest": normalized_digest,
                "target_digest": target_digest,
                "materialized_nodes": materialized,
                "allowed_roles": sorted(allowed_roles),
            }
        },
    )


__all__: Sequence[str] = (
    "FusionIssue",
    "FusionIssueCode",
    "FusionValidation",
    "WildcardRole",
    "certify_fusion_postprocessing",
    "validate_fusion_rsmi",
    "validate_endpoint_preservation",
    "validate_rbl_candidate",
    "validate_strict_rbl_candidate",
    "validate_wildcard_mapping_roles",
)
