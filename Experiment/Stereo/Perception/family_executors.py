"""Executors for stereo families outside connectivity-only perception."""

from __future__ import annotations

from typing import Any

import networkx as nx
from rdkit import Chem

from synkit.Graph.Stereo import descriptors_from_rdkit

_ADAPTER_FAMILIES = {
    "square_planar",
    "trigonal_bipyramidal",
    "octahedral",
}
_SHAPE_TAGS = {
    Chem.ChiralType.CHI_SQUAREPLANAR,
    Chem.ChiralType.CHI_TRIGONALBIPYRAMIDAL,
    Chem.ChiralType.CHI_OCTAHEDRAL,
}
_TETRAHEDRAL_TAGS = {
    Chem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
}


def neutralize_rdkit_configuration(molecule: Chem.Mol) -> Chem.Mol:
    """Copy an RDKit molecule while retaining carriers but erasing orientation."""
    working = Chem.Mol(molecule)
    for atom in working.GetAtoms():
        tag = atom.GetChiralTag()
        if tag in _TETRAHEDRAL_TAGS:
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        elif tag in _SHAPE_TAGS:
            atom.SetIntProp("_chiralPermutation", 0)
        if atom.HasProp("_CIPCode"):
            atom.ClearProp("_CIPCode")
    for bond in working.GetBonds():
        bond.SetStereo(Chem.BondStereo.STEREONONE)
        bond.SetBondDir(Chem.BondDir.NONE)
        if bond.HasProp("_CIPCode"):
            bond.ClearProp("_CIPCode")
    return working


def _result(
    *,
    passed: bool,
    reason: str | None,
    carrier_check: str,
    configuration_check: str = "not_applicable",
    expected_reason: str | None = None,
    observed: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    reason_check = "not_evaluated"
    if expected_reason is not None:
        reason_check = "passed" if reason == expected_reason else "failed"
    return {
        "status": "passed" if passed else "failed",
        "status_reason": None if passed else reason,
        "carrier_check": carrier_check,
        "configuration_check": configuration_check,
        "reason_check": reason_check,
        "observed": observed or [],
    }


def _serialize_descriptor(descriptor: Any) -> dict[str, Any]:
    value = {
        "family": descriptor.descriptor_class,
    }
    if hasattr(descriptor, "path"):
        value["support"] = {"kind": "path", "path": list(descriptor.path)}
    elif hasattr(descriptor, "plane_atoms"):
        value["support"] = {
            "kind": "plane",
            "plane_atoms": list(descriptor.plane_atoms),
            "pilot": descriptor.pilot,
        }
    else:
        value["support"] = {
            "kind": "atom",
            "center": descriptor.center,
            "references": list(descriptor.atoms[1:]),
        }
    return value


def _zero_based_descriptor(descriptor: Any, atom_count: int) -> Any:
    return descriptor.relabel(
        {identifier: identifier - 1 for identifier in range(1, atom_count + 1)}
    )


def _all_center_ligands_symmetry_related(molecule: Chem.Mol) -> bool:
    center = max(molecule.GetAtoms(), key=lambda atom: atom.GetDegree())
    ranks = Chem.CanonicalRankAtoms(molecule, breakTies=False)
    ligand_ranks = {ranks[neighbor.GetIdx()] for neighbor in center.GetNeighbors()}
    return len(ligand_ranks) == 1


def evaluate_configured_adapter(
    case: dict[str, Any],
    structure: dict[str, Any],
) -> dict[str, Any]:
    """Exercise configured and shape-declared/unknown RDKit adapters."""
    expected = case["expected"]
    expected_reason = expected["reason_code"]
    molecule = Chem.MolFromSmiles(structure["value"])
    if molecule is None:
        return _result(
            passed=False,
            reason="smiles_parse_failure",
            carrier_check="failed",
            expected_reason=expected_reason,
        )
    molecule = neutralize_rdkit_configuration(molecule)
    try:
        registry = descriptors_from_rdkit(molecule, require_atom_maps=False)
    except ValueError as error:
        reason = (
            "wrong_coordination_number"
            if "requires exactly" in str(error)
            else "adapter_error"
        )
        passed = expected["outcome"] == "unsupported" and reason == expected_reason
        return _result(
            passed=passed,
            reason=reason,
            carrier_check="passed" if passed else "failed",
            configuration_check="ignored",
            expected_reason=expected_reason,
            observed=[{"adapter_error": str(error), "reason": reason}],
        )

    descriptors = [
        _zero_based_descriptor(descriptor, molecule.GetNumAtoms())
        for descriptor in registry.values()
        if descriptor.descriptor_class == case["family"]
    ]
    observed = [_serialize_descriptor(descriptor) for descriptor in descriptors]
    if case["case_type"] in {"configured_positive", "unconfigured_positive"}:
        support = expected["support"]
        matching = [
            descriptor
            for descriptor in descriptors
            if descriptor.parity is None
            and descriptor.center == support["center"]
            and set(descriptor.atoms[1:]) == set(support["references"])
        ]
        passed = bool(matching)
        reason = (
            "shape_declared_configuration_unknown"
            if passed
            else "topology_perception_not_implemented"
        )
        return _result(
            passed=passed,
            reason=reason,
            carrier_check="passed" if passed else "failed",
            configuration_check="ignored",
            expected_reason=expected_reason,
            observed=observed,
        )
    if case["case_type"] == "negative_near_miss":
        symmetric = not descriptors and _all_center_ligands_symmetry_related(molecule)
        reason = "all_ligands_symmetry_related" if symmetric else None
        return _result(
            passed=symmetric,
            reason=reason,
            carrier_check="passed" if symmetric else "failed",
            expected_reason=expected_reason,
            observed=observed,
        )
    return _result(
        passed=False,
        reason="expected_adapter_rejection_missing",
        carrier_check="failed",
        expected_reason=expected_reason,
        observed=observed,
    )


def _formal_graph(structure: dict[str, Any]) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from((node["id"], dict(node)) for node in structure["nodes"])
    graph.add_edges_from(
        (left, right, {"kind": kind}) for left, right, kind in structure["edges"]
    )
    return graph


def _helical_observation(
    structure: dict[str, Any],
) -> tuple[str, dict[str, Any] | None]:
    graph = _formal_graph(structure)
    edge_kinds = {attributes["kind"] for *_edge, attributes in graph.edges(data=True)}
    if edge_kinds == {"single"}:
        return "flexible_open_chain", None
    endpoints = sorted(node for node, degree in graph.degree() if degree == 1)
    if not endpoints:
        return "multiple_shortest_carrier_paths", None
    path = tuple(nx.shortest_path(graph, endpoints[0], endpoints[1]))
    if len(path) != graph.number_of_nodes() or edge_kinds != {"aromatic"}:
        return "unsupported_formal_helix", None
    return "formal_graph_carrier", {"kind": "path", "path": list(path)}


def _planar_observation(structure: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    plane = tuple(
        node["id"] for node in structure["nodes"] if node.get("role") == "plane"
    )
    pilots = tuple(
        node["id"] for node in structure["nodes"] if node.get("role") == "pilot"
    )
    if len(pilots) != 1:
        return "missing_pilot_reference", None
    elements = {
        node["element"] for node in structure["nodes"] if node.get("role") == "plane"
    }
    if len(elements) == 1:
        return "plane_symmetry_unresolved", None
    return (
        "formal_graph_carrier",
        {"kind": "plane", "plane_atoms": list(plane), "pilot": pilots[0]},
    )


def evaluate_formal_sidecar(
    case: dict[str, Any],
    structure: dict[str, Any],
) -> dict[str, Any]:
    """Exercise formal carrier contracts and configured sidecar restoration."""
    expected = case["expected"]
    expected_reason = expected["reason_code"]
    if case["family"] == "helical":
        reason, support = _helical_observation(structure)
    else:
        reason, support = _planar_observation(structure)
    observed: list[dict[str, Any]] = []
    if support is not None:
        observed.append(
            {
                "family": case["family"],
                "support": support,
                "reason": reason,
            }
        )

    expected_present = expected["outcome"] == "carrier_present"
    carrier_passed = (support is not None) == expected_present
    if expected_present and support != expected["support"]:
        carrier_passed = False
    configuration_check = "not_applicable"
    if expected_present:
        configuration_check = "ignored"

    reason_matches = expected_reason is None or reason == expected_reason
    passed = carrier_passed and reason_matches
    return _result(
        passed=passed,
        reason=reason,
        carrier_check="passed" if carrier_passed else "failed",
        configuration_check=configuration_check,
        expected_reason=expected_reason,
        observed=observed,
    )


def evaluate_family_scope(
    case: dict[str, Any],
    structure: dict[str, Any],
    scope: str,
) -> dict[str, Any]:
    """Dispatch one non-topology case to its executable family contract."""
    if scope == "configured_adapter_only" and case["family"] in _ADAPTER_FAMILIES:
        return evaluate_configured_adapter(case, structure)
    if scope == "sidecar_only":
        return evaluate_formal_sidecar(case, structure)
    raise ValueError(f"No family executor for scope {scope!r}.")


__all__ = ["evaluate_family_scope", "neutralize_rdkit_configuration"]
