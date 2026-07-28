#!/usr/bin/env python3
"""Reproduce and explain the VS300 exact/topology chirality boundary.

This diagnostic does not alter production classification.  It records the two
different input contracts used by the exact configured-stereograph audit and
the ACS-specialized topology-completion classifier:

* the exact audit restores every source-declared ``@``/``@@`` tag;
* the specialized classifier accepts RDKit's sanitized local-stereo boundary
  and adds topology probes at eligible unrepresented sp3 centers.

For VS300 those contracts attach opposite local configurations at two cage
centers.  The restored source descriptors admit a mirror witness, while the
topology-probe bundle rejects that representative witness and the specialized
whole-molecule matcher proves that no mirror isomorphism remains.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from rdkit import Chem

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Experiment.Stereo.Chirality.exact_mirror import (  # noqa: E402
    _parse_supplied_configured_smiles,
)
from Experiment.Stereo.Chirality.published import (  # noqa: E402
    EXPECTED_SHA256,
    load_dataset,
)
from synkit.Chem.Molecule.chirality import (  # noqa: E402
    _complete_tetrahedral_topology,
    _indexed_copy,
    classify_molecular_chirality,
)
from synkit.Graph.Stereo import (  # noqa: E402
    classify_rdkit_stereograph_mirror,
    classify_stereoisomer_relation,
)
from synkit.Graph.Stereo.canonical import (  # noqa: E402
    _rdkit_graph_and_registry,
)
from synkit.IO.mol_to_graph import MolToGraph  # noqa: E402

SCHEMA = "synkit.vs300-global-stereo-diagnostic/1"
FIXTURE_SCHEMA = "synkit.global-stereo-minimal-fixtures/1"
RECORD_ID = "VS300"

MINIMAL_FIXTURES = (
    {
        "fixture_id": "global-minimal-positive",
        "role": "topology_completion_positive",
        "source_record": "VS291",
        "smiles": "C1C2(OCC1)OCCC2",
        "expected_exact_supplied": "achiral",
        "expected_topology_complete": "Chiral",
    },
    {
        "fixture_id": "global-minimal-enantiomer-a",
        "role": "configured_enantiomer",
        "smiles": "C1[C@@]2(OCC1)OCCC2",
        "expected_exact_supplied": "chiral",
        "expected_topology_complete": "Chiral",
    },
    {
        "fixture_id": "global-minimal-enantiomer-b",
        "role": "configured_enantiomer",
        "smiles": "C1[C@]2(OCC1)OCCC2",
        "expected_exact_supplied": "chiral",
        "expected_topology_complete": "Chiral",
    },
    {
        "fixture_id": "global-minimal-achiral-near-miss",
        "role": "achiral_heteroatom_erased_control",
        "smiles": "C1C2(CCC1)CCCC2",
        "expected_exact_supplied": "achiral",
        "expected_topology_complete": "Achiral",
    },
)


def _record() -> dict[str, str]:
    return next(row for row in load_dataset() if row["ID"] == RECORD_ID)


def _tag_inventory(molecule: Chem.Mol) -> list[dict[str, Any]]:
    return [
        {
            "atom": atom.GetIdx() + 1,
            "tag": str(atom.GetChiralTag()),
            "degree": atom.GetDegree(),
            "total_hydrogens": int(atom.GetTotalNumHs()),
        }
        for atom in molecule.GetAtoms()
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
    ]


def _descriptor_snapshot(descriptor: Any) -> dict[str, Any]:
    return {
        "wire": descriptor.to_dict(),
        "canonical_form": list(descriptor.canonical_form()),
    }


def _registry_snapshot(
    registry: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    return {
        key: _descriptor_snapshot(descriptor)
        for key, descriptor in sorted(registry.items())
    }


def _topology_completion_registry(
    molecule: Chem.Mol,
) -> tuple[dict[str, Any], tuple[int, ...]]:
    working = _indexed_copy(molecule)
    graph = MolToGraph(attr_profile="minimal").transform(
        working,
        use_index_as_atom_map=True,
    )
    registry = dict(graph.graph.get("stereo_descriptors", {}))
    completed = _complete_tetrahedral_topology(working, registry)
    return registry, completed


def _transport_audit(
    registry: Mapping[str, Any],
    mapping: Mapping[int, int],
) -> dict[str, Any]:
    mirrored = frozenset(descriptor.invert() for descriptor in registry.values())
    records = []
    for key, descriptor in sorted(registry.items()):
        transported = descriptor.relabel(mapping)
        records.append(
            {
                "source_key": key,
                "source_center": descriptor.center,
                "mapped_center": mapping[descriptor.center],
                "transported": transported.to_dict(),
                "matches_mirror_registry": transported in mirrored,
            }
        )
    return {
        "checks": records,
        "blocking_source_centers": [
            record["source_center"]
            for record in records
            if not record["matches_mirror_registry"]
        ],
    }


def _configuration_relation(source: Any, probe: Any) -> str:
    if source == probe:
        return "equivalent"
    if source == probe.invert():
        return "opposite"
    return "different_frame_or_configuration"


def build_vs300_diagnostic() -> dict[str, Any]:
    """Return deterministic evidence for the two VS300 input contracts."""
    row = _record()
    sanitized = Chem.MolFromSmiles(row["Input SMILES"])
    restored = _parse_supplied_configured_smiles(row["Input SMILES"])
    if sanitized is None or restored is None:
        raise ValueError("VS300 must parse under both audited input contracts.")

    exact = classify_rdkit_stereograph_mirror(
        restored,
        require_complete=False,
        identity_profile="chemical",
    )
    if exact.atom_mirror_isomorphism is None:
        raise ValueError("The current VS300 exact result must expose a mirror witness.")
    mirror_mapping = dict(exact.atom_mirror_isomorphism)

    _, source_registry = _rdkit_graph_and_registry(restored)
    topology_registry, completed = _topology_completion_registry(sanitized)
    source_by_center = {
        descriptor.center: descriptor for descriptor in source_registry.values()
    }
    topology_by_center = {
        descriptor.center: descriptor for descriptor in topology_registry.values()
    }
    shared_centers = sorted(source_by_center.keys() & topology_by_center.keys())

    sanitized_without_completion = classify_molecular_chirality(
        sanitized,
        stereo_complete=False,
    )
    sanitized_with_completion = classify_molecular_chirality(
        sanitized,
        stereo_complete=True,
    )
    restored_with_completion = classify_molecular_chirality(
        restored,
        stereo_complete=True,
    )

    return {
        "schema": SCHEMA,
        "dataset": {
            "record_id": RECORD_ID,
            "audited_sha256": EXPECTED_SHA256,
            "manual": row["manual"],
            "input_smiles": row["Input SMILES"],
            "atom_count": sanitized.GetNumAtoms(),
        },
        "baseline_mismatch": {
            "exact_configured_chemical": exact.status.value,
            "acs_reference": row["manual"].lower(),
            "classification": "global_topology_capability_boundary",
        },
        "input_contracts": {
            "rdkit_sanitized": {
                "configured_atom_tags": _tag_inventory(sanitized),
                "meaning": (
                    "RDKit removes source tags that are not conventional local "
                    "tetrahedral stereocenters."
                ),
            },
            "source_tags_restored": {
                "configured_atom_tags": _tag_inventory(restored),
                "meaning": (
                    "The exact audit preserves every source-declared atom tag "
                    "as one independent local tetrahedral descriptor."
                ),
            },
        },
        "classification_paths": {
            "exact_source_declared": {
                "status": exact.status.value,
                "descriptor_count": exact.descriptor_count,
                "method": exact.method,
                "original_digest": exact.original.canonical_digest,
                "mirror_digest": exact.mirror.canonical_digest,
                "representative_atom_mirror_isomorphism": [
                    list(pair) for pair in exact.atom_mirror_isomorphism
                ],
            },
            "sanitized_without_topology_completion": {
                "status": sanitized_without_completion.classification.value,
                "descriptor_count": sanitized_without_completion.descriptor_count,
            },
            "sanitized_with_topology_completion": {
                "status": sanitized_with_completion.classification.value,
                "descriptor_count": sanitized_with_completion.descriptor_count,
                "completed_centers": list(
                    sanitized_with_completion.completed_tetrahedral_centers
                ),
                "mirror_isomorphism": sanitized_with_completion.mirror_isomorphism,
            },
            "restored_with_topology_completion": {
                "status": restored_with_completion.classification.value,
                "descriptor_count": restored_with_completion.descriptor_count,
                "completed_centers": list(
                    restored_with_completion.completed_tetrahedral_centers
                ),
                "mirror_isomorphism": [
                    list(pair)
                    for pair in (restored_with_completion.mirror_isomorphism or ())
                ],
            },
        },
        "descriptor_comparison": {
            "source_restored": _registry_snapshot(source_registry),
            "topology_completion": _registry_snapshot(topology_registry),
            "shared_center_relations": [
                {
                    "center": center,
                    "relation": _configuration_relation(
                        source_by_center[center],
                        topology_by_center[center],
                    ),
                }
                for center in shared_centers
            ],
            "topology_completed_centers": list(completed),
        },
        "representative_mirror_witness_audit": {
            "source_restored": _transport_audit(
                source_registry,
                mirror_mapping,
            ),
            "topology_completion": _transport_audit(
                topology_registry,
                mirror_mapping,
            ),
            "interpretation": (
                "All restored source descriptors accept the exact mirror "
                "witness. The topology-probe bundle rejects it at the two cage "
                "centers exchanged by that witness; the specialized matcher "
                "then proves that no alternative mirror isomorphism remains."
            ),
        },
        "conclusion": {
            "what_is_proven": (
                "VS300 exercises two different configured-graph contracts. "
                "The 256/258 exact audit is reproducible and its achiral result "
                "is exact for the restored independent local descriptors."
            ),
            "what_is_missing": (
                "A general stereograph representation for the coupled global "
                "framework state used by topology completion."
            ),
            "implementation_guard": (
                "Do not replace source tags with arbitrary parity probes in "
                "the exact path. First model the probe bundle as a coupled "
                "global constraint with an explicit information state."
            ),
        },
    }


def build_minimal_fixtures() -> dict[str, Any]:
    """Evaluate the smallest public positive and a designed near-miss."""
    evaluated = []
    restored_molecules: dict[str, Chem.Mol] = {}
    for fixture in MINIMAL_FIXTURES:
        sanitized = Chem.MolFromSmiles(fixture["smiles"])
        restored = _parse_supplied_configured_smiles(fixture["smiles"])
        if sanitized is None or restored is None:
            raise ValueError(f"Cannot parse fixture {fixture['fixture_id']}.")
        exact = classify_rdkit_stereograph_mirror(
            restored,
            require_complete=False,
            identity_profile="chemical",
        )
        incomplete = classify_molecular_chirality(
            sanitized,
            stereo_complete=False,
        )
        complete = classify_molecular_chirality(
            sanitized,
            stereo_complete=True,
        )
        if exact.status.value != fixture["expected_exact_supplied"]:
            raise ValueError(f"Unexpected exact result for {fixture['fixture_id']}.")
        if complete.classification.value != fixture["expected_topology_complete"]:
            raise ValueError(
                f"Unexpected topology result for {fixture['fixture_id']}."
            )
        restored_molecules[fixture["fixture_id"]] = restored
        evaluated.append(
            {
                **fixture,
                "atom_count": sanitized.GetNumAtoms(),
                "sanitized_tags": _tag_inventory(sanitized),
                "restored_tags": _tag_inventory(restored),
                "observed": {
                    "exact_supplied": exact.status.value,
                    "without_topology_completion": (
                        incomplete.classification.value
                    ),
                    "with_topology_completion": complete.classification.value,
                    "completed_centers": list(
                        complete.completed_tetrahedral_centers
                    ),
                },
            }
        )

    left_id = "global-minimal-enantiomer-a"
    right_id = "global-minimal-enantiomer-b"
    left_graph, left_registry = _rdkit_graph_and_registry(
        restored_molecules[left_id]
    )
    right_graph, right_registry = _rdkit_graph_and_registry(
        restored_molecules[right_id]
    )
    relation = classify_stereoisomer_relation(
        left_graph,
        left_registry,
        right_graph,
        right_registry,
    )
    return {
        "schema": FIXTURE_SCHEMA,
        "fixtures": evaluated,
        "configured_pair": {
            "left": left_id,
            "right": right_id,
            "relation": relation.relation.value,
        },
        "interpretation": {
            "positive": (
                "The nine-atom VS291 constitution is the smallest ACS case "
                "whose sanitized graph changes from achiral to chiral under "
                "topology completion."
            ),
            "near_miss": (
                "Replacing both oxygen paths with carbon paths preserves the "
                "nine-atom bicyclic skeleton but restores a mirror symmetry."
            ),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    outputs = (
        (arguments.output, build_vs300_diagnostic()),
        (arguments.fixtures, build_minimal_fixtures()),
    )
    for path, payload in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "diagnostic": str(arguments.output),
                "fixtures": str(arguments.fixtures),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
