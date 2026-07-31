#!/usr/bin/env python3
"""Generate the source-stratified input inventory for stereo canonicalization.

This module deliberately performs no atom permutation and computes no
canonical certificate.  It freezes the candidate population and preserves the
scientific boundary of each source:

* ACS supplies configured molecular structures;
* CIP supplies external local stereo-unit truth, but its structures are not
  redistributed in this repository;
* RotA supplies positive axial-locus annotations without fixed handedness.
"""

from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
import sys
from typing import Any, Iterable

from rdkit import Chem

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Experiment.Stereo.Chirality.published import (  # noqa: E402
    DATASET as ACS_DATASET,
    EXPECTED_SHA256 as ACS_SHA256,
    load_dataset,
)
from synkit.Graph.Stereo import (  # noqa: E402
    AtropAxisStereo,
    AtropBondStereo,
    CumuleneAxisStereo,
    ExtendedCisTransStereo,
    HelicalStereo,
    OctahedralStereo,
    PlanarBondStereo,
    PlanarChiralityStereo,
    SquarePlanarStereo,
    TetrahedralStereo,
    TrigonalBipyramidalStereo,
    descriptors_from_rdkit,
)

STEREO_ROOT = ROOT / "Experiment" / "Stereo" / "Data"
CIP_ELEMENT_REPORT = STEREO_ROOT / "Perception" / "stereo_element_report.json"
ROTA_LOCUS_REPORT = STEREO_ROOT / "Perception" / "rota_locus_report.json"
CANON_DATA_ROOT = STEREO_ROOT / "Canonicalization"
DEFAULT_JSON = CANON_DATA_ROOT / "canonicalization_inventory.json"
DEFAULT_CSV = CANON_DATA_ROOT / "canonicalization_inventory.csv"

_DESCRIPTOR_FAMILIES = {
    TetrahedralStereo: ("tetrahedral", "atom_center"),
    SquarePlanarStereo: ("square_planar", "atom_center"),
    TrigonalBipyramidalStereo: (
        "trigonal_bipyramidal",
        "atom_center",
    ),
    OctahedralStereo: ("octahedral", "atom_center"),
    PlanarBondStereo: ("planar_bond", "bond_center"),
    AtropBondStereo: ("atrop_bond", "bond_axis"),
    AtropAxisStereo: ("atrop_axis", "axis"),
    CumuleneAxisStereo: ("cumulene_axis", "axis_path"),
    ExtendedCisTransStereo: (
        "extended_cis_trans",
        "extended_cumulene_path",
    ),
    HelicalStereo: ("helical", "path"),
    PlanarChiralityStereo: ("planar_chirality", "plane"),
}

_CIP_UNIT_MAPPINGS = {
    "TH": {
        "family": "tetrahedral",
        "carrier_kind": "atom_center",
        "mapping": "direct",
    },
    "CT": {
        "family": "planar_bond",
        "carrier_kind": "bond_center",
        "mapping": "direct",
    },
    "AT": {
        "family": "atrop_bond",
        "carrier_kind": "bond_axis",
        "mapping": "direct",
    },
    "HE": {
        "family": "helical",
        "carrier_kind": "path",
        "mapping": "direct",
    },
    "TH3": {
        "family": "cumulene_axis",
        "carrier_kind": "axis_path",
        "mapping": "extended_tetrahedral_candidate",
    },
    "TH5": {
        "family": "cumulene_axis",
        "carrier_kind": "axis_path",
        "mapping": "extended_tetrahedral_candidate",
    },
    "CT4": {
        "family": "extended_cis_trans",
        "carrier_kind": "extended_cumulene_path",
        "mapping": "direct",
    },
}

_ROTA_TYPE_MAPPINGS = {
    "Allene-like structure": {
        "family": "cumulene_axis",
        "carrier_kind": "axis_path",
        "mapping": "direct_locus",
    },
    "Biaryl structure": {
        "family": "atrop_bond",
        "carrier_kind": "bond_axis",
        "mapping": "direct_locus",
    },
    "Heterobiaryl (C-B)": {
        "family": "atrop_bond",
        "carrier_kind": "bond_axis",
        "mapping": "direct_locus",
    },
    "Heterobiaryl (C-C)": {
        "family": "atrop_bond",
        "carrier_kind": "bond_axis",
        "mapping": "direct_locus",
    },
    "Heterobiaryl (C-N)": {
        "family": "atrop_bond",
        "carrier_kind": "bond_axis",
        "mapping": "direct_locus",
    },
    "Nonbiaryl": {
        "family": "axis_locus_unresolved",
        "carrier_kind": "axis",
        "mapping": "source_locus_only",
    },
    "Chiral atom pair": {
        "family": "axis_locus_unresolved",
        "carrier_kind": "axis",
        "mapping": "source_locus_only",
    },
    "Spiral atom and chain": {
        "family": "axis_locus_unresolved",
        "carrier_kind": "path",
        "mapping": "source_locus_only",
    },
}

_CSV_FIELDS = (
    "source",
    "record_id",
    "source_categories",
    "families",
    "carrier_kinds",
    "configuration_evidence",
    "canonicalization_role",
    "multiplicity",
    "multiplicity_basis",
    "source_element_count",
    "extractable_configured_count",
    "reference_rs_count",
    "attached_tetrahedral_count",
    "reference_locus_count",
    "structure_available",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _multiplicity(count: int, *, zero: str = "none") -> str:
    if count == 0:
        return zero
    return "single" if count == 1 else "multiple"


def _descriptor_entry(identifier: str, descriptor: Any) -> dict[str, Any]:
    family_info = _DESCRIPTOR_FAMILIES.get(type(descriptor))
    if family_info is None:
        raise ValueError(
            f"Unregistered configured descriptor {type(descriptor).__name__}."
        )
    family, carrier_kind = family_info
    return {
        "descriptor_id": identifier,
        "descriptor_class": descriptor.descriptor_class,
        "family": family,
        "carrier_kind": carrier_kind,
        "descriptor": descriptor.to_dict(),
    }


def _acs_records(path: Path = ACS_DATASET) -> list[dict[str, Any]]:
    records = []
    for row in load_dataset(path):
        molecule = Chem.MolFromSmiles(row["Input SMILES"])
        if molecule is None:
            raise ValueError(f"RDKit rejected ACS case {row['ID']}.")
        registry = descriptors_from_rdkit(
            molecule,
            require_atom_maps=False,
        )
        elements = [
            _descriptor_entry(identifier, descriptor)
            for identifier, descriptor in sorted(registry.items())
        ]
        count = len(elements)
        if count == 0:
            continue
        families = sorted({item["family"] for item in elements})
        carrier_kinds = sorted({item["carrier_kind"] for item in elements})
        records.append(
            {
                "source": "acs_stereomolgraph",
                "record_id": row["ID"],
                "source_categories": [item["descriptor_class"] for item in elements]
                or ["none_extracted"],
                "families": families,
                "carrier_kinds": carrier_kinds,
                "configuration_evidence": "configured_structure",
                "canonicalization_role": "configured_certificate",
                "multiplicity": _multiplicity(count),
                "multiplicity_basis": "extractable_configured_elements",
                "source_element_count": count,
                "extractable_configured_count": count,
                "reference_rs_count": None,
                "attached_tetrahedral_count": None,
                "reference_locus_count": None,
                "structure_available": True,
                "global_chirality_label": row["manual"].lower(),
                "global_label_used_for_canonicalization": False,
                "atom_count": molecule.GetNumAtoms(),
                "configured_elements": elements,
            }
        )
    return records


def _cip_records(path: Path = CIP_ELEMENT_REPORT) -> list[dict[str, Any]]:
    report = _load_json(path)
    records = []
    for source in report["records"]:
        unit_tags = list(source["stereo_units"]) or ["none"]
        if unit_tags == ["none"]:
            continue
        mappings = [
            {"source_unit": unit, **_CIP_UNIT_MAPPINGS[unit]}
            for unit in unit_tags
            if unit != "none"
        ]
        reference_rs_count = len(source["reference_rs_positions"])
        attached_count = len(source["attached_configurations"])
        records.append(
            {
                "source": "cip_validation_suite",
                "record_id": source["id"],
                "source_categories": unit_tags,
                "families": sorted({item["family"] for item in mappings}),
                "carrier_kinds": sorted({item["carrier_kind"] for item in mappings}),
                "configuration_evidence": ("external_local_labels_not_redistributed"),
                "canonicalization_role": ("external_configured_input_pending_checkout"),
                "multiplicity": _multiplicity(
                    reference_rs_count,
                    zero="no_reference_rs",
                ),
                "multiplicity_basis": "reference_rs_positions_only",
                "source_element_count": None,
                "extractable_configured_count": None,
                "reference_rs_count": reference_rs_count,
                "attached_tetrahedral_count": attached_count,
                "reference_locus_count": None,
                "structure_available": False,
                "unit_mappings": mappings,
                "reference_rs_positions": source["reference_rs_positions"],
                "attached_tetrahedral_positions": (source["attached_configurations"]),
                "prior_reverse_renumbering_invariant": source["renumbering_invariant"],
                "inventory_limitation": (
                    "Single/multiple is defined only for reference R/S "
                    "positions. Exact element counts across CT, AT, HE, and "
                    "extended units require the audited external source file."
                ),
            }
        )
    return records


def _rota_records(path: Path = ROTA_LOCUS_REPORT) -> list[dict[str, Any]]:
    report = _load_json(path)
    records = []
    for source in report["records"]:
        source_type = source["chiral_type"]
        mapping = _ROTA_TYPE_MAPPINGS[source_type]
        loci = source.get("reference_pairs", [])
        count = len(loci)
        records.append(
            {
                "source": "chiralfinder_rota",
                "record_id": source["id"],
                "source_categories": [source_type],
                "families": [mapping["family"]],
                "carrier_kinds": [mapping["carrier_kind"]],
                "configuration_evidence": "locus_only_no_handedness",
                "canonicalization_role": "support_canonicalization_only",
                "multiplicity": _multiplicity(count),
                "multiplicity_basis": "reference_axis_loci",
                "source_element_count": count,
                "extractable_configured_count": 0,
                "reference_rs_count": None,
                "attached_tetrahedral_count": None,
                "reference_locus_count": count,
                "structure_available": True,
                "source_mapping": mapping,
                "reference_loci": loci,
                "reference_expanded_paths": source.get(
                    "reference_expanded_paths",
                    [],
                ),
                "prior_reverse_renumbering_invariant": source["renumbering_invariant"],
                "configured_certificate_eligible": False,
            }
        )
    return records


def _count_values(
    records: Iterable[dict[str, Any]],
    field: str,
) -> dict[str, int]:
    return dict(sorted(Counter(str(record[field]) for record in records).items()))


def _summary(
    acs: list[dict[str, Any]],
    cip: list[dict[str, Any]],
    rota: list[dict[str, Any]],
) -> dict[str, Any]:
    acs_family_elements = Counter(
        element["family"] for record in acs for element in record["configured_elements"]
    )
    cip_unit_records = Counter(
        unit for record in cip for unit in record["source_categories"]
    )
    rota_type_records = Counter(record["source_categories"][0] for record in rota)
    return {
        "records": {
            "acs_stereomolgraph": len(acs),
            "cip_validation_suite": len(cip),
            "chiralfinder_rota": len(rota),
            "total": len(acs) + len(cip) + len(rota),
        },
        "acs_stereomolgraph": {
            "multiplicity": _count_values(acs, "multiplicity"),
            "configured_family_elements": dict(sorted(acs_family_elements.items())),
            "configured_certificate_candidates": sum(
                record["canonicalization_role"] == "configured_certificate"
                for record in acs
            ),
            "excluded_graph_only_records": 258 - len(acs),
        },
        "cip_validation_suite": {
            "unit_tag_record_counts_nonexclusive": dict(
                sorted(cip_unit_records.items())
            ),
            "excluded_no_stereo_unit_records": 300 - len(cip),
            "reference_rs_multiplicity": _count_values(
                cip,
                "multiplicity",
            ),
            "attached_tetrahedral_multiplicity": dict(
                sorted(
                    Counter(
                        _multiplicity(
                            record["attached_tetrahedral_count"],
                        )
                        for record in cip
                    ).items()
                )
            ),
            "structure_dependent_processing_status": (
                "pending_audited_external_checkout"
            ),
        },
        "chiralfinder_rota": {
            "source_type_records": dict(sorted(rota_type_records.items())),
            "locus_multiplicity": _count_values(rota, "multiplicity"),
            "reference_loci": sum(record["reference_locus_count"] for record in rota),
            "configured_certificate_candidates": 0,
            "support_canonicalization_candidates": len(rota),
        },
    }


def build_inventory(
    *,
    acs_path: Path = ACS_DATASET,
    cip_report_path: Path = CIP_ELEMENT_REPORT,
    rota_report_path: Path = ROTA_LOCUS_REPORT,
) -> dict[str, Any]:
    """Build the pre-processing inventory without canonicalizing any graph."""
    acs = _acs_records(acs_path)
    cip = _cip_records(cip_report_path)
    rota = _rota_records(rota_report_path)
    cip_report = _load_json(cip_report_path)
    rota_report = _load_json(rota_report_path)
    return {
        "schema": "synkit.stereo-canonicalization-inventory/1",
        "task": "pre_benchmark_source_categorization",
        "processing_status": "no_permutations_or_certificates_computed",
        "inclusion_rule": (
            "Retain only records with configured stereo elements or explicit "
            "source stereo-locus/unit annotations; omit graph-only records."
        ),
        "generated_by": ("Experiment/Stereo/Canonicalization/inventory.py"),
        "source_integrity": {
            "acs_stereomolgraph": {
                "sha256": ACS_SHA256,
                "structures_available": True,
                "configured_truth": True,
            },
            "cip_validation_suite": {
                "sha256": cip_report["dataset"]["audited_sha256"],
                "structures_available": False,
                "configured_truth": True,
                "redistribution_allowed": False,
            },
            "chiralfinder_rota": {
                "sha256": rota_report["dataset"]["audited_sha256"],
                "structures_available": True,
                "configured_truth": False,
                "positive_locus_only": True,
            },
        },
        "family_definitions": {
            "atom_centered": [
                "tetrahedral",
                "square_planar",
                "trigonal_bipyramidal",
                "octahedral",
            ],
            "bond_centered": ["planar_bond"],
            "axis_or_path": [
                "atrop_bond",
                "atrop_axis",
                "cumulene_axis",
                "extended_cis_trans",
                "helical",
            ],
            "plane_centered": ["planar_chirality"],
        },
        "summary": _summary(acs, cip, rota),
        "records": [*acs, *cip, *rota],
    }


def _csv_value(value: Any) -> Any:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, list):
        return ";".join(str(item) for item in value)
    if value is None:
        return ""
    return value


def write_inventory(
    inventory: dict[str, Any],
    *,
    json_path: Path,
    csv_path: Path,
) -> None:
    """Write the rich JSON inventory and its record-level CSV projection."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for record in inventory["records"]:
            writer.writerow(
                {field: _csv_value(record.get(field)) for field in _CSV_FIELDS}
            )
