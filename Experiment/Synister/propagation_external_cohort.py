"""Build and independently recheck a provenance-bound external cohort."""

import argparse
import csv
import gzip
from hashlib import sha256
import json
from pathlib import Path

from Experiment.Synister.select_development import endpoint_key
from synkit.Chem.Mapper.identifiability import parse_reaction
from synkit.Chem.Mapper.numerical_scope import validate_study_domain
from synkit.Chem.Mapper.prediction_adapter import unmapped_input


ARTIFACT_NAMES = {
    "inputs.json",
    "selection.json",
    "real_selection.json",
    "references.json",
}
PILOT_SOURCE_FILES = {
    "pilot_structure.csv",
}
SALT = "synister-propagation-independent-v2"
STRATA = ("le20", "21to40", "41to60", "61to80", "gt80")
FRAMES = {
    "flower": "c1_selection_v1",
    "rhea": "c2_selection_v1",
}


def _digest(data):
    return sha256(data).hexdigest()


def _read(path):
    return json.loads(path.read_text())


def _inventory(root, artifact_paths=None):
    """Collect identities and endpoint pairs from all prior selected inputs."""
    artifacts = []
    endpoints, flower_groups, rhea_ids = set(), set(), set()
    if artifact_paths is None:
        paths = []
        for base in (
            root / "Experiment/Synister/runs",
            root / "paper/synister/evidence",
        ):
            if not base.exists():
                continue
            for path in base.rglob("*"):
                if not path.is_file():
                    continue
                if any(
                    part.startswith("propagation_external_")
                    for part in path.parts
                ):
                    continue
                if path.name in ARTIFACT_NAMES or path.name == "inputs.csv.gz":
                    paths.append(path)
                elif (
                    path.name in PILOT_SOURCE_FILES
                    and "historical_figure_replay" in path.parts
                ):
                    paths.append(path)
        paths.sort()
    else:
        paths = [root / relative for relative in artifact_paths]
        if any(not path.is_file() for path in paths):
            raise ValueError("An exclusion-inventory artifact is missing")
    for path in paths:
        raw = path.read_bytes()
        artifacts.append(
            {
                "path": str(path.relative_to(root)),
                "sha256": _digest(raw),
            }
        )

        if path.suffix == ".json":
            value = json.loads(raw)

            def visit(item):
                if isinstance(item, dict):
                    endpoint = item.get("endpoint_sha256")
                    if isinstance(endpoint, str) and len(endpoint) == 64:
                        endpoints.add(endpoint)
                    reaction = item.get("reaction") or item.get(
                        "original_reaction"
                    )
                    if isinstance(reaction, str) and ">>" in reaction:
                        try:
                            canonical = unmapped_input(reaction)
                            endpoints.add(endpoint_key(canonical))
                        except (ValueError, TypeError):
                            pass
                    group = item.get("source_sequence")
                    if group is not None:
                        flower_groups.add(str(group))
                    original = item.get("original_id") or item.get(
                        "reaction_id"
                    )
                    if (
                        isinstance(original, str)
                        and ":" in original
                        and not original.startswith("RHEA:")
                    ):
                        flower_groups.add(original.split(":", 1)[0])
                    if item.get("master_id") is not None:
                        rhea_ids.add(str(item["master_id"]))
                    for child in item.values():
                        visit(child)
                elif isinstance(item, list):
                    for child in item:
                        visit(child)

            visit(value)
        elif path.name == "inputs.csv.gz":
            with gzip.open(path, "rt") as stream:
                for row in csv.DictReader(stream):
                    original = row.get("reaction_id")
                    if (
                        isinstance(original, str)
                        and original.startswith("RHEA:")
                    ):
                        rhea_ids.add(original.split(":", 1)[1])
                    elif isinstance(original, str) and ":" in original:
                        flower_groups.add(original.split(":", 1)[0])
                    reaction = row.get("mapped_reaction") or row.get(
                        "reaction"
                    )
                    if isinstance(reaction, str):
                        if "|" in reaction:
                            reaction = reaction.rsplit("|", 1)[0]
                        try:
                            canonical = unmapped_input(reaction)
                            endpoints.add(endpoint_key(canonical))
                        except (ValueError, TypeError):
                            pass
        elif path.name == "pilot_structure.csv":
            with path.open() as stream:
                for row in csv.DictReader(stream):
                    original = row.get("reaction_id", "")
                    if ":" in original:
                        flower_groups.add(original.split(":", 1)[0])
    return {
        "artifacts": artifacts,
        "endpoint_sha256": sorted(endpoints),
        "flower_source_groups": sorted(flower_groups),
        "rhea_master_ids": sorted(rhea_ids),
    }


def _selected_inputs(root, inventory):
    excluded_endpoints = set(inventory["endpoint_sha256"])
    excluded_flower = set(inventory["flower_source_groups"])
    excluded_rhea = set(inventory["rhea_master_ids"])
    selected = []
    source_frames = {}
    for source, dirname in FRAMES.items():
        base = root / "Experiment/Synister/runs" / dirname
        frame_path = base / "frame.json"
        manifest_path = base / "manifest.json"
        audit_path = base / "selection_audit.json"
        frame = _read(frame_path)
        manifest = _read(manifest_path)
        audit = _read(audit_path)
        if audit.get("status") != "verified":
            raise ValueError(
                f"Source frame audit is not verified: {audit_path}"
            )
        if _digest(frame_path.read_bytes()) != manifest["frame_sha256"]:
            raise ValueError(f"Source frame hash mismatch: {frame_path}")
        source_frames[source] = {
            "directory": str(base.relative_to(root)),
            "frame_sha256": _digest(frame_path.read_bytes()),
            "manifest_sha256": _digest(manifest_path.read_bytes()),
            "selection_audit_sha256": _digest(audit_path.read_bytes()),
            "selection_audit_status": audit["status"],
        }
        candidates = []
        for row in frame:
            identity = (
                str(row["source_sequence"])
                if source == "flower"
                else str(row["master_id"])
            )
            source_exclusions = (
                excluded_flower if source == "flower" else excluded_rhea
            )
            if identity in source_exclusions:
                continue
            if row["endpoint_sha256"] in excluded_endpoints:
                continue
            n = int(row["heavy_atoms"])
            if not 1 <= n <= 256:
                continue
            candidates.append(row)
        rank_key = "source_sequence" if source == "flower" else "master_id"

        def rank(row):
            return (
                _digest(f"{SALT}\0{source}\0{row[rank_key]}".encode()),
                str(row[rank_key]),
            )

        for stratum in STRATA:
            in_stratum = [
                row
                for row in candidates
                if _stratum(int(row["heavy_atoms"])) == stratum
            ]
            in_stratum.sort(key=rank)
            if len(in_stratum) < 10:
                raise ValueError(
                    f"Only {len(in_stratum)} unused {source} candidates "
                    f"in {stratum}"
                )
            for row in in_stratum[:10]:
                reactant, product = parse_reaction(row["reaction"])
                validate_study_domain(reactant, product)
                if len(reactant.atomic_numbers) != len(product.atomic_numbers):
                    raise ValueError(
                        "External cohort contains an unbalanced pair"
                    )
                if len(reactant.atomic_numbers) != int(row["heavy_atoms"]):
                    raise ValueError(
                        "External cohort atom count differs from frame"
                    )
                if endpoint_key(row["reaction"]) != row["endpoint_sha256"]:
                    raise ValueError(
                        "External cohort endpoint key differs from frame"
                    )
                selected.append(
                    {
                        "benchmark_id": f"reaction_{100 + len(selected):03d}",
                        "reaction": row["reaction"],
                        "source_cohort": source,
                        "source_identity": str(row[rank_key]),
                        "endpoint_sha256": row["endpoint_sha256"],
                        "heavy_atoms": int(row["heavy_atoms"]),
                        "atom_stratum": stratum,
                        "selection_rank": rank(row)[0],
                    }
                )
    return selected, source_frames


def create_manifest(root, artifact_paths=None):
    """Freeze the first 50 disjoint candidates per source under a new rank."""
    root = Path(root).resolve()
    inventory = _inventory(root, artifact_paths)
    selected, source_frames = _selected_inputs(root, inventory)
    return {
        "schema_version": 1,
        "cohort": "synister_external_50_flower_50_rhea_v2",
        "purpose": "independent prospective performance evaluation",
        "selection_salt": SALT,
        "selection_rule": (
            "ten lowest SHA-256 ranks per atom-size stratum after source "
            "and endpoint exclusions"
        ),
        "heavy_atom_scope": [1, 256],
        "source_frames": source_frames,
        "exclusion_inventory": inventory,
        "inputs": selected,
    }


def _stratum(heavy_atoms):
    for limit, name in (
        (20, "le20"),
        (40, "21to40"),
        (60, "41to60"),
        (80, "61to80"),
    ):
        if heavy_atoms <= limit:
            return name
    return "gt80"


def verify_manifest(manifest, root):
    """Rebuild source inventory and selection; reject provenance drift."""
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported external cohort manifest schema")
    if manifest.get("cohort") != "synister_external_50_flower_50_rhea_v2":
        raise ValueError("Unexpected external cohort name")
    artifact_paths = [
        row["path"] for row in manifest["exclusion_inventory"]["artifacts"]
    ]
    actual = create_manifest(root, artifact_paths)
    if manifest != actual:
        raise ValueError(
            "External cohort differs from its reproducible selection"
        )
    return actual["inputs"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = create_manifest(args.root)
    with args.output.open("xb") as stream:
        stream.write(json.dumps(manifest, sort_keys=True, indent=2).encode())
        stream.write(b"\n")
    counts = {}
    for row in manifest["inputs"]:
        counts[row["source_cohort"]] = counts.get(row["source_cohort"], 0) + 1
    print(
        json.dumps(
            {"manifest": str(args.output), "selected": counts}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
