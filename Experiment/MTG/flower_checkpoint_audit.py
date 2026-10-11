#!/usr/bin/env python
"""Run a frozen CPU calibration cohort with the published FlowER-small model.

Use release 1.0.0, its settings and the original small checkpoint. Store rounded
matrices as well as decoded outputs; teacher-forced steps are not beam histories.
The cohort is selected and saved before inference, without inspecting predictions.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
from time import perf_counter
import zipfile

import networkx as nx
import numpy as np
from rdkit import Chem
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from synkit.Graph.Morphism import (  # noqa: E402
    ISOTOPE_ELECTRON_LLG_SCHEMA,
    LewisLabelledGraph,
    derive_electron_labeled_graph,
)
from synkit.Rule.Apply import RuleSpan, apply_dpo  # noqa: E402

COMMIT = "152c02cbfeccacd659c3bf60326d088f79ec0f3f"


def _digest(path):
    value = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _freeze(archive, path, sequences, samples, seed):
    """Select complete first sequence blocks in immutable archive order."""
    if path.exists():
        result = json.loads(path.read_text())
        assert result["protocol"]["sequence_limit"] == sequences
        assert result["protocol"]["samples_per_step"] == samples
        assert result["protocol"]["seed"] == seed
        assert result["source_archive_sha256"] == _digest(archive)
        return result
    rows, ids = [], []
    member = "data/flower_dataset/test.txt"
    with zipfile.ZipFile(archive) as zipped, zipped.open(member) as stream:
        for number, raw in enumerate(stream, 1):
            line = raw.decode().strip()
            sequence = line.rsplit("|", 1)[1]
            if sequence not in ids:
                if len(ids) >= sequences:
                    break
                ids.append(sequence)
            rows.append({"source_line": number, "sequence_id": sequence, "line": line})
    result = {
        "schema": "synkit.mtg-flower-frozen-calibration/1",
        "source_archive_sha256": _digest(archive),
        "member": member,
        "protocol": {
            "selection": "First complete sequence blocks in original test-file order; no outcome filtering.",
            "sequence_limit": sequences,
            "samples_per_step": samples,
            "seed": seed,
            "teacher_forcing": True,
            "device": "CPU",
            "ode": "Published dopri5, atol=rtol=1e-4.",
            "beam_search": False,
        },
        "sequence_ids": ids,
        "rows": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def matrix_state(matrix, molecule):
    """Translate an exact pair-bond matrix, without deriving electrons from SMILES."""
    if not np.array_equal(matrix, matrix.T):
        raise ValueError("Rounded matrix is asymmetric.")
    if not np.isfinite(matrix).all() or not np.array_equal(matrix, np.rint(matrix)):
        raise ValueError("Matrix has nonfinite or noninteger coordinates.")
    if (matrix < 0).any():
        raise ValueError("Matrix has negative electron coordinates.")
    graph = nx.Graph()
    table = Chem.GetPeriodicTable()
    for atom in molecule.GetAtoms():
        carrier = atom.GetAtomMapNum()
        electrons = int(matrix[carrier - 1, carrier - 1])
        graph.add_node(
            carrier,
            element=atom.GetSymbol(),
            isotope=atom.GetIsotope(),
            aromatic=False,
            hcount=atom.GetTotalNumHs(),
            radical=electrons % 2,
            lone_pairs=electrons // 2,
            valence_electrons=table.GetNOuterElecs(atom.GetAtomicNum()),
        )
    if set(graph) != set(range(1, len(matrix) + 1)):
        raise ValueError("This adapter requires contiguous supplied atom maps.")
    for first in range(len(matrix)):
        for second in range(first + 1, len(matrix)):
            order = int(matrix[first, second])
            if order > 3:
                raise ValueError(
                    "Bond occupancy exceeds this single/double/triple adapter."
                )
            if order:
                graph.add_edge(first + 1, second + 1, sigma_order=1, pi_order=order - 1)
    return LewisLabelledGraph.from_networkx(
        derive_electron_labeled_graph(graph), ISOTOPE_ELECTRON_LLG_SCHEMA
    )


def _origins(state):
    graph = state.to_networkx()
    return tuple(
        sorted(
            (n, tuple(sorted(k for k in graph[n] if graph.nodes[k]["element"] == "H")))
            for n, attrs in graph.nodes(data=True)
            if attrs["element"] == "O"
        )
    )


def _prediction(matrix, count, source, input_matrix, data_utils):
    row = {
        "count": int(count),
        "rounded_matrix": matrix.tolist(),
        "electron_sum_change": float(matrix.sum() - input_matrix.sum()),
    }
    origins = None
    try:
        left, right = matrix_state(input_matrix, source), matrix_state(matrix, source)
        rule = RuleSpan.from_mapping(
            left, right, {n: n for n in left.node_ids}, boundary="closed"
        )
        applied = apply_dpo(rule, left, {n: n for n in left.node_ids})
        assert applied.certificate.replay().valid
        row["matrix_status"] = "CERTIFIED_CLOSED_PAIR_BOND_STATE"
        origins = _origins(right)
        row["oxygen_hydrogen_origins"] = origins
    except ValueError as error:
        row["matrix_status"] = "UNSUPPORTED_OR_REFUSED_MATRIX"
        row["matrix_message"] = str(error)[:240]
    try:
        decoded = data_utils.BEmatrix_to_mol(source, matrix)
        smiles = Chem.MolToSmiles(decoded, isomericSmiles=False, allHsExplicit=True)
        parsed = Chem.MolFromSmiles(smiles, data_utils.ps)
        if parsed is None:
            raise ValueError("Published decoded SMILES fails sanitizing parse.")
        smiles = Chem.MolToSmiles(parsed, isomericSmiles=False, allHsExplicit=True)
        reconstructed = data_utils.get_BE_matrix(smiles)
        row["decode_status"] = "PARSED"
        row["mapped_smiles"] = smiles
        row["decoded_electron_sum_change"] = float(reconstructed.sum() - matrix.sum())
        row["exact_matrix_roundtrip"] = bool(np.array_equal(matrix, reconstructed))
        copy = Chem.Mol(parsed)
        for atom in copy.GetAtoms():
            atom.SetAtomMapNum(0)
        row["unmapped_projection"] = Chem.MolToSmiles(copy, isomericSmiles=False)
    except Exception as error:
        row["decode_status"] = "DECODING_REFUSED"
        row["decode_message"] = f"{type(error).__name__}: {error}"[:240]
    return row, origins


def run(args):
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.flower, text=True
    ).strip()
    assert commit == COMMIT, "Use the pinned original release for the small model."
    frozen = _freeze(args.data, args.cohort, args.sequences, args.samples, args.seed)
    sys.path.insert(0, str(args.flower))
    from settings import Args
    from train import init_model, init_loader
    from eval_multiGPU import predict_batch, custom_round
    from utils import data_utils

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    state = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    config = state["args"]
    assert config is Args  # Saved class references resolve against this release.
    config.device, config.local_rank = torch.device("cpu"), -1
    config.load_from, config.model_path = None, "/tmp/mtg-flower-model-work"
    config.num_workers, config.sample_size = 0, args.samples
    model, flow, _ = init_model(config)
    model.load_state_dict(
        {k.replace("module.", ""): v for k, v in state["state_dict"].items()}
    )
    model.eval()
    report = {
        "schema": "synkit.mtg-flower-checkpoint-calibration/1",
        "source": {
            "commit": commit,
            "checkpoint_sha256": _digest(args.checkpoint),
            "cohort_sha256": _digest(args.cohort),
        },
        "script_sha256": _digest(Path(__file__)),
        "protocol": frozen["protocol"],
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "threads": args.threads,
            "packages": {
                key: metadata.version(key)
                for key in (
                    "torch",
                    "numpy",
                    "rdkit",
                    "networkx",
                    "torchdiffeq",
                    "iteround",
                )
            },
        },
        "architecture": {
            key: getattr(config, key)
            for key in (
                "emb_dim",
                "enc_num_layers",
                "enc_heads",
                "enc_filter_size",
                "sigma",
                "rbf_high",
                "rbf_gap",
            )
        },
        "adaptations": (
            "CPU, zero loader workers, one source step per batch, model.eval and "
            "torch.no_grad as in published single-step evaluation. Current dependency "
            "versions recorded; not an exact historical environment reproduction."
        ),
        "rows": [],
        "claim_limits": (
            "Calibration only, selected by file order. Teacher forcing and template-imputed "
            "labels do not establish full predicted histories, physical mechanism truth or "
            "population performance."
        ),
    }
    for row_index, frozen_row in enumerate(frozen["rows"]):
        started = perf_counter()
        dataset = data_utils.ReactionDataset(
            config, [frozen_row["line"]], parallel=False
        )
        if dataset.data_size != 1:
            report["rows"].append({**frozen_row, "status": "UNSUPPORTED_INPUT"})
            continue
        loader = init_loader(
            config, dataset, batch_size=4096, shuffle=False, use_sort=False
        )
        batch = next(iter(loader))
        batch.to(config.device)
        with torch.no_grad():
            raw = predict_batch(config, row_index, batch, model, flow, 1)[-1]
            matrices, counts = torch.unique(
                custom_round(raw), dim=0, return_counts=True
            )
        input_matrix = batch.src_matrices[0].cpu().numpy()
        source = Chem.MolFromSmiles(dataset.src_smis[0], data_utils.ps)
        predictions, classes = [], defaultdict(list)
        for matrix, count in zip(matrices.cpu().numpy(), counts.cpu().numpy()):
            prediction, origins = _prediction(
                matrix, count, source, input_matrix, data_utils
            )
            predictions.append(prediction)
            if origins is not None and prediction["decode_status"] == "PARSED":
                classes[prediction["unmapped_projection"]].append(origins)
        item = {
            **frozen_row,
            "status": "INFERRED",
            "elapsed_seconds": perf_counter() - started,
            "input_matrix": input_matrix.tolist(),
            "predictions": predictions,
            "projection_origin_collisions": sum(
                len(set(values)) > 1 for values in classes.values()
            ),
            "continuous_electron_sum_max_absolute_error": float(
                torch.max(torch.abs(raw.sum((1, 2)) - float(input_matrix.sum())))
            ),
        }
        report["rows"].append(item)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"Step {row_index + 1}/{len(frozen['rows'])}: "
            f"{len(predictions)} unique matrices, {item['elapsed_seconds']:.2f}s",
            flush=True,
        )
    predictions = [p for row in report["rows"] for p in row.get("predictions", [])]
    report["summary"] = {
        "sequence_blocks": len(frozen["sequence_ids"]),
        "source_steps": len(frozen["rows"]),
        "inferred_steps": sum(r["status"] == "INFERRED" for r in report["rows"]),
        "samples": sum(p["count"] for p in predictions),
        "unique_predictions": len(predictions),
        "sample_weighted_matrix_status": dict(
            Counter(
                {
                    key: sum(
                        p["count"] for p in predictions if p["matrix_status"] == key
                    )
                    for key in {p["matrix_status"] for p in predictions}
                }
            )
        ),
        "sample_weighted_decode_status": dict(
            Counter(
                {
                    key: sum(
                        p["count"] for p in predictions if p["decode_status"] == key
                    )
                    for key in {p["decode_status"] for p in predictions}
                }
            )
        ),
        "rounded_electron_sum_changed_samples": sum(
            p["count"] for p in predictions if p["electron_sum_change"] != 0
        ),
        "parsed_electron_sum_changed_samples": sum(
            p["count"]
            for p in predictions
            if p.get("decoded_electron_sum_change", 0) != 0
        ),
        "origin_projection_collision_classes": sum(
            r.get("projection_origin_collisions", 0) for r in report["rows"]
        ),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flower", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequences", type=int, default=3)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()
