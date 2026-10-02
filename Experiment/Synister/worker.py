"""Single-case worker; process limits are imposed before scientific imports."""

import argparse
import json
import resource
import sys
import time
from dataclasses import asdict


def perform(task):
    from synkit.Chem.Mapper.identifiability import extract_label, parse_reaction
    from synkit.Chem.Mapper.prediction_adapter import align_mapped_prediction, predict_slap

    reaction, stage = task["reaction"], task["stage"]
    r, p = parse_reaction(reaction)
    if stage in ("r2_kekule", "r2_shell1", "r2_shell2"):
        from Experiment.Synister.representation_sensitivity import kekule_endpoint, ordering_control, search, weighted_cost
        if stage == "r2_kekule":
            left, right = reaction.split(">>")
            _, a = kekule_endpoint(left)
            _, b = kekule_endpoint(right)
            candidates = [(weighted_cost(a,b,m), tuple(m)) for m in task["prediction_mappings"]]
            mapping = list(min(candidates)[1]) if candidates else None
            result = search(r,p,objective_endpoints=(a,b),initial_mapping=mapping,
                            seconds=task["search_seconds"])
            result.update(endpoint_changed=[r != a,p != b], initial_mapping=mapping,
                          objective_endpoints=[asdict(a),asdict(b)],
                          ordering_controls=[ordering_control(left),ordering_control(right)])
            return result
        return search(r,p,target=task["target"],seconds=task["search_seconds"])
    if stage == "binary_exact":
        from Experiment.Synister.binary_sensitivity import search
        return search(r, p, task.get("initial_mapping"), task["search_seconds"])
    if stage == "annotation_replay":
        from Experiment.Synister.replay_annotations import replay_case
        return replay_case(task)
    if stage == "slap":
        prediction = predict_slap(reaction)
        return {"status": "valid", "prediction": prediction,
                "label": asdict(extract_label(r, p, prediction["mapping"]))}
    if stage == "rxnmapper":
        import torch
        from rxnmapper import RXNMapper

        torch.set_num_threads(1)
        torch.manual_seed(0)
        start = time.monotonic()
        mapper = RXNMapper()
        loaded = time.monotonic()
        raw = mapper.get_attention_guided_atom_maps([reaction])[0]
        predicted = time.monotonic()
        # Preserve raw model output even when validation fails.
        record = {"raw_prediction": raw, "model_load_seconds": loaded - start,
                  "inference_seconds": predicted - loaded}
        try:
            aligned = align_mapped_prediction(reaction, raw["mapped_rxn"])
            record.update(status="valid", prediction=asdict(aligned),
                          label=asdict(extract_label(r, p, aligned.mapping)))
        except (ValueError, KeyError) as exc:
            record.update(status="invalid_prediction", error=str(exc))
        return record
    if stage == "exact":
        from synkit.Chem.Mapper.exact.distance import enumerate_distance_mappings
        from synkit.Chem.Mapper.numerical_scope import validate_study_domain

        numerical_domain = validate_study_domain(r, p)
        initial_mapping = task.get("initial_mapping")
        initial_cost = None
        if initial_mapping is not None:
            initial_cost = str(extract_label(r, p, initial_mapping).weighted_distance)
        result = enumerate_distance_mappings(
            [r.graph(), p.graph()], CD="minimal", binary=False,
            max_bijections=None, max_mappings=100000,
            symmetry_pruning=True, symmetry_node_properties=("hcounts", "charges"),
            time_limit_seconds=task["search_seconds"],
            initial_mapping=initial_mapping,
        )
        # Conservatively require total search closure before treating the
        # reported minimum or emitted labels as a complete optimal set.
        closed = result.complete and result.status == "complete"
        labels = {}
        joint_labels = {}
        if closed:
            for mapping in result.mappings:
                label = extract_label(r, p, mapping)
                if label.weighted_distance != result.cost:
                    raise ValueError("Independent integer objective disagrees with search")
                key = tuple(sorted(label.changed_bonds))
                labels.setdefault(key, {"mapping": mapping, "label": asdict(label)})
                if task.get("export_joint_labels", False):
                    joint_labels.setdefault(label, {"mapping": mapping, "label": asdict(label)})
        return {
            "status": "complete" if closed else "unresolved",
            "numerical_domain": numerical_domain,
            "minimum_proved": closed, "enumeration_complete": closed,
            "labels_complete": closed, "minimum": result.cost,
            "solver_status": result.status, "truncation_reason": result.truncation_reason,
            "emitted_representatives": len(result.mappings),
            "mapping_scope": result.scope,
            "symmetry_search_complete": result.symmetry_search_complete,
            "initial_mapping": initial_mapping, "initial_cost": initial_cost,
            "labels": [labels[k] for k in sorted(labels)],
            "joint_labels": list(joint_labels.values()) if task.get("export_joint_labels", False) else None,
            "joint_labels_complete": closed and task.get("export_joint_labels", False),
            "note": "labels: one witness per fixed changed-bond set; joint_labels: complete joint-label witnesses only when explicitly requested and closed",
        }
    if stage == "score":
        from synkit.Chem.Mapper.evaluation import ExactBondEvaluator

        backend = task.get("score_backend", "full-group")
        if backend == "support-stabilizer":
            from synkit.Chem.Mapper.orbit_evaluation import SupportOrbitEvaluator
            scorer = SupportOrbitEvaluator(r, time_limit_seconds=task["score_seconds"])
        elif backend == "full-group":
            scorer = ExactBondEvaluator(r, time_limit_seconds=task["score_seconds"])
        else:
            raise ValueError("Unknown scoring backend")
        labels = [extract_label(r, p, item["mapping"]).changed_bonds for item in task["labels"]]
        a = extract_label(r, p, task["prediction_a"]).changed_bonds
        b = extract_label(r, p, task["prediction_b"]).changed_bonds
        lo, hi = scorer.paired_envelope(a, b, labels, labels_complete=True)
        def witness(w):
            return {"difference": str(w.difference), "label": sorted(w.label),
                    "a_score": str(w.a.score), "b_score": str(w.b.score),
                    "a_transporter": w.a.transporter, "b_transporter": w.b.transporter}
        return {"status": "complete", "lower": witness(lo), "upper": witness(hi),
                "width": str(hi.difference - lo.difference),
                "fixed_bond_labels": len(set(labels)),
                "bond_label_orbits": len({scorer.orbit_key(x) for x in labels}),
                "reactant_automorphisms": len(scorer.automorphisms) if backend == "full-group" else None,
                "symmetry_backend": backend,
                "isomorphism_extension_calls": getattr(scorer, "isomorphism_calls", None)}
    if stage == "annotations":
        from synkit.Chem.Mapper.annotation_evaluation import analyze_annotations

        reference, reference_status = None, {"status": "not_supplied"}
        if task.get("reference") is not None:
            try:
                aligned = align_mapped_prediction(reaction, task["reference"])
                reference = aligned.mapping
                reference_status = {"status": "valid", "alignment": asdict(aligned)}
            except ValueError as exc:
                reference_status = {"status": "invalid_reference", "reason": str(exc)}
        output = analyze_annotations(
            r, p, [x["mapping"] for x in task["joint_labels"]], task["prediction_a"], task["prediction_b"],
            joint_labels_complete=task["joint_labels_complete"], minimum=task["minimum"],
            reference_mapping=reference, metric_seconds=task.get("metric_seconds", 10),
            canonical_seconds=task.get("canonical_seconds", 0.25))
        output["reference_input"] = reference_status
        return output
    raise ValueError(f"Unknown stage: {stage}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory-gib", type=int, default=6)
    args = parser.parse_args()
    limit = args.memory_gib * 1024**3
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    started = time.monotonic()
    task = json.load(sys.stdin)
    try:
        result = perform(task)
    except Exception as exc:
        result = {"status": "error", "error_type": type(exc).__name__, "error": str(exc)}
    result["worker_seconds"] = time.monotonic() - started
    result["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
