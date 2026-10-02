"""R1 binary edge-presence objective with original weighted label extraction."""
from dataclasses import asdict

from synkit.Chem.Mapper.identifiability import extract_label
from synkit.Chem.Mapper.numerical_scope import validate_study_domain


def binary_cost(r, p, mapping):
    n = len(r.atomic_numbers)
    if sorted(mapping) != list(range(n)) or any(
            r.atomic_numbers[i] != p.atomic_numbers[mapping[i]] for i in range(n)):
        raise ValueError("Not an element-compatible bijection")
    a = {(i, j) for i, j, _ in r.bonds}
    b = {(i, j) for i, j, _ in p.bonds}
    return sum(((i, j) in a) != (tuple(sorted((mapping[i], mapping[j]))) in b)
               for i in range(n) for j in range(i+1, n))


def seed(r, p, predictions):
    candidates = [(binary_cost(r, p, x["prediction"]["mapping"]),
                   tuple(x["prediction"]["mapping"]), method)
                  for method, x in predictions.items() if x["status"] == "valid"]
    if not candidates:
        return None, None
    cost, mapping, method = min(candidates)
    return list(mapping), {"method": method, "binary_cost": cost,
                           "policy": "minimum-binary-cost_then_mapping_then_method"}


def search(r, p, initial_mapping=None, seconds=60, max_mappings=100000):
    from synkit.Chem.Mapper.exact.distance import enumerate_distance_mappings
    domain = validate_study_domain(r, p)
    result = enumerate_distance_mappings(
        [r.graph(), p.graph()], CD="minimal", binary=True, symmetry_pruning=False,
        max_bijections=None, max_mappings=max_mappings,
        initial_mapping=initial_mapping, time_limit_seconds=seconds)
    complete = result.complete and result.status == "complete"
    labels, joint = {}, {}
    if complete:
        for mapping in result.mappings:
            if binary_cost(r, p, mapping) != result.cost:
                raise ValueError("Independent binary objective disagrees with search")
            value = extract_label(r, p, mapping)
            record = {"mapping": list(mapping), "label": asdict(value)}
            labels.setdefault(tuple(sorted(value.changed_bonds)), record)
            joint.setdefault(value, record)
    return {"status": "complete" if complete else "unresolved",
            "objective": "binary_edge_presence", "label_endpoints": "original_weighted",
            "numerical_domain": domain, "minimum": result.cost,
            "minimum_proved": complete, "enumeration_complete": complete,
            "labels_complete": complete, "joint_labels_complete": complete,
            "solver_status": result.status, "truncation_reason": result.truncation_reason,
            "emitted_representatives": len(result.mappings), "mapping_scope": result.scope,
            "symmetry_pruning": False, "initial_mapping": initial_mapping,
            "initial_cost": binary_cost(r, p, initial_mapping) if initial_mapping is not None else None,
            "labels": [labels[k] for k in sorted(labels)], "joint_labels": list(joint.values())}
