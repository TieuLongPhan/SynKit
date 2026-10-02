"""Input-only cohort selection and checked worked-example data for benchmarks."""

from collections import defaultdict
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path


RECORD = Path(__file__).resolve().parents[2]/"paper/synister/evidence/worked_unmapped_flower84_v1/record.json"


def selection(inputs):
    buckets = defaultdict(list)
    for row in inputs:
        expected = sha256(row['reaction'].encode()).hexdigest()
        if row['selection_sha256'] != expected:
            raise ValueError('Input selection digest differs')
        buckets[row['source'], row['size_bin']].append(row)
    if set(buckets) != {(source, size) for source in ('FlowER', 'Rhea') for size in range(5)}:
        raise ValueError('Source/size strata missing')
    pilot, reserved = [], []
    for key in sorted(buckets):
        ordered = sorted(buckets[key], key=lambda row: row['selection_sha256'])
        if len(ordered) < 6:
            raise ValueError('Insufficient disjoint source/size inputs')
        pilot.extend(ordered[:2])
        reserved.extend(ordered[2:6])
    return pilot, reserved


def read_record():
    record = json.loads(RECORD.read_text())
    body = {k: v for k, v in record.items() if k != "record_sha256"}
    digest = sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if digest != record["record_sha256"]:
        raise ValueError("Worked reaction record has changed")
    a, b = record["endpoint_adjacency"]
    for cls in record["classes"]:
        m = cls["representative"]
        edits = [[i, j, a[i][j], b[m[i]][m[j]]] for i, j in combinations(range(len(m)), 2)
                 if a[i][j] != b[m[i]][m[j]]]
        if edits != cls["bond_edits"] or sum(abs(x-y) for _, _, x, y in edits) != 6:
            raise ValueError("Drawing ledger differs from saved mapping")
    if record["minimum_cd"] != 6 or [len(c["labeled_maps"]) for c in record["classes"]] != [4, 4]:
        raise ValueError("Worked example counts changed")
    return record
