"""Isolated synthetic-graph worker; uses Endpoint data, not chemical SMILES."""

from hashlib import sha256
import json
from pathlib import Path
import resource
import sys
import time


def endpoint(data):
    from synkit.Chem.Mapper.identifiability import Endpoint
    return Endpoint(tuple(data["atomic_numbers"]), tuple(data["charges"]),
                    tuple(data["hcounts"]), tuple(tuple(b) for b in data["bonds"]))


def perform(task):
    from Experiment.Synister.ablation_worker import enumerate_variant
    started = time.perf_counter()
    r, p = endpoint(task["reactant"]), endpoint(task["product"])
    parsed = time.perf_counter()
    result = enumerate_variant(r, p, variant=task["variant"], target=0,
                               seconds=max(0, task["seconds"]-(parsed-started)),
                               max_maps=task["max_maps"])
    exporting = time.perf_counter()
    maps = result.pop("mappings")
    data = json.dumps(maps, separators=(",", ":")).encode()
    with Path(task["map_path"]).open("xb") as stream:
        stream.write(data)
    result.update(mapping_sha256=sha256(data).hexdigest(), output_bytes=len(data),
                  endpoint_construction_seconds=parsed-started,
                  export_seconds=time.perf_counter()-exporting,
                  end_to_end_seconds=time.perf_counter()-started)
    return result


if __name__ == "__main__":
    task = json.load(sys.stdin)
    limit = int(task["memory_gib"] * 1024**3)
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    started = time.perf_counter()
    try:
        result = perform(task)
    except Exception as exc:
        result = {"complete": False, "minimum_proved": False, "termination": "worker_error",
                  "error_type": type(exc).__name__, "error": str(exc)}
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result.update(worker_seconds=time.perf_counter()-started,
                  cpu_seconds=usage.ru_utime+usage.ru_stime, peak_rss_kib=usage.ru_maxrss)
    print(json.dumps(result, allow_nan=False))
