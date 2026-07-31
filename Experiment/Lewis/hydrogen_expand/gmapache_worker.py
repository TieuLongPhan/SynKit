#!/usr/bin/env python3
"""Run the GranMapache hydrogen-reference backends in their pinned environment."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform

from Experiment.Lewis.hydrogen_expand.benchmark import (
    load_pickle,
    run_reference,
    select_reference_cases,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environment-output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, required=True)
    parser.add_argument("--case-timeout", type=float, required=True)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import gmapache

    dataset = load_pickle(args.dataset)
    if args.limit is not None:
        dataset = dataset[: args.limit]
    dataset, _ = select_reference_cases(dataset)

    rows = []
    for backend in ("an_gm", "rb_gm"):
        rows.extend(
            run_reference(
                dataset,
                args.repetitions,
                args.case_timeout,
                backend=backend,
            )
        )
    write_jsonl(args.output, rows)

    environment = {
        "python": platform.python_version(),
        "gmapache": importlib.metadata.version("gmapache"),
        "gmapache_extension_api": (
            "search_stable_extension"
            if hasattr(gmapache, "search_stable_extension")
            else "search_complete_induced_extension"
        ),
        "networkx": importlib.metadata.version("networkx"),
        "rdkit": importlib.metadata.version("rdkit"),
    }
    args.environment_output.write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
