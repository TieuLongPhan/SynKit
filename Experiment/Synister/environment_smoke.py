"""Record isolated-environment mapper/search smoke checks without using C1."""
import argparse
import json
from pathlib import Path
import sys

from Experiment.Synister.development import execute, environment, save
from Experiment.Synister.freeze_environment import closure, ROOTS
from Experiment.Synister.worked_oracle import REACTION


def run(output, lock):
    expected = json.loads(lock.read_text())["packages"]
    assert closure(ROOTS) == expected
    config = Path(sys.prefix) / "pyvenv.cfg"
    assert "include-system-site-packages = false" in config.read_text().lower()
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for stage in ("slap", "rxnmapper", "exact"):
        task = dict(reaction=REACTION, case_id="worked", stage=stage,
                    search_seconds=10, export_joint_labels=True)
        results.append(execute(task, 60, output))
    assert all(x["status"] == "valid" for x in results[:2]), results[:2]
    assert results[2]["status"] == "complete" and results[2]["minimum"] == 6
    save(output / "summary.json", {"status": "verified", "scope": "clean environment worked-example smoke, not C1 outcomes",
                                    "venv_config": config.read_text(), "environment": environment(),
                                    "stages": {x["stage"]: x["status"] for x in results}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    run(args.output, args.lock)
