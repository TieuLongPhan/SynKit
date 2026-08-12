"""Tests for retained reaction-level partial-expansion timings."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from Experiment.Lewis.partial_expand.plot import (
    load_runtime_samples,
    plot_runtime,
)
from Experiment.Lewis.partial_expand.repeat_external import (
    ROOT,
    isolate_historical_imports,
)
from Experiment.Lewis.partial_expand.timing_artifacts import (
    open_text,
    write_timing_artifact,
)


def _write_run(
    directory: Path,
    dataset: Path,
    method: str,
    repetition: int,
    values: dict[int, float],
) -> None:
    generated = directory / f"{method}-{repetition}-generated.jsonl.gz"
    with open_text(generated, "wt") as handle:
        for record_id, seconds in values.items():
            handle.write(
                json.dumps(
                    {
                        "method": method,
                        "record_id": record_id,
                        "status": "OUTPUT",
                        "generation_seconds": seconds,
                    }
                )
                + "\n"
            )
    write_timing_artifact(
        source=generated,
        output=(directory / f"general-{method}-run-{repetition:02d}-timings.json.gz"),
        dataset=dataset,
        method=method,
        repetition=repetition,
    )


def test_runtime_loader_reduces_five_measurements_per_reaction(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.json"
    dataset.write_text("[]\n")
    bases = {"rb1": 0.001, "rb2": 0.002, "gm": 0.003}
    for method, base in bases.items():
        for repetition in range(1, 6):
            _write_run(
                tmp_path,
                dataset,
                method,
                repetition,
                {
                    11: base + repetition * 0.0001,
                    22: base + 0.001 + repetition * 0.0001,
                },
            )

    samples = load_runtime_samples(tmp_path)

    assert samples["rb1"] == pytest.approx([1.3, 2.3])
    assert samples["rb2"] == pytest.approx([2.3, 3.3])
    assert samples["gm"] == pytest.approx([3.3, 4.3])


def test_runtime_loader_can_pool_executions_and_plot(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.json"
    dataset.write_text("[]\n")
    for method, base in {"rb1": 0.001, "rb2": 0.002, "gm": 0.003}.items():
        for repetition in range(1, 3):
            _write_run(
                tmp_path,
                dataset,
                method,
                repetition,
                {11: base, 22: base + repetition * 0.0001},
            )

    samples = load_runtime_samples(
        tmp_path,
        repetitions=2,
        reduction="pooled",
    )
    output = tmp_path / "runtime"
    plot_runtime(samples, output)

    assert all(len(values) == 4 for values in samples.values())
    assert output.with_suffix(".pdf").is_file()
    assert output.with_suffix(".png").is_file()


def test_historical_worker_excludes_active_synkit_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    historical_checkout = "/tmp/PartialAAMs-008"
    environment_site_packages = "/opt/conda/envs/aam/lib/python3.11/site-packages"
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(
        sys,
        "path",
        ["", str(ROOT), historical_checkout, environment_site_packages],
    )

    isolate_historical_imports()

    assert "" not in sys.path
    assert str(ROOT) not in sys.path
    assert historical_checkout in sys.path
    assert environment_site_packages in sys.path
