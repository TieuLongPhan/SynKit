"""FLOWER elementary-state graph combination tests."""

from __future__ import annotations

import gzip
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Experiment.Lewis.FLOWER.combine_mechanisms import (  # noqa: E402
    combine_block,
    combine_mechanisms,
)


def test_combine_block_emits_each_reachable_terminal() -> None:
    records, report = combine_block(
        "7",
        1,
        [
            ("A", "B"),
            ("B", "C"),
            ("B", "D"),
            ("C", "C"),
            ("D", "D"),
            ("X", "Y"),
        ],
    )

    assert [reaction for reaction, _ in records] == [
        "A>>C",
        "A>>D",
        "X>>Y",
    ]
    assert report["components"] == 2
    assert report["source_terminal_pairs"] == 3


def test_combine_block_reports_source_free_cycle() -> None:
    records, report = combine_block("PM", 1, [("A", "B"), ("B", "A")])

    assert records == []
    assert report["skipped_cyclic_components"] == 1
    assert report["skipped_samples"][0]["sources"] == 0


def test_combine_mechanisms_writes_split_and_manifest(tmp_path: Path) -> None:
    source = tmp_path / "train.txt"
    source.write_text(
        "A>>B|1\n" "B>>C|1\n" "B>>D|1\n" "C>>C|1\n" "D>>D|1\n" "X>>Y|PC\n",
        encoding="utf-8",
    )
    manifest = combine_mechanisms([source], tmp_path / "full")
    report = manifest["splits"][0]

    assert report["output"]["rows"] == 3
    assert report["rows_by_category"] == {"PC": 1, "numeric": 2}
    with gzip.open(report["output"]["path"], "rt", encoding="utf-8") as handle:
        reactions = [line.rsplit("|", 1)[0] for line in handle]
    assert reactions == ["A>>C", "A>>D", "X>>Y"]
