"""FLOWER batching and row-reader regression tests."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Experiment.Lewis.FLOWER.prepare_batches import prepare_batches  # noqa: E402
from Experiment.Lewis.FLOWER.prepare_resume import _durable_prefix  # noqa: E402
from Experiment.Lewis.FLOWER.replay import (  # noqa: E402
    case_identity,
    iter_rows,
    read_row_numbers,
)


def _write_rows(path: Path, labels: list[str]) -> list[str]:
    rows = [f"[C:{index}]>>[C:{index}]|{label}\n" for index, label in enumerate(labels)]
    path.write_text("".join(rows), encoding="utf-8")
    return rows


def test_prepare_batches_balances_concatenated_stream(tmp_path: Path) -> None:
    first = tmp_path / "train.txt"
    second = tmp_path / "test.txt"
    expected = _write_rows(first, ["a", "b", "c", "d"])
    expected += _write_rows(second, ["e", "f", "g"])
    output = tmp_path / "batches"

    manifest = prepare_batches([first, second], output, 3)

    assert manifest["total_rows"] == 7
    assert manifest["total_directional_replays"] == 14
    assert manifest["batch_rows"] == [3, 2, 2]
    observed: list[str] = []
    for report in manifest["batches"]:
        with gzip.open(report["path"], "rt", encoding="utf-8") as handle:
            observed.extend(handle.readlines())
    assert observed == expected
    expected_digest = hashlib.sha256("".join(expected).encode()).hexdigest()
    assert manifest["combined_content_sha256"] == expected_digest
    assert manifest["batches"][1]["source_rows"] == {
        "test.txt": 1,
        "train.txt": 1,
    }


def test_prepare_batches_is_byte_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "train.txt"
    _write_rows(source, ["1", "2", "3"])
    first = prepare_batches([source], tmp_path / "first", 2)
    second = prepare_batches([source], tmp_path / "second", 2)

    assert [item["sha256"] for item in first["batches"]] == [
        item["sha256"] for item in second["batches"]
    ]


def test_iter_rows_preserves_label_and_supports_window(tmp_path: Path) -> None:
    source = tmp_path / "batch.txt"
    _write_rows(source, ["one", "two", "three", "four"])

    assert list(iter_rows(source, offset=1, limit=2)) == [
        (2, "two", "[C:1]>>[C:1]"),
        (3, "three", "[C:2]>>[C:2]"),
    ]


def test_case_identity_is_portable_across_aggregate_bug_logs() -> None:
    identity = case_identity(
        Path("/server/data/batch-07-of-10.txt.gz"),
        321,
    )

    assert identity == {
        "case_id": "batch-07-of-10.txt.gz:321",
        "batch_file": "batch-07-of-10.txt.gz",
        "batch_row": 321,
    }


def test_iter_rows_supports_sparse_provenance_selection(
    tmp_path: Path,
) -> None:
    source = tmp_path / "batch.txt"
    _write_rows(source, ["one", "two", "three", "four", "five"])
    rows_file = tmp_path / "selected.rows"
    rows_file.write_text("5\n2\n2\n", encoding="utf-8")
    selected = read_row_numbers(rows_file)

    assert selected == {2, 5}
    assert list(iter_rows(source, row_numbers=selected)) == [
        (2, "two", "[C:1]>>[C:1]"),
        (5, "five", "[C:4]>>[C:4]"),
    ]


def test_durable_prefix_accepts_truncated_gzip_footer(
    tmp_path: Path,
) -> None:
    cases = tmp_path / "cases.jsonl.gz"
    with gzip.open(cases, "wt", encoding="utf-8") as handle:
        for row in range(1, 4):
            handle.write(json.dumps({"batch_row": row}) + "\n")
    cases.write_bytes(cases.read_bytes()[:-8])

    assert _durable_prefix(cases) == (3, False)


def test_prepare_batches_accepts_gzip_input(tmp_path: Path) -> None:
    source = tmp_path / "train.txt.gz"
    rows = "[C:1]>>[C:1]|one\n[C:2]>>[C:2]|two\n"
    with gzip.open(source, "wt", encoding="utf-8") as handle:
        handle.write(rows)

    manifest = prepare_batches([source], tmp_path / "batches", 1)

    report = manifest["inputs"][0]
    assert report["rows"] == 2
    expected_digest = hashlib.sha256(rows.encode()).hexdigest()
    assert report["content_sha256"] == expected_digest
