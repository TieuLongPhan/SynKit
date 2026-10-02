import csv

from scripts.audit_synister_flower_overlap import audit


def _write(path, headers, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def test_overlap_audit_excludes_whole_source_sequences(tmp_path):
    synister = tmp_path / "synister.csv"
    elementary = tmp_path / "elementary.csv"
    composite = tmp_path / "composite.csv"
    _write(synister, ["reaction_id"], [{"reaction_id": "1:1"}, {"reaction_id": "2:2"}])
    _write(elementary, ["original_id"], [{"original_id": "1:3"}, {"original_id": "3:1"}])
    _write(composite, ["original_id"], [{"original_id": "2:p1"}, {"original_id": "4:p1"}])

    result = audit(synister, elementary, composite)

    elementary_result, composite_result = result["candidates"]
    assert elementary_result["direct_identifier_overlap"] == 0
    assert elementary_result["source_sequence_overlap"] == 1
    assert elementary_result["rows_remaining_after_source_sequence_exclusion"] == 1
    assert composite_result["source_sequence_overlap"] == 1
    assert composite_result["source_sequences_remaining_after_exclusion"] == 1
