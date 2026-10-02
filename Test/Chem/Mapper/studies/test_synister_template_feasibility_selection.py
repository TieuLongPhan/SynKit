import csv

from scripts.select_synister_template_feasibility import select


def _write(path, headers, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def test_selection_excludes_source_groups_and_chooses_one_hash_ranked_row(tmp_path):
    synister = tmp_path / "synister.csv"
    elementary = tmp_path / "elementary.csv"
    _write(synister, ["reaction_id"], [{"reaction_id": "1:1"}])
    _write(
        elementary,
        ["r_id", "original_id", "ground_truth"],
        [
            {"r_id": "1", "original_id": "1:2", "ground_truth": "[C:1]>>[C:1]"},
            {"r_id": "2", "original_id": "2:1", "ground_truth": "[C:1]>>[C:1]"},
            {"r_id": "3", "original_id": "2:2", "ground_truth": "[C:1]>>[C:1]"},
            {"r_id": "4", "original_id": "3:1", "ground_truth": "[C:1][O:2]>>[C:1][O:2]"},
        ],
    )

    result = select(synister, elementary, maximum_atoms=1)

    assert result["selection"]["eligible_rows"] == 2
    assert result["selection"]["eligible_source_sequences"] == 1
    assert len(result["records"]) == 1
    assert result["records"][0]["source_sequence"] == "2"
