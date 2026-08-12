"""Safety contracts for the rebuilt paired RBL dataset."""

import hashlib
from pathlib import Path

from synkit.Chem.Reaction.standardize import Standardize
from synkit.IO import its_to_rsmi
from synkit.IO.data_io import load_from_pickle

from Experiment.RBL.benchmark import extract_normalized_rule
from Experiment.RBL.build_dataset import _file_manifest, build_records_with_report


def test_file_manifest_binds_path_size_and_content(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    payload = b"reaction\nC>>C\n"
    source.write_bytes(payload)

    manifest = _file_manifest(source)

    assert manifest["path"] == source.resolve().as_posix()
    assert manifest["bytes"] == len(payload)
    assert manifest["sha256"] == hashlib.sha256(payload).hexdigest()


def test_builder_stores_the_unambiguous_hcomplete_aam() -> None:
    aam = "[CH2:1]=[O:2].[H:3][H:4]>>[CH2:1]([H:3])[O:2][H:4]"
    raw_rows = [{"reactions": "C=O>>CO"}]
    aam_rows = [{"R-id": 0, "smart": aam}]

    records, report = build_records_with_report(raw_rows, aam_rows)

    assert report["reason_counts"] == {"accepted": 1}
    assert len(records) == 1
    assert "[H:3][H:4]" in records[0]["aam"]
    rule, audit = extract_normalized_rule(records[0])
    assert rule.rc.raw
    assert audit["hydrogen_completion"]["status"] == "unambiguous"
    assert audit["hydrogen_completion"]["exhaustive"] is True


def test_builder_rejects_charge_invalid_ground_truth() -> None:
    aam = (
        "[O:1]=[S:2](=[O:3])[c:4]1[cH:5][cH:6][c:7]([I:8])"
        "[cH:9][cH:10]1.[Na+:11]>>"
        "[O:1]=[S:2]([O-:3])[c:4]1[cH:5][cH:6][c:7]([I:8])"
        "[cH:9][cH:10]1.[Na+:11]"
    )
    raw = Standardize().fit(
        aam,
        remove_aam=True,
        ignore_stereo=True,
        remove_invalid=False,
    )

    records, report = build_records_with_report(
        [{"reactions": raw}],
        [{"R-id": 0, "smart": aam}],
    )

    assert records == []
    assert report["reason_counts"] == {"invalid_ground_truth_balance": 1}
    assert report["exclusions"][0]["element_balanced"] is True
    assert report["exclusions"][0]["charge_balanced"] is False


def test_builder_rejects_non_equivariant_hydrogen_completion() -> None:
    fixture = load_from_pickle("Data/Testcase/hydro/hydrogen_test.pkl.gz")[16]
    aam = its_to_rsmi(fixture["ITS"], format="typesGH")
    raw = Standardize().fit(
        aam,
        remove_aam=True,
        ignore_stereo=True,
        remove_invalid=False,
    )

    records, report = build_records_with_report(
        [{"reactions": raw}],
        [{"R-id": 0, "smart": aam}],
    )

    assert records == []
    assert report["reason_counts"] == {"hydrogen_completion_rejected": 1}
    assert report["exclusions"][0]["detail"] == "non_equivariant_rc"
    assert report["exclusions"][0]["candidates"] == 2
