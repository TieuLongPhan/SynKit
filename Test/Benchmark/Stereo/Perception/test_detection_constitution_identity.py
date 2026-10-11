"""Report overlap must erase configuration while preserving isotope identity."""

from pathlib import Path

import pytest

from Experiment.Stereo.Perception import full_detection


@pytest.mark.parametrize("dataset", ["acs", "rota", "cip"])
def test_detection_overlap_retains_isotopes_but_erases_stereo_and_atom_maps(monkeypatch, dataset):
    smiles = ["CC", "[13CH3]C", "[CH3:91][CH3:17]", "F[C@H](Cl)Br", "F[C@@H](Cl)Br"]
    if dataset == "acs":
        rows = [{"ID": str(i), "Input SMILES": value, "manual": "achiral"}
                for i, value in enumerate(smiles)]
        monkeypatch.setattr(full_detection, "load_dataset", lambda: rows)
        report, keys = full_detection._benchmark_acs(check_renumbering=False, timeout_seconds=5)
    elif dataset == "rota":
        rows = [{"SMILES": value, "label": "[]", "label_expanded": "[]",
                 "chiral_type": "Biaryl structure"} for value in smiles]
        monkeypatch.setattr(full_detection, "load_rota", lambda _path: rows)
        report, keys = full_detection._benchmark_rota(check_renumbering=False, timeout_seconds=5)
    else:
        rows = [{"ID": str(i), "SMILES": value, "recommended_labels": (),
                 "stereo_units": ()} for i, value in enumerate(smiles)]
        monkeypatch.setattr(full_detection, "load_cip", lambda _path: rows)
        report, keys, _ = full_detection._benchmark_cip(
            Path("unused"), check_renumbering=False, timeout_seconds=5,
        )
    assert report["parsed_records"] == len(smiles)
    assert keys[0] != keys[1]
    assert keys[0] == keys[2]
    assert keys[3] == keys[4]
    assert len(set(keys)) == 3
