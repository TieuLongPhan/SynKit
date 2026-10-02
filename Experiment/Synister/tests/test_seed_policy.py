from Experiment.Synister.coverage_followup import seed_from_predictions
from Experiment.Synister.worker import perform
from Experiment.Synister.worked_oracle import REACTION, run


def test_seed_selection_does_not_need_reference_or_candidates():
    mapping, metadata = seed_from_predictions("CO>>CO", {
        "b": {"status": "valid", "prediction": {"mapping": [0, 1]}},
        "a": {"status": "invalid_prediction"},
    })
    assert mapping == [0, 1] and metadata["method"] == "b"
    assert seed_from_predictions("CO>>CO", {"a": {"status": "hard_timeout"}}) == (None, None)


def test_seed_changes_neither_worked_minimum_nor_labels():
    maps = run()["minimizing_maps"]
    unseeded = perform({"reaction": REACTION, "stage": "exact", "search_seconds": 5})
    for mapping in (maps[0], maps[-1]):
        seeded = perform({"reaction": REACTION, "stage": "exact", "search_seconds": 5,
                          "initial_mapping": mapping})
        assert seeded["status"] == "complete"
        assert seeded["minimum"] == unseeded["minimum"] == 6
        def labels(result):
            return {tuple(tuple(x) for x in item["label"]["typed_bond_edits"]) for item in result["labels"]}
        assert labels(seeded) == labels(unseeded)
