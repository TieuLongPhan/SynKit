"""Check replay scoring and observed/withheld input boundaries independently."""

from types import SimpleNamespace

import pytest

from Experiment.Lewis.partial_replay import benchmark

from Experiment.Lewis.partial_replay.benchmark import (
    canonical_fragments, contains_fragments, initialize, prepare, process, summarize,
)

REACTION = "[CH3:1][Br:2].[OH-:3]>>[CH3:1][OH:3].[Br-:2]"


@pytest.fixture(scope="module")
def library(reactor_native_library):
    return str(reactor_native_library)


def test_endpoint_comparison_is_a_multiset_and_retains_charge_and_isotopes():
    assert canonical_fragments("[CH4:19].[OH2:3]") == canonical_fragments("O.C")
    assert canonical_fragments("C.C") != canonical_fragments("C")
    assert not contains_fragments(canonical_fragments("C"), canonical_fragments("C.C"))
    assert canonical_fragments("[OH-]") != canonical_fragments("[OH]")
    assert canonical_fragments("[13CH4]") != canonical_fragments("C")
    assert canonical_fragments("F[C@H](Cl)Br") == canonical_fragments("F[C@@H](Cl)Br")
    assert canonical_fragments("F[C@H](Cl)Br", stereo=True) != canonical_fragments("F[C@@H](Cl)Br", stereo=True)


@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_forward_and_backward_withheld_fragment_recovery_requires_completion(library, backend):
    initialize({"backend": backend, "library": library, "candidate_limit": 2048,
                "seconds": 10, "hydrogen_policy": "absolute"})
    row = process((0, {"r_id": "balanced-sn2", "aam": REACTION}))
    assert row["status"] == "attempted"
    for direction in ("fw", "bw"):
        for variant in ("full_control", "partial_all", "partial_largest"):
            result = row["runs"][variant][direction]
            assert result["status"] == "complete"
            assert result["exact"]
            assert ":" not in result["input"]  # reference maps never anchor bindings
        withheld = row["runs"]["partial_largest"][direction]
        assert withheld["withheld_components"] == 1
        assert withheld["exact_with_introduced_atoms"]
        assert not withheld["exact_with_ports"]


def test_missing_byproduct_is_not_silently_balanced_or_removed_from_denominator(library):
    initialize({"backend": "cpp", "library": library, "candidate_limit": 2048,
                "seconds": 10, "hydrogen_policy": "absolute"})
    row = process((0, {"r_id": "omitted-bromide", "aam": REACTION.split(".[Br-:2]")[0]}))
    assert row["status"] == "unsupported_atom_carrier"
    assert row["left_only_maps"] == [2]
    assert "runs" not in row
    report = summarize([row])
    assert report["rows"] == 1
    for metrics in report["metrics"].values():
        assert metrics["source_rows"] == 1
        assert metrics["unsupported_atom_carrier"] == 1
        assert metrics.get("exact", 0) == 0


def test_invalid_correspondence_is_a_preparation_error(library):
    initialize({"backend": "cpp", "library": library, "candidate_limit": 2048,
                "seconds": 10, "hydrogen_policy": "absolute"})
    row = process((0, {"aam": "[CH3:1][Br:1]>>[CH3:1][Br:2]"}))
    assert row["status"] == "preparation_error"


def test_absolute_and_relative_hydrogen_policies_are_distinct():
    absolute, *_ = prepare(REACTION, "absolute")
    relative, *_ = prepare(REACTION, "relative")
    assert absolute.rc.raw.nodes[3]["hcount"] == (1, 1)
    assert relative.rc.raw.nodes[3]["hcount"] == (0, 0)


@pytest.fixture
def without_interval_timer(monkeypatch):
    monkeypatch.setattr(benchmark, "signal", SimpleNamespace(signal=benchmark.signal.signal))


def test_replay_preparation_and_execution_without_posix_timers(without_interval_timer):
    initialize({"backend": "python", "library": None, "candidate_limit": 2048,
                "seconds": 10, "hydrogen_policy": "absolute"})
    balanced = process((0, {"aam": REACTION}))
    omitted = process((1, {"aam": REACTION.split(".[Br-:2]")[0]}))

    assert balanced["status"] == "attempted"
    assert all(run["status"] == "complete" and run["exact"]
               for runs in balanced["runs"].values() for run in runs.values())
    assert omitted["status"] == "unsupported_atom_carrier"
    assert omitted["left_only_maps"] == [2]


def test_replay_without_posix_timers_rejects_completed_work_over_budget(without_interval_timer, monkeypatch):
    readings = iter((10.0, 12.0))
    monkeypatch.setattr(benchmark, "perf_counter", lambda: next(readings))

    with pytest.raises(benchmark.ReplayTimeout, match="time budget"):
        with benchmark.deadline(1.0):
            pass
