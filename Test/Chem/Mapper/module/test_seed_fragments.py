"""Common-fragment seeds preserve feasibility and optional-work limits."""

import numpy as np

from synkit.Chem.Mapper.exact.seed_fragments import improve_fragment_seed
from synkit.Chem.Mapper.exact.seed_relaxation import improve_relaxed_seed_mapping


def test_fragment_cover_recovers_a_permuted_graph():
    a = np.zeros((8, 8))
    for i, j in ((1, 0), (2, 1), (3, 0), (4, 3), (5, 4), (6, 0), (7, 1)):
        a[i, j] = a[j, i] = 1
    p = [4, 1, 7, 2, 5, 6, 0, 3]
    b = a[np.ix_(p, p)]
    original = [1, 0, 4, 3, 6, 5, 2, 7]
    candidate, stats = improve_fragment_seed(a, b, [6] * 8, [6] * 8, original)
    assert sorted(candidate) == list(range(8))
    assert np.array_equal(a, b[np.ix_(candidate, candidate)])
    assert stats["anchored_atoms"] == 8


def test_fragment_budget_and_unsupported_matrix_preserve_seed():
    a = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    seed = [2, 1, 0]
    candidate, stats = improve_fragment_seed(
        a, a, [6] * 3, [6] * 3, seed, budget_seconds=0
    )
    assert candidate == seed
    assert stats["anchored_atoms"] == 0
    for matrix in (a * 0.1, np.triu(a)):
        candidate, stats = improve_fragment_seed(matrix, matrix, [6] * 3, [6] * 3, seed)
        assert candidate == seed
        assert "skipped" in stats


def test_relaxed_seed_respects_heuristic_anchors_without_restricting_output_type():
    rng = np.random.default_rng(54)
    a, b = (rng.choice([0, 0.5, 1.5], size=(6, 6)) for _ in range(2))
    labels = [6, 6, 6, 8, 8, 8]
    seed = [1, 2, 0, 5, 3, 4]
    candidate, _ = improve_relaxed_seed_mapping(
        a, b, labels, labels, seed, fixed_mapping={0: 1, 3: 5}
    )
    assert candidate[0] == 1 and candidate[3] == 5
    assert sorted(candidate) == list(range(6))
    assert all(labels[i] == labels[p] for i, p in enumerate(candidate))


def test_fragment_progress_budget_uses_cpu_not_scheduler_delay(monkeypatch):
    from types import SimpleNamespace
    import synkit.Chem.Mapper.exact.seed_fragments as fragments

    clock = SimpleNamespace(process_time=lambda: 0.02)
    monkeypatch.setattr(fragments, "time", clock)
    progress = fragments._FragmentProgress(0.025)
    assert progress(None, None)
    clock.process_time = lambda: 0.03
    assert not progress(None, None)


def test_fragment_slice_starts_after_setup_within_cover_budget(monkeypatch):
    from types import SimpleNamespace
    import synkit.Chem.Mapper.exact.seed_fragments as fragments

    clock = SimpleNamespace(process_time=lambda: 0.2)
    monkeypatch.setattr(fragments, "time", clock)
    progress = fragments._FragmentProgress(0.5)
    # Setup has already consumed much more than the 25 ms search allowance.
    assert progress(None, None)
    clock.process_time = lambda: 0.224
    assert progress(None, None)
    clock.process_time = lambda: 0.226
    assert not progress(None, None)


def test_fragment_setup_does_not_extend_overall_cover_budget(monkeypatch):
    from types import SimpleNamespace
    import synkit.Chem.Mapper.exact.seed_fragments as fragments

    monkeypatch.setattr(fragments, "time", SimpleNamespace(process_time=lambda: 0.6))
    assert not fragments._FragmentProgress(0.5)(None, None)
