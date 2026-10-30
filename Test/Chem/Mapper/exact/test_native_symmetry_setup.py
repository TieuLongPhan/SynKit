"""Representative search must not spend its deadline on unused group orders."""

import pytest

from Test.Chem.Mapper._helpers import graph
from synkit.Chem.Mapper.exact.native import candidates


def test_representative_preparation_uses_generators_without_counting_orders(
    native_library, monkeypatch
):
    pair = graph([[0, 1], [1, 0]]), graph([[0, 1], [1, 0]])
    monkeypatch.setattr(
        candidates, "bounded_automorphism_permutations",
        lambda *args, **kwargs: (((0, 1), (1, 0)), True),
    )

    def forbidden_order(*args, **kwargs):
        raise AssertionError("Representative preparation must not count group orders")

    monkeypatch.setattr(candidates, "permutation_group_order", forbidden_order)
    prepared = candidates.prepare_native_candidates(
        pair, 0, library_path=native_library, node_properties=(),
        allow_partial_symmetry=True,
    )
    assert prepared[6:8] == (None, None)
    assert prepared.symmetry_search_complete
    # Reporting paths that need exact multiplicities still request counting.
    with pytest.raises(AssertionError, match="must not count"):
        candidates.prepare_native_candidates(
            pair, 0, library_path=native_library, node_properties=()
        )
