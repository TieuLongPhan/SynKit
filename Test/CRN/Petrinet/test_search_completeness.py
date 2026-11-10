"""Search limits must not masquerade as complete structural enumeration."""

import warnings

import pytest

from synkit.CRN import SynCRN, native_backend
from synkit.CRN.backend import python_backend
from synkit.CRN.Petrinet.net import PetriNet
from synkit.CRN.Petrinet.persistence import siphon_persistence_details
from synkit.CRN.Petrinet.siphon_search import (
    SiphonSearchLimit,
    SiphonSearchWarning,
    minimal_siphons,
    minimal_traps,
)
from synkit.CRN.Petrinet.structure import find_siphons, find_traps


def _two_independent_pools():
    return PetriNet.from_syncrn(SynCRN.from_reaction_strings(["A>>B", "B>>A", "C>>D", "D>>C"]))


@pytest.mark.parametrize("search", [minimal_siphons, minimal_traps, find_siphons, find_traps])
def test_python_default_raises_when_search_is_incomplete(search):
    with python_backend(), pytest.raises(SiphonSearchLimit, match="incomplete"):
        search(_two_independent_pools(), max_nodes=1)


@pytest.mark.parametrize("search", [minimal_siphons, minimal_traps, find_siphons, find_traps])
def test_python_partial_results_require_explicit_opt_in_and_warn(search):
    with python_backend(), pytest.warns(SiphonSearchWarning, match="incomplete"):
        partial = search(_two_independent_pools(), max_nodes=1, strict_limit=False)
    assert isinstance(partial, list)


@pytest.mark.parametrize("search", [find_siphons, find_traps])
def test_native_default_raises_with_completeness_metadata(search, exact_library):
    with native_backend(exact_library), pytest.raises(SiphonSearchLimit) as caught:
        search(_two_independent_pools(), max_nodes=1)
    assert caught.value.result.complete is False
    assert caught.value.result.termination_reason != "complete"


@pytest.mark.parametrize("search", [find_siphons, find_traps])
def test_native_explicit_partial_mode_warns(search, exact_library):
    with native_backend(exact_library), pytest.warns(SiphonSearchWarning, match="incomplete"):
        partial = search(_two_independent_pools(), max_nodes=1, strict_limit=False)
    assert isinstance(partial, list)


@pytest.mark.parametrize("search", [find_siphons, find_traps])
def test_completed_search_has_both_pools_without_warning(search, exact_library):
    for backend in [python_backend(), native_backend(exact_library)]:
        with backend, warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            found = search(_two_independent_pools(), max_nodes=100)
        assert {frozenset(group) for group in found} == {frozenset({"A", "B"}), frozenset({"C", "D"})}
        assert not caught


def test_partial_enumeration_cannot_certify_persistence(exact_library):
    for backend in [python_backend(), native_backend(exact_library)]:
        with backend:
            result = siphon_persistence_details(_two_independent_pools(), max_nodes=1)
        assert result.persistence_ok is None
        assert result.complete is False
        assert result.termination_reason != "complete"
