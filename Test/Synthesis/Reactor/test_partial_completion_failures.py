"""Invalid extra subset completions must not discard observed forward replay."""

import pytest

from synkit.IO.chem_converter import rsmi_to_its
from synkit.Rule import SynRule
from synkit.Synthesis.Reactor import SynReactor, WildcardCompletionError, WildcardSearchLimitError
from synkit.Synthesis.Reactor.core import rewrite
from Test.Synthesis.Reactor.helpers import canonical_fragments

REACTION = "[CH3:1][I:2].[NH2:3][CH3:4]>>[CH3:1][NH:3][CH3:4]"


def reactor(host="CI.CN", **kwargs):
    rule = SynRule(rsmi_to_its(REACTION, core=True, format="tuple"), format="tuple", canon=False, implicit_h=False)
    return SynReactor(host, rule, explicit_h=False, partial=True, automorphism=False, stereo_mode="ignore", **kwargs)


@pytest.mark.parametrize("dedup", [False, True])
def test_invalid_missing_iodomethane_branch_retains_valid_full_replay(dedup):
    engine = reactor(dedup_its=dedup)
    outputs = {canonical_fragments(s.split(">>")[1]) for s in engine.smarts_list}
    assert canonical_fragments("CNC") in outputs
    assert engine.partial_completion_failures
    assert all("persist" in r["error"] for r in engine.partial_completion_failures)
    assert engine.smarts_list == engine.smarts_list


def test_all_invalid_completions_still_raise_without_cached_success():
    engine = reactor("CN")
    for _ in range(2):
        with pytest.raises(WildcardCompletionError, match="persist"):
            _ = engine.its_list
        assert engine._its is None
        assert engine.partial_completion_failures


def test_failure_of_a_full_input_binding_is_not_skipped(monkeypatch):
    def failure(*args, **kwargs):
        raise WildcardCompletionError("invalid full application")

    monkeypatch.setattr(rewrite, "_glue_graph", failure)
    with pytest.raises(WildcardCompletionError, match="invalid full application"):
        _ = reactor().its_list


def test_search_limit_is_not_treated_as_an_invalid_completion(monkeypatch):
    def failure(*args, **kwargs):
        raise WildcardSearchLimitError("incomplete search")

    monkeypatch.setattr(rewrite, "_glue_graph", failure)
    with pytest.raises(WildcardSearchLimitError, match="incomplete search"):
        _ = reactor().its_list
