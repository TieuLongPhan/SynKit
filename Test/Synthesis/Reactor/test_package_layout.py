"""Reactor package-boundary regressions."""

from importlib.util import find_spec

import pytest

from synkit.Synthesis import Reactor
from synkit.Synthesis.Reactor.output import deduplication, structural


@pytest.mark.parametrize(
    ("name", "module"),
    [
        ("SynReactor", "synkit.Synthesis.Reactor.core.engine"),
        ("Strategy", "synkit.Synthesis.Reactor.core.strategy"),
        ("BatchReactor", "synkit.Synthesis.Reactor.workflow.batch"),
        ("Benchmark", "synkit.Synthesis.Reactor.workflow.benchmark"),
        ("PostSyn", "synkit.Synthesis.Reactor.workflow.postprocess"),
        ("RuleFilter", "synkit.Synthesis.Reactor.workflow.rule_filter"),
        ("ImbaEngine", "synkit.Synthesis.Reactor.variants.imbalanced"),
        ("PartialEngine", "synkit.Synthesis.Reactor.variants.partial"),
    ],
)
def test_public_classes_have_one_canonical_implementation(
    name: str,
    module: str,
) -> None:
    assert getattr(Reactor, name).__module__ == module


@pytest.mark.parametrize(
    "module",
    [
        "syn_reactor",
        "graph_rewrite",
        "reactor_matching",
        "reactor_stereo",
        "deduplication",
        "structural_deduplication",
        "batch_reactor",
    ],
)
def test_flat_compatibility_modules_are_not_retained(module: str) -> None:
    assert find_spec(f"synkit.Synthesis.Reactor.{module}") is None


def test_structural_helpers_are_not_duplicated_in_general_deduplication() -> None:
    assert hasattr(structural, "_attach_exact_structural_signatures")
    assert not hasattr(deduplication, "_attach_exact_structural_signatures")
