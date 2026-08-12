"""SynRule uses the native LLG compositor without an optional MOD backend."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from synkit.Graph.Morphism import ELECTRON_LLG_SCHEMA
from synkit.Rule import SynRule
from synkit.Rule.Compose import (
    SynRuleCompositionError,
    SynRuleCompositionIssueCode,
    compose_synrules,
    search_synrule_compositions,
    synrule_to_span,
)


def _rules() -> tuple[SynRule, SynRule]:
    first = SynRule.from_smart(
        "[CH3:1][CH3:2]>>[CH2:1]=[CH2:2]",
        format="tuple",
        implicit_h=False,
    )
    second = SynRule.from_smart(
        "[CH2:1]=[CH2:2]>>[CH:1]#[CH:2]",
        format="tuple",
        implicit_h=False,
    )
    return first, second


def test_synrule_exposes_a_lossless_tuple_dpo_view() -> None:
    first, _ = _rules()

    direct = synrule_to_span(first)
    method = first.to_rule_span()

    assert direct.left.schema is ELECTRON_LLG_SCHEMA
    assert direct == method
    assert direct.interface.schema.node_state == ()


def test_synrule_composes_one_explicit_overlap_natively() -> None:
    first, second = _rules()

    direct = compose_synrules(first, second, {1: 1, 2: 2})
    method = first.compose(second, {1: 1, 2: 2})

    assert direct.certificate.replay().valid
    assert method.certificate.replay().valid
    assert direct.rule.left.is_isomorphic(method.rule.left)
    assert direct.rule.right.is_isomorphic(method.rule.right)


def test_synrule_candidate_api_retains_every_symmetric_overlap() -> None:
    first, second = _rules()

    direct = search_synrule_compositions(first, second)
    method = first.composition_candidates(second)

    assert (
        direct.raw_overlap_count,
        direct.accepted_count,
        direct.exact_class_count,
    ) == (
        7,
        7,
        3,
    )
    assert direct.match_matrix.counts == ((2, 1),)
    assert tuple(group.canonical_id for group in direct.classes) == tuple(
        group.canonical_id for group in method.classes
    )


def test_adapter_refuses_to_erase_stereo_semantics() -> None:
    first, _ = _rules()
    first.stereo_effects = {"atom:1": object()}

    with pytest.raises(SynRuleCompositionError) as error:
        first.to_rule_span()
    assert error.value.issues[0].code is SynRuleCompositionIssueCode.STEREO_UNSUPPORTED


def test_native_compose_submodule_has_no_mod_import() -> None:
    root = Path(__file__).resolve().parents[3]
    imported_roots = set()
    for source in (root / "synkit/Rule/Compose").glob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(
                    alias.name.partition(".")[0] for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.partition(".")[0])

    assert "mod" not in imported_roots
