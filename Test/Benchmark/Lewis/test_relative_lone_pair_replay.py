"""Relative lone-pair rule normalization and replay regression tests."""

from __future__ import annotations

from pathlib import Path
import sys

import networkx as nx

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Experiment.Lewis.common import (  # noqa: E402
    canonical_unmapped_reaction,
    canonical_unmapped_side,
)
from Experiment.Lewis.rule_replay.benchmark import (  # noqa: E402
    extract_rule,
    make_reactor,
)
from synkit.Graph.Matcher.subgraph_matcher import (  # noqa: E402
    electron_aware_node_match,
)
from synkit.Rule import SynRule  # noqa: E402
from synkit.Synthesis.Reactor.product_state import (  # noqa: E402
    _pair_electron_aware_node_attrs,
)

RECORD_886 = (
    "[CH3:1][n:2]1[n:3][n:4][n:5][c:6]1[S:7][CH2:8][CH2:9][CH2:10]"
    "[S:11](=[O:12])[c:13]1[cH:14][cH:15][cH:16][cH:17][n:18]1."
    "[O:19]=[CH:20][O:21][O:22][H:23]>>"
    "[CH3:1][n:2]1[n:3][n:4][n:5][c:6]1[S:7]("
    "[CH2:8][CH2:9][CH2:10][S:11](=[O:12])[c:13]1[cH:14][cH:15]"
    "[cH:16][cH:17][n:18]1)=[O:22].[O:19]=[CH:20][O:21][H:23]"
)


def _lone_pair_rule(pair: tuple[int, int]) -> SynRule:
    graph = nx.Graph()
    graph.add_node(
        1,
        element=("S", "S"),
        aromatic=(False, False),
        hcount=(0, 0),
        charge=(0, 0),
        radical=(0, 0),
        lone_pairs=pair,
        valence_electrons=(6, 6),
        neighbors=([], []),
        atom_map=(1, 1),
        present=(True, True),
    )
    return SynRule(
        graph,
        canon=False,
        implicit_h=False,
        format="tuple",
    )


def _apply_lone_pair_pair(
    host_lone_pairs: int,
    rule_pair: tuple[int, int],
) -> tuple[int, int]:
    host = {
        "element": "S",
        "aromatic": False,
        "hcount": 0,
        "charge": 0,
        "lone_pairs": host_lone_pairs,
        "typesGH": (
            ("S", False, 0, 0, []),
            ("S", False, 0, 0, []),
        ),
    }
    rule = {"lone_pairs": rule_pair}
    _pair_electron_aware_node_attrs(
        host,
        rule,
        relative_resources=frozenset({"lone_pairs"}),
    )
    return host["lone_pairs"]


def test_lone_pair_rule_is_normalized_to_consumption_and_supply() -> None:
    rule = _lone_pair_rule((2, 1))

    assert rule.rc.raw.nodes[1]["lone_pairs"] == (1, 0)
    assert rule.left.raw.nodes[1]["lone_pairs"] == 1
    assert rule.right.raw.nodes[1]["lone_pairs"] == 0
    assert rule.rc.raw.graph["relative_node_resources"] == ("lone_pairs",)


def test_relative_lone_pair_rewrite_applies_to_host_resource_level() -> None:
    pattern = {"element": "S", "lone_pairs": 1}
    assert not electron_aware_node_match(
        {"element": "S", "lone_pairs": 0},
        pattern,
        ("element", "lone_pairs"),
    )
    assert all(
        electron_aware_node_match(
            {"element": "S", "lone_pairs": value},
            pattern,
            ("element", "lone_pairs"),
        )
        for value in (1, 2, 3)
    )
    assert [_apply_lone_pair_pair(value, (1, 0)) for value in (1, 2, 3)] == [
        (1, 0),
        (2, 1),
        (3, 2),
    ]


def test_relative_lone_pair_rule_reverses_and_preserves_unchanged_state() -> None:
    reversed_rule = _lone_pair_rule((2, 1)).reversed()
    unchanged_rule = _lone_pair_rule((2, 2))

    assert reversed_rule.rc.raw.nodes[1]["lone_pairs"] == (0, 1)
    assert _apply_lone_pair_pair(0, (0, 1)) == (0, 1)
    assert unchanged_rule.rc.raw.nodes[1]["lone_pairs"] == (0, 0)
    assert _apply_lone_pair_pair(3, (0, 0)) == (3, 3)


def test_record_886_lwg_and_legacy_rules_recover_the_same_two_products() -> None:
    reactants, _ = RECORD_886.split(">>", 1)
    host = canonical_unmapped_side(reactants)
    expected = canonical_unmapped_reaction(RECORD_886)
    generated: dict[str, set[str]] = {}

    for representation in ("tuple", "typesGH"):
        rule = extract_rule(RECORD_886, representation)
        reactor = make_reactor(host, rule, representation, "forward", None)
        generated[representation] = {
            canonical_unmapped_reaction(reaction)
            for reaction in reactor.smarts_list
        }
        assert reactor.mapping_count == 2
        assert len(reactor.its_list) == 2
        assert expected in generated[representation]

    assert generated["tuple"] == generated["typesGH"]
