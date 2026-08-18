"""Minimal reaction-center parity checks for LWG rule replay."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Experiment.Lewis.common import (  # noqa: E402
    POLAR_DATASET,
    canonical_unmapped_reaction,
    canonical_unmapped_side,
    unique_standardized_reactions,
)
from Experiment.Lewis.rule_replay.benchmark import (  # noqa: E402
    extract_rule,
    extract_rule_and_hosts,
    load_rows,
    make_reactor,
)


def test_record_2110_minimal_lwg_rule_matches_legacy_products() -> None:
    reaction = next(
        row["reaction"] for row in load_rows(POLAR_DATASET) if row["record_id"] == 2110
    )
    host = canonical_unmapped_side(reaction.split(">>", 1)[0])
    generated: dict[str, set[str]] = {}

    for representation in ("tuple", "typesGH"):
        rule = extract_rule(reaction, representation)
        reactor = make_reactor(
            host,
            rule,
            representation,
            "forward",
            None,
        )
        assert reactor.product_deduplication == "structural"
        generated[representation] = {
            canonical_unmapped_reaction(product) for product in reactor.smarts_list
        }
        assert reactor.mapping_count == 8
        assert len(generated[representation]) == 8

    assert generated["tuple"] == generated["typesGH"]


def test_rule_extraction_reuses_parsed_endpoint_graphs() -> None:
    reaction = next(
        row["reaction"] for row in load_rows(POLAR_DATASET) if row["record_id"] == 2110
    )
    expected = canonical_unmapped_reaction(reaction)
    rule, hosts = extract_rule_and_hosts(reaction, "tuple")

    for direction, host in hosts.items():
        reactor = make_reactor(host, rule, "tuple", direction, None)
        generated = {
            canonical_unmapped_reaction(product) for product in reactor.smarts_list
        }
        assert expected in generated


def test_reused_endpoint_graphs_preserve_boron_product_state() -> None:
    reaction = next(
        row["reaction"] for row in load_rows(POLAR_DATASET) if row["record_id"] == 125
    )
    reactants, products = reaction.split(">>", 1)
    rule, reused_hosts = extract_rule_and_hosts(reaction, "tuple")
    canonical_hosts = {
        "forward": canonical_unmapped_side(reactants),
        "backward": canonical_unmapped_side(products),
    }

    for direction in ("forward", "backward"):
        canonical = make_reactor(
            canonical_hosts[direction],
            rule,
            "tuple",
            direction,
            None,
        )
        reused = make_reactor(
            reused_hosts[direction],
            rule,
            "tuple",
            direction,
            None,
        )
        canonical_outputs = unique_standardized_reactions(canonical.smarts_list)
        reused_outputs = unique_standardized_reactions(reused.smarts_list)
        assert canonical_outputs == reused_outputs


def test_deferred_quotients_collapse_after_endpoint_standardization() -> None:
    reaction = next(
        row["reaction"] for row in load_rows(POLAR_DATASET) if row["record_id"] == 572
    )
    rule, hosts = extract_rule_and_hosts(reaction, "tuple")
    structural = make_reactor(
        hosts["forward"].copy(),
        rule,
        "tuple",
        "forward",
        None,
        "structural",
    )
    deferred = make_reactor(
        hosts["forward"].copy(),
        rule,
        "tuple",
        "forward",
        None,
        "deferred",
    )

    assert len(structural.smarts_list) == 2
    assert len(deferred.smarts_list) == 4
    structural_outputs = unique_standardized_reactions(structural.smarts_list)
    deferred_outputs = unique_standardized_reactions(deferred.smarts_list)
    assert structural_outputs == deferred_outputs
