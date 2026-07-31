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
)
from Experiment.Lewis.rule_replay.benchmark import (  # noqa: E402
    extract_rule,
    load_rows,
    make_reactor,
)


def test_record_2110_minimal_lwg_rule_matches_legacy_products() -> None:
    reaction = next(
        row["reaction"]
        for row in load_rows(POLAR_DATASET)
        if row["record_id"] == 2110
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
        generated[representation] = {
            canonical_unmapped_reaction(product)
            for product in reactor.smarts_list
        }
        assert reactor.mapping_count == 8
        assert len(generated[representation]) == 8

    assert generated["tuple"] == generated["typesGH"]
