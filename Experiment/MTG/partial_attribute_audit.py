#!/usr/bin/env python
"""Exhaustively check a finite two-coordinate relabelling model.

The model has one persistent carrier, coordinate values 0 and 1, and explicit
undefinedness. It checks local label squares, not graph concurrency,
adhesivity, chemistry, or the soundness of the production DPO checker.
"""

from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path

LABELS = tuple(product((None, 0, 1), repeat=2))
TOTAL = tuple(product((0, 1), repeat=2))


def preserves(source, target):
    return all(a is None or a == b for a, b in zip(source, target))


def join(left, right):
    if any(a is not None and b is not None and a != b for a, b in zip(left, right)):
        return None
    return tuple(b if a is None else a for a, b in zip(left, right))


def meet(left, right):
    return tuple(a if a == b else None for a, b in zip(left, right))


def run():
    pushout_targets = pullback_targets = 0
    for left, right in product(LABELS, repeat=2):
        union = join(left, right)
        intersection = meet(left, right)
        for target in LABELS:
            if union is not None:
                assert (
                    preserves(left, target) and preserves(right, target)
                ) == preserves(union, target)
                pushout_targets += 1
            assert (preserves(target, left) and preserves(target, right)) == preserves(
                target, intersection
            )
            pullback_targets += 1

    cases = []
    for left, right, interface in product(TOTAL, TOTAL, LABELS):
        if not preserves(interface, left) or not preserves(interface, right):
            continue
        admitted_contexts = []
        for context in LABELS:
            left_square = (
                join(left, context) == left and meet(left, context) == interface
            )
            right_square = (
                join(context, right) == right and meet(context, right) == interface
            )
            if left_square and right_square:
                admitted_contexts.append(context)
        assert admitted_contexts == [interface]
        cases.append(
            {
                "left": left,
                "interface": interface,
                "right": right,
                "unique_natural_context": admitted_contexts[0],
                "retaining_left_context_is_natural": meet(left, left) == interface,
                "retaining_left_context_has_right_pushout": join(left, right)
                is not None,
            }
        )
    return {
        "schema": "synkit.mtg-partial-coordinate-research/1",
        "scope": "One persistent carrier with two partial coordinates over {0,1}.",
        "summary": {
            "compatible_rule_spans": len(cases),
            "unique_natural_contexts": len(cases),
            "pushout_universal_property_target_checks": pushout_targets,
            "pullback_universal_property_target_checks": pullback_targets,
            "retained_full_context_not_natural": sum(
                not c["retaining_left_context_is_natural"] for c in cases
            ),
            "retained_full_context_right_label_conflict": sum(
                not c["retaining_left_context_has_right_pushout"] for c in cases
            ),
        },
        "general_graph_theorem_verified": False,
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
