"""Compare boundary composition with independent literal component traversal."""

from hashlib import sha256
from itertools import combinations, product
import json
from pathlib import Path

from Experiment.MTG.component_boundary import compose, summarize

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "Experiment/MTG/component_boundary_protocol.json"
COLOURS = (("O", 0, "atom"), ("O", 18, "atom"), ("P", 0, "atom"), ("X", 0, "enzyme_port"))


def subsets(items):
    items = tuple(items)
    for mask in range(1 << len(items)):
        yield tuple(v for i, v in enumerate(items) if mask & (1 << i))


def literal(labels, edges):
    adjacency = {v: set() for v in labels}
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    unseen = set(labels)
    total = marked = 0
    while unseen:
        todo = [unseen.pop()]
        group = set(todo)
        while todo:
            for v in adjacency[todo.pop()]:
                if v in unseen:
                    unseen.remove(v)
                    group.add(v)
                    todo.append(v)
        if sum(labels[v][0] == "P" for v in group) == 1 and not any(
            labels[v][0] == "C" or labels[v][2] == "enzyme_port" for v in group
        ):
            total += 1
            marked += any(labels[v][0] == "O" and labels[v][1] == 18 for v in group)
    return total, marked


def fragments(prefix):
    result = []
    for size in range(3):
        nodes = tuple((prefix, k) for k in range(size))
        for colours in product(COLOURS, repeat=size):
            labels = dict(zip(nodes, colours))
            for edges in subsets(combinations(nodes, 2)):
                for ports in subsets(nodes):
                    result.append((labels, edges, frozenset(ports), summarize(labels, edges, ports)))
    return result


def refused(operation):
    try:
        operation()
    except ValueError:
        return True
    raise AssertionError("Unsupported input was not refused.")


def run():
    protocol = json.loads(PROTOCOL.read_text())
    assert protocol["mathematical_audit_only"]
    left, right = fragments("left"), fragments("right")
    pairs = observations = 0
    for ll, le, lp, ls in left:
        for rl, re, rp, rs in right:
            labels = {**ll, **rl}
            for connectors in subsets(product(sorted(lp), sorted(rp))):
                expected = literal(labels, le + re + connectors)
                merged = compose((ls, rs), connectors)
                closed = compose((merged,), ports=())
                assert merged.inventory() == closed.inventory() == expected
                pairs += 1
                observations += 2
    print(json.dumps({"phase": "disjoint", "pairs": pairs, "observations": observations}), flush=True)
    shared = brackets = 0
    for colours in product(COLOURS, repeat=3):
        labels = dict(enumerate(colours))
        for edges in subsets(combinations(range(3), 2)):
            aedges = tuple(edge for edge in edges if set(edge) <= {0, 1})
            bedges = tuple(edge for edge in edges if set(edge) <= {1, 2})
            connectors = tuple(edge for edge in edges if edge not in aedges + bedges)
            a = summarize({v: labels[v] for v in (0, 1)}, aedges, (0, 1))
            b = summarize({v: labels[v] for v in (1, 2)}, bedges, (1, 2))
            result = compose((a, b), connectors)
            assert result == summarize(labels, edges, range(3))
            assert result.inventory() == literal(labels, edges)
            shared += 1
            pieces = [summarize({v: labels[v]}, (), (v,)) for v in range(3)]
            ab = compose(pieces[:2], [e for e in edges if set(e) <= {0, 1}])
            bc = compose(pieces[1:], [e for e in edges if set(e) <= {1, 2}])
            left_bracket = compose((ab, pieces[2]), [e for e in edges if 2 in e])
            right_bracket = compose((pieces[0], bc), [e for e in edges if 0 in e])
            direct = compose(pieces, edges)
            assert left_bracket == right_bracket == direct == summarize(labels, edges, range(3))
            brackets += 1
    p = summarize({"p": COLOURS[2]}, (), ("p",))
    o = summarize({"o": COLOURS[1]}, (), ("o",))
    labelled = compose((p, o), (("p", "o"),))
    assert p.inventory() == (1, 0) and labelled.inventory() == (1, 1)
    # Equal current counts can have different answers after the same port attachment.
    at_port = summarize({"a": COLOURS[0], "h": COLOURS[2]}, (("a", "h"),), ("a",))
    away = summarize({"a": COLOURS[0], "h": COLOURS[2]}, (), ("a",))
    poison = summarize({"x": COLOURS[3]}, (), ("x",))
    assert at_port.inventory() == away.inventory() == (1, 0)
    assert compose((at_port, poison), (("a", "x"),)).inventory() == (0, 0)
    assert compose((away, poison), (("a", "x"),)).inventory() == (1, 0)
    carbon = summarize({"c": ("C", 0, "atom")}, (), ("c",))
    assert compose((p, carbon), (("p", "c"),)).inventory() == (0, 0)
    many_labels = {"a": COLOURS[0], **{f"private{k}": COLOURS[2] for k in range(3)}}
    many_edges = tuple(("a", f"private{k}") for k in range(3))
    many = summarize(many_labels, many_edges, ("a",))
    assert many.blocks[0].private_phosphorus == 2 and many.inventory() == (0, 0)
    assert compose((many, o), (("a", "o"),)).inventory() == literal(
        {**many_labels, "o": COLOURS[1]}, many_edges + (("a", "o"),)
    )
    retained = summarize({"p": COLOURS[2], "x": COLOURS[3]}, (("p", "x"),), ("p", "x"))
    assert retained.inventory() == (0, 0)
    assert literal({"p": COLOURS[2], "x": COLOURS[3]}, ()) == (1, 0)
    assert compose((retained,)).inventory() != literal({"p": COLOURS[2], "x": COLOURS[3]}, ())
    # A connectivity summary preserves its declared count, not new-bond absence.
    direct = summarize({0: COLOURS[0], 1: COLOURS[0]}, ((0, 1),), (0, 1))
    path = summarize({0: COLOURS[0], 1: COLOURS[0], 2: COLOURS[0]}, ((0, 2), (1, 2)), (0, 1))
    assert (direct.labels, direct.blocks, direct.closed_counts) == (path.labels, path.blocks, path.closed_counts)
    hidden = summarize({"p": COLOURS[2]}, (), ())
    bad_label = summarize({"p": COLOURS[1]}, (), ("p",))
    refusals = {
        "hidden_overlap": refused(lambda: compose((hidden, p))),
        "shared_label_conflict": refused(lambda: compose((p, bad_label))),
        "unknown_port": refused(lambda: compose((p,), (("p", "unknown"),))),
        "self_connection": refused(lambda: compose((p,), (("p", "p"),))),
        "forgotten_port": refused(lambda: compose((hidden,), ports=("p",))),
    }
    report = {
        "schema": "synkit.mtg-component-boundary-audit/1",
        "status": "PASS",
        "summary": {
            "fragments_per_side": len(left),
            "disjoint_gluings": pairs,
            "disjoint_inventory_checks": observations,
            "shared_port_gluings": shared,
            "three_fragment_bracket_checks": brackets,
        },
        "refusals": refusals,
        "controls": {
            "count_only_summary_fails": True,
            "shared_phosphorus_counted_once": True,
            "connectivity_summary_is_not_bond_absence_admission": True,
            "interior_deletion_outside_contract": True,
        },
        "new_chemical_execution": False,
        "novelty_established": False,
        "sha256": {
            str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__).resolve(), ROOT / "Experiment/MTG/component_boundary.py", PROTOCOL)
        },
    }
    output = ROOT / "Experiment/MTG/evidence/component_boundary.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "PASS", **report["summary"], "report": str(output)}))


if __name__ == "__main__":
    run()
