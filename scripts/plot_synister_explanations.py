"""Build three explanatory figures from the verified unmapped FlowER example.

The partial-cost pruning experiment is a separate literal control, not a trace
or performance measurement of the production solver. Centre and context panels
are derived from all eight archived minimum maps and one full ITS respectively.
"""

from collections import Counter, defaultdict
from itertools import combinations, permutations, product
import hashlib
import json
from pathlib import Path

import networkx as nx
import numpy as np

from plot_synister_worked_example import read_record
from synister_diagrams import Canvas, graph, shift

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper/synister"


def derive(record):
    a, b = map(np.array, record["endpoint_adjacency"])
    r, p = record["endpoint_inventories"]
    buckets = defaultdict(list)
    for atom in r:
        buckets[atom["element"]].append(atom["index"])
    images = {e: [x["index"] for x in p if x["element"] == e] for e in buckets}
    prefixes = {}
    minimum, optimizers = None, []
    for choices in product(*(permutations(images[e]) for e in buckets)):
        m = [None] * len(r)
        for indices, values in zip(buckets.values(), choices):
            for i, j in zip(indices, values):
                m[i] = j
        cost = sum(
            abs(int(a[i, j]) - int(b[m[i], m[j]]))
            for i, j in combinations(range(13), 2)
        )
        fixed = sum(
            abs(int(a[i, j]) - int(b[m[i], m[j]])) for i, j in combinations(range(6), 2)
        )
        row = prefixes.setdefault(
            tuple(m[:6]), dict(fixed_cost=fixed, minimum_completion=cost, completions=0)
        )
        row["minimum_completion"] = min(row["minimum_completion"], cost)
        row["completions"] += 1
        if minimum is None or cost < minimum:
            minimum, optimizers = cost, []
        if cost == minimum:
            optimizers.append(m)
    assert minimum == record["minimum_cd"] == 6
    assert sorted(optimizers) == record["oracle"]["all_minimum_maps"]
    assert (
        len(prefixes) == 480
        and sum(v["completions"] for v in prefixes.values()) == 5760
    )
    assert all(
        v["completions"] == 12 and v["fixed_cost"] <= v["minimum_completion"]
        for v in prefixes.values()
    )
    rejected = [key for key, row in prefixes.items() if row["fixed_cost"] > minimum]
    assert len(rejected) == 137
    assert not any(tuple(m[:6]) in rejected for m in optimizers)
    selected = next(key for key in rejected if prefixes[key]["fixed_cost"] == 7)
    selected_states = [
        [i, j, int(a[i, j]), int(b[selected[i], selected[j]])]
        for i, j in combinations(range(6), 2)
        if a[i, j] or b[selected[i], selected[j]]
    ]

    bond_counts, atom_counts = Counter(), Counter()
    for m in optimizers:
        for i, j in combinations(range(13), 2):
            if a[i, j] != b[m[i], m[j]]:
                bond_counts[i, j] += 1
        for i in range(13):
            if any(r[i][key] != p[m[i]][key] for key in ("hydrogens", "charge")):
                atom_counts[i] += 1
    assert sorted(bond_counts.values()) == [4] * 6 + [8] * 2
    assert atom_counts == Counter({1: 8, 8: 8, 11: 8, 12: 8})
    matrix = []
    bond_order = [(10, 11), (10, 12), (0, 1), (0, 8), (1, 6), (6, 8), (1, 10), (8, 10)]
    for cls in record["classes"]:
        rows = []
        for m in cls["labeled_maps"]:
            rows.append([[int(a[i, j]), int(b[m[i], m[j]])] for i, j in bond_order])
        assert all(row == rows[0] for row in rows)
        matrix.append(rows[0])

    m = record["classes"][0]["representative"]
    full = [
        [i, j, int(a[i, j]), int(b[m[i], m[j]])]
        for i, j in combinations(range(13), 2)
        if a[i, j] or b[m[i], m[j]]
    ]
    centre = {v for i, j, x, y in full if x != y for v in (i, j)} | set(atom_counts)
    union = nx.Graph()
    union.add_nodes_from(range(13))
    union.add_edges_from((i, j) for i, j, _, _ in full)
    radius_one = centre | {j for i in centre for j in union.neighbors(i)}
    # Independent shortest-path check of the radius definition.
    assert radius_one == {
        i
        for i in range(13)
        if min(nx.shortest_path_length(union, i, j) for j in centre) <= 1
    }
    boundary = []
    for i, j, x, y in full:
        if (i in radius_one) != (j in radius_one):
            inside, outside = (i, j) if i in radius_one else (j, i)
            boundary.append(
                dict(
                    inside=inside,
                    outside=outside,
                    element=r[outside]["element"],
                    before=x,
                    after=y,
                )
            )
    assert sorted(centre) == [1, 6, 8, 10, 11, 12]
    assert sorted(radius_one) == [0, 1, 4, 6, 7, 8, 9, 10, 11, 12]
    assert sorted(
        (x["inside"], x["element"], x["before"], x["after"]) for x in boundary
    ) == [(4, "C", 1, 1), (4, "O", 1, 1)]
    return dict(
        pruning=dict(
            target=minimum,
            selected_prefix=list(selected),
            selected_states=selected_states,
            prefixes=[
                dict(mapping=list(key), **row) for key, row in sorted(prefixes.items())
            ],
            pruned_prefixes=len(rejected),
            retained_prefixes=len(prefixes) - len(rejected),
            pruned_completions=12 * len(rejected),
            retained_completions=12 * (len(prefixes) - len(rejected)),
        ),
        centre=dict(
            bond_order=bond_order,
            class_states=matrix,
            bond_counts=[[i, j, n] for (i, j), n in sorted(bond_counts.items())],
            atom_counts=[[i, n] for i, n in sorted(atom_counts.items())],
            invariant_tagged_coordinates=6,
            possible_tagged_coordinates=12,
        ),
        context=dict(
            class_name="A",
            full_states=full,
            centre_vertices=sorted(centre),
            radius_one_vertices=sorted(radius_one),
            boundary=boundary,
            omitted_vertices=sorted(set(range(13)) - radius_one),
        ),
    )


def pruning_figure(data, record):
    c = Canvas(5.0)
    d = data["pruning"]
    for x, letter, title in (
        (0.1, "A", "Fix six atom images"),
        (5.4, "B", "Count the fixed edits"),
        (10.9, "C", "Cut an entire subtree"),
    ):
        c.head(x, -0.08, letter, title)
    m = d["selected_prefix"]
    product_indices = sorted(m)
    top = [(0.6 + 0.73 * i, -1.45) for i in range(6)]
    bottom = [(0.6 + 0.73 * i, -3.10) for i in range(6)]
    for i, j in enumerate(m):
        c.line(
            top[i],
            bottom[product_indices.index(j)],
            "synflow,draw=npgBlue!45,line width=.5pt",
        )
    for points, indices, inventory in (
        (top, range(6), record["endpoint_inventories"][0]),
        (bottom, product_indices, record["endpoint_inventories"][1]),
    ):
        for (x, y), i in zip(points, indices):
            element = inventory[i]["element"]
            color = "figInk" if element == "C" else "chem" + element
            c.text(x, y, element, "synchem,text=" + color)
            c.text(x, y - 0.32, str(i), "synidx")
    c.text(2.45, -0.89, "Reactant indices", "synsmall")
    c.text(2.45, -3.95, "Product indices", "synsmall")
    c.text(2.45, -4.55, "7 atoms still free", "synsmall")
    # The induced paired graph on the six assigned reactant vertices.
    coords = [
        (5.9, -1.25),
        (7.6, -1.25),
        (9.5, -2.80),
        (7.6, -2.80),
        (6.75, -3.65),
        (5.9, -2.80),
    ]
    graph(
        c,
        coords,
        d["selected_states"],
        record["endpoint_inventories"][0][:6],
        paired=True,
    )
    for i in (0, 3, 4):
        c.text(*coords[i], "C", "synchem")
    for i, (x, y) in enumerate(coords):
        c.text(x + 0.22, y - 0.23, str(i), "synidx")
    c.text(7.8, -4.40, r"$\mathrm{LB}=7> C=6$", "synlabel,text=figBroken")
    # Counts concern this explicit six-atom partition, not a runtime trace.
    c.text(13.25, -1.10, "480 prefixes", "synlabel")
    for x, count, color, label, n in (
        (11.95, d["pruned_prefixes"], "figBroken", "prune", d["pruned_completions"]),
        (
            14.55,
            d["retained_prefixes"],
            "npgBlue",
            "continue",
            d["retained_completions"],
        ),
    ):
        c.line((13.25, -1.4), (x, -2.0), "synflow,draw=" + color)
        c.text(
            x, -2.45, str(count), f"synatom,minimum size=10mm,draw={color},text={color}"
        )
        c.text(x, -3.15, label, "synsmall,text=" + color)
        c.text(x, -3.85, r"$\times12$", "synsmall")
        c.text(x, -4.45, f"{n:,} maps", "synlabel,text=" + color)
    c.save("exact_pruning.tex")


def centre_figure(data, record):
    c = Canvas(7.6)
    d = data["centre"]
    c.head(0.1, -0.08, "A", "Which changes survive every mapping?")
    labels = [
        r"S--Cl$_1$",
        r"S--Cl$_2$",
        r"Me--O$_a$",
        r"Me--O$_b$",
        r"C--O$_a$",
        r"C--O$_b$",
        r"S--O$_a$",
        r"S--O$_b$",
    ]
    for col, label in enumerate(labels):
        x = 3.05 + 1.6 * col
        c.text(x, -0.88, label, "synsmall")
        for row in range(2):
            yy = -1.48 - 0.67 * row
            before, after = d["class_states"][row][col]
            color = (
                "figBroken"
                if after < before
                else "npgGreen" if after > before else "figHair"
            )
            c.add(
                rf"\fill[{color}!12,rounded corners=2pt] ({x-.70},{yy-.26}) rectangle ({x+.70},{yy+.26});"
            )
            text = rf"${before}\!\to\!{after}$" if before != after else r"$\cdot$"
            c.text(x, yy, text, "synlabel,text=" + color)
        count = next(
            n for i, j, n in d["bond_counts"] if (i, j) == tuple(d["bond_order"][col])
        )
        c.text(
            x,
            -2.82,
            f"{count}/8",
            "synlabel,text=" + ("npgBlue" if count == 8 else "npgOrange!90!black"),
        )
    c.text(1.10, -1.48, "A: 4 maps", "synsmall")
    c.text(1.10, -2.15, "B: 4 maps", "synsmall")
    c.text(1.10, -2.82, "Frequency", "synsmall")
    c.line((0.1, -3.25), (15.9, -3.25), "draw=figHair,line width=.5pt")
    c.head(0.1, -3.42, "B", "The possible center")
    c.head(8.1, -3.42, "C", "Invariant and variable coordinates")
    coords = {
        0: (3.15, -4.30),
        1: (1.4, -5.28),
        6: (3.15, -5.28),
        8: (4.9, -5.28),
        10: (3.15, -6.28),
        11: (2.15, -7.0),
        12: (4.15, -7.0),
    }
    # Frequency styles differ deliberately from paired bond-state styles above.
    for i, j, n in d["bond_counts"]:
        style = (
            "draw=npgBlue,line width=1.2pt"
            if n == 8
            else "draw=npgOrange,line width=1pt,dashed"
        )
        c.line(coords[i], coords[j], style)
    atom_names = {0: "Me", 1: "O", 6: "C", 8: "O", 10: "S", 11: "Cl", 12: "Cl"}
    for i, (x, y) in coords.items():
        if i in dict(d["atom_counts"]):
            c.add(rf"\draw[npgBlue,line width=.7pt] ({x},{y}) circle (.26);")
        el = record["endpoint_inventories"][0][i]["element"]
        c.text(
            x,
            y,
            atom_names[i],
            "synchem,text=" + ("figInk" if el == "C" else "chem" + el),
        )
    c.text(1.14, -5.03, "a", "synidx")
    c.text(5.17, -5.03, "b", "synidx,text=npgPurple")
    # Actual tagged coordinates: 4 unary atoms and 2 bonds in I; 6 further bonds in U.
    c.add(
        r"\draw[npgOrange!75,fill=npgOrange!3,rounded corners=10pt] (8.65,-4.18) rectangle (15.65,-6.85);"
    )
    c.add(
        r"\draw[npgBlue,fill=npgBlue!5,rounded corners=8pt] (10.10,-4.70) rectangle (14.20,-6.28);"
    )
    c.text(9.13, -4.49, "$U$", "synlabel,text=npgOrange!85!black")
    c.text(10.48, -5.0, "$I$", "synlabel,text=npgBlue")
    for j in range(4):
        c.dot(11.10 + 0.64 * j, -5.12, "npgBlue", 0.085)
    for x in (11.43, 12.93):
        c.line((x - 0.27, -5.80), (x + 0.27, -5.80), "draw=npgBlue,line width=1.4pt")
    for x, y in (
        (9.35, -5.25),
        (9.35, -6.05),
        (14.90, -5.25),
        (14.90, -6.05),
        (11.15, -6.60),
        (13.05, -6.60),
    ):
        c.line((x - 0.28, y), (x + 0.28, y), "draw=npgOrange,line width=1.2pt,dashed")
    c.text(12.16, -7.25, r"$|I|=6,\quad |U\setminus I|=6$", "synlabel")
    c.save("centre_certainty.tex")


def context_figure(data, record):
    c = Canvas(5.8)
    d = data["context"]
    coords = [
        (0, 1.75),
        (1.1, 2.3),
        (0, 0.6),
        (1.1, 0),
        (2.2, 0.6),
        (3.3, 0),
        (2.2, 1.75),
        (2.2, 2.7),
        (3.3, 2.3),
        (5.5, 2.3),
        (4.4, 1.75),
        (4.4, 0.6),
        (5.5, 1.15),
    ]
    for k, (letter, title, retained) in enumerate(
        (
            ("A", r"Full ITS $\Upsilon$", set(range(13))),
            ("B", r"Center $\Gamma$", set(d["centre_vertices"])),
            ("C", "Radius 1 + boundary", set(d["radius_one_vertices"])),
        )
    ):
        offset = 5.40 * k
        c.head(offset + 0.1, -0.08, letter, title)
        points = shift(coords, offset + 0.45, -3.50, 0.72)
        indices = sorted(retained)
        lookup = {v: i for i, v in enumerate(indices)}
        states = [
            (lookup[i], lookup[j], x, y)
            for i, j, x, y in d["full_states"]
            if i in retained and j in retained
        ]
        graph(
            c,
            [points[i] for i in indices],
            states,
            [record["endpoint_inventories"][0][i] for i in indices],
            paired=True,
        )
        for i in retained & {0, 3, 4, 6}:
            c.text(*points[i], "C", "synchem")
        if k == 2:
            for boundary in d["boundary"]:
                inside = points[boundary["inside"]]
                dest = points[boundary["outside"]]
                c.line(inside, dest, "draw=npgBlue,densely dotted,line width=.8pt")
                c.text(
                    dest[0],
                    dest[1] - 0.30,
                    rf"$({boundary['element']};{boundary['before']},{boundary['after']})$",
                    "synsmall,text=npgBlue,fill=white,inner sep=1pt",
                )
        c.text(offset + 2.45, -4.28, f"{len(retained)} atoms", "synsmall")
    c.line((4.75, -2.6), (5.32, -2.6), "synflow")
    c.line((10.12, -2.6), (10.70, -2.6), "synflow")
    c.line((1.30, -5.02), (1.95, -5.02), "synbroken")
    c.text(2.80, -5.02, "removed", "synsmall")
    c.line((4.10, -5.02), (4.75, -5.02), "synformed")
    c.text(5.55, -5.02, "formed", "synsmall")
    c.line((7.0, -5.02), (7.65, -5.02), "synbond,draw=figMuted!75")
    c.text(8.65, -5.02, "retained", "synsmall")
    c.line(
        (10.15, -5.02), (10.80, -5.02), "draw=npgBlue,densely dotted,line width=.8pt"
    )
    c.text(12.25, -5.02, "boundary record", "synsmall")
    c.text(13.25, -5.60, r"$(\mathrm{element};b_R,b_P)$", "synsmall,text=npgBlue")
    c.save("context_radius.tex")


def main():
    record = read_record()
    data = derive(record)
    pruning_figure(data, record)
    centre_figure(data, record)
    context_figure(data, record)
    payload = dict(
        schema_version=1,
        worked_record_sha256=record["record_sha256"],
        source_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                ROOT / "scripts/synister_diagrams.py",
                PAPER / "figures/figure_style.tex",
            )
        },
        **data,
    )
    (PAPER / "figures/source_data/explanation_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(
        "Explanatory controls: 5,760 maps; 137 pruned prefixes; 6 invariant / 12 possible tags; centre 6 / radius-1 10 atoms."
    )


if __name__ == "__main__":
    main()
