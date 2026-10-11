"""Component-count observers under immutable-fragment gluing, not rule admission."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Block:
    ports: frozenset
    private_phosphorus: int
    forbidden: bool
    labelled: bool


@dataclass(frozen=True)
class BoundarySummary:
    nodes: frozenset
    labels: tuple
    blocks: tuple
    closed_counts: tuple

    @property
    def ports(self):
        return frozenset(dict(self.labels))

    def inventory(self):
        labels = dict(self.labels)
        total, marked = self.closed_counts
        for block in self.blocks:
            p = block.private_phosphorus + sum(labels[v][0] == "P" for v in block.ports)
            if p == 1 and not block.forbidden:
                total += 1
                marked += int(block.labelled)
        return total, marked


class Union:
    def __init__(self, nodes):
        self.parent = {n: n for n in nodes}

    def find(self, node):
        parent = self.parent[node]
        if parent != node:
            self.parent[node] = self.find(parent)
        return self.parent[node]

    def join(self, left, right):
        self.parent[self.find(left)] = self.find(right)

    def groups(self):
        groups = {}
        for n in self.parent:
            groups.setdefault(self.find(n), set()).add(n)
        return list(groups.values())


def _canonical(blocks):
    return tuple(sorted(blocks, key=lambda b: repr(sorted(b.ports, key=repr))))


def _closed(p, forbidden, labelled):
    return (1, int(labelled)) if p == 1 and not forbidden else (0, 0)


def summarize(labels, edges, ports):
    """Labels are invariant (element, isotope, carrier_kind) triples."""
    labels = dict(labels)
    if any(len(v) != 3 for v in labels.values()):
        raise ValueError("Known invariant label triples are required.")
    ports = frozenset(ports)
    if not ports <= labels.keys():
        raise ValueError("Boundary atoms must be known.")
    union = Union(labels)
    for a, b in edges:
        if a == b or a not in labels or b not in labels:
            raise ValueError("Edges need two distinct known endpoints.")
        union.join(a, b)
    blocks = []
    closed = [0, 0]
    for group in union.groups():
        border = frozenset(group & ports)
        forbidden = any(labels[v][0] == "C" or labels[v][2] == "enzyme_port" for v in group)
        marked = any(labels[v][0] == "O" and labels[v][1] == 18 for v in group)
        if border:
            p = min(2, sum(labels[v][0] == "P" for v in group - ports))
            blocks.append(Block(border, p, forbidden, marked))
        else:
            count = _closed(sum(labels[v][0] == "P" for v in group), forbidden, marked)
            closed = [a + b for a, b in zip(closed, count)]
    return BoundarySummary(
        frozenset(labels), tuple(sorted(((v, labels[v]) for v in ports), key=repr)), _canonical(blocks), tuple(closed)
    )


def _collect(summaries):
    labels = {}
    all_nodes = set()
    blocks = {}
    owners = {}
    closed = [0, 0]
    for k, summary in enumerate(summaries):
        for old in summaries[:k]:
            if not (old.nodes & summary.nodes) <= (old.ports & summary.ports):
                raise ValueError("Every shared atom must be exposed on both fragments.")
        all_nodes.update(summary.nodes)
        for v, label in summary.labels:
            if v in labels and labels[v] != label:
                raise ValueError("Shared boundary atom labels disagree.")
            labels[v] = label
        closed = [a + b for a, b in zip(closed, summary.closed_counts)]
        for j, block in enumerate(summary.blocks):
            key = (k, j)
            blocks[key] = block
            for v in block.ports:
                owners.setdefault(v, []).append(key)
    return labels, all_nodes, blocks, owners, closed


def compose(summaries, connections=(), *, ports=None):
    """Join immutable fragments on named common ports and supplied port edges.

    Original source graphs, conditions and certificates remain authoritative.
    This operation cannot establish DPO admission or delete a fragment edge.
    """
    labels, all_nodes, blocks, owners, closed = _collect(tuple(summaries))
    union = Union(blocks)
    for refs in owners.values():
        for other in refs[1:]:
            union.join(refs[0], other)
    for a, b in connections:
        if a == b or a not in owners or b not in owners:
            raise ValueError("Connections need two distinct currently exposed ports.")
        union.join(owners[a][0], owners[b][0])
    ports = frozenset(labels) if ports is None else frozenset(ports)
    if not ports <= labels.keys():
        raise ValueError("A forgotten or unknown atom cannot be re-exposed.")
    result = []
    for group in union.groups():
        border = frozenset(v for key in group for v in blocks[key].ports)
        p = min(
            2, sum(blocks[key].private_phosphorus for key in group) + sum(labels[v][0] == "P" for v in border - ports)
        )
        forbidden = any(blocks[key].forbidden for key in group)
        marked = any(blocks[key].labelled for key in group)
        remaining = border & ports
        if remaining:
            result.append(Block(remaining, p, forbidden, marked))
        else:
            count = _closed(p, forbidden, marked)
            closed = [a + b for a, b in zip(closed, count)]
    return BoundarySummary(
        frozenset(all_nodes),
        tuple(sorted(((v, labels[v]) for v in ports), key=repr)),
        _canonical(result),
        tuple(closed),
    )
