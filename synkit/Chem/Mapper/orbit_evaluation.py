"""Exact bond-label orbits via point-stabilizer transversals.

Only the action on vertices supporting the label is needed. For a base
b_1,...,b_k of those vertices let H_j fix its first j points. One witnessed
transversal T_j for H_{j-1}/H_j gives G = T_1 ... T_k H_k. Since H_k fixes the
label, its orbit is obtained by applying T_k,...,T_1 and deduplicating images.
Thus irrelevant spectator permutations need never be listed. This uses
standard group decomposition, not a new graph-isomorphism algorithm.
"""

import math
from time import monotonic

import networkx as nx

from .evaluation import ExactBondEvaluator, IncompleteSymmetry, ScoreWitness, _graph, bond_f1, transport_bonds


class SupportOrbitEvaluator(ExactBondEvaluator):
    """Exact quotient scorer without explicit enumeration of the entire group.

    Every transporter is a full attributed-graph automorphism. Failure to
    complete any orbit query raises; no partial group is reported as exact.
    Deadlines are cooperative around isomorphism calls and need an external
    process deadline for a hard wall-clock guarantee.
    """

    def __init__(self, reactant, *, max_label_images=100000, time_limit_seconds=30):
        if (not isinstance(max_label_images, int) or max_label_images < 1
                or not math.isfinite(time_limit_seconds) or time_limit_seconds <= 0):
            raise ValueError("Require positive finite time and label-image limits")
        self.n = len(reactant.atomic_numbers)
        self.graph = _graph(reactant)
        self.identity = tuple(range(self.n))
        self.deadline = monotonic() + time_limit_seconds
        self.max_label_images = max_label_images
        self._orbits = {}
        self._transversals = {}
        self.isomorphism_calls = 0
        self.colors = self._refine_colors()

    def _check(self):
        if monotonic() > self.deadline:
            raise IncompleteSymmetry("Label-orbit deadline exceeded")

    def _refine_colors(self):
        def number(signatures):
            numbers = {value: i for i, value in enumerate(sorted(set(signatures)))}
            return tuple(numbers[x] for x in signatures)
        colors = number([self.graph.nodes[i]["color"] for i in range(self.n)])
        for _ in range(self.n):
            self._check()
            refined = number([(colors[i], tuple(sorted(
                (self.graph.edges[i, j]["order"], colors[j]) for j in self.graph[i])))
                              for i in range(self.n)])
            if len(set(refined)) == len(set(colors)):
                return refined
            colors = refined
        return colors

    def _extension(self, fixed, source, target):
        self._check()
        if source == target:
            return self.identity
        left, right = self.graph.copy(), self.graph.copy()
        for i in range(self.n):
            left.nodes[i]["constraint"] = (self.colors[i], -1)
            right.nodes[i]["constraint"] = (self.colors[i], -1)
        for i in fixed:
            left.nodes[i]["constraint"] = right.nodes[i]["constraint"] = (self.colors[i], i)
        left.nodes[source]["constraint"] = (self.colors[source], source)
        right.nodes[target]["constraint"] = (self.colors[target], source)
        matcher = nx.algorithms.isomorphism.GraphMatcher(
            left, right,
            node_match=nx.algorithms.isomorphism.categorical_node_match("constraint", None),
            edge_match=nx.algorithms.isomorphism.categorical_edge_match("order", None),
        )
        self.isomorphism_calls += 1
        mapping = next(matcher.isomorphisms_iter(), None)
        self._check()
        if mapping is None:
            return None
        return tuple(mapping[i] for i in range(self.n))

    def _transversal(self, fixed, source):
        key = (fixed, source)
        if key not in self._transversals:
            values = []
            for target in range(self.n):
                if target in fixed or self.colors[target] != self.colors[source]:
                    continue
                witness = self._extension(fixed, source, target)
                if witness is not None:
                    values.append(witness)
            self._transversals[key] = tuple(values)
        return self._transversals[key]

    def orbit(self, label):
        """Return each distinct image with a verified full-graph transporter."""
        self._check()
        label = self._validate(label)
        if label not in self._orbits:
            base = self._support(label)
            levels = [self._transversal(base[:j], source) for j, source in enumerate(base)]
            images = {label: self.identity}
            for transversal in reversed(levels):
                next_images = {}
                for t in transversal:
                    for current, witness in images.items():
                        self._check()
                        image = self._transport(current, t)
                        if image not in next_images:
                            next_images[image] = tuple(t[witness[i]] for i in range(self.n))
                            if len(next_images) > self.max_label_images:
                                raise IncompleteSymmetry("Label-image cap exceeded")
                images = next_images
            self._orbits[label] = images
        return self._orbits[label]

    def _support(self, label):
        return tuple(sorted({i for pair in label for i in pair}))

    def _transport(self, label, permutation):
        return transport_bonds(label, permutation)

    def score(self, prediction, label):
        prediction = self._validate(prediction)
        best = None
        for image, witness in self.orbit(label).items():
            self._check()
            score = bond_f1(prediction, image)
            if best is None or score > best.score:
                best = ScoreWitness(score, witness)
        return best

    def orbit_key(self, label):
        return min(tuple(sorted(image)) for image in self.orbit(label))
