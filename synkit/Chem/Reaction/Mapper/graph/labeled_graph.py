from __future__ import annotations

from collections import defaultdict

from .synkit_adapter import stable_int_token, synkit_wl_node_labels


class LabeledGraph:
    """Indexed graph with labels and metadata."""

    SYNKIT_WL_NODE_LIMIT = 256

    def __init__(self, graph, labels):
        self.graph = self._normalise_graph(graph, len(labels))
        self.labels = list(labels)
        self.label2idxs = defaultdict(list)
        self.props = {}
        self.reindex_labels()

        self._ini_labels = list(self.labels)
        self._ini_wl_labels = None
        self._irred_labels = {}

    @staticmethod
    def _normalise_graph(graph, n_nodes):
        normalised = {idx: dict(graph.get(idx, {})) for idx in range(n_nodes)}
        for src, nbrs in graph.items():
            normalised.setdefault(src, {}).update(nbrs)
        return normalised

    def reindex_labels(self):
        """Rebuild label -> indices."""
        label_index = defaultdict(list)
        for idx, label in enumerate(self.labels):
            label_index[label].append(idx)
        self.label2idxs = label_index

    def build_label2idxs(self):
        """Alias for reindex_labels."""
        self.reindex_labels()

    def copy(self):
        """Copy graph state."""
        clone = type(self).__new__(type(self))
        clone.graph = {idx: dict(nbrs) for idx, nbrs in self.graph.items()}
        clone.labels = list(self.labels)
        clone.label2idxs = defaultdict(
            list,
            {label: list(idxs) for label, idxs in self.label2idxs.items()},
        )
        clone.props = self.props
        clone._ini_labels = self._ini_labels
        clone._ini_wl_labels = self._ini_wl_labels
        clone._irred_labels = dict(self._irred_labels)
        return clone

    def set_prop(self, name, prop):
        """Set metadata."""
        self.props[name] = prop

    def binarize_graph(self):
        """Set all edge weights to 1."""
        for nbrs in self.graph.values():
            for nbr in list(nbrs):
                nbrs[nbr] = 1

    def neighborhood_signature(self, idx, labels=None, binary=False):
        """WL neighborhood signature."""
        labels = self.labels if labels is None else labels
        grouped = defaultdict(list)
        for nbr, weight in self.graph.get(idx, {}).items():
            grouped[labels[nbr]].append(1 if binary else weight)
        return (
            labels[idx],
            tuple(
                sorted(
                    (label, tuple(sorted(weights, reverse=True)))
                    for label, weights in grouped.items()
                )
            ),
        )

    def one_wl_round(self, labels=None, binary=False):
        """One deterministic WL round."""
        labels = self.labels if labels is None else labels
        return [
            stable_int_token(self.neighborhood_signature(idx, labels, binary))
            for idx in range(len(labels))
        ]

    def get_WL_labels(self):
        """Stabilized WL colors."""
        if len(self.labels) <= self.SYNKIT_WL_NODE_LIMIT:
            synkit_labels = synkit_wl_node_labels(self)
            if synkit_labels is not None:
                return synkit_labels

        colors = list(self.labels)
        for _ in range(max(1, 2 * len(colors))):
            refined = self.one_wl_round(colors)
            if len(set(refined)) == len(set(colors)):
                return colors
            colors = refined
        return colors
