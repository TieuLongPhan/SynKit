import networkx as nx
import hashlib
from synkit.IO import setup_logging

logger = setup_logging()


class NautyCanonicalizer:
    __slots__ = ("node_attrs", "edge_attrs")

    def __init__(self, node_attrs=None, edge_attrs=None):
        self.node_attrs = list(node_attrs) if node_attrs else []
        self.edge_attrs = list(edge_attrs) if edge_attrs else []

    @staticmethod
    def _freeze(x):
        if isinstance(x, list):
            return tuple(NautyCanonicalizer._freeze(v) for v in x)
        if isinstance(x, dict):
            return frozenset(
                (k, NautyCanonicalizer._freeze(v)) for k, v in sorted(x.items())
            )
        return x

    def canonical_form(
        self,
        G,
        return_aut=False,
        remap_aut=False,
        return_orbits=False,
        return_perm=False,
        max_depth=None,
    ):
        """
        Compute canonical form of graph G with optional automorphisms, orbits, and early stopping.

        :param G: NetworkX graph to canonicalize.
        :param return_aut: bool, whether to return list of automorphism permutations.
        Default: False.
        :param remap_aut: bool, whether to remap automorphisms to canonical labels
        (only valid if return_aut=True). Default: False.
        :param return_orbits: bool, whether to return node orbits (symmetry groups). Default: False.
        :param return_perm: bool, whether to return canonical permutation (ordering of nodes). Default: False.
        :param max_depth: int or None, max recursion depth for backtracking search (early stopping).
        Default: None (unlimited).
        :return: tuple containing requested results and a boolean early_stop flag indicating if search terminated early.
                 The order of outputs is (G_canon, perm?, automorphisms?, orbits?, early_stop).
        """
        logger.debug(
            f"Starting canonical_form: max_depth={max_depth},"
            + f" return_aut={return_aut}, remap_aut={remap_aut},"
            + f" return_orbits={return_orbits}, return_perm={return_perm}"
        )

        best = {"label": None, "perm": None}
        aut_perms = []

        initial_partition = self._initial_partition(G)
        logger.debug(f"Initial partition: {initial_partition}")

        early_stop_occurred = self._search(
            G, initial_partition, [], best, aut_perms, depth=0, max_depth=max_depth
        )

        perm = best["perm"]
        if perm is None:
            logger.error(
                f"Canonical form not found: search stopped early (max_depth={max_depth} too small)."
            )
            raise RuntimeError(
                f"Canonical form not found: search stopped early (max_depth={max_depth} too small)."
            )

        mapping = {v: i + 1 for i, v in enumerate(perm)}
        G_can = nx.relabel_nodes(G, mapping, copy=True)
        self._update_atom_map(G_can)

        results = [G_can]
        if return_perm:
            results.append(perm)

        if return_aut:
            if remap_aut:
                remapped = [[mapping[v] for v in p] for p in aut_perms]
                results.append(remapped)
            else:
                results.append(aut_perms)

        if return_orbits:
            orbits = self.compute_orbits(aut_perms)
            if remap_aut and return_aut:
                orbits = [set(mapping[v] for v in orbit) for orbit in orbits]
            results.append(orbits)

        results.append(early_stop_occurred)

        logger.debug(
            f"canonical_form completed, early_stop_occurred={early_stop_occurred}"
        )
        return tuple(results) if len(results) > 2 else results[0]

    def _update_atom_map(self, G):
        for n in G.nodes():
            G.nodes[n]["atom_map"] = n

    def _initial_partition(self, G):
        if not self.node_attrs:
            return [sorted(G.nodes())]
        buckets = {}
        for v in G.nodes():
            key = tuple(
                self._freeze(G.nodes[v].get(attr, None)) for attr in self.node_attrs
            )
            buckets.setdefault(key, []).append(v)
        return [sorted(nodes) for _, nodes in sorted(buckets.items())]

    def _node_signature(self, G, v, partition):
        node_attrs = tuple(
            self._freeze(G.nodes[v].get(a, None)) for a in self.node_attrs
        )
        degree = G.degree[v]

        nbr_part_counts = []
        for cell in partition:
            count = sum(1 for nbr in G.neighbors(v) if nbr in cell)
            nbr_part_counts.append(count)
        nbr_part_counts = tuple(nbr_part_counts)

        edge_attr_multiset = []
        for nbr in G.neighbors(v):
            attrs = G[v][nbr]
            edge_attrs = []
            for a in self.edge_attrs:
                val = attrs.get(a, None)
                if a == "order" and isinstance(val, tuple):
                    val = tuple(sorted(round(float(x), 3) for x in val))
                edge_attrs.append(self._freeze(val))
            edge_attr_multiset.append(tuple(edge_attrs))
        edge_attr_multiset = tuple(sorted(edge_attr_multiset))

        return (node_attrs, degree, nbr_part_counts, edge_attr_multiset)

    def _refine(self, G, partition):
        changed = True
        while changed:
            changed = False
            new_partition = []
            sig_cache = {}
            for cell in partition:
                if len(cell) <= 1:
                    new_partition.append(cell)
                    continue
                sigs = {}
                for v in cell:
                    if v not in sig_cache:
                        sig_cache[v] = self._node_signature(G, v, partition)
                    sig = sig_cache[v]
                    sigs.setdefault(sig, []).append(v)
                if len(sigs) > 1:
                    changed = True
                    for sig in sorted(sigs):
                        new_partition.append(sorted(sigs[sig]))
                else:
                    new_partition.append(cell)
            partition = new_partition
        return partition

    def _search(self, G, partition, prefix, best, aut_perms, depth=0, max_depth=None):
        if max_depth is not None and depth > max_depth:
            logger.debug(
                f"Early stopping at depth {depth} due to max_depth={max_depth}"
            )
            return True  # early stop triggered

        partition = self._refine(G, partition)
        if all(len(c) == 1 for c in partition):
            perm = prefix + [v for c in partition for v in c]
            label = self._build_label(G, perm)
            if best["label"] is None or label < best["label"]:
                best["label"], best["perm"] = label, perm
                aut_perms.clear()
                aut_perms.append(perm)
                logger.debug(f"New best label found at depth {depth}")
            elif label == best["label"]:
                aut_perms.append(perm)
                logger.debug(f"Equivalent label found at depth {depth}")
            return False

        idx = next(i for i, c in enumerate(partition) if len(c) > 1)
        cell = partition[idx]
        sorted_cell = sorted(cell, key=lambda n: G.nodes[n].get("atom_map", n))

        for v in sorted_cell:
            rest = [w for w in cell if w != v]
            # fmt: off
            new_partition = (
                partition[:idx]
                + [[v]]
                + ([sorted(rest)] if rest else [])
                + partition[idx + 1:]
            )
            # fmt: on
            candidate_prefix = prefix + [v]

            partial_label = self._build_partial_label(G, candidate_prefix)

            if best["label"] is not None and partial_label > best["label"]:
                logger.debug(f"Pruning branch at depth {depth} due to partial label")
                continue  # prune branch early

            if self._search(
                G,
                new_partition,
                candidate_prefix,
                best,
                aut_perms,
                depth=depth + 1,
                max_depth=max_depth,
            ):
                return True  # propagate early stop upward

        return False

    def _build_label(self, G, perm):
        node_segment = "|".join(
            ":".join(
                str(self._freeze(G.nodes[v].get(attr, ""))) for attr in self.node_attrs
            )
            for v in perm
        )
        n = len(perm)
        edge_bits = []
        for i in range(n):
            vi = perm[i]
            for j in range(i + 1, n):
                vj = perm[j]
                if G.has_edge(vi, vj):
                    attrs = G[vi][vj]
                    frozen_attrs = tuple(
                        self._freeze(attrs.get(a, "")) for a in self.edge_attrs
                    )
                    edge_bits.append("1:" + ":".join(str(x) for x in frozen_attrs))
                else:
                    edge_bits.append("0:" + ":".join("" for _ in self.edge_attrs))
        edge_segment = "|".join(edge_bits)
        return node_segment + "||" + edge_segment

    def _build_partial_label(self, G, prefix):
        node_segment = "|".join(
            ":".join(
                str(self._freeze(G.nodes[v].get(attr, ""))) for attr in self.node_attrs
            )
            for v in prefix
        )
        suffix = "{" * 1000  # lexicographically larger than any label char
        return node_segment + suffix

    def compute_orbits(self, aut_perms):
        if not aut_perms:
            return []

        orbit_map = {}
        orbits = []

        def union_orbits(i, j):
            if i == j:
                return
            o1 = orbits[i]
            o2 = orbits[j]
            if len(o1) < len(o2):
                i, j = j, i
                o1, o2 = o2, o1
            o1.update(o2)
            orbits[j] = set()
            for v in o2:
                orbit_map[v] = i

        first_perm = aut_perms[0]
        for idx, node in enumerate(first_perm):
            orbit_map[node] = idx
            orbits.append({node})

        for perm in aut_perms:
            for idx, node in enumerate(perm):
                union_orbits(idx, orbit_map[node])

        return [o for o in orbits if o]

    def graph_signature(self, G):
        G_canon = self.canonical_form(G)
        label = self._build_label(G_canon, sorted(G_canon.nodes()))
        return hashlib.sha256(label.encode("utf-8")).hexdigest()
