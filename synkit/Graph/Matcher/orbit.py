from __future__ import annotations

from typing import Dict, FrozenSet, Hashable, Iterable, List, Set


class OrbitAccuracy:
    """
    Compare two orbit partitions (approximate vs exact) and compute accuracy metrics.

    The class is intentionally small and OOP-styled: most methods are chainable
    (return ``self``) and computed results are exposed via properties.

    Parameters
    ----------
    approx_orbits :
        Iterable of frozenset-like objects (each containing node identifiers).
        Each element represents one orbit (set of node ids) from the *approximate*
        partition. The iterable is consumed and a *copy* (list of frozensets) is
        stored internally.
    exact_orbits :
        Iterable of frozenset-like objects (each containing node identifiers).
        Each element represents one orbit from the *exact* partition.

    Raises
    ------
    ValueError
        If the union of nodes covered by the two partitions differs (i.e. they do
        not refer to the same node set), a :class:`ValueError` is raised with a
        short diagnostic listing nodes missing in either partition.

    Attributes
    ----------
    nodes : set
        The set of all node identifiers (union of both partitions). Available
        after initialization.
    metrics : dict
        Computed metrics (see :meth:`compute`) exposed as a dictionary via the
        :pyattr:`metrics` property.
    confusion_map : dict
        A mapping ``approx_orbit_index -> { exact_orbit_index: overlap_count }``
        exposed via the :pyattr:`confusion_map` property.

    Notes
    -----
    - Node identifiers must be hashable (ints, str, ...).
    - The input iterables are not modified; the class stores its own frozenset
      copies.
    - :meth:`compute` is chainable and returns ``self``; read metrics via the
      :pyattr:`metrics` property.

    Examples
    --------
    .. code-block:: python

        approx = [frozenset({1}), frozenset({2, 3})]
        exact  = [frozenset({1}), frozenset({2, 3})]

        oa = OrbitAccuracy(approx, exact).compute()
        print(oa)                       # -> <OrbitAccuracy nodes=3 approx_orbits=2 exact_orbits=2>
        print(oa.metrics)               # -> {'node_exact_match_fraction': 1.0, ...}
        print(oa.confusion_map)         # -> {0: {0: 1}, 1: {1: 2}}
    """

    def __init__(
        self,
        approx_orbits: Iterable[FrozenSet[Hashable]],
        exact_orbits: Iterable[FrozenSet[Hashable]],
    ) -> None:
        # store copies as frozensets to guarantee immutability semantics
        self._approx_raw: List[FrozenSet[Hashable]] = [
            frozenset(s) for s in approx_orbits
        ]
        self._exact_raw: List[FrozenSet[Hashable]] = [
            frozenset(s) for s in exact_orbits
        ]
        self._validate()
        self._build_mappings()
        self._metrics: Dict[str, float] = {}
        self._confusion: Dict[int, Dict[int, int]] = {}

    def __repr__(self) -> str:
        return (
            f"<OrbitAccuracy nodes={len(self.nodes)} "
            f"approx_orbits={len(self._approx_raw)} exact_orbits={len(self._exact_raw)}>"
        )

    # ---- Convenience / help ----
    def help(self) -> str:
        """
        Return a short usage/help string.

        Returns
        -------
        str
            Brief one-line instructions on how to use the class.
        """
        return (
            "Instantiate with approx and exact orbit iterables of frozensets. "
            "Call .compute() then read .metrics or .confusion_map."
        )

    # ---- Internal validation / mapping builders ----
    def _validate(self) -> None:
        """
        Ensure both partitions cover the same node set.

        Raises
        ------
        ValueError
            If the union of nodes in ``approx_orbits`` differs from that in
            ``exact_orbits``. The raised message includes nodes missing in
            either partition for quick diagnostics.
        """
        approx_nodes: Set[Hashable] = (
            set().union(*self._approx_raw) if self._approx_raw else set()
        )
        exact_nodes: Set[Hashable] = (
            set().union(*self._exact_raw) if self._exact_raw else set()
        )
        if approx_nodes != exact_nodes:
            missing_in_approx = exact_nodes - approx_nodes
            missing_in_exact = approx_nodes - exact_nodes
            msg_parts = []
            if missing_in_approx:
                msg_parts.append(
                    f"nodes missing in approx: {sorted(missing_in_approx)}"
                )
            if missing_in_exact:
                msg_parts.append(f"nodes missing in exact: {sorted(missing_in_exact)}")
            raise ValueError(
                "Orbit partitions do not cover the same node set. "
                + " ".join(msg_parts)
            )
        self.nodes: Set[Hashable] = approx_nodes

    def _build_mappings(self) -> None:
        """Build node -> orbit index mappings for both partitions (internal use)."""
        self._node_to_approx: Dict[Hashable, int] = {}
        self._node_to_exact: Dict[Hashable, int] = {}
        for i, orb in enumerate(self._approx_raw):
            for n in orb:
                self._node_to_approx[n] = i
        for j, orb in enumerate(self._exact_raw):
            for n in orb:
                self._node_to_exact[n] = j

    # ---- Core API ----
    def compute(self, brute_force_pairs: bool = True) -> "OrbitAccuracy":
        """
        Compute all metrics and build the confusion map.

        This method is chainable and returns ``self``; call :pyattr:`metrics` or
        :pyattr:`confusion_map` afterwards to access results.

        Parameters
        ----------
        brute_force_pairs : bool, optional
            If True (default) compute pairwise accuracy by checking all unordered
            node pairs (O(N^2)). For very large node sets a combinatorial
            method (based on orbit sizes) may be preferred; this implementation
            defaults to brute-force because typical orbit counts are moderate.

        Returns
        -------
        OrbitAccuracy
            Returns ``self`` to enable chaining.
        """
        self._compute_node_exact_match_fraction()
        self._compute_confusion()
        self._compute_purity()
        if brute_force_pairs:
            self._compute_pairwise_accuracy_bruteforce()
        else:
            self._compute_pairwise_accuracy_combinatorial_fallback()
        return self

    def _compute_node_exact_match_fraction(self) -> None:
        """Compute fraction of nodes whose approx orbit exactly equals the exact orbit."""
        correct = 0
        for n in self.nodes:
            approx_idx = self._node_to_approx[n]
            exact_idx = self._node_to_exact[n]
            if self._approx_raw[approx_idx] == self._exact_raw[exact_idx]:
                correct += 1
        total = len(self.nodes) or 1
        self._metrics["node_exact_match_fraction"] = correct / total

    def _compute_confusion(self) -> None:
        """Build confusion matrix as a dict: approx_idx -> { exact_idx: overlap_count }."""
        conf: Dict[int, Dict[int, int]] = {}
        for ai, aorb in enumerate(self._approx_raw):
            row: Dict[int, int] = {}
            for ej, eorb in enumerate(self._exact_raw):
                inter = len(aorb & eorb)
                if inter:
                    row[ej] = inter
            conf[ai] = row
        self._confusion = conf

    def _compute_purity(self) -> None:
        """
        Compute purity: weighted average (by approx-orbit size) of the largest
        overlap between each approx-orbit and any exact-orbit.
        """
        total_nodes = len(self.nodes) or 1
        weight_sum = 0
        for ai, aorb in enumerate(self._approx_raw):
            if not aorb:
                continue
            overlaps = self._confusion.get(ai, {})
            best_overlap = max(overlaps.values()) if overlaps else 0
            weight_sum += best_overlap
        self._metrics["purity"] = weight_sum / total_nodes

    def _compute_pairwise_accuracy_bruteforce(self) -> None:
        """Brute-force O(N^2) pairwise agreement calculation (same-orbit vs different-orbit)."""
        nodes = list(self.nodes)
        n = len(nodes)
        if n < 2:
            self._metrics["pairwise_accuracy"] = 1.0
            return
        matches = 0
        total = 0
        for i in range(n):
            ni = nodes[i]
            for j in range(i + 1, n):
                nj = nodes[j]
                approx_same = self._node_to_approx[ni] == self._node_to_approx[nj]
                exact_same = self._node_to_exact[ni] == self._node_to_exact[nj]
                if approx_same == exact_same:
                    matches += 1
                total += 1
        self._metrics["pairwise_accuracy"] = matches / total if total else 1.0

    def _compute_pairwise_accuracy_combinatorial_fallback(self) -> None:
        """
        Placeholder for a combinatorial pairwise method (not implemented).
        Currently delegates to the brute-force implementation.
        """
        self._compute_pairwise_accuracy_bruteforce()

    # ---- Properties / accessors ----
    @property
    def metrics(self) -> Dict[str, float]:
        """
        Return computed metrics.

        Returns
        -------
        dict
            Copy of the metrics dictionary. Call :meth:`compute` first to populate.
        """
        return dict(self._metrics)

    @property
    def confusion_map(self) -> Dict[int, Dict[int, int]]:
        """
        Return the confusion map: approx_orbit_index -> { exact_orbit_index: count }.

        Returns
        -------
        dict
            Copy of the internal confusion mapping. Call :meth:`compute` first.
        """
        # return a shallow copy to avoid accidental external mutation
        return {k: dict(v) for k, v in self._confusion.items()}

    @property
    def approx_orbits(self) -> List[FrozenSet[Hashable]]:
        """
        Return the stored approx-orbits as a list of frozensets.

        Returns
        -------
        list
            Internal copy of the approximate orbit list.
        """
        return list(self._approx_raw)

    @property
    def exact_orbits(self) -> List[FrozenSet[Hashable]]:
        """
        Return the stored exact-orbits as a list of frozensets.

        Returns
        -------
        list
            Internal copy of the exact orbit list.
        """
        return list(self._exact_raw)

    # ---- Human-readable report ----
    def report(self, max_rows: int = 10) -> str:
        """
        Produce a short human-readable report summarising computed metrics and the
        top confusion rows.

        Parameters
        ----------
        max_rows : int, optional
            Maximum number of confusion rows to include in the textual report.
            Default is 10.

        Returns
        -------
        str
            A multi-line string summarising the results. Call :meth:`compute`
            before calling this method.
        """
        lines: List[str] = []
        m = self.metrics
        lines.append(repr(self))
        lines.append("Metrics:")
        for k in ("node_exact_match_fraction", "pairwise_accuracy", "purity"):
            if k in m:
                lines.append(f"  {k}: {m[k]:.6f}")
        lines.append("Top confusion rows (approx_orbit_idx -> {exact_idx:count}):")
        for ai in sorted(self._confusion)[:max_rows]:
            lines.append(f"  {ai} -> {self._confusion[ai]}")
        return "\n".join(lines)
