import networkx as nx
from collections import Counter, OrderedDict
from typing import Optional, Dict, Any, List, Iterable
from statistics import mean, median, stdev
from joblib import Parallel, delayed
from synkit.IO.debug import setup_logging

logger = setup_logging()


def _safe_stats(values: List[float]) -> Dict[str, Optional[float]]:
    """Robust numeric summary for small lists."""
    if not values:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "stdev": None,
            "count": 0,
        }
    try:
        smin = min(values)
        smax = max(values)
        smean = mean(values)
        smedian = median(values)
        sstdev = stdev(values) if len(values) > 1 else 0.0
        return {
            "min": smin,
            "max": smax,
            "mean": smean,
            "median": smedian,
            "stdev": sstdev,
            "count": len(values),
        }
    except Exception as e:
        logger.debug("stat error: %s", e)
        return {
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "stdev": None,
            "count": len(values),
        }


class Topology:
    """
    Chemical-aware Topology descriptor for a single NX graph representing:
    - reaction centre, ITS, or CGR-like condensed reaction graph.

    Node-level attributes commonly inspected:
      - element / atom / label / symbol
      - aromatic / is_aromatic (bool)
      - formal_charge, charge_change, valence_change
      - atom_map / map_idx / atom_map_number / map
      - stereo, radical, unpaired_electrons

    Edge-level attributes commonly inspected:
      - order / bond_order / bond_type / type
      - aromatic / is_aromatic
      - delta_order / order_change / delta / delta_bond (for CGR)
      - is_reaction / changed / reactive / stereo_change

    :param graph: networkx.Graph (or None)
    :param graph_id: optional identifier (string) for bookkeeping
    :param max_nodes_for_expensive: skip very expensive ops when graph larger than this
    """

    def __init__(
        self,
        graph: Optional[nx.Graph] = None,
        graph_id: Optional[str] = None,
        max_nodes_for_expensive: int = 2000,
    ):
        self._graph = graph
        self.graph_id = graph_id
        self.max_nodes_for_expensive = int(max_nodes_for_expensive)

        # Basic counters (populated in analyze)
        self.node_count = 0
        self.edge_count = 0
        self.is_directed = False
        self.has_selfloops = False
        self.number_of_isolates = 0

        # Chemical descriptors
        self.element_counts: OrderedDict = OrderedDict()
        self.heavy_atom_count: int = 0
        self.heteroatom_count: int = 0
        self.heteroatom_types: OrderedDict = OrderedDict()
        self.mapped_atom_count: int = 0
        self.mapped_fraction: float = 0.0

        # Charges
        self.formal_charge_counts: Dict[int, int] = {}
        self.charge_change_counts: Dict[int, int] = {}

        # Aromaticity & rings
        self.aromatic_atom_count: int = 0
        self.aromatic_bond_count: int = 0
        self.ring_count: int = 0
        self.ring_sizes: List[int] = []
        self.aromatic_ring_count: int = 0
        self.fused_ring_systems: int = 0  # heuristic: count of fused components

        # Bonds and bond-changes
        self.bond_order_counts: Dict[Any, int] = {}
        self.bond_order_change_hist: Dict[Any, int] = {}
        self.bonds_formed: int = 0
        self.bonds_broken: int = 0
        self.changed_edge_count: int = 0

        # Reaction center localization (if CGR/marked edges exist)
        self.rc_node_count: int = 0
        self.rc_edge_count: int = 0
        self.rc_attachment_points: int = 0
        self.rc_boundary_size: int = 0

        # stereochemistry/radicals
        self.stereo_changes: int = 0
        self.radical_centers: int = 0

        # heuristics: leaving groups / nucleophiles
        self.leaving_group_count: int = 0
        self.nucleophile_count: int = 0

        # valence/degree changes (if present)
        self.node_valence_change_stats: Dict[str, Optional[float]] = {}

        # simple deterministic fingerprint (stable short string)
        self.simple_fingerprint: Optional[str] = None

        # connectivity / components
        self.is_connected: Optional[bool] = None
        self.components_count: int = 0
        self.largest_component_size: int = 0

        # meta
        self.description: Optional[str] = None
        self._computed = False

    # ---------------------------
    # Validation / helpers
    # ---------------------------
    @staticmethod
    def _validate_graph(g: Optional[nx.Graph]) -> None:
        if g is not None and not isinstance(g, nx.Graph):
            raise TypeError("Topology expects networkx.Graph or None")

    @staticmethod
    def _iter_edge_order(edge_data: dict):
        """Try common keys for order: 'order', 'bond_type', 'bond_order'"""
        for k in ("order", "bond_order", "bond_type", "b_order"):
            if k in edge_data:
                return edge_data[k]
        return edge_data.get("type", None)

    @staticmethod
    def _iter_edge_delta(edge_data: dict):
        """Try common keys for delta/change: 'delta_order','order_change','delta'"""
        for k in ("delta_order", "order_change", "delta", "delta_bond"):
            if k in edge_data:
                return edge_data[k]
        # boolean flag indicating changed edge
        if (
            edge_data.get("is_reaction")
            or edge_data.get("changed")
            or edge_data.get("reactive")
        ):
            return 1
        return 0

    @staticmethod
    def _node_map_index(node_data: dict):
        """Look for common atom-map keys."""
        for k in ("atom_map", "map_idx", "atom_map_number", "map"):
            if k in node_data:
                return node_data[k]
        return None

    # ---------------------------
    # Chemical descriptor computations
    # ---------------------------
    def _compute_basic_counts(self, G: nx.Graph) -> None:
        """Compute node/edge counts, directedness, selfloops."""
        self.node_count = G.number_of_nodes()
        self.edge_count = G.number_of_edges()
        self.is_directed = G.is_directed()
        # networkx exposes number_of_selfloops as a module-level function, not a Graph method
        self.has_selfloops = nx.number_of_selfloops(G) > 0
        self.number_of_isolates = sum(1 for n in G.nodes() if G.degree(n) == 0)

    def _compute_element_and_mapping(self, G: nx.Graph) -> None:
        elems = []
        mapped = 0
        charges = []
        aromatic_atoms = 0
        radicals = 0
        valence_changes = []
        for _, data in G.nodes(data=True):
            # element detection
            el = (
                data.get("element")
                or data.get("atom")
                or data.get("label")
                or data.get("symbol")
            )
            if el is not None:
                elems.append(el)
            # mapping
            if self._node_map_index(data) is not None:
                mapped += 1
            # charges
            if "formal_charge" in data:
                try:
                    charges.append(int(data.get("formal_charge", 0)))
                except Exception:
                    pass
            if "charge_change" in data:
                val = data.get("charge_change")
                if val is not None:
                    # record as integer if possible
                    try:
                        self.charge_change_counts[int(val)] = (
                            self.charge_change_counts.get(int(val), 0) + 1
                        )
                    except Exception:
                        pass
            # aromatic / radical / valence change
            if data.get("aromatic") or data.get("is_aromatic"):
                aromatic_atoms += 1
            if data.get("radical") or data.get("unpaired_electrons"):
                radicals += 1
            if "valence_change" in data:
                try:
                    valence_changes.append(float(data["valence_change"]))
                except Exception:
                    pass

        self.element_counts = OrderedDict(sorted(dict(Counter(elems)).items()))
        self.heavy_atom_count = sum(
            v
            for k, v in self.element_counts.items()
            if str(k).upper() != "H" and k is not None
        )
        # heteroatoms = atoms that are not C/H (common heuristic)
        self.heteroatom_count = sum(
            v
            for k, v in self.element_counts.items()
            if k is not None and str(k).upper() not in ("C", "H")
        )
        # heteroatom types counts
        hetero = {
            k: v
            for k, v in self.element_counts.items()
            if k is not None and str(k).upper() not in ("C", "H")
        }
        self.heteroatom_types = (
            OrderedDict(sorted(hetero.items())) if hetero else OrderedDict()
        )

        self.mapped_atom_count = int(mapped)
        self.mapped_fraction = (mapped / self.node_count) if self.node_count else 0.0

        if charges:
            self.formal_charge_counts = dict(Counter(charges))
        else:
            self.formal_charge_counts = {}

        self.aromatic_atom_count = int(aromatic_atoms)
        self.radical_centers = int(radicals)
        self.node_valence_change_stats = (
            _safe_stats(valence_changes) if valence_changes else {}
        )

    def _compute_bonds_and_changes(self, G: nx.Graph) -> None:
        bond_orders = []
        bonds_formed = 0
        bonds_broken = 0
        changed_edges = 0
        aromatic_bonds = 0

        for u, v, data in G.edges(data=True):
            bo = self._iter_edge_order(data)
            if bo is not None:
                bond_orders.append(bo)
                self.bond_order_counts[bo] = self.bond_order_counts.get(bo, 0) + 1
            if data.get("aromatic") or data.get("is_aromatic"):
                aromatic_bonds += 1

            delta = self._iter_edge_delta(data)
            if delta is None:
                delta = 0
            # interpret numeric deltas if present
            try:
                dval = float(delta)
            except Exception:
                dval = 1.0 if bool(delta) else 0.0

            if dval != 0:
                changed_edges += 1
                # positive -> bond formed or order increase (heuristic)
                if dval > 0:
                    bonds_formed += 1
                elif dval < 0:
                    bonds_broken += 1
                # histogram
                key = int(dval) if float(dval).is_integer() else dval
                self.bond_order_change_hist[key] = (
                    self.bond_order_change_hist.get(key, 0) + 1
                )

        self.bond_order_counts = dict(self.bond_order_counts)
        self.bond_order_change_hist = dict(self.bond_order_change_hist)
        self.bonds_formed = int(bonds_formed)
        self.bonds_broken = int(bonds_broken)
        self.changed_edge_count = int(changed_edges)
        self.aromatic_bond_count = int(aromatic_bonds)

    def _compute_rings_and_aromaticity(self, G: nx.Graph) -> None:
        # use NX cycle basis for ring sizes (heuristic for chemical rings)
        try:
            if self.node_count == 0:
                self.ring_count = 0
                self.ring_sizes = []
                self.aromatic_ring_count = 0
                self.fused_ring_systems = 0
                return

            cycles = nx.minimum_cycle_basis(G)
            sizes = sorted(len(c) for c in cycles)
            self.ring_sizes = sizes
            self.ring_count = len(sizes)

            # crude aromatic ring detection: ring considered aromatic if majority of atoms flagged aromatic
            arom_ring = 0
            fused_components = 0
            # find overlapping cycles -> fused
            if cycles:
                # number of fused systems: connected components in ring adjacency graph
                ring_adj = nx.Graph()
                ring_adj.add_nodes_from(range(len(cycles)))
                for i, a in enumerate(cycles):
                    for j, b in enumerate(cycles):
                        if j <= i:
                            continue
                        if set(a) & set(b):
                            ring_adj.add_edge(i, j)
                fused_components = (
                    nx.number_connected_components(ring_adj)
                    if ring_adj.number_of_nodes() > 0
                    else 0
                )
                for cyc in cycles:
                    arom_atoms = sum(
                        1
                        for n in cyc
                        if G.nodes[n].get("aromatic") or G.nodes[n].get("is_aromatic")
                    )
                    if arom_atoms >= len(cyc) // 2:
                        arom_ring += 1
                self.fused_ring_systems = fused_components
            self.aromatic_ring_count = int(arom_ring)
        except Exception as e:
            logger.debug("ring/analyis failed: %s", e)
            self.ring_count = 0
            self.ring_sizes = []
            self.aromatic_ring_count = 0
            self.fused_ring_systems = 0

    def _localize_reaction_center(self, G: nx.Graph) -> None:
        """
        If the graph encodes reaction changes on edges/nodes (CGR), localize the RC:
        RC edges = edges with non-zero delta/order_change or flagged 'is_reaction'/'changed'.
        RC nodes = nodes incident to RC edges.
        Compute attachment points (nodes in RC with neighbors outside RC).
        """
        rc_edges = set()
        for u, v, data in G.edges(data=True):
            if self._iter_edge_delta(data):
                rc_edges.add((u, v))
            elif data.get("is_reaction") or data.get("changed") or data.get("reactive"):
                rc_edges.add((u, v))

        rc_nodes = set()
        for u, v in rc_edges:
            rc_nodes.add(u)
            rc_nodes.add(v)

        self.rc_edge_count = len(rc_edges)
        self.rc_node_count = len(rc_nodes)

        if not rc_nodes:
            self.rc_attachment_points = 0
            self.rc_boundary_size = 0
            return

        attachment_points = 0
        boundary = 0
        for n in rc_nodes:
            neighs = set(G.neighbors(n))
            outside = neighs - rc_nodes
            if outside:
                attachment_points += 1
                boundary += len(outside)
        self.rc_attachment_points = int(attachment_points)
        self.rc_boundary_size = int(boundary)

        # heuristics: leaving groups ~ nodes in rc_nodes that lose bonds (negative delta on incident edges)
        leaving = 0
        nucleophile = 0
        for n in rc_nodes:
            for nbr in G.neighbors(n):
                data = G.get_edge_data(n, nbr, default={})
                d = self._iter_edge_delta(data)
                try:
                    dval = float(d)
                except Exception:
                    dval = 1.0 if bool(d) else 0.0
                # negative delta -> bond lost near node => candidate leaving-group atom
                if dval < 0:
                    # preferentially heteroatoms are leaving groups (heuristic)
                    el = G.nodes[n].get("element") or G.nodes[n].get("atom")
                    if el and str(el).upper() not in ("C", "H"):
                        leaving += 1
                if dval > 0:
                    # bond formation near hetero leads to nucleophilic attack heuristic
                    el = G.nodes[n].get("element") or G.nodes[n].get("atom")
                    if el and str(el).upper() not in ("C", "H"):
                        nucleophile += 1
        self.leaving_group_count = int(leaving)
        self.nucleophile_count = int(nucleophile)

    def _detect_stereo_and_radical_changes(self, G: nx.Graph) -> None:
        stereo = 0
        radicals = 0
        for _, data in G.nodes(data=True):
            if data.get("stereo_change") or data.get("stereo") == "changed":
                stereo += 1
            if data.get("radical") or data.get("unpaired_electrons"):
                radicals += 1
        for _, _, data in G.edges(data=True):
            if data.get("stereo_change") or data.get("stereo") == "changed":
                stereo += 1
        self.stereo_changes = int(stereo)
        # radical_centers already computed partly in element pass; keep max
        self.radical_centers = max(self.radical_centers, int(radicals))

    def _make_simple_fingerprint(self, G: nx.Graph) -> None:
        """
        Deterministic short fingerprint using sorted element counts and
        sorted bond-order counts + ring-count. Useful as a quick hash.
        """
        pieces = []
        for el, c in self.element_counts.items():
            pieces.append(f"{el}{c}")
        bo_parts = []
        for bo, c in sorted(self.bond_order_counts.items(), key=lambda x: str(x[0])):
            bo_parts.append(f"{bo}:{c}")
        pieces.append("BO|" + "|".join(bo_parts))
        pieces.append(
            f"RINGS|{self.ring_count}|" + ",".join(str(s) for s in self.ring_sizes)
        )
        self.simple_fingerprint = "|".join(pieces)

    # ---------------------------
    # Main analyze flow (chainable)
    # ---------------------------
    def analyze(self) -> "Topology":
        """
        Populate all chemical-aware descriptors (chainable).
        """
        self._validate_graph(self._graph)
        self._computed = False

        if self._graph is None or self._graph.number_of_nodes() == 0:
            # set sensible defaults for empty
            self.description = "Empty graph"
            self._computed = True
            return self

        G = self._graph

        try:
            self._compute_basic_counts(G)
            self._compute_element_and_mapping(G)
            self._compute_bonds_and_changes(G)
            self._compute_rings_and_aromaticity(G)
            self._localize_reaction_center(G)
            self._detect_stereo_and_radical_changes(G)

            # connectivity / components
            try:
                self.is_connected = nx.is_connected(G)
            except Exception:
                # directed graphs or others
                try:
                    self.is_connected = (
                        nx.is_weakly_connected(G) if G.is_directed() else False
                    )
                except Exception:
                    self.is_connected = False
            try:
                comps = list(nx.connected_components(G))
                self.components_count = len(comps)
                self.largest_component_size = max((len(c) for c in comps), default=0)
            except Exception:
                self.components_count = 0
                self.largest_component_size = 0

            # fingerprint & description
            self._make_simple_fingerprint(G)
            self.description = f"id={self.graph_id or 'graph'} nodes={self.node_count} edges={self.edge_count} rings={self.ring_count} changed_edges={self.changed_edge_count}"

            self._computed = True
        except Exception as e:
            logger.exception("Chemical Topology analyze() failed: %s", e)
            self._computed = False

        return self

    # ---------------------------
    # Lightweight static/class utilities for quick one-liners
    # ---------------------------
    @classmethod
    def ring_info(
        cls,
        G: nx.Graph,
        *,
        method: str = "min_basis",
        sizes: bool = True,
        membership: bool = False,
        filter_heavy: bool = True,
        max_nodes: int = 20000,
    ) -> Dict[str, Any]:
        """
        Quick ring information for a graph (STATIC/CLASS utility).

        Parameters
        ----------
        G : networkx.Graph or None
        method : {'cyclomatic','min_basis','cycle_basis'}
        sizes : bool - return ring sizes (list of ints)
        membership : bool - return node membership lists for each ring
        filter_heavy : bool - drop hydrogens first if nodes have element attrs
        max_nodes : int - guard to avoid expensive basis methods on huge graphs
        """
        if G is None:
            return {
                "node_count": 0,
                "edge_count": 0,
                "components": 0,
                "cyclomatic_number": 0,
                "ring_count": 0,
                "ring_sizes": [] if sizes else None,
                "rings": [] if membership else None,
            }

        # operate on a simple Graph
        H = nx.Graph(G) if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)) else G

        # optionally filter hydrogens
        if filter_heavy:
            if H.number_of_nodes() > 0:
                sample_attrs = next(iter(H.nodes(data=True)), (None, {}))[1]
                if any(
                    k in sample_attrs for k in ("element", "atom", "label", "symbol")
                ):
                    nodes_to_keep = [
                        n
                        for n, d in H.nodes(data=True)
                        if str(d.get("element") or d.get("atom") or "").upper() != "H"
                    ]
                    H = H.subgraph(nodes_to_keep).copy()

        n = H.number_of_nodes()
        m = H.number_of_edges()
        try:
            comps = list(nx.connected_components(H))
            c = len(comps)
        except Exception:
            try:
                comps = list(nx.weakly_connected_components(H))
                c = len(comps)
            except Exception:
                comps = [set(H.nodes())]
                c = 1

        cyclo = m - n + c
        result: Dict[str, Any] = {
            "node_count": int(n),
            "edge_count": int(m),
            "components": int(c),
            "cyclomatic_number": int(cyclo),
            "ring_count": 0,
            "ring_sizes": [] if sizes else None,
            "rings": [] if membership else None,
        }

        if method == "cyclomatic":
            result["ring_count"] = max(0, int(cyclo))
            return result

        if n > max_nodes:
            raise ValueError(
                f"Graph too large ({n} nodes) for basis method; use method='cyclomatic' or increase max_nodes."
            )

        # collect cycles
        cycles: List[List[Any]] = []
        if method == "min_basis":
            cycles = nx.minimum_cycle_basis(H)
        elif method == "cycle_basis":
            for comp in comps:
                sub = H.subgraph(comp)
                cycles.extend(nx.cycle_basis(sub))
        else:
            raise ValueError(
                "Unsupported method; choose 'cyclomatic'|'min_basis'|'cycle_basis'"
            )

        result["ring_count"] = len(cycles)
        if sizes:
            result["ring_sizes"] = sorted(len(c) for c in cycles)
        if membership:
            result["rings"] = [list(c) for c in cycles]
        return result

    @classmethod
    def quick_descriptors(
        cls,
        G: nx.Graph,
        *,
        include: Iterable[str] = ("counts", "rings"),
        ring_method: str = "min_basis",
        ring_max_nodes: int = 20000,
        filter_heavy_for_rings: bool = True,
    ) -> Dict[str, Any]:
        """
        Quick, pick-and-choose small descriptor bag for chemistry pipelines.

        Supported include keys:
          - "counts"       : node_count, edge_count, components, cyclomatic_number
          - "rings"        : ring_count, ring_sizes (uses ring_method)
          - "mapping"      : mapped_atom_count, mapped_fraction (node attrs)
          - "charges"      : formal_charge_counts (node attrs 'formal_charge')
          - "bond_changes" : bonds_formed, bonds_broken, changed_edge_count (edge attrs)
        """
        out: Dict[str, Any] = {}
        if G is None:
            return out

        # Basic counts (cheap)
        n = G.number_of_nodes()
        m = G.number_of_edges()
        try:
            comps = list(nx.connected_components(G))
            c = len(comps)
        except Exception:
            try:
                comps = list(nx.weakly_connected_components(G))
                c = len(comps)
            except Exception:
                comps = [set(G.nodes())]
                c = 1
        cyclo = m - n + c

        if "counts" in include:
            out.update(
                {
                    "node_count": int(n),
                    "edge_count": int(m),
                    "components": int(c),
                    "cyclomatic_number": int(cyclo),
                }
            )

        if "rings" in include:
            ring_info = cls.ring_info(
                G,
                method=ring_method,
                sizes=True,
                membership=False,
                filter_heavy=filter_heavy_for_rings,
                max_nodes=ring_max_nodes,
            )
            out.update(
                {
                    "ring_count": ring_info["ring_count"],
                    "ring_sizes": ring_info.get("ring_sizes", []),
                }
            )

        if "mapping" in include:
            mapped = 0
            for _, d in G.nodes(data=True):
                if cls._node_map_index(d) is not None:
                    mapped += 1
            out["mapped_atom_count"] = int(mapped)
            out["mapped_fraction"] = float(mapped / n) if n else 0.0

        if "charges" in include:
            charges = []
            for _, d in G.nodes(data=True):
                if "formal_charge" in d:
                    try:
                        charges.append(int(d.get("formal_charge", 0)))
                    except Exception:
                        pass
            out["formal_charge_counts"] = dict(Counter(charges)) if charges else {}

        if "bond_changes" in include:
            bonds_formed = 0
            bonds_broken = 0
            changed = 0
            bo_counts = Counter()
            for u, v, d in G.edges(data=True):
                delta = cls._iter_edge_delta(d)
                try:
                    dval = float(delta)
                except Exception:
                    dval = 1.0 if bool(delta) else 0.0
                if dval != 0.0:
                    changed += 1
                    if dval > 0:
                        bonds_formed += 1
                    elif dval < 0:
                        bonds_broken += 1
                bo = cls._iter_edge_order(d)
                if bo is not None:
                    bo_counts[bo] += 1
            out.update(
                {
                    "bonds_formed": int(bonds_formed),
                    "bonds_broken": int(bonds_broken),
                    "changed_edge_count": int(changed),
                    "bond_order_counts": dict(bo_counts),
                }
            )

        return out

    @classmethod
    def chemical_quick(
        cls,
        G: nx.Graph,
        *,
        features: Optional[Iterable[str]] = None,
        ring_method: str = "min_basis",
        ring_max_nodes: int = 20000,
    ) -> Dict[str, Any]:
        """
        Convenience wrapper around quick_descriptors with a chemistry-aware default set.
        Default features: counts, rings, mapping, bond_changes.
        """
        if features is None:
            features = ("counts", "rings", "mapping", "bond_changes")
        return cls.quick_descriptors(
            G, include=features, ring_method=ring_method, ring_max_nodes=ring_max_nodes
        )

    # ---------------------------
    # Exports & properties
    # ---------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable dict with the computed chemical descriptors."""
        return {
            "graph_id": self.graph_id,
            "node_count": int(self.node_count),
            "edge_count": int(self.edge_count),
            "is_directed": bool(self.is_directed),
            "has_selfloops": bool(self.has_selfloops),
            "element_counts": dict(self.element_counts),
            "heavy_atom_count": int(self.heavy_atom_count),
            "heteroatom_count": int(self.heteroatom_count),
            "heteroatom_types": dict(self.heteroatom_types),
            "mapped_atom_count": int(self.mapped_atom_count),
            "mapped_fraction": float(self.mapped_fraction),
            "formal_charge_counts": dict(self.formal_charge_counts),
            "charge_change_counts": dict(self.charge_change_counts),
            "aromatic_atom_count": int(self.aromatic_atom_count),
            "aromatic_bond_count": int(self.aromatic_bond_count),
            "ring_count": int(self.ring_count),
            "ring_sizes": list(self.ring_sizes),
            "aromatic_ring_count": int(self.aromatic_ring_count),
            "fused_ring_systems": int(self.fused_ring_systems),
            "bond_order_counts": dict(self.bond_order_counts),
            "bond_order_change_hist": dict(self.bond_order_change_hist),
            "bonds_formed": int(self.bonds_formed),
            "bonds_broken": int(self.bonds_broken),
            "changed_edge_count": int(self.changed_edge_count),
            "rc_node_count": int(self.rc_node_count),
            "rc_edge_count": int(self.rc_edge_count),
            "rc_attachment_points": int(self.rc_attachment_points),
            "rc_boundary_size": int(self.rc_boundary_size),
            "stereo_changes": int(self.stereo_changes),
            "radical_centers": int(self.radical_centers),
            "leaving_group_count": int(self.leaving_group_count),
            "nucleophile_count": int(self.nucleophile_count),
            "node_valence_change_stats": (
                dict(self.node_valence_change_stats)
                if isinstance(self.node_valence_change_stats, dict)
                else self.node_valence_change_stats
            ),
            "is_connected": bool(self.is_connected),
            "components_count": int(self.components_count),
            "largest_component_size": int(self.largest_component_size),
            "simple_fingerprint": self.simple_fingerprint,
            "description": self.description,
        }

    def summary(self) -> Dict[str, Any]:
        """Compact summary for logging/inspection."""
        return {
            "graph_id": self.graph_id,
            "nodes": self.node_count,
            "edges": self.edge_count,
            "rings": self.ring_count,
            "changed_edges": self.changed_edge_count,
            "mapped_frac": self.mapped_fraction,
            "fingerprint": self.simple_fingerprint,
        }

    def __repr__(self) -> str:
        return f"<TopologyChem id={self.graph_id!r} nodes={self.node_count} edges={self.edge_count} rings={self.ring_count} changed_edges={self.changed_edge_count}>"

    # ---------------------------
    # Batch helper
    # ---------------------------
    @classmethod
    def process_graphs_in_parallel(
        cls,
        graphs: List[nx.Graph],
        graph_ids: Optional[List[str]] = None,
        n_jobs: int = 4,
        verbose: int = 0,
    ):
        """
        Parallel batch processing. Returns list of to_dict() for each graph.
        """
        if graph_ids is None:
            graph_ids = [None] * len(graphs)

        def _proc(g, gid):
            t = cls(g, graph_id=gid)
            t.analyze()
            return t.to_dict()

        results = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(_proc)(g, gid) for g, gid in zip(graphs, graph_ids)
        )
        return results
