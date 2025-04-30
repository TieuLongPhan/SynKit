from __future__ import annotations

"""synreactor.py
================
A lightweight *pure‑Python* reactor that applies a **SynRule** template to a
substrate molecule represented as a :class:`~synkit.Graph.syn_graph.SynGraph`.
The class performs three high‑level steps:

1. **Canonicalise inputs** – substrate + template are normalised via
   :class:`~synkit.Graph.canon_graph.GraphCanonicaliser` so that hash/lookup
   semantics are stable.
2. **Sub‑graph matching** – finds every monomorphism of the template’s *left*
   pattern into the substrate using VF2 (with inline H‑count constraints).
3. **Graph composition** – glues the reaction‑core (RC) onto the substrate,
   explicitly migrates hydrogens, and serialises each mapping to a canonical
   SMARTS string.

The implementation is self‑contained (uses only NetworkX + SynKit helpers) and
avoids heavy RDKit calls until after graph assembly.
"""

import networkx as nx
from copy import deepcopy
from collections import defaultdict
from networkx.algorithms.isomorphism import GraphMatcher
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
from synkit.IO.chem_converter import (
    smiles_to_graph,
    rsmi_to_its,
    graph_to_smi,
)

from synkit.Graph.syn_rule import SynRule
from synkit.Graph.syn_graph import SynGraph
from synkit.Graph.canon_graph import GraphCanonicaliser
from synkit.Graph.ITS.its_decompose import its_decompose


__all__ = ["SynReactor"]

NodeId = Any
EdgeAttr = Mapping[str, Any]
MappingDict = Dict[NodeId, NodeId]


class SynReactor:
    """
    Reactor that applies a SynRule-style reaction template to a substrate molecule.

    This class wraps substrate and template inputs, finds valid subgraph mappings,
    builds intermediate template structures (ITS), and generates canonical SMARTS.

    Attributes
    ----------
    graph : SynGraph
        Substrate molecule as a SynGraph.
    rule : SynRule
        The reaction template as a SynRule.
    invert : bool
        Whether to apply the template in reverse (products → reactants).
    its : List[nx.Graph]
        ITS graphs for each mapping.
    smarts : List[str]
        Canonical SMARTS strings for each ITS.

    Methods
    -------
    from_smiles()
        Alternate constructor using a SMILES string.
    help()
        Print a summary of reactor configuration.
    mapping_count
        Number of valid subgraph mappings.
    smiles_list
        List of product SMILES from each mapping.
    substrate_smiles
        Canonical SMILES of the input substrate.
    """

    @classmethod
    def from_smiles(
        cls,
        smiles: str,
        template: Union[str, nx.Graph, SynRule],
        *,
        invert: bool = False,
        canonicaliser: Optional[GraphCanonicaliser] = None,
        explicit_h: bool = True,
    ) -> "SynReactor":
        """

        Parameters
        ----------
        smiles : str
            SMILES representation of the substrate.
        template : Union[str, nx.Graph, SynRule]
            Reaction template as ITS-SMARTS, an ITS graph, or a SynRule.
        invert : bool, optional
            If True, apply the template in reverse (default False).
        canonicaliser : Optional[GraphCanonicaliser], optional
            Custom GraphCanonicaliser (default: new instance).

        Returns
        -------
        SynReactor
            Configured SynReactor instance.
        """
        return cls(
            smiles,
            template,
            invert=invert,
            canonicaliser=canonicaliser,
            explicit_h=explicit_h,
        )

    # ------------------------------------------------------------------
    # Initialiser
    # ------------------------------------------------------------------
    def __init__(
        self,
        substrate: Union[str, nx.Graph, SynGraph],
        template: Union[str, nx.Graph, SynRule],
        *,
        invert: bool = False,
        canonicaliser: Optional[GraphCanonicaliser] = None,
        explicit_h: bool = True,
    ) -> None:
        """
        Initialize SynReactor: wrap inputs, canonicalise, find mappings, build ITS and SMARTS.

        Parameters
        ----------
        input : Union[str, nx.Graph, SynGraph]
            Substrate as SMILES, raw NX graph, or SynGraph.
        template : Union[str, nx.Graph, SynRule]
            Reaction template as ITS-SMARTS, raw ITS graph, or SynRule.
        invert : bool
            If True, apply template in reverse (default False).
        canonicaliser : Optional[GraphCanonicaliser]
            Custom canonicaliser; created if None.
        """
        self._canonicaliser: GraphCanonicaliser = canonicaliser or GraphCanonicaliser()
        self.invert: bool = invert

        # ---------- wrap / canonicalise inputs ----------------------
        self.graph: SynGraph = self._wrap_input(substrate)
        self.rule: SynRule = self._wrap_template(template)

        # ---------- find pattern → host mappings --------------------
        self._mappings: List[Dict[Any, Any]] = self._find_subgraph_mappings(
            host=self.graph.raw,
            pattern=self.rule.left.raw,
            node_attrs=["element", "charge"],
            edge_attrs=["order"],
        )

        # ---------- build ITS graphs for each mapping ---------------
        self.its: List[nx.Graph] = [
            self._glue_graph(self.graph.raw, self.rule.rc.raw, m)
            for m in self._mappings
        ]
        if explicit_h:
            self.its = [self._explicit_h(g) for g in self.its]

        # ---------- canonical SMARTS --------------------------------
        self.smarts: List[str] = [self._to_smarts(g) for g in self.its]

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def help(self) -> None:
        """
        Print a compact summary of the reactor's configuration:
        substrate, template, inversion flag, mapping count, and SMARTS list.
        """
        print("SynReactor\n----------")
        print("Substrate :", self.graph)
        print("Template  :", self.rule)
        print("Inverted  :", self.invert)
        print("Mappings  :", self.mapping_count)
        print("SMARTS    :")
        for s in self.smarts:
            print("  ", s)

    # ================================================================== #
    # Convenience properties                                             #
    # ================================================================== #
    def __str__(self) -> str:
        """String repr indicating substrate atom count and mapping count."""
        return (
            f"<SynReactor substrate_atoms={self.graph.raw.number_of_nodes()} "
            f"mappings={len(self._mappings)}>"
        )

    __repr__ = __str__

    @property
    def mapping_count(self) -> int:
        """int: Number of valid subgraph mappings found."""
        return len(self._mappings)

    @property
    def smiles_list(self) -> List[str]:
        """List[str]: Product SMILES strings from each SMARTS mapping."""
        return [s.split(">>")[-1] for s in self.smarts]

    @property
    def substrate_smiles(self) -> str:
        """str: Canonical SMILES representation of the input substrate."""
        return graph_to_smi(self.graph.raw)

    # ------------------------------------------------------------------
    # Private utilities – wrapping / canonicalising
    # ------------------------------------------------------------------
    @staticmethod
    def _wrap_input(obj: Union[str, nx.Graph, SynGraph]) -> SynGraph:
        """
        Convert input to a SynGraph.

        Parameters
        ----------
        obj : Union[str, nx.Graph, SynGraph]
            SMILES, raw NX graph, or SynGraph.

        Returns
        -------
        SynGraph
            Wrapped substrate graph.

        Raises
        ------
        TypeError
            If unsupported type.
        """
        if isinstance(obj, SynGraph):
            return obj
        if isinstance(obj, nx.Graph):
            return SynGraph(obj, GraphCanonicaliser())
        if isinstance(obj, str):
            graph = smiles_to_graph(obj, use_index_as_atom_map=False)
            return SynGraph(graph, GraphCanonicaliser())
        raise TypeError(f"Unsupported input type: {type(obj)}")

    def _wrap_template(self, tpl: Union[str, nx.Graph, "SynRule"]) -> SynRule:
        if isinstance(tpl, SynRule):
            return tpl
        if isinstance(tpl, nx.Graph):
            return SynRule(tpl, canonicaliser=self._canonicaliser)
        if isinstance(tpl, str):
            return SynRule(rsmi_to_its(tpl), canonicaliser=self._canonicaliser)
        raise TypeError(f"Unsupported template type: {type(tpl)}")

    # ------------------------------------------------------------------
    # Sub‑graph matching (VF2 with inline H‑count)
    # ------------------------------------------------------------------

    @staticmethod
    def _find_subgraph_mappings(
        host: nx.Graph,
        pattern: nx.Graph,
        *,
        node_attrs: List[str],
        edge_attrs: List[str],
    ) -> List[Dict[Any, Any]]:
        """
        Return all pattern→host monomorphisms obeying hcount constraints.

        Now enforces host.hcount >= pattern.hcount in the node_match itself.
        """

        def node_match(nh: EdgeAttr, np: EdgeAttr) -> bool:
            if any(nh.get(k) != np.get(k) for k in node_attrs):
                return False
            return nh.get("hcount", 0) >= np.get("hcount", 0)

        def edge_match(eh: EdgeAttr, ep: EdgeAttr) -> bool:
            return all(eh.get(k) == ep.get(k) for k in edge_attrs)

        gm = GraphMatcher(host, pattern, node_match=node_match, edge_match=edge_match)
        return [
            {p: h for h, p in iso.items()} for iso in gm.subgraph_monomorphisms_iter()
        ]

    @staticmethod
    def _node_glue(
        host_n: Dict[str, Any], pat_n: Dict[str, Any], key: str = "typesGH"
    ) -> None:
        """
        Merge pattern hydrogen deltas onto a host node's typesGH attribute.

        Parameters
        ----------
        host_n : Dict[str, Any]
            Host node’s attributes (modified in place).
        pat_n : Dict[str, Any]
            Pattern node’s attributes (provides deltas).
        key : str
            Key under which typesGH tuple-pair is stored.
        """
        host_r, host_p = host_n[key]
        pat_r, pat_p = pat_n[key]
        delta = pat_r[2] - pat_p[2]

        new_r = host_r[:2] + (host_r[2],) + host_r[3:]
        new_p = host_p[:2] + (host_r[2] - delta,) + (pat_p[3],) + host_p[4:]
        host_n[key] = (new_r, new_p)

        if "h_pairs" in pat_n:
            host_n["h_pairs"] = pat_n["h_pairs"]

    @staticmethod
    def _glue_graph(host: nx.Graph, rc: nx.Graph, mapping: MappingDict) -> nx.Graph:
        its = deepcopy(host)

        def _default_tg(a: Dict[str, Any]) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
            tpl = (
                a.get("element", "*"),
                a.get("aromatic", False),
                a.get("hcount", 0),
                a.get("charge", 0),
                a.get("neighbors", []),
            )
            return tpl, tpl

        for _, data in its.nodes(data=True):
            data.setdefault("typesGH", _default_tg(data))
        for u, v, data in its.edges(data=True):
            o = data.get("order", 1.0)
            data["order"] = (o, o)
            data.setdefault("standard_order", 0.0)

        for _, data in rc.nodes(data=True):
            data.setdefault("typesGH", _default_tg(data))

        # merge nodes
        for rc_n, host_n in mapping.items():
            if its.has_node(host_n):
                SynReactor._node_glue(its.nodes[host_n], rc.nodes[rc_n])

        # merge edges with additive order if rc contribution (0,x)
        for u, v, rc_attr in rc.edges(data=True):
            hu, hv = mapping.get(u), mapping.get(v)
            if hu is None or hv is None:
                continue
            if not its.has_edge(hu, hv):
                its.add_edge(hu, hv, **dict(rc_attr))
            else:
                host_attr = its[hu][hv]
                rc_order = rc_attr.get("order", (0, 0))
                if rc_order[0] == 0:  # additive only on product side
                    ho = host_attr["order"]
                    host_attr["order"] = (ho[0], round(ho[1] + rc_order[1]))
                    host_attr["standard_order"] += rc_attr.get("standard_order", 0.0)
                else:
                    host_attr.update(rc_attr)
        return its

    @staticmethod
    def _explicit_h(rc: nx.Graph) -> nx.Graph:
        """
        Insert explicit hydrogen nodes based on h_pairs and typesGH deltas,
        correctly handling atoms that participate in multiple H-pairs.

        Steps:
        1. Record each heavy atom’s original Δ = (hl − hr).
        2. Build a connectivity graph linking atoms sharing any h_pair ID.
        3. In each connected component, donors (Δ>0) send exactly that many H’s to
           recipients (Δ<0), one unit at a time.
        4. Create one new H node per unit migration, with directed edges.
        5. Zero out all original heavy-atom Δ’s afterward.
        """
        # rc = rc_graph.copy()
        next_id = max((n for n in rc.nodes if isinstance(n, int)), default=-1) + 1

        # 1) Record original Δ per atom and collect h_pair memberships.
        orig_delta: Dict[int, int] = {}
        pair_to_nodes: Dict[int, List[int]] = defaultdict(list)
        for n, d in rc.nodes(data=True):
            h_pairs = d.get("h_pairs", [])
            hl = d["typesGH"][0][2]
            hr = d["typesGH"][1][2]
            orig_delta[n] = hl - hr
            for pid in h_pairs:
                if n not in pair_to_nodes[pid]:
                    pair_to_nodes[pid].append(n)
        # 2) Build connectivity graph of atoms sharing any PID.
        conn = nx.Graph()
        for nodes in pair_to_nodes.values():
            conn.add_nodes_from(nodes)
            # fmt: off
            conn.add_edges_from(
                [(u, v) for i, u in enumerate(nodes) for v in nodes[i + 1:]]
            )
            # fmt: on

        # 3) Schedule one migration per Δ unit in each component.
        migrations: List[Tuple[int, int]] = []
        for comp in nx.connected_components(conn):
            donors = [(n, orig_delta[n]) for n in comp if orig_delta[n] > 0]
            recips = [(n, -orig_delta[n]) for n in comp if orig_delta[n] < 0]
            for donor, count in donors:
                for _ in range(count):
                    recv_idx = next(i for i, r in enumerate(recips) if r[1] > 0)
                    recv, rcap = recips[recv_idx]
                    recips[recv_idx] = (recv, rcap - 1)
                    migrations.append((donor, recv))

        # 4) Create explicit H node for each scheduled migration.
        for src, dst in migrations:
            h = next_id
            next_id += 1
            rc.add_node(
                h,
                element="H",
                aromatic=False,
                charge=0,
                atom_map=0,
                hcount=0,
                typesGH=(("H", False, 0, 0, []), ("H", False, 0, 0, [])),
            )
            rc.add_edge(src, h, order=(1, 0), standard_order=1)
            rc.add_edge(h, dst, order=(0, 1), standard_order=-1)

        affected = {n for nodes in pair_to_nodes.values() for n in nodes}
        for n in affected:
            t0, t1 = rc.nodes[n]["typesGH"]
            rc.nodes[n]["typesGH"] = (t0[:2] + (0,) + t0[3:], t1[:2] + (0,) + t1[3:])
        return rc

    # ------------------------------------------------------------------
    # Utility – SMARTS serialisation
    # ------------------------------------------------------------------
    @staticmethod
    def _to_smarts(its: nx.Graph) -> str:
        l, r = its_decompose(its)
        return f"{graph_to_smi(l)}>>{graph_to_smi(r)}"
