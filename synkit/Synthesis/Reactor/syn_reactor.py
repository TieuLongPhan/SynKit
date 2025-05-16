from __future__ import annotations

from copy import deepcopy
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import networkx as nx


from synkit.IO.chem_converter import (
    smiles_to_graph,
    rsmi_to_its,
    graph_to_smi,
)
from synkit.IO import setup_logging
from synkit.Chem.Reaction.rsmi_utils import reverse_reaction

from synkit.Rule import SynRule
from synkit.Graph.syn_graph import SynGraph
from synkit.Graph.canon_graph import GraphCanonicaliser
from synkit.Graph.ITS.its_decompose import its_decompose
from synkit.Graph.ITS.its_construction import ITSConstruction
from synkit.Graph.Matcher.subgraph_matcher import SubgraphSearchEngine
from synkit.Graph.Hyrogen._misc import h_to_implicit, h_to_explicit, has_XH
from synkit.Graph import remove_wildcard_nodes, add_wildcard_subgraph_for_unmapped
from synkit.Synthesis.Reactor.strategy import Strategy


# ──────────────────────────────────────────────────────────────────────────────
# Typing aliases
# ──────────────────────────────────────────────────────────────────────────────
NodeId = Any
EdgeAttr = Mapping[str, Any]
MappingDict = Dict[NodeId, NodeId]

# ──────────────────────────────────────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────────────────────────────────────

log = setup_logging(task_type="synreactor")


# ──────────────────────────────────────────────────────────────────────────────
# SynReactor core
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class SynReactor:
    """
    A **hardened** and **typed** re-write of the original SynReactor,
    preserving 100% API compatibility but with:

    - **Safer**  – avoids mutating inputs, validates arguments, logs diagnostics
    - **Faster** – lazy builds and optional thread‐pool for explicit‐H work
    - **Cleaner** – exhaustive doc-strings, full typing, single‐purpose helpers

    Parameters
    ----------
    substrate : Union[str, nx.Graph, SynGraph]
        The input reaction substrate, as SMILES, a raw Graph, or a SynGraph.
    template : Union[str, nx.Graph, SynRule]
        Reaction template (SMILES/SMARTS, raw Graph, or SynRule).
    invert : bool, optional
        False for forward prediction; True for backward (target→precursors).
    canonicaliser : Optional[GraphCanonicaliser], optional
        If provided, used to canonicalise intermediary graphs.
    explicit_h : bool, default=True
        If True, render all H’s explicitly in the reaction‐center SMARTS.
    implicit_temp : bool, default=False
        If True, treat the input template as an implicit‐H pattern (forces explicit_h=False).
    strategy : Strategy or str, default=Strategy.ALL
        Graph‐matching strategy: ALL, COMPONENT (comp), or BACKTRACK (bt).

    Attributes (cached)
    -------------------
    graph : SynGraph
        The substrate, wrapped as a SynGraph.
    rule : SynRule
        The reaction template, wrapped as a SynRule.
    mappings : List[MappingDict]
        All subgraph‐mapping dicts found (according to `strategy`).
    its_list : List[nx.Graph]
        ITS graphs built for each mapping.
    smarts_list : List[str]
        SMARTS strings generated from each ITS graph (inverted if `invert`).
    """

    substrate: Union[str, nx.Graph, SynGraph]
    template: Union[str, nx.Graph, SynRule]
    invert: bool = False
    canonicaliser: GraphCanonicaliser | None = None
    explicit_h: bool = True
    implicit_temp: bool = False
    strategy: Strategy | str = Strategy.ALL

    # Private caches – populated on demand -------------------------------
    _graph: SynGraph | None = field(init=False, default=None, repr=False)
    _rule: SynRule | None = field(init=False, default=None, repr=False)
    _mappings: List[MappingDict] | None = field(init=False, default=None, repr=False)
    _its: List[nx.Graph] | None = field(init=False, default=None, repr=False)
    _smarts: List[str] | None = field(init=False, default=None, repr=False)
    _flag_pattern_has_explicit_H: bool = field(init=False, default=False, repr=False)

    def __post_init__(self):
        # Enforce consistency: explicit_h only valid when implicit_temp is True
        if self.implicit_temp and self.explicit_h:
            raise ValueError(
                "`explicit_h` cannot be True when `implicit_temp` is False."
            )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_smiles(
        cls,
        smiles: str,
        template: Union[str, nx.Graph, SynRule],
        *,
        invert: bool = False,
        canonicaliser: Optional[GraphCanonicaliser] = None,
        explicit_h: bool = True,
        implicit_temp: bool = False,
        strategy: Strategy | str = Strategy.ALL,
    ) -> "SynReactor":
        """
        Alternate constructor: build a SynReactor directly from SMILES.

        :param smiles: SMILES string for the substrate.
        :type smiles: str
        :param template: Reaction template (SMILES/SMARTS string, Graph, or SynRule).
        :type template: str or networkx.Graph or SynRule
        :param invert: If True, perform backward prediction (target→precursors).
                       Defaults to False (forward prediction).
        :type invert: bool
        :param canonicaliser: Optional GraphCanonicaliser to use for internal graphs.
        :type canonicaliser: GraphCanonicaliser or None
        :param explicit_h: If True, keep explicit hydrogens in the reaction center.
        :type explicit_h: bool
        :param implicit_temp: If True, treat the template as implicit-H (forces explicit_h=False).
        :type implicit_temp: bool
        :param strategy: Matching strategy: ALL, 'comp', or 'bt'. Defaults to ALL.
        :type strategy: Strategy or str
        :returns: A new `SynReactor` instance.
        :rtype: SynReactor
        """
        return cls(
            substrate=smiles,
            template=template,
            invert=invert,
            canonicaliser=canonicaliser,
            explicit_h=explicit_h,
            implicit_temp=implicit_temp,
            strategy=strategy,
        )

    # ------------------------------------------------------------------
    # Public read‑only properties (lazily computed) ----------------------
    # ------------------------------------------------------------------
    @property
    def graph(self) -> SynGraph:  # noqa: D401 – read‑only property
        """
        Lazily wrap the substrate into a SynGraph.

        :returns: The reaction substrate as a `SynGraph`.
        :rtype: SynGraph
        """
        if self._graph is None:
            self._graph = self._wrap_input(self.substrate)
        return self._graph

    @property
    def rule(self) -> SynRule:  # noqa: D401
        """
        Lazily wrap the template into a SynRule.

        :returns: The reaction template as a `SynRule`.
        :rtype: SynRule
        """
        if self._rule is None:
            self._rule = self._wrap_template(self.template)
        return self._rule

    # ------------------------------------------------------------------
    # Mapping / ITS / SMARTS (computed once, cached) --------------------
    # ------------------------------------------------------------------
    @property
    def mappings(self) -> List[MappingDict]:
        """
        Find subgraph mappings between substrate and template.

        :returns: A list of node-mapping dictionaries.
        :rtype: list of dict
        """
        if self._mappings is None:
            log.debug("Finding sub‑graph mappings (strategy=%s)", self.strategy)
            pattern_graph = self.rule.left.raw

            # Detect explicit‑H constraints on the pattern and pre‑process
            if has_XH(pattern_graph):
                self._flag_pattern_has_explicit_H = True
                # self.strategy = Strategy.ALL # force to find all in implicit case
                pattern_graph = h_to_implicit(pattern_graph)
            if self.implicit_temp:
                pattern_graph = remove_wildcard_nodes(pattern_graph)
            self._mappings = SubgraphSearchEngine.find_subgraph_mappings(
                host=self.graph.raw,
                pattern=pattern_graph,
                node_attrs=["element", "charge"],
                edge_attrs=["order"],
                strategy=Strategy.from_string(self.strategy),
            )
            log.info("%d mapping(s) discovered", len(self._mappings))
        return self._mappings

    @property
    def its_list(self) -> List[nx.Graph]:
        """
        Build ITS graphs for each subgraph mapping.

        :returns: A list of ITS (Internal Transition State) graphs.
        :rtype: list of networkx.Graph
        """
        if self._its is None:
            # Build ITS for each mapping -------------------------------
            host_raw = self.graph.raw
            rc_raw = self.rule.rc.raw
            self._its = []
            for m in self.mappings:
                its_batch = self._glue_graph(
                    host_raw,
                    rc_raw,
                    m,
                    self._flag_pattern_has_explicit_H,
                    self.rule.left.raw,
                    Strategy.from_string(self.strategy),
                )
                self._its.extend(its_batch)

            if self.explicit_h:
                self._its = [self._explicit_h(g) for g in self._its]
            log.debug("Built %d ITS graph(s)", len(self._its))
        return self._its

    @property
    def smarts_list(self) -> List[str]:
        """
        Serialise each ITS graph to a reaction-SMARTS string.

        :returns: A list of SMARTS strings (inverted if `invert=True`).
        :rtype: list of str
        """
        if self._smarts is None:
            self._smarts = [self._to_smarts(g) for g in self.its_list]
            self._smarts = [value for value in self._smarts if value]
            if self.invert:
                self._smarts = [reverse_reaction(rsmi) for rsmi in self._smarts]
        return self._smarts

    # Backward‑compat aliases (original attribute names) ----------------
    smarts = property(lambda self: self.smarts_list)
    its = property(lambda self: self.its_list)
    _mappings_prop = property(lambda self: self.mappings, doc="Alias for compatibility")

    # Convenience re‑exports -------------------------------------------
    mapping_count = property(lambda self: len(self.mappings), doc="Number of mappings")
    smiles_list = property(lambda self: [s.split(">>")[-1] for s in self.smarts_list])
    substrate_smiles = property(lambda self: graph_to_smi(self.graph.raw))

    # ------------------------------------------------------------------
    # String‑likes ------------------------------------------------------
    # ------------------------------------------------------------------
    def __str__(self) -> str:  # pragma: no cover
        return (
            f"<SynReactor atoms={self.graph.raw.number_of_nodes()} "
            f"mappings={self.mapping_count}>"
        )

    __repr__ = __str__

    # ------------------------------------------------------------------
    # Public helper -----------------------------------------------------
    # ------------------------------------------------------------------
    def help(
        self, print_results=False
    ) -> None:  # pragma: no cover – human‑oriented output
        print("SynReactor")
        print("  Substrate :", self.substrate_smiles)
        print("  Template  :", self.rule)
        print("  Implicit Template :", self.implicit_temp)
        print("  Invert rule  :", self.invert)
        print("  Strategy  :", Strategy.from_string(self.strategy).value)
        print("  Predictions  :", self.mapping_count)
        if print_results:
            for i, s in enumerate(self.smarts_list, 1):
                print(f"  SMARTS[{i:02d}] : {s}")
        else:
            print(f"  First result : {self.smarts_list[0]}")

    # ==================================================================
    # Private – wrapping / canonicalising
    # ==================================================================
    def _wrap_input(self, obj: Union[str, nx.Graph, SynGraph]) -> SynGraph:
        if isinstance(obj, SynGraph):
            return obj
        if isinstance(obj, nx.Graph):
            return SynGraph(obj, self.canonicaliser or GraphCanonicaliser())
        if isinstance(obj, str):
            graph = smiles_to_graph(
                obj, use_index_as_atom_map=False, drop_non_aam=False
            )
            return SynGraph(graph, self.canonicaliser or GraphCanonicaliser())
        raise TypeError(f"Unsupported substrate type: {type(obj)}")

    def _wrap_template(self, tpl: Union[str, nx.Graph, SynRule]) -> SynRule:
        # Return early when incoming SynRule matches desired orientation
        if not self.invert and isinstance(tpl, SynRule):
            return tpl

        # Convert to raw graph ------------------------------------------------
        if isinstance(tpl, SynRule):
            graph = tpl.rc.raw  # raw reaction‑core graph
        elif isinstance(tpl, nx.Graph):
            graph = tpl
        elif isinstance(tpl, str):
            graph = rsmi_to_its(tpl)
        else:  # pragma: no cover
            raise TypeError(f"Unsupported template type: {type(tpl)}")

        # Invert if asked -----------------------------------------------------
        if self.invert:
            if self.implicit_temp:
                graph = self._invert_template(graph, balance_its=True)
                return SynRule(
                    graph,
                    canonicaliser=self.canonicaliser or GraphCanonicaliser(),
                    implicit_h=False,
                )
            else:
                graph = self._invert_template(graph, balance_its=False)
                return SynRule(
                    graph, canonicaliser=self.canonicaliser or GraphCanonicaliser()
                )
        else:
            if self.implicit_temp:
                return SynRule(
                    graph,
                    canonicaliser=self.canonicaliser or GraphCanonicaliser(),
                    implicit_h=False,
                )
            return SynRule(
                graph, canonicaliser=self.canonicaliser or GraphCanonicaliser()
            )

    @staticmethod
    def _invert_template(tpl: nx.Graph, balance_its: bool = True) -> nx.Graph:
        l, r = its_decompose(tpl)
        return ITSConstruction().ITSGraph(r, l, balance_its=balance_its)

    # ==================================================================
    # Aux – glue, explicit‑H, SMARTS
    # ==================================================================
    @staticmethod
    def _node_glue(
        host_n: Dict[str, Any], pat_n: Dict[str, Any], key: str = "typesGH"
    ) -> None:
        host_r, host_p = host_n[key]
        pat_r, pat_p = pat_n[key]
        delta = pat_r[2] - pat_p[2]
        if pat_r[0] == "*":
            new_r = (pat_r[0],) + host_r[1:2] + (host_r[2],) + host_r[3:]
        else:
            new_r = host_r[:2] + (host_r[2],) + host_r[3:]
        if pat_p[0] == "*":
            new_p = (
                (pat_p[0],)
                + host_p[:2]
                + (host_r[2] - delta,)
                + (pat_p[3],)
                + host_p[4:]
            )
        else:
            new_p = host_p[:2] + (host_r[2] - delta,) + (pat_p[3],) + host_p[4:]
        # if pat_r[0] == '*':
        #     host_r[0] = '*'
        # if pat_p[0] == '*':
        #     host_p[0] = '*'
        host_n[key] = (new_r, new_p)

        if "h_pairs" in pat_n:
            host_n["h_pairs"] = pat_n["h_pairs"]

    @staticmethod
    def _get_explicit_map(
        host: nx.Graph,
        mapping: MappingDict,
        pattern_explicit: nx.Graph | None = None,
        strategy: Strategy = Strategy.ALL,
    ):
        expand_nodes = [v for _, v in mapping.items()]
        host_explicit = h_to_explicit(host, expand_nodes)
        mappings = SubgraphSearchEngine.find_subgraph_mappings(
            host=host_explicit,
            pattern=pattern_explicit or nx.Graph(),
            node_attrs=["element", "charge"],
            edge_attrs=["order"],
            strategy=strategy,
        )
        return mappings, host_explicit

    @staticmethod
    def _glue_graph(
        host: nx.Graph,
        rc: nx.Graph,
        mapping: MappingDict,
        pattern_has_explicit_H: bool = False,
        pattern_explicit: nx.Graph | None = None,
        strategy: Strategy = Strategy.ALL,
    ) -> List[nx.Graph]:
        list_its: List[nx.Graph] = []
        host_g = deepcopy(host)

        def _default_tg(a: Dict[str, Any]) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
            tpl = (
                a.get("element", "*"),
                a.get("aromatic", False),
                a.get("hcount", 0),
                a.get("charge", 0),
                a.get("neighbors", []),
            )
            return tpl, tpl

        for _, data in host_g.nodes(data=True):
            data.setdefault("typesGH", _default_tg(data))

        if pattern_has_explicit_H:
            mappings, host_g = SynReactor._get_explicit_map(
                host_g, mapping, pattern_explicit, strategy
            )
        else:
            mappings = [mapping]

        # Iterate over remappings --------------------------------------
        for m in mappings:

            its = deepcopy(host_g)
            # This should only work for implict cases
            if len(m.keys()) < rc.number_of_nodes():
                its, m = add_wildcard_subgraph_for_unmapped(its, rc, m)

            for _, _, data in its.edges(data=True):
                o = data.get("order", 1.0)
                data["order"] = (o, o)
                data.setdefault("standard_order", 0.0)

            for _, data in rc.nodes(data=True):
                data.setdefault("typesGH", _default_tg(data))

            # merge nodes -------------------------------------------
            for rc_n, host_n in m.items():
                if its.has_node(host_n):
                    SynReactor._node_glue(its.nodes[host_n], rc.nodes[rc_n])

            # merge edges (additive order) ---------------------------
            for u, v, rc_attr in rc.edges(data=True):
                hu, hv = m.get(u), m.get(v)
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
                        host_attr["standard_order"] += rc_attr.get(
                            "standard_order", 0.0
                        )
                    else:
                        host_attr.update(rc_attr)
            list_its.append(its)
        return list_its

    # --------------------- explicit‑H handling -------------------------
    @staticmethod
    def _explicit_h(rc: nx.Graph) -> nx.Graph:
        next_id = max((n for n in rc.nodes if isinstance(n, int)), default=-1) + 1
        orig_delta: Dict[int, int] = {}
        pair_to_nodes: Dict[int, List[int]] = defaultdict(list)

        for n, d in rc.nodes(data=True):
            h_pairs = d.get("h_pairs", [])
            hl, hr = d["typesGH"][0][2], d["typesGH"][1][2]
            orig_delta[n] = hl - hr
            for pid in h_pairs:
                if n not in pair_to_nodes[pid]:
                    pair_to_nodes[pid].append(n)

        conn = nx.Graph()
        for nodes in pair_to_nodes.values():
            conn.add_nodes_from(nodes)
            # fmt: off
            conn.add_edges_from(
                (u, v) for i, u in enumerate(nodes) for v in nodes[i + 1:]
            )
            # fmt: on

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

        affected = [n for nodes in pair_to_nodes.values() for n in nodes]
        for n in affected:
            t0, t1 = rc.nodes[n]["typesGH"]
            delta_h = t0[2] - t1[2]
            if delta_h >= 0:
                t0_h, t1_h = t0[2] - 1, t1[2]
            else:
                t0_h, t1_h = t0[2], t1[2] - 1
            rc.nodes[n]["typesGH"] = (
                t0[:2] + (t0_h,) + t0[3:],
                t1[:2] + (t1_h,) + t1[3:],
            )
        return rc

    # --------------------- SMARTS serialisation -----------------------
    @staticmethod
    def _to_smarts(its: nx.Graph) -> str:
        left, right = its_decompose(its)
        left = remove_wildcard_nodes(left)
        right = remove_wildcard_nodes(right)
        r_smi = graph_to_smi(left)
        p_smi = graph_to_smi(right)
        if r_smi is None or p_smi is None:
            return None
        return f"{r_smi}>>{p_smi}"
