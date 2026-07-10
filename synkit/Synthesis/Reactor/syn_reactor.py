from __future__ import annotations

from copy import deepcopy
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import networkx as nx
from rdkit import Chem
from networkx.algorithms.isomorphism import (
    GraphMatcher,
    categorical_edge_match,
    categorical_node_match,
)


from synkit.IO.chem_converter import (
    ITSFormat,
    _get_preserved_hydrogen_maps,
    detect_its_format,
    smiles_to_graph,
    rsmi_to_its,
    graph_to_smi,
)
from synkit.IO import setup_logging
from synkit.Chem.utils import reverse_reaction

from synkit.Rule import SynRule
from synkit.Graph.syn_graph import SynGraph
from synkit.Graph.canon_graph import GraphCanonicaliser
from synkit.Graph.ITS.its_decompose import its_decompose
from synkit.Graph.ITS.its_reverter import ITSReverter
from synkit.Graph.ITS.its_construction import ITSConstruction
from synkit.Graph.Matcher.automorphism import (
    Automorphism,
)
from synkit.Graph.Matcher.dedup_matches import deduplicate_matches_with_anchor
from synkit.Graph.Matcher.auto_est import AutoEst
from synkit.Graph.Matcher.graph_cluster import GraphCluster
from synkit.Graph.Matcher.partial_matcher import PartialMatcher
from synkit.Graph.Matcher.subgraph_matcher import SubgraphSearchEngine
from synkit.Graph.Matcher.subgraph_matcher import resolve_template_match_attrs
from synkit.Graph.Feature.wl_hash import WLHash
from synkit.Graph.Mech.electron_accounting import (
    graph_to_sanitized_kekule_mol,
    refresh_electron_fields,
)
from synkit.IO.graph_to_mol import GraphToMol
from synkit.IO.mol_to_graph import MolToGraph
from synkit.Graph.Hyrogen._misc import (
    h_to_implicit,
    h_to_explicit,
    has_XH,
    implicit_hydrogen,
)
from synkit.Graph import (
    remove_wildcard_nodes,
    add_wildcard_subgraph_for_unmapped,
    has_wildcard_node,
)
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

ITS_STRUCTURAL_NODE_ATTRS = [
    "element",
    "aromatic",
    "hcount",
    "charge",
    "radical",
    "lone_pairs",
    "valence_electrons",
    "present",
]
ITS_STRUCTURAL_EDGE_ATTRS = ["order", "kekule_order", "sigma_order", "pi_order"]


# ──────────────────────────────────────────────────────────────────────────────
# SynReactor core
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class SynReactor:
    """A hardened and typed re-write of the original SynReactor, preserving API
    compatibility while offering safer, faster, and cleaner behavior.

    :param substrate: The input reaction substrate, as a SMILES string,
        a raw NetworkX graph, or a SynGraph.
    :type substrate: Union[str, nx.Graph, SynGraph]
    :param template: Reaction template, provided as SMILES/SMARTS, a raw
        NetworkX graph, or a SynRule.
    :type template: Union[str, nx.Graph, SynRule]
    :param invert: Whether to invert the reaction (predict precursors).
        Defaults to False.
    :type invert: bool
    :param canonicaliser: Optional canonicaliser for intermediate
        graphs. If None, a default GraphCanonicaliser is used.
    :type canonicaliser: Optional[GraphCanonicaliser]
    :param explicit_h: If True, render all hydrogens explicitly in the
        reaction-center SMARTS. Defaults to True.
    :type explicit_h: bool
    :param implicit_temp: If True, treat the input template as
        implicit-H (forces explicit_h=False). Defaults to False.
    :type implicit_temp: bool
    :param strategy: Matching strategy, one of Strategy.ALL, 'comp', or
        'bt'. Defaults to Strategy.ALL.
    :type strategy: Strategy or str
    :param partial: If True, use a partial matching fallback. Defaults
        to False.
    :type partial: bool
    :param template_format: ITS representation used when ``template`` is a
        reaction string. Defaults to ``"typesGH"`` for compatibility.
    :type template_format: ITSFormat
    :param electron_diagnostics: If True, expose per-result electron-accounting
        diagnostics without changing generated products.
    :type electron_diagnostics: bool
    :ivar _graph: Cached SynGraph for the substrate.
    :vartype _graph: Optional[SynGraph]
    :ivar _rule: Cached SynRule for the template.
    :vartype _rule: Optional[SynRule]
    :ivar _mappings: Cached list of subgraph-mapping dicts.
    :vartype _mappings: Optional[List[MappingDict]]
    :ivar _its: Cached list of ITS graphs.
    :vartype _its: Optional[List[nx.Graph]]
    :ivar _smarts: Cached list of SMARTS strings.
    :vartype _smarts: Optional[List[str]]
    :ivar _flag_pattern_has_explicit_H: Internal flag indicating
        explicit-H constraints.
    :vartype _flag_pattern_has_explicit_H: bool
    """

    substrate: Union[str, nx.Graph, SynGraph]
    template: Union[str, nx.Graph, SynRule]
    invert: bool = False
    canonicaliser: GraphCanonicaliser | None = None
    explicit_h: bool = True
    implicit_temp: bool = False
    strategy: Strategy | str = Strategy.ALL
    partial: bool = False
    template_format: ITSFormat = "typesGH"
    electron_diagnostics: bool = False
    embed_threshold: Optional[int] = None
    embed_pre_filter: bool = False
    automorphism: bool = True

    # Private caches – populated on demand -------------------------------
    _graph: SynGraph | None = field(init=False, default=None, repr=False)
    _rule: SynRule | None = field(init=False, default=None, repr=False)
    _mappings: List[MappingDict] | None = field(init=False, default=None, repr=False)
    _its: List[nx.Graph] | None = field(init=False, default=None, repr=False)
    _smarts: List[str] | None = field(init=False, default=None, repr=False)
    _host_for_matching: nx.Graph | None = field(init=False, default=None, repr=False)
    _flag_pattern_has_explicit_H: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        """Validate and enforce consistency of `explicit_h` and
        `implicit_temp`.

        :raises ValueError: If `explicit_h` is True while `implicit_temp` is False.
        """
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
        automorphism: bool = False,
        strategy: Strategy | str = Strategy.ALL,
        template_format: ITSFormat = "typesGH",
        electron_diagnostics: bool = False,
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
        :param template_format: ITS representation used when ``template`` is a
            reaction string. Defaults to ``"typesGH"``.
        :type template_format: ITSFormat
        :param electron_diagnostics: If True, expose per-result electron
            diagnostics without changing products.
        :type electron_diagnostics: bool
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
            automorphism=automorphism,
            template_format=template_format,
            electron_diagnostics=electron_diagnostics,
        )

    # ------------------------------------------------------------------
    # Public read‑only properties (lazily computed) ----------------------
    # ------------------------------------------------------------------
    @property
    def graph(self) -> SynGraph:  # noqa: D401 – read‑only property
        """Lazily wrap the substrate into a SynGraph.

        :returns: The reaction substrate as a `SynGraph`.
        :rtype: SynGraph
        """
        if self._graph is None:
            self._graph = self._wrap_input(self.substrate)
        return self._graph

    @property
    def rule(self) -> SynRule:  # noqa: D401
        """Lazily wrap the template into a SynRule.

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
        """Return unique sub‑graph mappings, optionally pruned via automorphisms."""
        if self._mappings is None:
            log.debug("Finding sub‑graph mappings (strategy=%s)", self.strategy)
            pattern_graph = self.rule.left.raw
            # Handle explicit‑H constraints
            if has_XH(pattern_graph):
                self._flag_pattern_has_explicit_H = True
                pattern_graph = h_to_implicit(pattern_graph)
            # Handle wildcard‑node patterns
            if has_wildcard_node(pattern_graph):
                pattern_graph = remove_wildcard_nodes(pattern_graph)

            pattern_graph = self._with_aromatic_n_pi_roles(pattern_graph)
            matching_host = self._with_aromatic_n_pi_roles(self._matching_host_graph())
            node_attrs, edge_attrs = resolve_template_match_attrs(pattern_graph)

            # --- Choose matcher ------------------------------------------------
            if self.partial:
                max_results = (
                    self.embed_threshold / 100 if self.embed_threshold else None
                )
                matcher = PartialMatcher(
                    host=matching_host,
                    pattern=pattern_graph,
                    node_attrs=node_attrs,
                    edge_attrs=edge_attrs,
                    strategy=Strategy.from_string(self.strategy),
                    threshold=self.embed_threshold,
                    pre_filter=self.embed_pre_filter,
                    max_results=max_results,
                    prune_auto=True,
                )
                raw_maps = matcher.get_mappings()
            else:
                raw_maps = SubgraphSearchEngine.find_subgraph_mappings(
                    host=matching_host,
                    pattern=pattern_graph,
                    node_attrs=node_attrs,
                    edge_attrs=edge_attrs,
                    strategy=Strategy.from_string(self.strategy),
                    threshold=self.embed_threshold,
                    pre_filter=self.embed_pre_filter,
                )

            # --- Automorphism pruning ----------------------------------------
            if self.automorphism and raw_maps:
                automorphism_pattern = self._automorphism_pattern_graph(pattern_graph)
                auto = Automorphism(
                    automorphism_pattern,
                    node_attr_keys=self._automorphism_node_attrs(
                        automorphism_pattern,
                        node_attrs,
                    ),
                    edge_attr_keys=edge_attrs,
                )
                host_auto = Automorphism(
                    self._matching_host_graph(),
                    node_attr_keys=node_attrs,
                    edge_attr_keys=edge_attrs,
                )
                self._mappings = deduplicate_matches_with_anchor(
                    raw_maps,
                    pattern_orbits=auto.orbits,
                    pattern_anchor=auto.anchor_component,
                    host_orbits=host_auto.orbits,
                    host_anchor=host_auto.anchor_component,
                )
                self._mappings = self._deduplicate_equivalent_free_components(
                    self._mappings,
                    automorphism_pattern,
                    auto.anchor_component,
                    self._automorphism_node_attrs(
                        automorphism_pattern,
                        node_attrs,
                    ),
                    edge_attrs,
                )
                log.debug(
                    "Automorphism pruning: %d → %d unique mapping(s)",
                    len(raw_maps),
                    len(self._mappings),
                )
            else:
                auto = AutoEst(
                    pattern_graph,
                    node_attrs=["element", "charge", "aromatic", "hcount"],
                    edge_attrs=["order"],
                )
                auto.fit()
                self._mappings = deduplicate_matches_with_anchor(
                    raw_maps,
                    pattern_orbits=auto.orbits,
                    pattern_anchor=auto.anchor_component,
                )
                # self._mappings = raw_maps

            log.info("%d mapping(s) discovered", len(self._mappings))
        return self._mappings

    @staticmethod
    def _with_aromatic_n_pi_roles(graph: nx.Graph) -> nx.Graph:
        """Label aromatic nitrogens by incident aromatic pi-bond count."""
        decorated = graph.copy()
        for node, attrs in decorated.nodes(data=True):
            if attrs.get("element") != "N" or not attrs.get("aromatic", False):
                continue
            attrs["aromatic_n_pi_count"] = sum(
                1
                for _, _, edge in decorated.edges(node, data=True)
                if edge.get("order") == 1.5 and edge.get("pi_order") == 1.0
            )
        return decorated

    @property
    def its_list(self) -> List[nx.Graph]:
        """Build ITS graphs for each subgraph mapping.

        :returns: A list of ITS (Internal Transition State) graphs.
        :rtype: list of networkx.Graph
        """
        if self._its is None:
            # Build ITS for each mapping -------------------------------
            host_raw = self._matching_host_graph()
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
                    embed_threshold=self.embed_threshold,
                    embed_pre_filter=False,
                )
                self._its.extend(its_batch)

            if self.explicit_h:
                self._its = [self._explicit_h(g) for g in self._its]
            self._its = self._deduplicate_structural_its(self._its)
            log.debug("Built %d ITS graph(s)", len(self._its))
        return self._its

    @property
    def smarts_list(self) -> List[str]:
        """Serialise each ITS graph to a reaction-SMARTS string.

        :returns: A list of SMARTS strings (inverted if `invert=True`).
        :rtype: list of str
        """
        if self._smarts is None:
            self._smarts = [self._to_smarts(g) for g in self.its_list]
            self._smarts = [value for value in self._smarts if value]
            if self.invert:
                self._smarts = [reverse_reaction(rsmi) for rsmi in self._smarts]
            self._smarts = list(dict.fromkeys(self._smarts))
        return self._smarts

    @property
    def diagnostics(self) -> List[Dict[str, Any]]:
        """Return optional electron-accounting diagnostics for built ITS graphs."""
        if not self.electron_diagnostics:
            return []

        reports: List[Dict[str, Any]] = []
        for index, its in enumerate(self.its_list):
            if its.graph.get("electron_aware_rewrite", False):
                mismatches = {}
                for node, attrs in its.nodes(data=True):
                    charge_mismatch = attrs.get("charge_mismatch")
                    mismatch = (
                        charge_mismatch[1]
                        if isinstance(charge_mismatch, tuple)
                        and len(charge_mismatch) == 2
                        else charge_mismatch
                    )
                    if mismatch:
                        template_charge = attrs.get("template_charge")
                        recomputed_charge = attrs.get("recomputed_charge")
                        mismatches[node] = {
                            "charge": (
                                template_charge[1]
                                if isinstance(template_charge, tuple)
                                and len(template_charge) == 2
                                else template_charge
                            ),
                            "recomputed_charge": (
                                recomputed_charge[1]
                                if isinstance(recomputed_charge, tuple)
                                and len(recomputed_charge) == 2
                                else recomputed_charge
                            ),
                        }
            else:
                product = self._product_graph_for_diagnostics(its)
                refreshed = refresh_electron_fields(product)
                mismatches = {
                    node: {
                        "charge": attrs.get("charge"),
                        "recomputed_charge": attrs.get("recomputed_charge"),
                    }
                    for node, attrs in refreshed.nodes(data=True)
                    if attrs.get("charge_mismatch")
                }
            reports.append(
                {
                    "index": index,
                    "electron_aware_rewrite": bool(
                        its.graph.get("electron_aware_rewrite", False)
                    ),
                    "mismatch_count": len(mismatches),
                    "mismatches": mismatches,
                }
            )
        return reports

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
            graph = rsmi_to_its(tpl, format=self.template_format)
        else:  # pragma: no cover
            raise TypeError(f"Unsupported template type: {type(tpl)}")
        # graph = normalize_h_pair_graph(graph)

        format = detect_its_format(graph)

        # Invert if asked -----------------------------------------------------
        if self.invert:
            if self.implicit_temp:
                graph = self._invert_template(
                    graph,
                    balance_its=True,
                    format=format,
                )
                return SynRule(
                    graph,
                    canonicaliser=self.canonicaliser or GraphCanonicaliser(),
                    format=format,
                )
            else:
                graph = self._invert_template(
                    graph,
                    balance_its=False,
                    format=format,
                )
                return SynRule(
                    graph,
                    canonicaliser=self.canonicaliser or GraphCanonicaliser(),
                    format=format,
                )
        else:
            if self.implicit_temp:
                return SynRule(
                    graph,
                    canonicaliser=self.canonicaliser or GraphCanonicaliser(),
                    format=format,
                )
            return SynRule(
                graph,
                canonicaliser=self.canonicaliser or GraphCanonicaliser(),
                format=format,
            )

    def _matching_host_graph(self) -> nx.Graph:
        """Return the host graph normalized to the active rule representation."""
        if self._host_for_matching is None:
            host = self.graph.raw
            if getattr(self.rule, "_format", None) == "tuple":
                host = self._implicit_heavy_hydrogens(host)
            self._host_for_matching = host
        return self._host_for_matching

    @staticmethod
    def _implicit_heavy_hydrogens(graph: nx.Graph) -> nx.Graph:
        """Convert ordinary heavy-atom-bound explicit H nodes into hcount."""
        normalized = graph.copy()
        removable = []
        for node, attrs in normalized.nodes(data=True):
            if attrs.get("element") != "H":
                continue
            neighbors = list(normalized.neighbors(node))
            heavy_neighbors = [
                nbr for nbr in neighbors if normalized.nodes[nbr].get("element") != "H"
            ]
            if heavy_neighbors and len(heavy_neighbors) == len(neighbors):
                removable.append((node, heavy_neighbors))

        for h, heavy_neighbors in removable:
            if not normalized.has_node(h):
                continue
            for heavy in heavy_neighbors:
                normalized.nodes[heavy]["hcount"] = (
                    normalized.nodes[heavy].get("hcount", 0) + 1
                )
            normalized.remove_node(h)
        return normalized

    @staticmethod
    def _invert_template(
        tpl: nx.Graph,
        balance_its: bool = True,
        format: ITSFormat | None = None,
    ) -> nx.Graph:
        resolved_format = format or detect_its_format(tpl)
        if resolved_format == "tuple":
            reverter = ITSReverter(tpl)
            l, r = reverter.to_reactant_graph(), reverter.to_product_graph()
            return ITSConstruction().construct(
                r,
                l,
                balance_its=balance_its,
            )
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

        for key in ("h_pairs", "h_pairs_left", "h_pairs_right", "h_pair_atom_maps"):
            if key in pat_n:
                host_n[key] = pat_n[key]

    @staticmethod
    def _get_explicit_map(
        host: nx.Graph,
        mapping: MappingDict,
        pattern_explicit: nx.Graph | None = None,
        strategy: Strategy = Strategy.ALL,
        embed_threshold: float = None,
        embed_pre_filter: bool = False,
    ):
        expand_nodes = [v for _, v in mapping.items()]
        host_explicit = h_to_explicit(host, expand_nodes)
        mappings = SubgraphSearchEngine.find_subgraph_mappings(
            host=host_explicit,
            pattern=pattern_explicit or nx.Graph(),
            node_attrs=["element", "charge"],
            edge_attrs=["order"],
            strategy=strategy,
            threshold=embed_threshold,
            pre_filter=embed_pre_filter,
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
        embed_threshold: float = None,
        embed_pre_filter: bool = False,
    ) -> List[nx.Graph]:
        list_its: List[nx.Graph] = []
        host_g = deepcopy(host)
        electron_aware = SynReactor._is_electron_aware_template(rc)

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
        if electron_aware:
            SynReactor._ensure_host_atom_maps(host_g)

        if pattern_has_explicit_H:
            mappings, host_g = SynReactor._get_explicit_map(
                host_g,
                mapping,
                pattern_explicit,
                strategy,
                embed_threshold,
                embed_pre_filter,
            )
            if electron_aware:
                SynReactor._ensure_host_atom_maps(host_g)
        else:
            mappings = [mapping]

        # Iterate over remappings --------------------------------------
        for m in mappings:

            its = deepcopy(host_g)
            # This should only work for implict cases
            if len(m.keys()) < rc.number_of_nodes():
                its, m = add_wildcard_subgraph_for_unmapped(
                    its,
                    rc,
                    m,
                    tuple_mode=electron_aware,
                )

            for _, _, data in its.edges(data=True):
                o = data.get("order", 1.0)
                data["order"] = (o, o)
                if electron_aware:
                    sigma = data.get("sigma_order", 1.0 if o else 0.0)
                    pi = data.get("pi_order", max(0.0, float(o) - 1.0))
                    data["sigma_order"] = (sigma, sigma)
                    data["pi_order"] = (pi, pi)
                data.setdefault("standard_order", 0.0)

            for _, data in rc.nodes(data=True):
                data.setdefault("typesGH", _default_tg(data))

            # merge nodes -------------------------------------------
            for rc_n, host_n in m.items():
                if its.has_node(host_n):
                    SynReactor._node_glue(its.nodes[host_n], rc.nodes[rc_n])
                    if electron_aware:
                        SynReactor._pair_electron_aware_node_attrs(
                            its.nodes[host_n],
                            rc.nodes[rc_n],
                        )

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
                        if electron_aware:
                            host_sigma = host_attr.get("sigma_order", (0.0, 0.0))
                            host_pi = host_attr.get("pi_order", (0.0, 0.0))
                            rc_sigma = rc_attr.get("sigma_order", (0.0, 0.0))
                            rc_pi = rc_attr.get("pi_order", (0.0, 0.0))
                            host_attr["sigma_order"] = (
                                host_sigma[0],
                                host_sigma[1] + rc_sigma[1],
                            )
                            host_attr["pi_order"] = (
                                host_pi[0],
                                host_pi[1] + rc_pi[1],
                            )
                        host_attr["standard_order"] += rc_attr.get(
                            "standard_order", 0.0
                        )
                    else:
                        host_attr.update(rc_attr)
            if electron_aware:
                SynReactor._refresh_product_electron_fields(its)
            its.graph["electron_aware_rewrite"] = electron_aware
            list_its.append(its)
        return list_its

    @staticmethod
    def _is_electron_aware_template(rc: nx.Graph) -> bool:
        """Return whether an RC carries sigma/pi rewrite state."""
        return any(
            "sigma_order" in data and "pi_order" in data
            for _, _, data in rc.edges(data=True)
        )

    @staticmethod
    def _automorphism_node_attrs(
        pattern: nx.Graph,
        node_attrs: List[str],
    ) -> List[str]:
        """Keep pruning at least as role-aware as the stored template data."""
        attrs = list(node_attrs)
        for attr in ("aromatic", "neighbors", "_rewrite_role"):
            if attr not in attrs and any(
                attr in data for _, data in pattern.nodes(data=True)
            ):
                attrs.append(attr)
        return attrs

    def _automorphism_pattern_graph(self, pattern: nx.Graph) -> nx.Graph:
        """Decorate tuple patterns with product-side rewrite roles for pruning."""
        if getattr(self.rule, "_format", None) != "tuple":
            return pattern

        decorated = pattern.copy()
        rc = self.rule.rc.raw
        for node, attrs in decorated.nodes(data=True):
            rc_attrs = rc.nodes.get(node)
            if not rc_attrs:
                continue
            types = rc_attrs.get("typesGH")
            if isinstance(types, tuple) and len(types) == 2:
                attrs["_rewrite_role"] = self._chemical_rewrite_role(types[1])
        return decorated

    @staticmethod
    def _chemical_rewrite_role(role: Any) -> Any:
        """Drop provenance-only atom-map identity from tuple rewrite roles."""
        if isinstance(role, tuple) and len(role) >= 9:
            chemical_role = role[:-1]
            if chemical_role[0] == "H":
                return chemical_role[:-1] + ((),)
            return chemical_role
        return role

    @staticmethod
    def _prepare_its_for_structural_cluster(its: nx.Graph) -> nx.Graph:
        """Attach one combined edge signature for exact ITS clustering."""
        prepared = deepcopy(its)
        if prepared.graph.get("electron_aware_rewrite", False):
            SynReactor._refresh_product_electron_fields(prepared)
        aromatic_nodes = {
            node
            for u, v, attrs in prepared.edges(data=True)
            if attrs.get("order") == (1.5, 1.5)
            for node in (u, v)
        }
        for node in aromatic_nodes:
            template_charge = prepared.nodes[node].get("template_charge")
            if isinstance(template_charge, tuple) and len(template_charge) == 2:
                prepared.nodes[node]["charge"] = template_charge
        for _, _, attrs in prepared.edges(data=True):
            edge_values = []
            aromatic_unchanged = attrs.get("order") == (1.5, 1.5)
            for name in ITS_STRUCTURAL_EDGE_ATTRS:
                value = attrs.get(name)
                if aromatic_unchanged and name in {
                    "kekule_order",
                    "sigma_order",
                    "pi_order",
                }:
                    value = "aromatic_phase"
                edge_values.append(value)
            attrs["_its_edge_sig"] = tuple(edge_values)
        return prepared

    @staticmethod
    def _deduplicate_structural_its(its_graphs: List[nx.Graph]) -> List[nx.Graph]:
        """Keep one representative for each structurally unique ITS graph."""
        if len(its_graphs) < 2:
            return its_graphs

        hasher = WLHash(
            node=ITS_STRUCTURAL_NODE_ATTRS,
            edge="_its_edge_sig",
        )
        buckets: Dict[str, List[Tuple[int, nx.Graph]]] = defaultdict(list)
        for index, its in enumerate(its_graphs):
            prepared = SynReactor._prepare_its_for_structural_cluster(its)
            signature = hasher.weisfeiler_lehman_graph_hash(prepared)
            buckets[signature].append((index, prepared))

        cluster = GraphCluster(
            node_label_names=ITS_STRUCTURAL_NODE_ATTRS,
            node_label_default=["*", False, 0, 0, 0, 0, 0, ()],
            edge_attribute="_its_edge_sig",
        )
        representative_indices: List[int] = []
        for bucket in buckets.values():
            if len(bucket) == 1:
                representative_indices.append(bucket[0][0])
                continue
            prepared = [prepared for _, prepared in bucket]
            classes, _ = cluster.iterative_cluster(prepared)
            for cls in classes:
                representative_indices.append(bucket[min(cls)][0])

        representative_indices.sort()
        return [its_graphs[index] for index in representative_indices]

    def _deduplicate_equivalent_free_components(
        self,
        mappings: List[MappingDict],
        pattern: nx.Graph,
        anchor: Optional[frozenset[NodeId]],
        node_attrs: List[str],
        edge_attrs: List[str],
    ) -> List[MappingDict]:
        """Collapse swaps of equivalent disconnected non-anchor tuple components."""
        if getattr(self.rule, "_format", None) != "tuple" or len(mappings) < 2:
            return mappings

        anchor = anchor or frozenset()
        free_components = [
            frozenset(component)
            for component in nx.connected_components(pattern)
            if not set(component) & set(anchor)
        ]
        if len(free_components) < 2:
            return mappings

        component_groups: List[List[frozenset[NodeId]]] = []
        for component in free_components:
            for group in component_groups:
                if self._components_are_equivalent(
                    pattern,
                    component,
                    group[0],
                    node_attrs,
                    edge_attrs,
                ):
                    group.append(component)
                    break
            else:
                component_groups.append([component])

        swappable_groups = [group for group in component_groups if len(group) > 1]
        if not swappable_groups:
            return mappings

        swappable_nodes = set().union(
            *[set().union(*group) for group in swappable_groups]
        )
        seen = set()
        unique: List[MappingDict] = []
        for mapping in mappings:
            fixed = tuple(
                sorted(
                    (node, host)
                    for node, host in mapping.items()
                    if node not in swappable_nodes
                )
            )
            component_bags = tuple(
                tuple(
                    sorted(
                        tuple(sorted(mapping[node] for node in component))
                        for component in group
                    )
                )
                for group in swappable_groups
            )
            signature = (fixed, component_bags)
            if signature in seen:
                continue
            seen.add(signature)
            unique.append(mapping)
        return unique

    @staticmethod
    def _components_are_equivalent(
        pattern: nx.Graph,
        left: frozenset[NodeId],
        right: frozenset[NodeId],
        node_attrs: List[str],
        edge_attrs: List[str],
    ) -> bool:
        """Return whether two disconnected pattern components have one role shape."""
        left_graph = pattern.subgraph(left)
        right_graph = pattern.subgraph(right)
        node_defaults = [0 if attr == "charge" else "*" for attr in node_attrs]
        edge_defaults = [1.0 for _ in edge_attrs]
        matcher = GraphMatcher(
            left_graph,
            right_graph,
            node_match=categorical_node_match(node_attrs, node_defaults),
            edge_match=categorical_edge_match(edge_attrs, edge_defaults),
        )
        return matcher.is_isomorphic()

    @staticmethod
    def _pair_electron_aware_node_attrs(
        host_n: Dict[str, Any],
        rc_n: Dict[str, Any],
    ) -> None:
        """Store direct paired node attrs after legacy-compatible node glue."""
        _, product_types = host_n["typesGH"]
        rc_present = rc_n.get("present")
        reactant_is_absent = (
            isinstance(rc_present, tuple) and len(rc_present) == 2 and not rc_present[0]
        )
        legacy_product_values = {
            "element": product_types[0],
            "aromatic": product_types[1],
            "hcount": product_types[2],
            "neighbors": product_types[4],
        }

        for key, product_value in legacy_product_values.items():
            left_value = host_n.get(key)
            product_is_absent = (
                isinstance(rc_present, tuple)
                and len(rc_present) == 2
                and not rc_present[1]
            )
            rc_value = rc_n.get(key)
            if (
                reactant_is_absent
                and isinstance(rc_value, tuple)
                and len(rc_value) == 2
            ):
                product_value = rc_value[1]
            if key == "element" and product_value == "*" and not product_is_absent:
                product_value = left_value
            host_n[key] = (left_value, product_value)

        for key in ("radical", "lone_pairs", "valence_electrons"):
            rc_value = rc_n.get(key)
            if isinstance(rc_value, tuple) and len(rc_value) == 2:
                left_value = host_n.get(key)
                if left_value is None:
                    left_value = rc_value[0]
                host_n[key] = (left_value, rc_value[1])

        host_n["template_charge"] = (host_n.get("charge"), product_types[3])

        # Electron-authoritative RCs derive charge at the product boundary.
        # Keep the reactant-side value temporarily so mutation does not copy
        # the RC's product charge label.
        host_n["charge"] = (host_n.get("charge"), host_n.get("charge"))

        if "atom_map" in host_n:
            host_n["atom_map"] = (host_n["atom_map"], host_n["atom_map"])
        if isinstance(rc_present, tuple) and len(rc_present) == 2:
            host_n["present"] = (bool(host_n.get("present", True)), rc_present[1])
        else:
            host_n["present"] = (True, True)

    @staticmethod
    def _ensure_host_atom_maps(host: nx.Graph) -> None:
        """Assign stable fresh atom maps to unmapped host atoms."""
        for node, attrs in host.nodes(data=True):
            if attrs.get("atom_map") in (None, 0):
                attrs["atom_map"] = node

    @staticmethod
    def _refresh_product_electron_fields(
        its: nx.Graph,
    ) -> None:
        """Refresh product-side electron fields from the scalar product graph."""
        product = SynReactor._prepared_electron_product_graph(its)
        refreshed = refresh_electron_fields(product)
        for node, attrs in refreshed.nodes(data=True):
            current_charge = its.nodes[node].get("charge")
            left_charge = (
                current_charge[0]
                if isinstance(current_charge, tuple) and len(current_charge) == 2
                else current_charge
            )
            product_charge = SynReactor._electron_product_charge(its, node, attrs)
            if product_charge is not None:
                its.nodes[node]["charge"] = (left_charge, product_charge)

            template_charge = its.nodes[node].get("template_charge")
            if isinstance(template_charge, tuple) and len(template_charge) == 2:
                attrs["charge_mismatch"] = template_charge[1] != attrs.get(
                    "recomputed_charge"
                )

            for key in ("bond_order_sum", "recomputed_charge", "charge_mismatch"):
                if key in attrs:
                    current = its.nodes[node].get(key)
                    left_value = (
                        current[0]
                        if isinstance(current, tuple) and len(current) == 2
                        else current
                    )
                    its.nodes[node][key] = (left_value, attrs[key])
        for u, v, attrs in refreshed.edges(data=True):
            for key in ("kekule_order", "sigma_order", "pi_order"):
                if key not in attrs:
                    continue
                current = its.edges[u, v].get(key)
                left_value = (
                    current[0]
                    if isinstance(current, tuple) and len(current) == 2
                    else current
                )
                its.edges[u, v][key] = (left_value, attrs[key])

    @staticmethod
    def _prepared_electron_product_graph(its: nx.Graph) -> nx.Graph:
        """Build the scalar product graph used for electron recomputation."""
        product = ITSReverter(its).to_product_graph()
        preserved_hydrogens = _get_preserved_hydrogen_maps(its, "tuple")
        product = implicit_hydrogen(product, set(preserved_hydrogens))
        return SynReactor._reperceive_product_kekule_phase(product, its)

    @staticmethod
    def _electron_product_charge(
        its: nx.Graph,
        node: Any,
        product_attrs: Mapping[str, Any],
    ) -> Any:
        """Choose the product charge used for electron-aware serialization.

        Non-aromatic tuple products are electron-authoritative and use the
        recomputed formal charge. Aromatic tuple products are still an open
        representation boundary: if the template explicitly carries a product
        charge, preserve it instead of inventing cationic aromatic carbons from
        an incomplete Kekule phase.
        """
        if node in its:
            template_charge = its.nodes[node].get("template_charge")
            aromatic = product_attrs.get("aromatic", its.nodes[node].get("aromatic"))
            if (
                aromatic is True
                and isinstance(template_charge, tuple)
                and len(template_charge) == 2
            ):
                return template_charge[1]
        return product_attrs.get("recomputed_charge")

    @staticmethod
    def _reperceive_product_kekule_phase(product: nx.Graph, its: nx.Graph) -> nx.Graph:
        """Refresh aromatic sigma/pi phase from full product presentation bonds."""
        if not any(data.get("order") == 1.5 for _, _, data in product.edges(data=True)):
            return product

        probe = product.copy()
        for node, attrs in probe.nodes(data=True):
            template_charge = its.nodes[node].get("template_charge")
            if isinstance(template_charge, tuple) and len(template_charge) == 2:
                attrs["charge"] = template_charge[1]

        try:
            mol = GraphToMol(edge_attributes={"order": "order"}).graph_to_mol(
                probe,
                sanitize=True,
                use_h_count=True,
            )
            reperceived = MolToGraph(attr_profile="minimal").transform(
                mol,
                use_index_as_atom_map=True,
            )
        except Exception:
            return product

        refreshed = product.copy()
        for u, v in refreshed.edges():
            if not reperceived.has_edge(u, v):
                continue
            for key in ("kekule_order", "sigma_order", "pi_order"):
                if key in reperceived[u][v]:
                    refreshed[u][v][key] = reperceived[u][v][key]
        return refreshed

    @staticmethod
    def _product_graph_for_diagnostics(its: nx.Graph) -> nx.Graph:
        """Return the product graph matching the rewrite representation."""
        if its.graph.get("electron_aware_rewrite", False):
            return ITSReverter(its).to_product_graph()
        return its_decompose(its)[1]

    # --------------------- explicit‑H handling -------------------------
    @staticmethod
    def _explicit_h(rc: nx.Graph) -> nx.Graph:
        if bool(rc.graph.get("electron_aware_rewrite", False)):
            return SynReactor._explicit_h_tuple(rc)

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

    @staticmethod
    def _explicit_h_tuple(rc: nx.Graph) -> nx.Graph:
        """Materialize only hydrogens that were explicit in the template."""
        next_id = max((n for n in rc.nodes if isinstance(n, int)), default=-1) + 1
        pair_left: Dict[int, int] = {}
        pair_right: Dict[int, int] = {}
        for n, data in rc.nodes(data=True):
            for pair_id in data.get("h_pairs_left", []):
                pair_left[pair_id] = n
            for pair_id in data.get("h_pairs_right", []):
                pair_right[pair_id] = n

        explicit_pairs = sorted(set(pair_left) & set(pair_right))
        used_maps = {
            value
            for _, data in rc.nodes(data=True)
            for atom_map in [data.get("atom_map")]
            for value in (
                atom_map
                if isinstance(atom_map, tuple)
                else (() if atom_map in (None, 0) else (atom_map,))
            )
        }
        for pair_id in explicit_pairs:
            src = pair_left[pair_id]
            dst = pair_right[pair_id]
            h = next_id
            next_id += 1
            preferred_map = rc.nodes[src].get("h_pair_atom_maps", {}).get(
                pair_id
            ) or rc.nodes[dst].get("h_pair_atom_maps", {}).get(pair_id)
            atom_map = preferred_map if preferred_map not in used_maps else h
            while atom_map in used_maps:
                atom_map += 1
            used_maps.add(atom_map)
            rc.add_node(
                h,
                element=("H", "H"),
                aromatic=(False, False),
                charge=(0, 0),
                atom_map=(atom_map, atom_map),
                hcount=(0, 0),
                radical=(0, 0),
                lone_pairs=(0, 0),
                valence_electrons=(1, 1),
                neighbors=([], []),
                present=(True, True),
                typesGH=(("H", False, 0, 0, []), ("H", False, 0, 0, [])),
            )
            if src == dst:
                rc.add_edge(
                    src,
                    h,
                    order=(1.0, 1.0),
                    kekule_order=(1.0, 1.0),
                    sigma_order=(1.0, 1.0),
                    pi_order=(0.0, 0.0),
                    standard_order=0.0,
                )
                continue
            rc.add_edge(
                src,
                h,
                order=(1.0, 0.0),
                kekule_order=(1.0, 0.0),
                sigma_order=(1.0, 0.0),
                pi_order=(0.0, 0.0),
                standard_order=1.0,
            )
            rc.add_edge(
                h,
                dst,
                order=(0.0, 1.0),
                kekule_order=(0.0, 1.0),
                sigma_order=(0.0, 1.0),
                pi_order=(0.0, 0.0),
                standard_order=-1.0,
            )

        for pair_id in explicit_pairs:
            src = pair_left[pair_id]
            dst = pair_right[pair_id]
            if src == dst:
                h0, h1 = rc.nodes[src]["hcount"]
                rc.nodes[src]["hcount"] = (h0 - 1, h1 - 1)
                continue
            src_h0, src_h1 = rc.nodes[src]["hcount"]
            dst_h0, dst_h1 = rc.nodes[dst]["hcount"]
            rc.nodes[src]["hcount"] = (src_h0 - 1, src_h1)
            rc.nodes[dst]["hcount"] = (dst_h0, dst_h1 - 1)

        for n in set(pair_left.values()) | set(pair_right.values()):
            if "typesGH" in rc.nodes[n]:
                t0, t1 = rc.nodes[n]["typesGH"]
                rc.nodes[n]["typesGH"] = (
                    t0[:2] + (rc.nodes[n]["hcount"][0],) + t0[3:],
                    t1[:2] + (rc.nodes[n]["hcount"][1],) + t1[3:],
                )

        SynReactor._ensure_tuple_atom_maps(rc)
        return rc

    @staticmethod
    def _ensure_tuple_atom_maps(graph: nx.Graph) -> None:
        """Assign stable paired atom maps to tuple nodes lacking visible maps."""
        for node, attrs in graph.nodes(data=True):
            atom_map = attrs.get("atom_map")
            if atom_map in (None, 0) or atom_map == (0, 0):
                attrs["atom_map"] = (node, node)

    # --------------------- SMARTS serialisation -----------------------
    @staticmethod
    def _to_smarts(its: nx.Graph) -> str:
        electron_aware = bool(its.graph.get("electron_aware_rewrite", False))
        if electron_aware:
            reverter = ITSReverter(its)
            left = reverter.to_reactant_graph()
            right = reverter.to_product_graph()
            preserved_hydrogens = _get_preserved_hydrogen_maps(its, "tuple")
        else:
            left, right = its_decompose(its)
            preserved_hydrogens = []
        left = remove_wildcard_nodes(left)
        right = remove_wildcard_nodes(right)
        r_smi = graph_to_smi(left, preserve_atom_maps=preserved_hydrogens)
        if electron_aware:
            product_candidates = [
                right,
                SynReactor._prepared_electron_product_graph(its),
            ]
            p_smi = None
            for product in product_candidates:
                product = refresh_electron_fields(product)
                for node, attrs in product.nodes(data=True):
                    product_charge = SynReactor._electron_product_charge(
                        its,
                        node,
                        attrs,
                    )
                    if product_charge is not None:
                        attrs["charge"] = product_charge
                if any(
                    attrs.get("order") == 1.5
                    for _, _, attrs in product.edges(data=True)
                ):
                    p_smi = graph_to_smi(product)
                    if p_smi is not None:
                        break
                try:
                    p_smi = Chem.MolToSmiles(graph_to_sanitized_kekule_mol(product))
                    break
                except Exception:
                    p_smi = None
        else:
            p_smi = graph_to_smi(right)
        if r_smi is None or p_smi is None:
            return None
        return f"{r_smi}>>{p_smi}"
