import networkx as nx
from copy import deepcopy
from collections import defaultdict
from networkx.algorithms.isomorphism import GraphMatcher
from typing import Union, List, Dict, Any, Tuple, Optional
from synkit.IO.chem_converter import (
    smiles_to_graph,
    rsmi_to_its,
    graph_to_smi,
)

from synkit.Graph.syn_rule import SynRule
from synkit.Graph.syn_graph import SynGraph
from synkit.Graph.canon_graph import GraphCanonicaliser
from synkit.Graph.ITS.its_decompose import its_decompose


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
    ) -> "SynReactor":
        """
        Alternate constructor to create a SynReactor from a SMILES string.

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
        return cls(smiles, template, invert=invert, canonicaliser=canonicaliser)

    def __init__(
        self,
        input: Union[str, nx.Graph, SynGraph],
        template: Union[str, nx.Graph, SynRule],
        *,
        invert: bool = False,
        canonicaliser: Optional[GraphCanonicaliser] = None,
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

        # Wrap substrate
        self.graph: SynGraph = self._wrap_input(input)

        # Wrap template
        if isinstance(template, SynRule):
            self.rule: SynRule = template
        elif isinstance(template, nx.Graph):
            self.rule = SynRule(template, canonicaliser=self._canonicaliser)
        elif isinstance(template, str):
            its_graph = rsmi_to_its(template)
            self.rule = SynRule(its_graph, canonicaliser=self._canonicaliser)
        else:
            raise TypeError(f"Unsupported template type: {type(template)}")

        # Find subgraph mappings
        self._mappings: List[Dict[Any, Any]] = self._find_subgraph_mappings(
            host=self.graph.raw,
            pattern=self.rule.left.raw,
            node_attrs=["element", "charge"],
            edge_attrs=["order"],
        )

        # Build ITS graphs per mapping
        self.its: List[nx.Graph] = [
            self._explicit_h(self._glue_graph(self.graph.raw, self.rule.rc.raw, m))
            for m in self._mappings
        ]

        # Generate canonical SMARTS
        self.smarts: List[str] = []
        for its_graph in self.its:
            left_g, right_g = its_decompose(its_graph)
            smarts = f"{graph_to_smi(left_g)}>>{graph_to_smi(right_g)}"
            self.smarts.append(smarts)

    def help(self) -> None:
        """
        Print a compact summary of the reactor's configuration:
        substrate, template, inversion flag, mapping count, and SMARTS list.
        """
        print("SynReactor Summary")
        print(f"  Substrate: {self.graph}")
        print(f"  Template : {self.rule}")
        print(f"  Inverted : {self.invert}")
        print(f"  Mappings : {len(self._mappings)}")
        print("  SMARTS  :")
        for s in self.smarts:
            print(f"    {s}")

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

        Parameters
        ----------
        host : nx.Graph
            Host substrate graph.
        pattern : nx.Graph
            Pattern graph to match.
        node_attrs : List[str]
            Node attribute keys to match.
        edge_attrs : List[str]
            Edge attribute keys to match.

        Returns
        -------
        List[Dict[Any, Any]]
            List of mappings pattern_node → host_node.
        """

        def node_match(n1: Dict[str, Any], n2: Dict[str, Any]) -> bool:
            return all(n1.get(k) == n2.get(k) for k in node_attrs)

        def edge_match(e1: Dict[str, Any], e2: Dict[str, Any]) -> bool:
            return all(e1.get(k) == e2.get(k) for k in edge_attrs)

        gm = GraphMatcher(host, pattern, node_match=node_match, edge_match=edge_match)
        mappings: List[Dict[Any, Any]] = []
        for h2p in gm.subgraph_monomorphisms_iter():
            p2h = {p: h for h, p in h2p.items()}
            if all(
                host.nodes[p2h[p]].get("hcount", 0) >= pattern.nodes[p].get("hcount", 0)
                for p in p2h
            ):
                mappings.append(p2h)
        return mappings

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
        new_p = host_p[:2] + (host_r[2] - delta,) + host_p[3:]
        host_n[key] = (new_r, new_p)

        if "h_pairs" in pat_n:
            host_n["h_pairs"] = pat_n["h_pairs"]

    @staticmethod
    def _glue_graph(host: nx.Graph, rc: nx.Graph, mapping: Dict[Any, Any]) -> nx.Graph:
        """
        Glue reaction-core onto host, merging node and edge data.

        Parameters
        ----------
        host : nx.Graph
            Substrate graph.
        rc : nx.Graph
            Reaction-core graph.
        mapping : Dict[Any, Any]
            rc-node → host-node mapping.

        Returns
        -------
        nx.Graph
            Glued graph.
        """
        its_graph = deepcopy(host)

        def default_tg(
            attrs: Dict[str, Any],
        ) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
            tpl = (
                attrs.get("element", "*"),
                attrs.get("aromatic", False),
                attrs.get("hcount", 0),
                attrs.get("charge", 0),
                attrs.get("neighbors", []),
            )
            return tpl, tpl

        for _, a in its_graph.nodes(data=True):
            a.setdefault("typesGH", default_tg(a))
        for u, v, a in its_graph.edges(data=True):
            o = a.get("order", 1.0)
            a["order"] = (o, o)
            a.setdefault("standard_order", 0.0)

        for _, a in rc.nodes(data=True):
            a.setdefault("typesGH", default_tg(a))

        for rc_n, host_n in mapping.items():
            if its_graph.has_node(host_n):
                SynReactor._node_glue(its_graph.nodes[host_n], rc.nodes[rc_n])

        for u, v, eattr in rc.edges(data=True):
            hu, hv = mapping.get(u), mapping.get(v)
            if hu is None or hv is None:
                continue
            if not its_graph.has_edge(hu, hv):
                its_graph.add_edge(hu, hv)
            its_graph[hu][hv].update(eattr)

        return its_graph

    @staticmethod
    def _explicit_h(rc_graph: nx.Graph) -> nx.Graph:
        """
        Insert explicit hydrogen nodes based on h_pairs and typesGH deltas.

        Parameters
        ----------
        rc_graph : nx.Graph
            Graph after glue step.

        Returns
        -------
        nx.Graph
            Graph with explicit H atoms added.

        Raises
        ------
        ValueError
            If h-pair deltas are inconsistent.
        """
        rc = deepcopy(rc_graph)
        next_id = max((n for n in rc.nodes if isinstance(n, int)), default=-1) + 1

        pair_to_nodes: Dict[int, List[int]] = defaultdict(list)
        for n, d in rc.nodes(data=True):
            for pid in d.get("h_pairs", []):
                pair_to_nodes[pid].append(n)

        for pid, nodes in pair_to_nodes.items():
            if len(nodes) != 2:
                continue
            i, j = nodes
            hl_i = rc.nodes[i]["typesGH"][0][2]
            hr_i = rc.nodes[i]["typesGH"][1][2]
            hl_j = rc.nodes[j]["typesGH"][0][2]
            hr_j = rc.nodes[j]["typesGH"][1][2]

            delta_i = hl_i - hr_i
            delta_j = hl_j - hr_j
            if delta_i + delta_j != 0:
                raise ValueError(
                    f"Inconsistent H-pair {pid}: Δi={delta_i}, Δj={delta_j}"
                )
            if delta_i == 0:
                continue

            mag = abs(delta_i)
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
            if delta_i > 0:
                rc.add_edge(i, h, order=(mag, 0), standard_order=mag)
                rc.add_edge(h, j, order=(0, mag), standard_order=-mag)
            else:
                rc.add_edge(j, h, order=(mag, 0), standard_order=mag)
                rc.add_edge(h, i, order=(0, mag), standard_order=-mag)

            for node in (i, j):
                t0, t1 = rc.nodes[node]["typesGH"]
                rc.nodes[node]["typesGH"] = (
                    t0[:2] + (0,) + t0[3:],
                    t1[:2] + (0,) + t1[3:],
                )
        return rc
