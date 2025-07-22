import networkx as nx
from typing import Dict, Any, Optional, Tuple, Hashable


class PartialITS:
    """Utility class for building **partial** Imaginary‑Transition‑State (ITS)
    graphs from a pair of reactant/product `networkx` graphs.

    The resulting ITS graph contains

    * a **union** of nodes from *G* (reactant) and *H* (product),
    * a per‑node attribute ``typesGH`` – a 2‑tuple ``(attrs_from_G, attrs_from_H)`` –
      where missing sides are filled by the present one,
    * edges categorised as **unchanged**, **broken** or **formed** and stored as
      an ``order`` tuple ``(o_G, o_H)``, and
    * a convenience edge attribute ``standard_order = o_G - o_H`` (optionally
      zeroed when |Δ| < 1 to ignore aromaticity changes).
    """

    # ---------------------------------------------------------------------
    # Helper – node‐attribute retrieval
    # ---------------------------------------------------------------------
    @staticmethod
    def _get_node_attr_tuple(
        graph: nx.Graph,
        node: Hashable,
        defaults: Dict[str, Any],
    ) -> Tuple[Any, ...]:
        """Return a tuple containing *all* attributes in *defaults* order.

        :param graph: graph to query.
        :param node: node identifier.
        :param defaults: mapping of attribute → default value.
        :returns: tuple in the order of *defaults.keys()*.
        """
        return tuple(
            graph.nodes[node].get(attr, default) for attr, default in defaults.items()
        )

    # ------------------------------------------------------------------
    # Helper – standard_order
    # ------------------------------------------------------------------
    @staticmethod
    def _attach_standard_order(
        graph: nx.Graph,
        ignore_aromaticity: bool = False,
    ) -> nx.Graph:
        """Attach ``standard_order`` edge attribute in‑place.

        :param graph: ITS graph with ``order`` tuples.
        :param ignore_aromaticity: if *True*, set Δ=0 when |Δ|<1.
        :returns: *graph* (for chaining).
        """
        for u, v, data in graph.edges(data=True):
            o_g, o_h = data.get("order", (0, 0))
            delta = o_g - o_h
            if ignore_aromaticity and abs(delta) < 1:
                delta = 0
            graph[u][v]["standard_order"] = delta
        return graph

    # ------------------------------------------------------------------
    # Helper – edge insertion logic
    # ------------------------------------------------------------------
    @staticmethod
    def _populate_edges(
        its: nx.Graph,
        G: nx.Graph,
        H: nx.Graph,
    ) -> None:
        """Populate *its* with ``order`` tuples following the rules.

        Unchanged (present in both): ``(o, o)``
        Broken (present only in G): ``(o, 0)`` *when one end survives*
        Formed (present only in H): ``(0, o)`` *when one end survives*
        Unchanged-external (only in one, *both* ends external): ``(o, o)``
        """
        common = set(G.nodes()) & set(H.nodes())
        seen: set[Tuple[Hashable, Hashable]] = set()

        def add(u: Hashable, v: Hashable, order: Tuple[float, float]):
            if (u, v) in seen or (v, u) in seen:
                return
            its.add_edge(u, v, order=order)
            seen.add((u, v))

        # Pass 1 – edges from G
        for u, v, d in G.edges(data=True):
            o_g = d.get("order", 0)
            if H.has_edge(u, v):  # unchanged (core)
                add(u, v, (o_g, o_g))
            else:
                if (u in common) ^ (v in common):  # broken
                    add(u, v, (o_g, 0))
                else:  # unchanged non‑core (G only)
                    add(u, v, (o_g, o_g))

        # Pass 2 – edges unique to H
        for u, v, d in H.edges(data=True):
            if G.has_edge(u, v):
                continue  # already handled
            o_h = d.get("order", 0)
            if (u in common) ^ (v in common):  # formed
                add(u, v, (0, o_h))
            else:  # unchanged non‑core (H only)
                add(u, v, (o_h, o_h))

    @staticmethod
    def balance_valences(graph: nx.Graph) -> nx.Graph:
        """
        Balances valences in a NetworkX graph by adding wildcard '*' nodes for atoms
        that have missing bonds according to their broken bonds and hydrogen counts.

        :param graph: NetworkX Graph with node attributes:
                    - element: str, chemical symbol
                    - charge: int, formal charge
                    - typesGH: tuple of descriptors (element, aromatic, hcount, h_change, connections)
                    - atom_map: int, unique identifier (node key)
        :type graph: nx.Graph
        :return: Modified graph with wildcard nodes added
        :rtype: nx.Graph
        """
        # Copy to avoid modifying the original
        G = graph.copy()
        # Determine next wildcard index (integer keys only)
        existing_ids = [n for n in G.nodes if isinstance(n, int)]
        next_id = max(existing_ids, default=0) + 1

        # Iterate over original nodes
        for atom in list(G.nodes):
            data = G.nodes[atom]
            # Skip wildcards
            if data.get("element") == "*":
                continue
            # Sum of positive standard_order values (broken bonds)
            broken = sum(
                d.get("standard_order", 0)
                for _, _, d in G.edges(atom, data=True)
                if d.get("standard_order", 0) > 0
            )
            if broken <= 0:
                continue
            # Available hydrogen counts from typesGH descriptors (index 2)
            h_counts = [desc[2] for desc in data["typesGH"]]
            # If any descriptor has hydrogen >= broken, no wildcard needed
            if max(h_counts, default=0) >= broken:
                continue
            # Need wildcard for remaining broken bonds
            wc_id = next_id
            next_id += 1
            # Add wildcard node with two GH types: one providing valence and one default
            G.add_node(
                wc_id,
                element="*",
                charge=0,
                typesGH=(("*", False, broken, 0, []), ("*", False, 0, 0, [])),
                atom_map=wc_id,
            )
            # Forming bond with wildcard: dynamic order=broken, negative standard_order
            G.add_edge(
                atom, wc_id, order=(0.0, float(broken)), standard_order=-float(broken)
            )
        return G

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @staticmethod
    def construct(
        G: nx.Graph,
        H: nx.Graph,
        *,
        ignore_aromaticity: bool = False,
        attributes_defaults: Optional[Dict[str, Any]] = None,
        balance: bool = True,
    ) -> nx.Graph:
        """Return a partial ITS graph for *G* → *H*.

        :param G: reactant graph.
        :param H: product graph.
        :keyword ignore_aromaticity: if *True*, set ``standard_order`` to 0 when
                                     |Δ|<1.
        :keyword attributes_defaults: mapping of attribute → default value used
                                      for the ``typesGH`` tuples.  If *None*, a
                                      small sensible default set is used.
        :returns: an ITS graph with nodes, ``typesGH`` tuples and annotated
                  edges.
        """
        # ------------------------------------------------------------------
        # Set defaults
        # ------------------------------------------------------------------
        if attributes_defaults is None:
            attributes_defaults = {
                "element": "*",
                "aromatic": False,
                "hcount": 0,
                "charge": 0,
                "neighbors": [],
            }

        # ------------------------------------------------------------------
        # Build node union
        # ------------------------------------------------------------------
        its = nx.Graph()
        its.add_nodes_from(G.nodes(data=True))
        its.add_nodes_from((n, d) for n, d in H.nodes(data=True) if n not in its)

        # ------------------------------------------------------------------
        # typesGH per node
        # ------------------------------------------------------------------
        types: Dict[Hashable, Tuple[Tuple[Any, ...], Tuple[Any, ...]]] = {}
        for n in its.nodes():
            in_g, in_h = n in G.nodes(), n in H.nodes()
            attrs_g = PartialITS._get_node_attr_tuple(
                G if in_g else H, n, attributes_defaults
            )
            attrs_h = (
                PartialITS._get_node_attr_tuple(H, n, attributes_defaults)
                if in_h
                else attrs_g
            )
            if not in_h:
                attrs_h = attrs_g
            types[n] = (attrs_g, attrs_h)
        nx.set_node_attributes(its, types, "typesGH")

        # ------------------------------------------------------------------
        # Edges with order tuples
        # ------------------------------------------------------------------
        PartialITS._populate_edges(its, G, H)

        # ------------------------------------------------------------------
        # Attach standard_order and return
        # ------------------------------------------------------------------
        its = PartialITS._attach_standard_order(its, ignore_aromaticity)
        if balance:
            its = PartialITS.balance_valences(its)
        return its
