import re
import networkx as nx
from typing import Optional, List, Any

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

__all__ = ["get_rc", "its_decompose"]


# def get_rc(
#     ITS: nx.Graph,
#     element_key: List[str] = ["element", "charge", "typesGH", "atom_map"],
#     bond_key: str = "order",
#     standard_key: str = "standard_order",
#     disconnected: bool = False,
# ) -> nx.Graph:
#     """Extract the reaction-center (RC) subgraph from an ITS graph.

#     This function identifies:
#       1. All bonds whose standard order (difference between ITS orders) is non-zero.
#       2. All H–H bonds, ensuring they are included even if no order change is detected.
#       3. (Optional) Additional nodes with charge changes and reconnection of edges
#          if `disconnected=True`.

#     :param ITS: The integrated transition-state graph with composite node/edge attributes.
#     :type ITS: nx.Graph
#     :param element_key: List of node‐attribute keys to copy into the RC graph.
#     :type element_key: List[str]
#     :param bond_key: Edge attribute key representing the tuple of bond orders.
#     :type bond_key: str
#     :param standard_key: Edge attribute key for the computed standard_order.
#     :type standard_key: str
#     :param disconnected: If True, also include nodes with charge changes and
#                          reconnect any ITS edges between RC nodes.
#     :type disconnected: bool
#     :returns: A new graph containing only the reaction-center nodes and edges.
#     :rtype: nx.Graph

#     :example:
#     >>> ITS = nx.Graph()
#     >>> # ... populate ITS with 'order', 'standard_order', 'typesGH', etc. ...
#     >>> RC = get_rc(ITS, disconnected=True)
#     >>> isinstance(RC, nx.Graph)
#     True
#     """
#     rc = nx.Graph()
#     _add_bond_order_changes(ITS, rc, element_key, bond_key, standard_key)

#     # 1.5) H-H bonds (force inclusion, with fallback typesGH)
#     for u, v, data in ITS.edges(data=True):
#         elem_u = ITS.nodes[u].get("element")
#         elem_v = ITS.nodes[v].get("element")
#         if elem_u == "H" and elem_v == "H":
#             for n in (u, v):
#                 node_data = dict(ITS.nodes[n])
#                 if "typesGH" not in node_data:
#                     node_data["typesGH"] = (
#                         ("H", False, 0, 0, []),
#                         ("*", False, 0, 0, []),
#                     )
#                 # Ensure typesGH is available even if not in original element_key
#                 final_attrs = {k: node_data[k] for k in element_key if k in node_data}
#                 final_attrs["typesGH"] = node_data["typesGH"]
#                 rc.add_node(n, **final_attrs)

#             rc.add_edge(
#                 u,
#                 v,
#                 **{
#                     bond_key: data.get(bond_key),
#                     standard_key: data.get(standard_key),
#                 },
#             )
#     if disconnected:
#         _add_charge_change_nodes(ITS, rc, element_key)
#         _reconnect_rc_edges(ITS, rc, bond_key, standard_key)

#     return rc


# def get_rc(
#     ITS: nx.Graph,
#     element_key: list[str] = ["element", "charge", "typesGH", "atom_map"],
#     bond_key: str = "order",
#     standard_key: str = "standard_order",
#     disconnected: bool = False,
#     keep_mtg: bool = False,
# ) -> nx.Graph:
#     """
#     Extract the reaction-center (RC) subgraph from an ITS graph.

#     This function identifies:
#       1. All bonds whose standard order is non-zero.
#       2. (Optional) All bonds labeled with 'is_mtg=True' if keep_mtg is True.
#       3. All H-H bonds, ensuring they are included even if no order change is detected.
#       4. (Optional) Additional nodes with charge changes and reconnection of edges
#          if `disconnected=True`.

#     :param ITS: The integrated transition-state graph with composite node/edge attributes.
#     :type ITS: nx.Graph
#     :param element_key: List of node-attribute keys to copy into the RC graph.
#     :type element_key: List[str]
#     :param bond_key: Edge attribute key representing the tuple of bond orders.
#     :type bond_key: str
#     :param standard_key: Edge attribute key for the computed standard_order.
#     :type standard_key: str
#     :param disconnected: If True, also include nodes with charge changes and
#                          reconnect any ITS edges between RC nodes.
#     :type disconnected: bool
#     :param keep_mtg: If True, also include edges where 'is_mtg' attribute is True.
#     :type keep_mtg: bool
#     :returns: A new graph containing only the reaction-center nodes and edges.
#     :rtype: nx.Graph
#     """
#     rc = nx.Graph()
#     # 1) Bonds with standard order change or mechanistic transition
#     for u, v, data in ITS.edges(data=True):
#         std = data.get(standard_key)
#         is_mtg_attr = data.get("is_mtg", False)
#         include = False
#         if isinstance(std, (int, float)) and std != 0:
#             include = True
#         if keep_mtg and is_mtg_attr:
#             include = True
#         if not include:
#             continue
#         # add nodes
#         for n in (u, v):
#             if not rc.has_node(n):
#                 node_data = dict(ITS.nodes[n])
#                 final_attrs = {k: node_data[k] for k in element_key if k in node_data}
#                 rc.add_node(n, **final_attrs)
#         # add edge
#         edge_attrs = {
#             bond_key: data.get(bond_key),
#             standard_key: std,
#             "is_mtg": is_mtg_attr,
#         }
#         rc.add_edge(u, v, **edge_attrs)

#     # 2) H-H bonds (force inclusion, with fallback typesGH)
#     for u, v, data in ITS.edges(data=True):
#         elem_u = ITS.nodes[u].get("element")
#         elem_v = ITS.nodes[v].get("element")
#         if elem_u == "H" and elem_v == "H":
#             for n in (u, v):
#                 if not rc.has_node(n):
#                     node_data = dict(ITS.nodes[n])
#                     if "typesGH" not in node_data:
#                         node_data["typesGH"] = (
#                             ("H", False, 0, 0, []),
#                             ("*", False, 0, 0, []),
#                         )
#                     final_attrs = {
#                         k: node_data[k] for k in element_key if k in node_data
#                     }
#                     final_attrs["typesGH"] = node_data["typesGH"]
#                     rc.add_node(n, **final_attrs)
#             if not rc.has_edge(u, v):
#                 rc.add_edge(
#                     u,
#                     v,
#                     **{
#                         bond_key: data.get(bond_key),
#                         standard_key: data.get(standard_key),
#                         "is_mtg": data.get("is_mtg", False),
#                     },
#                 )

#     if disconnected:
#         _add_charge_change_nodes(ITS, rc, element_key)
#         _reconnect_rc_edges(ITS, rc, bond_key, standard_key)

#     return rc

# import networkx as nx
# from typing import List, Any


def get_rc(
    ITS: nx.Graph,
    element_key: List[str] = ["element", "charge", "typesGH", "atom_map"],
    bond_key: str = "order",
    standard_key: str = "standard_order",
    disconnected: bool = False,
    keep_mtg: bool = False,
) -> nx.Graph:
    """
    Extract the reaction-center (RC) subgraph from an ITS graph.
    """
    rc = nx.Graph()
    _add_changed_bonds(ITS, rc, element_key, bond_key, standard_key, keep_mtg)
    _add_hh_bonds(ITS, rc, element_key, bond_key, standard_key)
    if disconnected:
        _add_charge_change_nodes(ITS, rc, element_key)
        _reconnect_rc_edges(ITS, rc, bond_key, standard_key)
    return rc


def _add_changed_bonds(
    ITS: nx.Graph,
    rc: nx.Graph,
    element_key: List[str],
    bond_key: str,
    standard_key: str,
    keep_mtg: bool,
) -> None:
    """
    Add bonds with non-zero standard order or mechanistic transitions.
    """
    for u, v, data in ITS.edges(data=True):
        std = data.get(standard_key)
        is_mtg_attr = data.get("is_mtg", False)
        if not _should_include_edge(std, is_mtg_attr, keep_mtg):
            continue
        _ensure_node(rc, ITS, u, element_key)
        _ensure_node(rc, ITS, v, element_key)
        rc.add_edge(
            u,
            v,
            **{bond_key: data.get(bond_key), standard_key: std, "is_mtg": is_mtg_attr},
        )


def _add_hh_bonds(
    ITS: nx.Graph,
    rc: nx.Graph,
    element_key: List[str],
    bond_key: str,
    standard_key: str,
) -> None:
    """
    Force inclusion of H-H bonds, with fallback for typesGH.
    """
    for u, v, data in ITS.edges(data=True):
        if _is_hh_pair(ITS, u, v):
            for n in (u, v):
                _ensure_node_hh(rc, ITS, n, element_key)
            if not rc.has_edge(u, v):
                rc.add_edge(
                    u,
                    v,
                    **{
                        bond_key: data.get(bond_key),
                        standard_key: data.get(standard_key),
                        "is_mtg": data.get("is_mtg", False),
                    },
                )


def _should_include_edge(
    std: Any,
    is_mtg_attr: bool,
    keep_mtg: bool,
) -> bool:
    """
    Determine if an edge should be included based on standard order and mechanistic flag.
    """
    if isinstance(std, (int, float)) and std != 0:
        return True
    if keep_mtg and is_mtg_attr:
        return True
    return False


def _is_hh_pair(ITS: nx.Graph, u: Any, v: Any) -> bool:
    """
    Check if both nodes of an edge are hydrogen.
    """
    return ITS.nodes[u].get("element") == "H" and ITS.nodes[v].get("element") == "H"


def _ensure_node(
    rc: nx.Graph,
    ITS: nx.Graph,
    node: Any,
    element_key: List[str],
) -> None:
    """
    Add a node to RC with selected attributes if not already present.
    """
    if not rc.has_node(node):
        node_data = ITS.nodes[node]
        final_attrs = {k: node_data[k] for k in element_key if k in node_data}
        rc.add_node(node, **final_attrs)


def _ensure_node_hh(
    rc: nx.Graph,
    ITS: nx.Graph,
    node: Any,
    element_key: List[str],
) -> None:
    """
    Add H node to RC, ensuring typesGH fallback if missing.
    """
    if not rc.has_node(node):
        node_data = dict(ITS.nodes[node])
        if "typesGH" not in node_data:
            node_data["typesGH"] = (("H", False, 0, 0, []), ("*", False, 0, 0, []))
        final_attrs = {k: node_data[k] for k in element_key if k in node_data}
        final_attrs["typesGH"] = node_data["typesGH"]
        rc.add_node(node, **final_attrs)


def _carry_node_attrs(src: nx.Graph, dst: nx.Graph, n: int, keys: List[str]) -> None:
    """Copy node *n* from *src* to *dst* with only *keys* attributes."""
    if dst.has_node(n):
        return
    attrs = {k: src.nodes[n][k] for k in keys if k in src.nodes[n]}
    dst.add_node(n, **attrs)


def _add_charge_change_nodes(
    ITS: nx.Graph,
    rc: nx.Graph,
    keys: List[str],
) -> None:
    """Step 3a – add nodes whose *typesGH* shows a charge change."""
    for n, data in ITS.nodes(data=True):
        gh = data.get("typesGH")
        if (
            isinstance(gh, (list, tuple))
            and len(gh) >= 2
            and gh[0][3] != gh[1][3]
            and not rc.has_node(n)
        ):
            _carry_node_attrs(ITS, rc, n, keys)


def _reconnect_rc_edges(
    ITS: nx.Graph,
    rc: nx.Graph,
    bond_key: str,
    standard_key: str,
) -> None:
    """Step 3b – re-add any original ITS edge between nodes already in RC."""
    for u, v, data in ITS.edges(data=True):
        if rc.has_node(u) and rc.has_node(v) and not rc.has_edge(u, v):
            rc.add_edge(
                u,
                v,
                **{bond_key: data.get(bond_key), standard_key: data.get(standard_key)},
            )


def _add_bond_order_changes(
    ITS: nx.Graph,
    rc: nx.Graph,
    keys: List[str],
    bond_key: str,
    standard_key: str,
) -> None:
    """Step 1 – bond-order-change edges and their nodes."""
    for u, v, data in ITS.edges(data=True):
        old, new = data.get(bond_key, (None, None))
        if old == new:
            continue
        for n in (u, v):
            _carry_node_attrs(ITS, rc, n, keys)
        rc.add_edge(
            u, v, **{bond_key: data[bond_key], standard_key: data.get(standard_key)}
        )


def its_decompose(its_graph: nx.Graph, nodes_share="typesGH", edges_share="order"):
    """Decompose an ITS graph into two separate reactant (G) and product (H)
    graphs.

    Nodes and edges in `its_graph` carry composite attributes:
      - Each node has `its_graph.nodes[nodes_share] = (node_attrs_G, node_attrs_H)`.
      - Each edge has `its_graph.edges[edges_share] = (order_G, order_H)`.

    This function splits those tuples to reconstruct the original G and H graphs.

    :param its_graph: The ITS graph with composite node/edge attributes.
    :type its_graph: nx.Graph
    :param nodes_share: Node attribute key storing (G_attrs, H_attrs) tuples.
    :type nodes_share: str
    :param edges_share: Edge attribute key storing (order_G, order_H) tuples.
    :type edges_share: str
    :returns: A tuple of two graphs (G, H) reconstructed from the ITS.
    :rtype: Tuple[nx.Graph, nx.Graph]

    :example:
    >>> its = nx.Graph()
    >>> # ... set its.nodes[n]['typesGH'] and its.edges[e]['order'] ...
    >>> G, H = its_decompose(its)
    >>> isinstance(G, nx.Graph) and isinstance(H, nx.Graph)
    True
    """
    G = nx.Graph()
    H = nx.Graph()

    # Decompose nodes
    for node, data in its_graph.nodes(data=True):
        if nodes_share in data:
            node_attr_g, node_attr_h = data[nodes_share]
            # Unpack node attributes for G
            G.add_node(
                node,
                element=node_attr_g[0],
                aromatic=node_attr_g[1],
                hcount=node_attr_g[2],
                charge=node_attr_g[3],
                # neighbors=node_attr_g[4],
                atom_map=node,
            )
            if len(node_attr_h) > 0:
                # Unpack node attributes for H
                H.add_node(
                    node,
                    element=node_attr_h[0],
                    aromatic=node_attr_h[1],
                    hcount=node_attr_h[2],
                    charge=node_attr_h[3],
                    # neighbors=node_attr_h[4],
                    atom_map=node,
                )

    # Decompose edges
    for u, v, data in its_graph.edges(data=True):
        if edges_share in data:
            order_g, order_h = data[edges_share]
            if order_g > 0:  # Assuming 0 means no edge in G
                G.add_edge(u, v, order=order_g)
            if order_h > 0:  # Assuming 0 means no edge in H
                H.add_edge(u, v, order=order_h)

    return G, H


def compare_graphs(
    graph1: nx.Graph,
    graph2: nx.Graph,
    node_attrs: list = ["element", "aromatic", "hcount", "charge", "neighbors"],
    edge_attrs: list = ["order"],
) -> bool:
    """Compare two graphs based on specified node and edge attributes.

    Parameters:
    - graph1 (nx.Graph): The first graph to compare.
    - graph2 (nx.Graph): The second graph to compare.
    - node_attrs (list): A list of node attribute names to include in the comparison.
    - edge_attrs (list): A list of edge attribute names to include in the comparison.

    Returns:
    - bool: True if both graphs are identical with respect to the specified attributes,
    otherwise False.
    """
    # Compare node sets
    if set(graph1.nodes()) != set(graph2.nodes()):
        return False

    # Compare nodes based on attributes
    for node in graph1.nodes():
        if node not in graph2:
            return False
        node_data1 = {attr: graph1.nodes[node].get(attr, None) for attr in node_attrs}
        node_data2 = {attr: graph2.nodes[node].get(attr, None) for attr in node_attrs}
        if node_data1 != node_data2:
            return False

    # Compare edge sets with sorted tuples
    if set(tuple(sorted(edge)) for edge in graph1.edges()) != set(
        tuple(sorted(edge)) for edge in graph2.edges()
    ):
        return False

    # Compare edges based on attributes
    for edge in graph1.edges():
        # Sort the edge for consistent comparison
        sorted_edge = tuple(sorted(edge))
        if sorted_edge not in graph2.edges():
            return False
        edge_data1 = {attr: graph1.edges[edge].get(attr, None) for attr in edge_attrs}
        edge_data2 = {
            attr: graph2.edges[sorted_edge].get(attr, None) for attr in edge_attrs
        }
        if edge_data1 != edge_data2:
            return False

    return True


def enumerate_tautomers(reaction_smiles: str) -> Optional[List[str]]:
    """Enumerates possible tautomers for reactants while canonicalizing the
    products in a reaction SMILES string. This function first splits the
    reaction SMILES string into reactants and products. It then generates all
    possible tautomers for the reactants and canonicalizes the product
    molecule. The function returns a list of reaction SMILES strings for each
    tautomer of the reactants combined with the canonical product.

    Parameters:
    - reaction_smiles (str): A SMILES string of the reaction formatted as
    'reactants>>products'.

    Returns:
    - List[str] | None: A list of SMILES strings for the reaction, with each string
    representing a different
    - tautomer of the reactants combined with the canonicalized products. Returns None if
    an error occurs or if invalid SMILES strings are provided.

    Raises:
    - ValueError: If the provided SMILES strings cannot be converted to molecule objects,
    indicating invalid input.
    """
    try:
        # Split the input reaction SMILES string into reactants and products
        reactants_smiles, products_smiles = reaction_smiles.split(">>")

        # Convert SMILES strings to molecule objects
        reactants_mol = Chem.MolFromSmiles(reactants_smiles)
        products_mol = Chem.MolFromSmiles(products_smiles)

        if reactants_mol is None or products_mol is None:
            raise ValueError(
                "Invalid SMILES string provided for reactants or products."
            )

        # Initialize tautomer enumerator

        enumerator = rdMolStandardize.TautomerEnumerator()

        # Enumerate tautomers for the reactants and canonicalize the products
        try:
            reactants_can = enumerator.Enumerate(reactants_mol)
        except Exception as e:
            print(f"An error occurred: {e}")
            reactants_can = [reactants_mol]
        products_can = products_mol

        # Convert molecule objects back to SMILES strings
        reactants_can_smiles = [Chem.MolToSmiles(i) for i in reactants_can]
        products_can_smiles = Chem.MolToSmiles(products_can)

        # Combine each reactant tautomer with the canonical product in SMILES format
        rsmi_list = [i + ">>" + products_can_smiles for i in reactants_can_smiles]
        if len(rsmi_list) == 0:
            return [reaction_smiles]
        else:
            # rsmi_list.remove(reaction_smiles)
            rsmi_list.insert(0, reaction_smiles)
            return rsmi_list

    except Exception as e:
        print(f"An error occurred: {e}")
        return [reaction_smiles]


def mapping_success_rate(list_mapping_data):
    """Calculate the success rate of entries containing atom mappings in a list
    of data strings.

    Parameters:
    - list_mapping_in_data (list of str): List containing strings to be searched for atom
    mappings.

    Returns:
    - float: The success rate of finding atom mappings in the list as a percentage.

    Raises:
    - ValueError: If the input list is empty.
    """
    atom_map_pattern = re.compile(r":\d+")
    if not list_mapping_data:
        raise ValueError("The input list is empty, cannot calculate success rate.")

    success = sum(
        1 for entry in list_mapping_data if re.search(atom_map_pattern, entry)
    )
    rate = 100 * (success / len(list_mapping_data))

    return round(rate, 2)
