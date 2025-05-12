import re
import networkx as nx
from typing import Optional, List

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


def get_rc(
    ITS: nx.Graph,
    element_key: List[str] = ["element", "charge", "typesGH", "atom_map"],
    bond_key: str = "order",
    standard_key: str = "standard_order",
    disconnected: bool = False,
) -> nx.Graph:
    """
    Extract the reaction center (RC) from ITS graph.

    Enhancements:
    - Adds nodes and edges where bond order changes (core logic).
    - If disconnected=True:
        - Adds nodes with charge change based on typesGH.
        - Reconnects any ITS edge between two RC nodes.
    - NEW: Always includes H-H bonds in RC. Adds default typesGH if missing.
    """
    rc = nx.Graph()
    _add_bond_order_changes(ITS, rc, element_key, bond_key, standard_key)

    # 1.5) H-H bonds (force inclusion, with fallback typesGH)
    for u, v, data in ITS.edges(data=True):
        elem_u = ITS.nodes[u].get("element")
        elem_v = ITS.nodes[v].get("element")
        if elem_u == "H" and elem_v == "H":
            for n in (u, v):
                node_data = dict(ITS.nodes[n])
                if "typesGH" not in node_data:
                    node_data["typesGH"] = (
                        ("H", False, 0, 0, []),
                        ("*", False, 0, 0, []),
                    )
                # Ensure typesGH is available even if not in original element_key
                final_attrs = {k: node_data[k] for k in element_key if k in node_data}
                final_attrs["typesGH"] = node_data["typesGH"]
                rc.add_node(n, **final_attrs)

            rc.add_edge(
                u,
                v,
                **{
                    bond_key: data.get(bond_key),
                    standard_key: data.get(standard_key),
                },
            )
    if disconnected:
        _add_charge_change_nodes(ITS, rc, element_key)
        _reconnect_rc_edges(ITS, rc, bond_key, standard_key)

    return rc


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


# def get_rc(
#     ITS: nx.Graph,
#     element_key: List[str] = ["element", "charge", "typesGH", "atom_map"],
#     bond_key: str = "order",
#     standard_key: str = "standard_order",
#     disconnected: bool = False,
# ) -> nx.Graph:
#     """
#     Extract the reaction center (RC) from ITS graph.

#     Enhancements:
#     - Adds nodes and edges where bond order changes (core logic).
#     - If disconnected=True:
#         - Adds nodes with charge change based on typesGH.
#         - Reconnects any ITS edge between two RC nodes.
#     - NEW: Always includes H-H bonds in RC. Adds default typesGH if missing.
#     """
#     rc = nx.Graph()

#     # 1) edges with bond-order change
#     for u, v, data in ITS.edges(data=True):
#         old, new = data.get(bond_key, [None, None])
#         if old != new:
#             for n in (u, v):
#                 if not rc.has_node(n):
#                     rc.add_node(
#                         n,
#                         **{
#                             k: ITS.nodes[n][k] for k in element_key if k in ITS.nodes[n]
#                         },
#                     )
#             rc.add_edge(
#                 u,
#                 v,
#                 **{bond_key: data.get(bond_key), standard_key: data.get(standard_key)},
#             )

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
#         # 2) nodes with typesGH-based charge change
#         for n, data in ITS.nodes(data=True):
#             gh = data.get("typesGH")
#             if (
#                 isinstance(gh, (list, tuple))
#                 and len(gh) >= 2
#                 and len(gh[0]) > 3
#                 and len(gh[1]) > 3
#                 and gh[0][3] != gh[1][3]
#             ):
#                 if not rc.has_node(n):
#                     rc.add_node(n, **{k: data[k] for k in element_key if k in data})

#         # 3) reconnect RC nodes
#         for u, v, data in ITS.edges(data=True):
#             if rc.has_node(u) and rc.has_node(v) and not rc.has_edge(u, v):
#                 rc.add_edge(
#                     u,
#                     v,
#                     **{
#                         bond_key: data.get(bond_key),
#                         standard_key: data.get(standard_key),
#                     },
#                 )

#     return rc


# def get_rc(
#     ITS: nx.Graph,
#     element_key: List[str] = ["element", "charge", "typesGH", "atom_map"],
#     bond_key: str = "order",
#     standard_key: str = "standard_order",
#     disconnected: bool = False,
# ) -> nx.Graph:
#     """
#     Extract the reaction center (RC) from ITS by:

#     1. Always adding any edge whose bond order changes
#        (bond_key[0] != bond_key[1]), plus its two end-nodes.
#     2. [if disconnected=True] Adding any node whose 'typesGH' record shows a charge change
#        (typesGH[0][3] != typesGH[1][3]), even if isolated.
#     3. [if disconnected=True] Re-adding any ITS edge between two nodes already in RC
#        (to preserve connectivity), carrying over bond_key & standard_key.

#     Parameters:
#     - ITS (nx.Graph): input ITS graph.
#     - element_key (List[str]): node attrs to carry over.
#     - bond_key (str): edge attr key for bond order.
#     - standard_key (str): edge attr key for standard order.
#     - disconnected (bool): if True, include “charge-change” nodes (step 2) and
#       reconnect any edges among RC nodes (step 3). If False, only performs step 1.
#     """
#     rc = nx.Graph()

#     # 1) edges with bond-order change
#     for u, v, data in ITS.edges(data=True):
#         old, new = data.get(bond_key, [None, None])
#         if old != new:
#             for n in (u, v):
#                 if not rc.has_node(n):
#                     rc.add_node(
#                         n,
#                         **{
#                             k: ITS.nodes[n][k] for k in element_key if k in ITS.nodes[n]
#                         },
#                     )
#             rc.add_edge(
#                 u,
#                 v,
#                 **{bond_key: data.get(bond_key), standard_key: data.get(standard_key)},
#             )

#     if disconnected:
#         # 2) nodes with a typesGH-based charge change
#         for n, data in ITS.nodes(data=True):
#             gh = data.get("typesGH")
#             if (
#                 isinstance(gh, (list, tuple))
#                 and len(gh) >= 2
#                 and len(gh[0]) > 3
#                 and len(gh[1]) > 3
#                 and gh[0][3] != gh[1][3]
#             ):
#                 if not rc.has_node(n):
#                     rc.add_node(n, **{k: data[k] for k in element_key if k in data})

#         # 3) re-add any ITS edge between RC nodes to preserve connectivity
#         for u, v, data in ITS.edges(data=True):
#             if rc.has_node(u) and rc.has_node(v) and not rc.has_edge(u, v):
#                 rc.add_edge(
#                     u,
#                     v,
#                     **{
#                         bond_key: data.get(bond_key),
#                         standard_key: data.get(standard_key),
#                     },
#                 )

#     return rc


def its_decompose(its_graph: nx.Graph, nodes_share="typesGH", edges_share="order"):
    """
    Decompose an ITS graph into two separate graphs G and H based on shared
    node and edge attributes.

    Parameters:
    - its_graph (nx.Graph): The integrated transition state (ITS) graph.
    - nodes_share (str): Node attribute key that stores tuples with node attributes
    or G and H.
    - edges_share (str): Edge attribute key that stores tuples with edge attributes
    for G and H.

    Returns:
    - Tuple[nx.Graph, nx.Graph]: A tuple containing the two graphs G and H.
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
                neighbors=node_attr_g[4],
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
                    neighbors=node_attr_h[4],
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
    """
    Compare two graphs based on specified node and edge attributes.

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
    """
    Enumerates possible tautomers for reactants while canonicalizing the products in a
    reaction SMILES string. This function first splits the reaction SMILES string into
    reactants and products. It then generates all possible tautomers for the reactants and
    canonicalizes the product molecule. The function returns a list of reaction SMILES
    strings for each tautomer of the reactants combined with the canonical product.

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
    """
    Calculate the success rate of entries containing atom mappings in a list of data
    strings.

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
