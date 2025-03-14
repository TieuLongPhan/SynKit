import re
import networkx as nx
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from typing import Optional, List


def get_rc(
    ITS: nx.Graph,
    element_key: list = ["element", "charge", "typesGH", "atom_map"],
    bond_key: str = "order",
    standard_key: str = "standard_order",
) -> nx.Graph:
    """
    Extracts the reaction center (RC) graph from a given ITS graph by identifying edges
    where the bond order changes, indicating a reaction event.

    Parameters:
    - ITS (nx.Graph): The ITS graph to extract the RC from.
    - element_key (list): List of node attribute keys for atom properties.
    Defaults to ['element', 'charge', 'typesGH'].
    - bond_key (str): Edge attribute key for bond order. Defaults to 'order'.
    - standard_key (str): Edge attribute key for standard order information.
    Defaults to 'standard_order'.

    Returns:
    - nx.Graph: A new graph representing the reaction center of the ITS.
    """
    rc = nx.Graph()
    for n1, n2, data in ITS.edges(data=True):
        if data.get(bond_key, [None, None])[0] != data.get(bond_key, [None, None])[1]:
            rc.add_node(
                n1, **{k: ITS.nodes[n1][k] for k in element_key if k in ITS.nodes[n1]}
            )
            rc.add_node(
                n2, **{k: ITS.nodes[n2][k] for k in element_key if k in ITS.nodes[n2]}
            )
            rc.add_edge(
                n1, n2, **{bond_key: data[bond_key], standard_key: data[standard_key]}
            )
    return rc


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


def expand_hydrogens(graph: nx.Graph) -> nx.Graph:
    """
    For each node in the graph that has an 'hcount' attribute greater than zero,
    adds the specified number of hydrogen nodes and connects them with edges that
    have specific attributes.

    Parameters
    - graph (nx.Graph): A graph representing a molecule with nodes that can
    include 'element', 'hcount', 'charge', and 'atom_map' attributes.

    Returns:
    - nx.Graph: A new graph with hydrogen atoms expanded.
    """
    new_graph = graph.copy()  # Create a copy to modify and return
    atom_map = (
        max(data["atom_map"] for _, data in graph.nodes(data=True))
        if graph.nodes
        else 0
    )

    # Iterate through each node to process potential hydrogens
    for node, data in graph.nodes(data=True):
        hcount = data.get("hcount", 0)
        if hcount > 0:
            for _ in range(hcount):
                atom_map += 1
                hydrogen_node = {
                    "element": "H",
                    "charge": 0,
                    "atom_map": atom_map,
                }
                new_graph.add_node(atom_map, **hydrogen_node)
                new_graph.add_edge(node, atom_map, order=(1.0, 1.0), standard_order=0.0)

    return new_graph
