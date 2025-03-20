import networkx as nx
from copy import copy
from typing import Set
import warnings


def implicit_hydrogen(
    graph: nx.Graph, preserve_atom_maps: Set[int], reindex: bool = False
) -> nx.Graph:
    """
    Adds implicit hydrogens to a molecular graph and removes non-preserved hydrogens.
    This function operates on a deep copy of the input graph to avoid in-place modifications.
    It counts hydrogen neighbors for each non-hydrogen node and adjusts based on
    hydrogens that need to be preserved. Non-preserved hydrogen nodes are removed from the graph.

    Parameters:
    - graph (nx.Graph): A NetworkX graph representing the molecule, where each node has an 'element'
      attribute for the element type (e.g., 'C', 'H') and an 'atom_map' attribute for atom mapping.
    - preserve_atom_maps (Set[int]): Set of atom map numbers for hydrogens that should be preserved.
    - reindex (bool): If true, reindexes node indices and atom maps sequentially after modifications.

    Returns:
    - nx.Graph: A new NetworkX graph with updated hydrogen atoms, where non-preserved hydrogens
      have been removed and hydrogen counts adjusted for non-hydrogen atoms.
    """
    # Create a deep copy of the graph to avoid in-place modifications
    new_graph = copy(graph)

    # First pass: count hydrogen neighbors for each non-hydrogen node
    for node, data in new_graph.nodes(data=True):
        if data["element"] != "H":  # Skip hydrogen atoms
            count_h_explicit = sum(
                1
                for neighbor in new_graph.neighbors(node)
                if new_graph.nodes[neighbor]["element"] == "H"
            )
            count_h_implicit = data["hcount"]
            new_graph.nodes[node]["hcount"] = count_h_explicit + count_h_implicit

    # List of hydrogens to preserve based on atom map
    preserved_hydrogens = [
        node
        for node, data in new_graph.nodes(data=True)
        if data["element"] == "H" and data["atom_map"] in preserve_atom_maps
    ]

    # Adjust hydrogen counts for preserved hydrogens
    for hydrogen in preserved_hydrogens:
        for neighbor in new_graph.neighbors(hydrogen):
            if (
                new_graph.nodes[neighbor]["element"] != "H"
            ):  # Only adjust non-hydrogen neighbors
                new_graph.nodes[neighbor]["hcount"] -= 1

    # Remove non-preserved hydrogen nodes from the graph
    hydrogen_to_remove = [
        node
        for node, data in new_graph.nodes(data=True)
        if data["element"] == "H" and node not in preserved_hydrogens
    ]
    new_graph.remove_nodes_from(hydrogen_to_remove)

    # Reindex the graph if reindex=True
    if reindex:
        # Create new mapping and update node indices and atom maps
        mapping = {node: idx + 1 for idx, node in enumerate(new_graph.nodes())}
        new_graph = nx.relabel_nodes(new_graph, mapping)  # Relabel nodes

        # Update atom maps to reflect new node indices
        for node, data in new_graph.nodes(data=True):
            data["atom_map"] = node  # Sync atom map with node index

    return new_graph


def explicit_hydrogen(graph: nx.Graph) -> nx.Graph:
    """
    Adds explicit hydrogens to the molecular graph based on hydrogen counts ('hcount') for non-hydrogen
    atoms and increases the 'atom_map' attribute for each hydrogen added. This function assumes that
    'hcount' is present for each atom (representing how many hydrogens should be added) and that the
    'atom_map' for existing atoms is valid.

    Parameters:
    - graph (nx.Graph): A NetworkX graph representing the molecule, where each node has an 'element'
      attribute for the element type (e.g., 'C', 'H'), 'hcount' for the number of hydrogens to add,
      and 'atom_map' for atom mapping.

    Returns:
    - nx.Graph: A new NetworkX graph with explicit hydrogen atoms added and 'atom_map' updated.
    """
    warnings.warn(
        "This function can only work with single graph and cannot guarantee the mapping between G and H"
    )
    # Create a deep copy of the graph to avoid in-place modifications
    new_graph = copy(graph)

    # Find the maximum atom_map currently in the graph
    max_atom_map = max(
        [
            data["atom_map"]
            for node, data in new_graph.nodes(data=True)
            if "atom_map" in data
        ],
        default=0,
    )

    # Prepare a list of nodes that will need explicit hydrogens
    hydrogen_id = max_atom_map + 1  # Start adding hydrogens from max atom_map + 1
    hydrogen_additions = []  # To keep track of hydrogens to add

    # First, collect all nodes that need hydrogens
    for node, data in new_graph.nodes(data=True):
        if data["element"] != "H":  # Skip hydrogens
            hcount = data.get("hcount", 0)  # Number of hydrogens to add
            for _ in range(hcount):
                hydrogen_additions.append((node, hydrogen_id))
                hydrogen_id += 1  # Increment for next hydrogen

    # Now, add the hydrogens and update the graph
    for parent, hydrogen_atom_map in hydrogen_additions:
        hydrogen_node = f"H_{hydrogen_atom_map}"
        new_graph.add_node(hydrogen_node, element="H", atom_map=hydrogen_atom_map)
        new_graph.add_edge(
            parent, hydrogen_node
        )  # Connect the hydrogen to its parent atom

    return new_graph


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
