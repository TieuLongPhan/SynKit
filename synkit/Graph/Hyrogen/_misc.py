from copy import copy
import networkx as nx
from operator import eq
from typing import List, Any, Set, Tuple
from networkx.algorithms.isomorphism import generic_node_match, generic_edge_match

from synkit.Graph.Feature.graph_descriptors import GraphDescriptor


def has_XH(G: nx.Graph) -> bool:
    """
    Check whether the graph contains any heavy atom–hydrogen bond.

    A heavy atom is any atom whose 'element' attribute is not 'H'.
    This function searches for any edge that connects a heavy atom to a hydrogen atom.

    Parameters
    ----------
    G : nx.Graph
        A graph where each node has an 'element' attribute indicating the atom type.

    Returns
    -------
    bool
        True if at least one edge connects a hydrogen atom ('H') to a heavy atom (element ≠ 'H').
        False otherwise.
    """
    for u, v, _ in G.edges(data=True):
        el_u = G.nodes[u].get("element")
        el_v = G.nodes[v].get("element")
        if (el_u != "H" and el_v == "H") or (el_v != "H" and el_u == "H"):
            return True
    return False


def has_HH(G: nx.Graph) -> bool:
    """
    Check whether the graph contains any heavy atom–hydrogen bond.

    A heavy atom is any atom whose 'element' attribute is not 'H'.
    This function searches for any edge that connects a heavy atom to a hydrogen atom.

    Parameters
    ----------
    G : nx.Graph
        A graph where each node has an 'element' attribute indicating the atom type.

    Returns
    -------
    bool
        True if at least one edge connects a hydrogen atom ('H') to a heavy atom (element ≠ 'H').
        False otherwise.
    """
    for u, v, _ in G.edges(data=True):
        el_u = G.nodes[u].get("element")
        el_v = G.nodes[v].get("element")
        if el_u == el_v == "H":
            return True
    return False


def h_to_implicit(G: nx.Graph) -> nx.Graph:
    """
    Convert explicit hydrogen atoms to implicit counts on heavy atoms.

    For each hydrogen atom ('element' == 'H'), its neighbor (assumed to be a heavy atom)
    will have its 'hcount' attribute incremented. The hydrogen nodes are then removed.

    Parameters
    ----------
    G : nx.Graph
        Input graph with explicit hydrogen atoms as nodes (element='H').
        Heavy atoms must have 'element' and optionally 'hcount' attributes.

    Returns
    -------
    nx.Graph
        A copy of the original graph with hydrogen atoms removed and their counts
        added to the corresponding heavy atoms' 'hcount' attribute.
    """
    H2 = G.copy()
    h_nodes = [n for n, d in H2.nodes(data=True) if d.get("element") == "H"]

    for h in h_nodes:
        neighbors = list(H2.neighbors(h))
        for heavy in neighbors:
            if H2.nodes[heavy].get("element") != "H":
                H2.nodes[heavy]["hcount"] = H2.nodes[heavy].get("hcount", 0) + 1
        H2.remove_node(h)

    return H2


def h_to_explicit(G: nx.Graph, nodes: List[int] = None) -> nx.Graph:
    """
    Convert implicit hydrogen counts on heavy atoms into explicit hydrogen nodes.

    For each node ID in `nodes`, this function reads the node's 'hcount', adds that many
    new hydrogen nodes, connects them to the node with a single bond (order=1.0), and
    decrements the node's 'hcount'. Optionally updates the 'typesGH' field if present.

    Parameters
    ----------
    G : nx.Graph
        Input graph with heavy atoms containing 'hcount' indicating implicit hydrogens.

    nodes : List[int]
        List of node IDs (typically heavy atoms) on which to expand implicit hydrogens.

    Returns
    -------
    nx.Graph
        A copy of the graph with new explicit hydrogen nodes added and connected
        to the specified heavy atoms.
    """
    if nodes is None or len(nodes) == 0:
        nodes = G.nodes()
    H2 = G.copy()
    max_node = max(H2.nodes) if H2.nodes else 0

    for heavy in nodes:
        if heavy not in H2:
            continue
        count = H2.nodes[heavy].get("hcount", 0)
        if count <= 0:
            continue

        for _ in range(count):
            max_node += 1
            H2.add_node(
                max_node,
                element="H",
                aromatic=False,
                hcount=0,
                charge=0,
                atom_map=0,
                typesGH=(("H", False, 0, 0, []), ("H", False, 0, 0, [])),
            )
            H2.add_edge(heavy, max_node, order=1.0)

        H2.nodes[heavy]["hcount"] -= count

        # Optionally adjust the typesGH field if it exists
        if "typesGH" in H2.nodes[heavy]:
            tgh = H2.nodes[heavy]["typesGH"]
            tgh_list = [list(row) for row in tgh]
            tgh_list[0][2] -= count  # Assume hcount is stored at position [0][2]
            H2.nodes[heavy]["typesGH"] = tuple(tuple(row) for row in tgh_list)

    return H2


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


# def explicit_hydrogen(graph: nx.Graph) -> nx.Graph:
#     """
#     Adds explicit hydrogens to the molecular graph based on hydrogen counts ('hcount') for non-hydrogen
#     atoms and increases the 'atom_map' attribute for each hydrogen added. This function assumes that
#     'hcount' is present for each atom (representing how many hydrogens should be added) and that the
#     'atom_map' for existing atoms is valid.

#     Parameters:
#     - graph (nx.Graph): A NetworkX graph representing the molecule, where each node has an 'element'
#       attribute for the element type (e.g., 'C', 'H'), 'hcount' for the number of hydrogens to add,
#       and 'atom_map' for atom mapping.

#     Returns:
#     - nx.Graph: A new NetworkX graph with explicit hydrogen atoms added and 'atom_map' updated.
#     """
#     warnings.warn(
#         "This function can only work with single graph and cannot guarantee the mapping between G and H"
#     )
#     # Create a deep copy of the graph to avoid in-place modifications
#     new_graph = copy(graph)

#     # Find the maximum atom_map currently in the graph
#     max_atom_map = max(
#         [
#             data["atom_map"]
#             for node, data in new_graph.nodes(data=True)
#             if "atom_map" in data
#         ],
#         default=0,
#     )

#     # Prepare a list of nodes that will need explicit hydrogens
#     hydrogen_id = max_atom_map + 1  # Start adding hydrogens from max atom_map + 1
#     hydrogen_additions = []  # To keep track of hydrogens to add

#     # First, collect all nodes that need hydrogens
#     for node, data in new_graph.nodes(data=True):
#         if data["element"] != "H":  # Skip hydrogens
#             hcount = data.get("hcount", 0)  # Number of hydrogens to add
#             for _ in range(hcount):
#                 hydrogen_additions.append((node, hydrogen_id))
#                 hydrogen_id += 1  # Increment for next hydrogen

#     # Now, add the hydrogens and update the graph
#     for parent, hydrogen_atom_map in hydrogen_additions:
#         hydrogen_node = f"H_{hydrogen_atom_map}"
#         new_graph.add_node(hydrogen_node, element="H", atom_map=hydrogen_atom_map)
#         new_graph.add_edge(
#             parent, hydrogen_node
#         )  # Connect the hydrogen to its parent atom

#     return new_graph


# def expand_hydrogens(graph: nx.Graph) -> nx.Graph:
#     """
#     For each node in the graph that has an 'hcount' attribute greater than zero,
#     adds the specified number of hydrogen nodes and connects them with edges that
#     have specific attributes.

#     Parameters
#     - graph (nx.Graph): A graph representing a molecule with nodes that can
#     include 'element', 'hcount', 'charge', and 'atom_map' attributes.

#     Returns:
#     - nx.Graph: A new graph with hydrogen atoms expanded.
#     """
#     new_graph = graph.copy()  # Create a copy to modify and return
#     atom_map = (
#         max(data["atom_map"] for _, data in graph.nodes(data=True))
#         if graph.nodes
#         else 0
#     )

#     # Iterate through each node to process potential hydrogens
#     for node, data in graph.nodes(data=True):
#         hcount = data.get("hcount", 0)
#         if hcount > 0:
#             for _ in range(hcount):
#                 atom_map += 1
#                 hydrogen_node = {
#                     "element": "H",
#                     "charge": 0,
#                     "atom_map": atom_map,
#                 }
#                 new_graph.add_node(atom_map, **hydrogen_node)
#                 new_graph.add_edge(node, atom_map, order=(1.0, 1.0), standard_order=0.0)

#     return new_graph


def check_equivariant_graph(
    its_graphs: List[nx.Graph],
) -> Tuple[List[Tuple[int, int]], int]:
    """
    Checks for isomorphism among a list of ITS graphs.

    Parameters:
    - its_graphs (List[nx.Graph]): A list of ITS graphs.

    Returns:
    - List[Tuple[int, int]]: A list of tuples representing pairs of indices of
    isomorphic graphs.
    """
    nodeLabelNames = ["typesGH"]
    nodeLabelDefault = [()]
    nodeLabelOperator = [eq]
    nodeMatch = generic_node_match(nodeLabelNames, nodeLabelDefault, nodeLabelOperator)
    edgeMatch = generic_edge_match("order", 1, eq)

    classified = []

    for i in range(1, len(its_graphs)):
        # Compare the first graph with each subsequent graph
        if nx.is_isomorphic(
            its_graphs[0], its_graphs[i], node_match=nodeMatch, edge_match=edgeMatch
        ):
            classified.append((0, i))
    return classified, len(classified)


def check_explicit_hydrogen(graph: nx.Graph) -> tuple:
    """
    Counts the explicit hydrogen nodes in the given graph and collects their IDs.

    Parameters:
    - graph (nx.Graph): The graph to inspect.

    Returns:
    tuple: A tuple containing the number of hydrogen nodes and a list of their node IDs.
    """
    hydrogen_nodes = [
        node_id
        for node_id, attr in graph.nodes(data=True)
        if attr.get("element") == "H"
    ]
    return len(hydrogen_nodes), hydrogen_nodes


def check_hcount_change(react_graph: nx.Graph, prod_graph: nx.Graph) -> int:
    """
    Computes the maximum change in hydrogen count ('hcount') between corresponding nodes
    in the reactant and product graphs. It considers both hydrogen formation and breakage.

    Parameters:
    - react_graph (nx.Graph): The graph representing reactants.
    - prod_graph (nx.Graph): The graph representing products.

    Returns:
    int: The maximum hydrogen change observed across all nodes.
    """
    # max_hydrogen_change = 0
    hcount_break, _ = check_explicit_hydrogen(react_graph)
    hcount_form, _ = check_explicit_hydrogen(prod_graph)

    for node_id, attrs in react_graph.nodes(data=True):
        react_hcount = attrs.get("hcount", 0)
        if node_id in prod_graph:
            prod_hcount = prod_graph.nodes[node_id].get("hcount", 0)
        else:
            prod_hcount = 0

        if react_hcount >= prod_hcount:
            hcount_break += react_hcount - prod_hcount
        else:
            hcount_form += prod_hcount - react_hcount

        max_hydrogen_change = max(hcount_break, hcount_form)

    return max_hydrogen_change


def get_cycle_member_rings(G: nx.Graph, type="minimal") -> List[int]:
    """
    Identifies all cycles in the given graph using cycle bases to ensure no overlap
    and returns a list of the sizes of these cycles (member rings),
    sorted in ascending order.

    Parameters:
    - G (nx.Graph): The NetworkX graph to be analyzed.

    Returns:
    - List[int]: A sorted list of cycle sizes (member rings) found in the graph.
    """
    if not isinstance(G, nx.Graph):
        raise TypeError("Input must be a networkx Graph object.")

    if type == "minimal":
        cycles = nx.minimum_cycle_basis(G)
    else:
        cycles = nx.cycle_basis(G)
    member_rings = [len(cycle) for cycle in cycles]

    member_rings.sort()

    return member_rings


def get_priority(reaction_centers: List[Any]) -> List[int]:
    """
    Evaluate reaction centers for specific graph characteristics, selecting indices based
    on the shortest reaction paths and maximum ring sizes, and adjusting for certain
    graph types by modifying the ring information.

    Parameters:
    - reaction_centers: List[Any], a list of reaction centers where each center should be
    capable of being analyzed for graph type and ring sizes.

    Returns:
    - List[int]: A list of indices from the original list of reaction centers that meet
    the criteria of having the shortest reaction steps and/or the largest ring sizes.
    Returns indices with minimum reaction steps if no indices meet both criteria.
    """
    # Extract topology types and ring sizes from reaction centers
    topo_type = [
        GraphDescriptor.check_graph_type(center) for center in reaction_centers
    ]
    cyclic = [
        get_cycle_member_rings(center, "fundamental") for center in reaction_centers
    ]

    # Adjust ring information based on the graph type
    for index, graph_type in enumerate(topo_type):
        if graph_type in ["Acyclic", "Complex Cyclic"]:
            cyclic[index] = [0] + cyclic[index]

    # Determine minimum reaction steps
    reaction_steps = [len(rings) for rings in cyclic]
    min_reaction_step = min(reaction_steps)

    # Filter indices with the minimum reaction steps
    indices_shortest = [
        i for i, steps in enumerate(reaction_steps) if steps == min_reaction_step
    ]

    # Filter indices with the maximum ring size
    max_size = max(
        max(rings) for rings in cyclic if rings
    )  # Safeguard against empty sublists
    prior_indices = [i for i, rings in enumerate(cyclic) if max(rings) == max_size]

    # Combine criteria for final indices
    final_indices = [index for index in prior_indices if index in indices_shortest]

    # Fallback to shortest indices if no indices meet both criteria
    if not final_indices:
        return indices_shortest

    return final_indices
