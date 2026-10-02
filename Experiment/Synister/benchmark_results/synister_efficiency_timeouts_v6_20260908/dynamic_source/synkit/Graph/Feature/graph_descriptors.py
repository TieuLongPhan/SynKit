import networkx as nx
from joblib import Parallel, delayed
from typing import List, Dict, Any, Union
from collections import Counter, OrderedDict
from synkit.IO.debug import setup_logging
from synkit.Graph.Feature.graph_signature import GraphSignature

logger = setup_logging()


class GraphDescriptor:
    def __init__(self) -> None:
        pass

    @staticmethod
    def is_graph_empty(graph: Union[nx.Graph, dict, list, Any]) -> bool:
        """Determine if a graph representation is empty.

        :param graph: A graph representation which can be
                      a NetworkX graph, a dictionary, a list, or an object with an 'is_empty' method.
        :type graph: Union[nx.Graph, dict, list, Any]

        :return: True if the graph is empty, False otherwise.
        :rtype: bool

        :raises TypeError: If the graph representation is not supported.
        """
        if isinstance(graph, nx.Graph):
            return graph.number_of_nodes() == 0
        elif isinstance(graph, dict):
            return len(graph) == 0
        elif isinstance(graph, list):
            return all(len(row) == 0 for row in graph)
        elif hasattr(graph, "is_empty"):
            return graph.is_empty()
        else:
            raise TypeError("Unsupported graph representation")

    @staticmethod
    def is_acyclic_graph(G: nx.Graph) -> bool:
        """Determines if the given graph is acyclic.

        :param G: The graph to be checked.
        :type G: nx.Graph

        :return: True if the graph is acyclic, False otherwise.
        :rtype: bool
        """
        GraphDescriptor._validate_graph_input(G)
        return nx.is_tree(G) if not GraphDescriptor.is_graph_empty(G) else False

    @staticmethod
    def is_single_cyclic_graph(G: nx.Graph) -> bool:
        """Determines if the given graph has exactly one cycle.

        :param G: The graph to be checked.
        :type G: nx.Graph

        :return: True if the graph is single cyclic, False otherwise.
        :rtype: bool
        """
        GraphDescriptor._validate_graph_input(G)
        if GraphDescriptor.is_graph_empty(G) or not nx.is_connected(G):
            return False

        cycles = nx.cycle_basis(G)
        if cycles and set(G.nodes()) == {node for cycle in cycles for node in cycle}:
            return G.number_of_edges() == G.number_of_nodes()
        return False

    @staticmethod
    def is_complex_cyclic_graph(G: nx.Graph) -> bool:
        """Determines if the graph is complex cyclic with multiple cycles.

        :param G: The graph to be checked.
        :type G: nx.Graph

        :return: True if the graph is complex cyclic, False otherwise.
        :rtype: bool
        """
        GraphDescriptor._validate_graph_input(G)
        if GraphDescriptor.is_graph_empty(G) or not nx.is_connected(G):
            return False

        cycles = nx.minimum_cycle_basis(G)
        nodes_in_cycles = {node for cycle in cycles for node in cycle}
        return len(cycles) > 1 and nodes_in_cycles == set(G.nodes())

    @staticmethod
    def check_graph_type(G: nx.Graph) -> str:
        """Classifies the graph as acyclic, single cyclic, or complex cyclic.

        :param G: The graph to be checked.
        :type G: nx.Graph

        :return: The classification result.
        :rtype: str
        """
        GraphDescriptor._validate_graph_input(G)
        if GraphDescriptor.is_graph_empty(G):
            return "Empty Graph"
        elif GraphDescriptor.is_acyclic_graph(G):
            return "Acyclic"
        elif GraphDescriptor.is_single_cyclic_graph(G):
            return "Single Cyclic"
        elif GraphDescriptor.is_complex_cyclic_graph(G):
            return "Combinatorial Cyclic"
        else:
            return "Complex Cyclic"

    @staticmethod
    def get_cycle_member_rings(G: nx.Graph, type="minimal") -> List[int]:
        """Identifies all cycles in the given graph using cycle bases to ensure
        no overlap and returns a list of the sizes of these cycles (member
        rings), sorted in ascending order.

        :param G: The NetworkX graph to be analyzed.
        :type G: nx.Graph

        :return: A sorted list of cycle sizes (member rings) found in the graph.
        :rtype: List[int]
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

    @staticmethod
    def get_element_count(graph: nx.Graph) -> Dict[str, int]:
        """Counts occurrences of each element in the graph nodes.

        :param graph: A NetworkX graph with 'element' attribute in nodes.
        :type graph: nx.Graph

        :return: An ordered dictionary with element counts.
        :rtype: Dict[str, int]
        """
        element_counts = Counter(data["element"] for _, data in graph.nodes(data=True))
        return OrderedDict(sorted(element_counts.items()))

    @staticmethod
    def get_descriptors(
        entry: Dict,
        reaction_centers: str = "RC",
        its: str = "ITS",
        condensed: bool = True,
    ) -> Dict:
        """Enhance an entry dictionary with topology type and reaction type
        descriptors.

        :param entry: A dictionary with reaction data.
        :type entry: Dict
        :param reaction_centers: Key for accessing reaction center data.
        :type reaction_centers: str
        :param its: Key for accessing ITS (Intermediate Transition State) data.
        :type its: str

        :return: The enhanced entry with additional descriptors.
        :rtype: Dict
        """
        graph = GraphDescriptor._extract_graph(entry, reaction_centers)
        its_graph = GraphDescriptor._extract_graph(entry, its)

        if not graph or not its_graph:
            return entry  # Early exit if graphs are missing

        # Set initial topology descriptor for the reaction center graph
        entry["topo"] = GraphDescriptor.check_graph_type(graph)
        entry["cycle"] = GraphDescriptor.get_cycle_member_rings(graph)
        entry["atom_count"] = GraphDescriptor.get_element_count(graph)
        entry["its_count"] = GraphDescriptor.get_element_count(its_graph)

        # Determine the reaction type based on the topology type
        entry["rtype"] = (
            "Elementary"
            if entry["topo"] in ["Single Cyclic", "Acyclic"]
            else "Complicated"
        )

        GraphDescriptor._adjust_cycle_and_step(entry, "cycle", entry["topo"])
        entry["signature_rc"] = GraphSignature(graph).create_graph_signature()

        # Initialize ITS descriptors and call adjust
        topo_its = GraphDescriptor.check_graph_type(its_graph)
        cycle_its = GraphDescriptor.get_cycle_member_rings(its_graph)
        entry["cycle_its"] = cycle_its  # Ensure key is initialized
        GraphDescriptor._adjust_cycle_and_step(
            entry, "cycle_its", topo_its, its_prefix="its"
        )

        entry["signature_its"] = GraphSignature(its_graph).create_graph_signature()

        return entry

    @staticmethod
    def _extract_graph(entry: Dict, key: str) -> Union[nx.Graph, None]:
        """Extracts a graph from an entry dictionary based on the specified
        key.

        :param entry: The dictionary containing graph data.
        :type entry: Dict
        :param key: The key for accessing graph data.
        :type key: str

        :return: The extracted graph or None if unavailable.
        :rtype: Union[nx.Graph, None]
        """
        data = entry.get(key)
        if isinstance(data, tuple):
            try:
                return data[2]
            except IndexError:
                logger.error(f"No graph data available at index 2 for entry {entry}")
        elif isinstance(data, nx.Graph):
            return data
        else:
            logger.error(f"Unsupported data type for {key} in entry {entry}")
        return None

    @staticmethod
    def _adjust_cycle_and_step(
        entry: Dict, cycle_key: str, topo_type: str, its_prefix: str = ""
    ) -> None:
        """Adjusts cycle and step descriptors based on the graph topology type.

        :param entry: The entry dictionary to update.
        :type entry: Dict
        :param cycle_key: The key for the cycle descriptor.
        :type cycle_key: str
        :param topo_type: The topology type.
        :type topo_type: str
        :param its_prefix: Prefix for ITS-specific descriptors.
        :type its_prefix: str
        """
        step_key = f"rstep_{its_prefix}" if its_prefix else "rstep"

        # Initialize the step key in the dictionary to avoid KeyError
        if cycle_key not in entry:
            entry[cycle_key] = []

        if topo_type == "Acyclic":
            entry[cycle_key] = [0]
        elif topo_type == "Complex Cyclic":
            entry[cycle_key] = [0] + entry[cycle_key]

        entry[step_key] = len(entry[cycle_key])

    @staticmethod
    def _validate_graph_input(G: nx.Graph) -> None:
        """Validates that the input is a NetworkX graph.

        :param G: The graph to validate.
        :type G: nx.Graph

        :raises TypeError: If G is not a NetworkX Graph.
        """
        if not isinstance(G, nx.Graph):
            raise TypeError("Input must be a NetworkX Graph object.")

    @staticmethod
    def process_entries_in_parallel(
        entries: List[Dict],
        reaction_centers: str = "RC",
        its: str = "ITS",
        condensed: bool = True,
        n_jobs: int = 4,
        verbose: int = 0,
    ) -> List[Dict]:
        """Processes a list of entries in parallel to enhance each entry with
        descriptors.

        :param entries: List of dictionaries containing reaction data to enhance.
        :type entries: List[Dict]
        :param reaction_centers: Key to retrieve reaction center graph data from each
                                 entry dictionary.
        :type reaction_centers: str
        :param its: Key to retrieve ITS (Intermediate Transition State) graph data from
                    each entry dictionary.
        :type its: str
        :param condensed: If True, condenses node signatures with counts.
        :type condensed: bool
        :param n_jobs: Number of jobs to run in parallel. -1 uses all processors.
        :type n_jobs: int
        :param verbose: The verbosity level for joblib's Parallel.
        :type verbose: int

        :return: A list of enhanced dictionaries with added descriptors.
        :rtype: List[Dict]
        """
        return Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(GraphDescriptor.get_descriptors)(
                entry, reaction_centers, its, condensed
            )
            for entry in entries
        )


def check_graph_connectivity(graph: nx.Graph) -> str:
    """Check the connectivity of a NetworkX graph.

    This function assesses whether all nodes in the graph are connected by some path,
    applicable to undirected graphs.

    :param graph: A NetworkX graph object.
    :type graph: nx.Graph

    :return: Returns 'Connected' if the graph is connected, otherwise 'Disconnected'.
    :rtype: str

    :raises NetworkXNotImplemented: If graph is directed and does not support is_connected.
    """
    if nx.is_connected(graph):
        return "Connected"
    else:
        return "Disconnected."
