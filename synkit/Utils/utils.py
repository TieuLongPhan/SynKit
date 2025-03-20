import random
from typing import Dict, List, Optional
from datetime import datetime
import networkx as nx
from typing import Iterable


def stratified_random_sample(
    data: List[Dict[str, any]],
    property_key: str,
    samples_per_class: int = 1,
    seed: Optional[int] = 42,
    bypass: bool = False,
) -> List[Dict[str, any]]:
    """
    Stratifies and samples data from a list of dictionaries based on a
    specified property key.

    Parameters:
    - data (List[Dict[str, any]]): The data to sample from, a list of dictionaries.
    - property_key (str): The key in the dictionaries to stratify by.
    - samples_per_class (int): The number of samples to take from each class.
    Defaults to 1.
    - seed (Optional[int], optional): The seed for the random number generator
    for reproducibility. Defaults to 42.
    - bypass (bool, optional): If True, classes with fewer than
    `samples_per_class` entries will be skipped without raising an error.
    Defaults to False.

    Returns:
    - List[Dict[str, any]]: A list of sampled dictionaries, where each entry corresponds
    to a sampled item.

    Raises:
    - ValueError: If a class has fewer than `samples_per_class` entries
    and `bypass` is False.
    """

    if seed is not None:
        random.seed(seed)

    stratified_data = {}
    for item in data:
        key = item.get(property_key)
        if key is None:
            continue  # Exclude data items where the specified key is not present
        if key in stratified_data:
            stratified_data[key].append(item)
        else:
            stratified_data[key] = [item]

    sampled_data = []
    for key, items in stratified_data.items():
        class_size = len(items)
        if class_size >= samples_per_class:
            sampled_data.extend(random.sample(items, samples_per_class))
        elif bypass:
            continue  # Skip this group entirely if not enough data and bypass is True
        else:
            raise ValueError(
                f"Not enough data to sample {samples_per_class} items for class '{key}', "
                f"only {class_size} available."
            )

    return sampled_data


def calculate_processing_time(start_time_str: str, end_time_str: str) -> float:
    """
    Calculates the processing time in seconds between two timestamps.

    Parameters:
    - start_time_str (str): A string representing the start time in the format
    'YYYY-MM-DD HH:MM:SS,fff'.
    - end_time_str (str): A string representing the end time in the same format as
    start_time_str.

    Returns:
    - float: The duration between the start and end time in seconds.

    Raises:
    - ValueError: If the input strings do not match the expected format.
    """
    datetime_format = "%Y-%m-%d %H:%M:%S,%f"

    start_time = datetime.strptime(start_time_str, datetime_format)
    end_time = datetime.strptime(end_time_str, datetime_format)

    duration = end_time - start_time

    return duration.total_seconds()


def remove_explicit_hydrogen(
    Graph: nx.Graph, excluded_indices: Iterable[int]
) -> nx.Graph:
    """
    Processes a molecular graph by calculating hydrogen count ('h_count') for each node and
    removing hydrogen nodes that are not specified in the excluded indices.

    Parameters
    ----------
    Graph : nx.Graph
        The input graph with nodes expected to have an 'element' attribute.
    excluded_indices : Iterable[int]
        Indices of hydrogen nodes to be preserved and excluded from 'h_count' calculations.

    Returns
    -------
    nx.Graph
        The modified graph where each node has an 'h_count' attribute indicating the count
        of hydrogen neighbors, and specific hydrogens have been removed unless listed in
        excluded_indices.

    Notes
    -----
    This function operates on a copy of the input graph and does not alter the original.
    """
    G = Graph.copy()

    # Calculate h_count for each node
    for node in list(G.nodes):
        h_count = 0
        for neighbor in G.neighbors(node):
            if (
                G.nodes[neighbor].get("element") == "H"
                and neighbor not in excluded_indices
            ):
                h_count += 1
        G.nodes[node]["hcount"] = h_count

    # Remove hydrogen nodes not in excluded indices
    nodes_to_remove = [
        n
        for n in G.nodes
        if G.nodes[n].get("element") == "H" and n not in excluded_indices
    ]
    G.remove_nodes_from(nodes_to_remove)

    return G


def fix_implicit_hydrogen(Graph: nx.Graph, indices: Iterable[int]) -> nx.Graph:
    """
    Adjusts the 'h_count' attribute of specific nodes in a molecular graph,
    decreasing it based on the presence of neighboring hydrogen atoms that are also
    included in the specified indices. This function works on a copy
    of the provided graph and returns the modified copy.

    Parameters
    ----------
    - Graph (nx.Graph): The input graph where nodes have an 'element' attribute
    and possibly an 'hcount'.
    - indices (Iterable[int]): Indices of nodes to check for neighboring hydrogen atoms
    that are also in the indices list.

    Returns
    -------
    - nx.Graph: A modified copy of the original graph with adjusted
    'hcount' for specific nodes.

    Notes
    -----
    Ensure the 'hcount' exists and is appropriately structured before using this
    function. It is assumed that 'hcount' is a mutable integer that can be directly
    decremented.
    """
    G = Graph.copy()  # Work on a copy of the graph to preserve the original
    valid_indices = set(indices).intersection(
        G.nodes
    )  # Ensure node indices exist in the graph

    for node in valid_indices:
        if "hcount" in G.nodes[node]:  # Ensure the node has an 'h_count' to modify
            for neighbor in G.neighbors(node):
                # Check if neighbor is hydrogen and also in indices
                if G.nodes[neighbor].get("element") == "H" and neighbor in indices:
                    if G.nodes[node].get("element") != "H":
                        G.nodes[node]["hcount"] -= 1

    return G
