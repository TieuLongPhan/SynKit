from collections import Counter
from synkit.Chem.Reaction.standardize import Standardize
from synkit.Graph.Cluster.graph_cluster import GraphCluster
from synkit.IO.chem_converter import rsmi_to_its, gml_to_smart


def _get_unique_aam(list_aam: list) -> list:
    """
    Retrieves the unique atom-atom mappings (AAM) by clustering a list of ITS graphs.

    This function first converts each item in the provided list of AAM strings to an ITS graph
    using the `rsmi_to_its` function. Then, it performs iterative clustering of the ITS graphs
    based on matching nodes and edges, returning a list of unique AAMs based on the clustering results.

    Parameters:
    - list_aam (list): A list of AAM strings that will be converted to ITS graphs and clustered.

    Returns:
    - list: A list of unique AAMs based on the iterative clustering process.

    Raises:
    - Exception: If an error occurs during the conversion or clustering process, an exception is raised.
    """
    its_list = [rsmi_to_its(i) for i in list_aam]

    cluster = GraphCluster()

    cls, _ = cluster.iterative_cluster(
        its_list,
        attributes=None,
        nodeMatch=cluster.nodeMatch,
        edgeMatch=cluster.edgeMatch,
    )
    unique = []
    for subset in cls:
        unique.append(list_aam[list(subset)[0]])
    return unique


@staticmethod
def _deduplicateGraphs(initial) -> list:
    """
    Deduplicates a list of molecular graphs by checking for isomorphisms.

    This method checks each graph in the `initial` list against the others for isomorphism,
    and removes duplicates by keeping only one representative for each unique graph.

    Parameters:
    - initial (list): A list of molecular graphs to be deduplicated.

    Returns:
    - list: A list of unique molecular graphs, with duplicates removed.

    Raises:
    - None: No exceptions are raised by this method.
    """
    res = []
    for cand in initial:
        for a in res:
            if cand.isomorphism(a) != 0:
                res.append(a)  # the one we had already
                break
        else:
            # didn't find any isomorphic, use the new one
            res.append(cand)
    return res


def _get_connected_subgraphs(gml: str, invert: bool = False):
    """
    Given a GML string, this function returns the number of connected subgraphs based
    on the 'smart' representation split or a list of subgraphs, depending on the invert flag.

    Parameters:
    - gml: str, the GML string to be converted into a 'smart' format.
    - invert: bool, determines the output behavior:
      - If True, returns the count of subgraphs in the second part (p).
      - If False, returns the list of subgraphs from the first part (r).

    Returns:
    - A list of subgraphs if invert is False, or an integer count if invert is True.
    """
    # Validate GML input (ensure it's a valid non-empty string)
    if not isinstance(gml, str) or not gml.strip():
        raise ValueError("Invalid GML string provided.")

    # Convert GML to 'smart' representation
    smart, _ = gml_to_smart(gml, sanitize=False)

    # Split the 'smart' string by the delimiter '>>' to get the left (r) and right (p) parts
    try:
        left_part, right_part = smart.split(">>")
    except ValueError:
        raise ValueError("GML string does not contain the expected '>>' delimiter.")

    # Handle the result based on the invert flag
    if invert:
        # Return the count of subgraphs in the right part (p)
        return len(right_part.split("."))
    else:
        # Return a list of subgraphs from the left part (r)
        return len(left_part.split("."))


def _get_reagent(original_smiles: list, output_rsmi: str, invert: bool = False):
    """
    Identifies reagents present in the original SMILES list that are absent in the processed output SMILES string.

    Parameters:
    - original_smiles: list of SMILES strings representing the original reagents.
    - output_rsmi: SMILES string of the reaction, which is standardized and split to obtain new SMILES strings.
    - invert: bool, flag to choose between reactants or products for comparison.

    Returns:
    - List of SMILES strings found in original but not in the new list.
    """
    output_rsmi = Standardize().fit(output_rsmi)
    reactants, products = output_rsmi.split(">>")
    smiles = products.split(".") if invert else reactants.split(".")

    # Use Counter to find differences
    original_count = Counter(original_smiles)
    new_count = Counter(smiles)
    reagent_difference = (
        original_count - new_count
    )  # Subtract counts to find unique in original

    # Extract unique reagents
    unique_reagents = list(reagent_difference.elements())

    return unique_reagents


def _add_reagent(rsmi: str, reagents: list):
    """
    Modifies the SMILES representation of a reaction by adding additional reagents.

    Parameters:
    - rsmi: str, the SMILES reaction string, expected to contain '>>' separating reactants and products.
    - reagents: list, a list of reagent SMILES strings to be added.

    Returns:
    - str: a new SMILES string with reagents added to both reactants and products.
    """
    if not reagents:
        return rsmi  # Return original if no reagents are added

    try:
        reactants, products = rsmi.split(">>")
    except ValueError:
        raise ValueError("Input SMILES string does not contain '>>'")

    # Prepare the reagents string only once
    reagents_string = ".".join(reagents)

    # Incorporate reagents into both reactants and products
    modified_reactants = (
        f"{reactants}.{reagents_string}" if reactants else reagents_string
    )
    modified_products = f"{products}.{reagents_string}" if products else reagents_string

    return f"{modified_reactants}>>{modified_products}"
