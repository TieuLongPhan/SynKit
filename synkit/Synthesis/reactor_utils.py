from typing import List
from collections import Counter
from synkit.Chem.Reaction.standardize import Standardize
from synkit.Graph.Matcher.graph_cluster import GraphCluster
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
    smart = gml_to_smart(gml, sanitize=False)

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


def _get_reagent_rsmi(rsmi: str) -> List[str]:
    """
    Identifies reagents that appear in both the reactant and product sides of
    a reaction SMILES string, suggesting these elements are unchanged
    by the chemical reaction.

    Parameters:
    - rsmi (str): A reaction SMILES string formatted as "reactants>>products".

    Returns:
    - List[str]: A list of unique reagents that appear on both sides of the reaction, unchanged.
    """
    # Standardize the input reaction SMILES
    rsmi = Standardize().fit(rsmi)

    # Splitting the standardized rSMI into reactants and products
    reactants, products = rsmi.split(">>")
    reactants = reactants.split(".")
    products = products.split(".")

    # Count occurrences of each molecule in reactants and products
    reactants_count = Counter(reactants)
    products_count = Counter(products)

    # Find common elements in reactants and products
    common_elements = reactants_count & products_count  # Use intersection

    # Extract and return the unique reagents that are common
    unique_reagents = list(common_elements.elements())

    return unique_reagents


def _remove_reagent(rsmi: str) -> str:
    """
    Removes common molecules from the reactants and products in a SMILES reaction string.

    This function identifies the molecules that appear on both sides of the reaction
    (reactants and products) and removes one occurrence of each common molecule from
    both sides.

    Parameters:
    - rsmi (str): A SMILES string representing a chemical reaction in the form:
    'reactant1.reactant2...>>product1.product2...'

    Returns:
    - str: A new SMILES string with the common molecules removed, in the form:
    'reactant1.reactant2...>>product1.product2...'

    Example:
    >>> remove_reagent_from_smiles('CC=O.CC=O.CCC=O>>CC=CO.CC=O.CC=O')
    'CCC=O>>CC=CO'
    """

    # Split the input SMILES string into reactants and products
    reactants, products = rsmi.split(">>")

    # Split the reactants and products by '.' to separate molecules
    reactant_molecules = reactants.split(".")
    product_molecules = products.split(".")

    # Count the occurrences of each molecule in reactants and products
    reactant_count = Counter(reactant_molecules)
    product_count = Counter(product_molecules)

    # Find common molecules between reactants and products
    common_molecules = set(reactant_count) & set(product_count)

    # Remove common molecules by the minimum occurrences in both reactants and products
    for molecule in common_molecules:
        common_occurrences = min(reactant_count[molecule], product_count[molecule])

        # Decrease the count by the common occurrences
        reactant_count[molecule] -= common_occurrences
        product_count[molecule] -= common_occurrences

    # Rebuild the lists of reactant and product molecules after removal
    filtered_reactant_molecules = [
        molecule for molecule, count in reactant_count.items() for _ in range(count)
    ]
    filtered_product_molecules = [
        molecule for molecule, count in product_count.items() for _ in range(count)
    ]

    # Join the remaining molecules back into SMILES strings
    new_reactants = ".".join(filtered_reactant_molecules)
    new_products = ".".join(filtered_product_molecules)

    # Return the updated reaction string
    return f"{new_reactants}>>{new_products}"


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


def _calculate_max_depth(reaction_tree, current_node=None, depth=0):
    """
    Calculate the maximum depth of a reaction tree.

    Parameters:
    - reaction_tree (dict): A dictionary where keys are reaction SMILES (RSMI)
    and values are lists of product reactions.
    - current_node (str): The current node in the tree being explored (reaction SMILES).
    - depth (int): The current depth of the tree.

    Returns:
    - int: The maximum depth of the tree.
    """
    # If current_node is None, start from the root node (first key in the reaction tree)
    if current_node is None:
        current_node = list(reaction_tree.keys())[0]

    # Get the products of the current node (reaction)
    products = reaction_tree.get(current_node, [])

    # If no products, we are at a leaf node, return the current depth
    if not products:
        return depth

    # Recursively calculate the depth for each product and return the maximum
    max_subtree_depth = max(
        _calculate_max_depth(reaction_tree, product, depth + 1) for product in products
    )
    return max_subtree_depth


def _find_all_paths(
    reaction_tree,
    target_products,
    current_node,
    target_depth,
    current_depth=0,
    path=None,
):
    """
    Recursively find all paths from the root to the maximum depth in the reaction tree.

    Parameters:
    - reaction_tree (dict): A dictionary of reaction SMILES with products.
    - current_node (str): The current node (reaction SMILES).
    - target_depth (int): The depth at which the product matches the root's product.
    - current_depth (int): The current depth of the search.
    - path (list): The current path in the tree.

    Returns:
    - List of all paths to the max depth.
    """
    if path is None:
        path = []

    # Add the current node (reaction SMILES) to the path
    path.append(current_node)

    # If we have reached the target depth, check the product
    if current_depth == target_depth:
        # Extract products of the current node
        current_products = sorted(
            current_node.split(">>")[1].split("."), key=len
        )  # Sort by length of SMILES
        largest_current_product = current_products[-1] if current_products else None

        # Process target_products to get the largest product

        sorted_target_products = sorted(
            target_products, key=len
        )  # target_products should be a string here

        largest_target_product = (
            sorted_target_products[-1] if sorted_target_products else None
        )

        # Compare the largest elements
        return [path] if largest_current_product == largest_target_product else []

    # If we haven't reached the target depth, recurse on the products
    paths = []
    for product in reaction_tree.get(current_node, []):
        paths.extend(
            _find_all_paths(
                reaction_tree,
                target_products,
                product,
                target_depth,
                current_depth + 1,
                path.copy(),
            )
        )
    return paths
