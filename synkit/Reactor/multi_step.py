from typing import List, Dict, Tuple
from synkit.Chem.Reaction.standardize import Standardize
from synkit.Reactor.core_engine import CoreEngine

std = Standardize()


def perform_multi_step_reaction(
    gml_list: List[str], order: List[int], rsmi: str
) -> Tuple[List[List[str]], Dict[str, List[str]]]:
    """
    Applies a sequence of multi-step reactions to a starting SMILES string. The function
    processes each reaction step in a specified order, and returns both the intermediate
    and final products, as well as a mapping of reactant SMILES to their
    corresponding products.

    Parameters:
    - gml_list (List[str]): A list of reaction rules (in GML format) to be applied.
    Each element corresponds to a reaction step.
    - order (List[int]): A list of integers that defines the order in which the
    reaction steps should be applied. Each integer is an index referring to the position
    of a reaction rule in the `gml_list`.
    - rsmi (str): The starting reaction SMILES string, representing the reactants for the
    first reaction.

    Returns:
    - Tuple[List[List[str]], Dict[str, List[str]]]:
        - A list of lists of SMILES strings, where each inner list contains the
        RSMI generated  at each reaction step.
        - A dictionary mapping each RSMI string to the resulting products after applying
          the reaction rules. The keys are the input RSMIs, and the values are the
          resulting product  SMILES strings.
    """

    # Initialize CoreEngine for reaction processing
    core = CoreEngine()
    # Initialize a dictionary to hold reaction results
    reaction_results = {}

    # List to store the results of each reaction step
    all_steps: List[List[str]] = []
    result: List[str] = [rsmi]  # Initial result is the input SMILES string

    # Loop over the reaction steps in the specified order
    for i, j in enumerate(order):
        # Get the reaction SMILES (RSMI) for the current step
        current_step_gml = gml_list[j]
        new_result: List[str] = []  # List to hold products for this step

        # Apply the reaction for each current reactant SMILES
        for current_rsmi in result:
            smi_lst = (
                current_rsmi.split(">>")[0].split(
                    "."
                )  # Split reactants at the first step
                if i == 0
                else current_rsmi.split(">>")[1].split(
                    "."
                )  # Split products for subsequent steps
            )

            # Perform the reaction using the CoreEngine
            o = core.perform_reaction(current_step_gml, smi_lst)

            # Apply standardization on the products
            o = [std.fit(i) for i in o]

            # Collect the new results (products) from this reaction step
            new_result.extend(o)

            # Record the reaction results in the dictionary, mapping input RSMI to output products
            if len(o) > 0:
                reaction_results[current_rsmi] = o

        # Update the result list for the next step
        result = new_result

        # Append the results of this step to the overall steps list
        all_steps.append(result)

    # Return the results: a list of all steps and a dictionary of reaction results
    return all_steps, reaction_results


def calculate_max_depth(reaction_tree, current_node=None, depth=0):
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
        calculate_max_depth(reaction_tree, product, depth + 1) for product in products
    )
    return max_subtree_depth


# def find_all_paths(
#     reaction_tree,
#     target_products,
#     current_node,
#     target_depth,
#     current_depth=0,
#     path=None,
# ):
#     """
#     Recursively find all paths from the root to the maximum depth in the reaction tree.

#     Parameters:
#     - reaction_tree (dict): A dictionary of reaction SMILES with products.
#     - current_node (str): The current node (reaction SMILES).
#     - target_depth (int): The depth at which the product matches the root's product.
#     - current_depth (int): The current depth of the search.
#     - path (list): The current path in the tree.

#     Returns:
#     - List of all paths to the max depth.
#     """
#     if path is None:
#         path = []

#     # Add the current node (reaction SMILES) to the path
#     path.append(current_node)

#     # If we have reached the target depth, check the product
#     if current_depth == target_depth:
#         # Extract products of the current node
#         products = sorted(current_node.split(">>")[1].split("."))
#         return [path] if products == target_products else []

#     # If we haven't reached the target depth, recurse on the products
#     paths = []
#     for product in reaction_tree.get(current_node, []):
#         paths.extend(
#             find_all_paths(
#                 reaction_tree,
#                 target_products,
#                 product,
#                 target_depth,
#                 current_depth + 1,
#                 path.copy(),
#             )
#         )

#     return paths


def find_all_paths(
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
            find_all_paths(
                reaction_tree,
                target_products,
                product,
                target_depth,
                current_depth + 1,
                path.copy(),
            )
        )
    return paths
