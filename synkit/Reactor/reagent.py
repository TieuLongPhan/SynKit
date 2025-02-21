from collections import Counter
from synkit.Chem.Reaction.standardize import Standardize

std = Standardize()


def remove_reagent_from_smiles(rsmi: str) -> str:
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


def add_catalysis(reaction_smiles, catalyst_smiles):
    """
    Adds a catalyst to both the reactant and product sides of a reaction SMILES string.
    If the catalyst is None or an empty string, the function returns the
    original reaction SMILES string.

    Parameters:
    - reaction_smiles (str): The SMILES string of the reaction ("reactants>>products").
    - catalyst_smiles (str): The SMILES string of the catalyst,
    which can be None or empty.

    Returns:
    - str: Modified reaction SMILES string with the catalyst included on both sides,
    or the original if no valid catalyst is provided.
    """
    # Check if the catalyst is None or empty
    if catalyst_smiles is None or catalyst_smiles == "":
        return reaction_smiles

    # Split the reaction SMILES into reactants and products
    reactants, products = reaction_smiles.split(">>")

    # Add the catalyst to both reactants and products
    reactants_with_cat = ".".join([reactants, catalyst_smiles])
    products_with_cat = ".".join([products, catalyst_smiles])

    # Combine the modified reactants and products back into a reaction SMILES
    new_reaction_smiles = ">>".join([reactants_with_cat, products_with_cat])

    return new_reaction_smiles
