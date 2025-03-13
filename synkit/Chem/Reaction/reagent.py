from rxnmapper import RXNMapper
from typing import Tuple, Optional
from synkit.IO.debug import setup_logging

logger = setup_logging()
rxn_mapper = RXNMapper()


class Reagent:
    def __init__(self) -> None:
        pass

    @staticmethod
    def map_with_rxn_mapper(
        reaction_smiles: str, rxn_mapper: RXNMapper = rxn_mapper
    ) -> str:
        """
        Maps a reaction SMILES string using the RXNMapper.

        Parameters:
        - reaction_smiles (str): The SMILES string of the reaction to be mapped.
        - rxn_mapper (RXNMapper): The RXNMapper instance used for mapping.

        Returns:
        - str: The atom-mapped reaction SMILES string, or the original reaction if mapping fails.
        """
        try:
            # Perform the atom mapping using RXNMapper
            mapped_rxn = rxn_mapper.get_attention_guided_atom_maps(
                [reaction_smiles], canonicalize_rxns=False
            )[0]["mapped_rxn"]
            # Ensure we only take the mapped reaction if any extra information exists
            return mapped_rxn.split(" ")[0] if " " in mapped_rxn else mapped_rxn
        except Exception as e:
            logger.error(
                f"RXNMapper mapping failed for reaction '{reaction_smiles}': {e}"
            )
            return reaction_smiles

    @staticmethod
    def clean_reagents(
        reaction_smiles: str, return_clean_reaction: bool = True
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Identifies and removes reagents from a reaction based on atom mappings.

        Parameters:
        - reaction_smiles (str): The reaction in SMILES format.
        - return_clean_reaction (bool): If True, returns the cleaned reaction and the detected reagents.
          If False, only returns the detected reagents.

        Returns:
        - Tuple[Optional[str], Optional[str]]: A tuple containing the cleaned reaction (if requested) and the reagents.
          If an error occurs, both elements of the tuple will be `None`.
        """
        try:
            # Generate atom-atom mappings for the reaction
            mapped_rsmi = Reagent.map_with_rxn_mapper(reaction_smiles)

            # Split the original and mapped reactions into reactants and products
            precursor, product = reaction_smiles.split(">>")
            precursor1, product1 = mapped_rsmi.split(">>")

            # Step 1: Identify unmapped reactants (reagents)
            reactant_elements = precursor.split(".")
            mapped_reactant_elements = precursor1.split(".")
            clean_reactants = [
                r for r in reactant_elements if r not in mapped_reactant_elements
            ]
            reagents_react = [
                r for r in reactant_elements if r in mapped_reactant_elements
            ]

            # Step 2: Identify unmapped products (reagents)
            product_elements = product.split(".")
            mapped_product_elements = product1.split(".")
            clean_products = [
                p for p in product_elements if p not in mapped_product_elements
            ]
            reagents_prod = [
                p for p in product_elements if p in mapped_product_elements
            ]

            # Combine the reagents detected from both reactants and products
            reagents = reagents_react + reagents_prod
            reagents_str = ".".join(reagents) if reagents else None

            # Cleaned reaction without reagents
            clean_reaction = (
                (".".join(clean_reactants) + ">>" + ".".join(clean_products))
                if clean_reactants or clean_products
                else None
            )

            if return_clean_reaction:
                return clean_reaction, reagents_str
            else:
                return None, reagents_str

        except Exception as e:
            logger.error(f"Error processing reaction: {e}")
            return None, None
