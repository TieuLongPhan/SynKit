from synkit.IO.chem_converter import (
    rsmi_to_graph,
    graph_to_rsmi,
    smiles_to_graph,
)

from synkit.Graph.ITS.its_decompose import its_decompose
from synkit.Graph.ITS.its_construction import ITSConstruction
from synkit.Graph.ITS.its_builder import ITSBuilder
from synkit.Chem.Reaction.standardize import Standardize

std = Standardize()


class ITSExpand:
    """
    A class for partially expanding reaction SMILES (RSMI) by applying transformation
    rules based on the reaction center (RC) graph.

    This class provides methods for expanding a given RSMI by identifying the
    reaction center (RC), applying transformation rules, and standardizing atom mappings
    to generate a full AAM RSMI.

    Methods:
    - expand(rsmi: str) -> str:
        Expands a reaction SMILES string by identifying the reaction center (RC),
        applying transformation rules, and standardizing atom mappings.

    - graph_expand(partial_its: nx.Graph, rsmi: str) -> str:
        Expands a reaction SMILES string using an Imaginary Transition State
        (ITS) graph and applies the transformation rule based on the reaction center (RC).
    """

    def __init__(self) -> None:
        """
        Initializes the PartialExpand class.

        This constructor currently does not initialize any instance-specific attributes.
        """
        pass

    @staticmethod
    def expand_aam_with_its(rsmi: str, use_G: bool = True, light_weight=True) -> str:
        """
        Expands a partial reaction SMILES string to a full reaction SMILES by reconstructing
        intermediate transition states (ITS) and decomposing them back into reactants and products.

        Parameters:
        - rsmi (str): The reaction SMILES string that potentially contains a partial mapping of atoms.
        - use_G (bool, optional): A flag to determine which part of the reaction SMILES to expand.
        If True, uses the reactants' part for expansion; if False, uses the products' part.

        Returns:
        - str: The expanded reaction SMILES string with a complete mapping of all atoms involved
        in the reaction.

        Note:
        - This function assumes that the input reaction SMILES is formatted correctly and split
        into reactants and products separated by '>>'.
        - The function relies on graph transformation methods to construct the ITS graph, decompose it,
        and finally convert the resulting graph back into a SMILES string.
        """
        # Split the reaction SMILES based on the use_G flag
        smi = rsmi.split(">>")[0] if use_G else rsmi.split(">>")[1]

        # Convert reaction SMILES to graph representation of reactants and products
        r, p = rsmi_to_graph(rsmi)

        # Construct the Imaginary Transition State (ITS) graph from reactants and products
        rc = ITSConstruction().ITSGraph(r, p)
        # rc = get_rc(rc)

        # Convert a SMILES string to graph; parameters are indicative and function should exist
        G = smiles_to_graph(
            smi,
            sanitize=True,
            drop_non_aam=False,
            use_index_as_atom_map=False,
        )

        # Rebuild the ITS graph from the generated graph and the reconstructed ITS
        its = ITSBuilder().ITSGraph(G, rc)

        # Decompose the ITS graph back into modified reactants and products
        r, p = its_decompose(its)

        # Convert the modified reactants and products back into a reaction SMILES string
        return graph_to_rsmi(r, p, its, True, False)
