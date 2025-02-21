import networkx as nx
from synutility.SynIO.Format.nx_to_gml import NXToGML
from synutility.SynIO.Format.chemical_conversion import (
    rsmi_to_graph,
    graph_to_rsmi,
    smiles_to_graph,
)

from synutility.SynAAM.misc import its_decompose, get_rc
from synutility.SynAAM.its_construction import ITSConstruction
from synutility.SynAAM.its_builder import ITSBuilder
from synutility.SynChem.Reaction.standardize import Standardize
from synutility.SynAAM.inference import aam_infer

std = Standardize()


class PartialExpand:
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
    def graph_expand(partial_its: nx.Graph, rsmi: str) -> str:
        """
        Expands a reaction SMILES string by applying transformation rules using an
        ITS graph based on the reaction center (RC) graph.

        This method extracts the reaction center (RC) from the ITS graph, decomposes it
        into reactant and product graphs, generates a GML rule for transformation,
        and applies the rule to the RSMI string.

        Parameters:
        - partial_its (nx.Graph): The Intermediate Transition State (ITS) graph.
        - rsmi (str): The input reaction SMILES string to be expanded.

        Returns:
        - str: The transformed reaction SMILES string after applying the
        transformation rules.
        """
        # Extract the reaction center (RC) graph from the ITS graph
        rc = get_rc(partial_its)

        # Decompose the RC into reactant and product graphs
        r_graph, p_graph = its_decompose(rc)

        # Transform the graph into a GML rule
        rule = NXToGML().transform((r_graph, p_graph, rc))

        # Apply the transformation rule to the RSMI
        transformed_rsmi = aam_infer(rsmi, rule)[0]

        return transformed_rsmi

    @staticmethod
    def expand_aam_with_transform(rsmi: str) -> str:
        """
        Expands a reaction SMILES string by identifying the reaction center (RC),
        applying transformation rules, and standardizing the atom mappings.

        This method constructs the Intermediate Transition State (ITS) graph from the
        input RSMI, applies the reaction transformation rules using `graph_expand`,
        and returns the transformed reaction SMILES string.

        Parameters:
        - rsmi (str): The input reaction SMILES string to be expanded.

        Returns:
        - str: The transformed reaction SMILES string after applying the
        transformation rules.

        Raises:
        - Exception: If an error occurs during the expansion process, the original RSMI
        is returned.
        """
        try:
            # Convert RSMI to reactant and product graphs
            r_graph, p_graph = rsmi_to_graph(rsmi, light_weight=True, sanitize=False)

            # Construct the ITS graph from the reactant and product graphs
            its = ITSConstruction().ITSGraph(r_graph, p_graph)

            # Standardize smiles
            rsmi = std.fit(rsmi)
            # Apply graph expansion and return the result
            return PartialExpand.graph_expand(its, rsmi)

        except Exception as e:
            # Log the error and return the original RSMI if something goes wrong
            print(f"An error occurred during RSMI expansion: {e}")
            return None

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

        # Construct the Intermediate Transition State (ITS) graph from reactants and products
        rc = ITSConstruction().ITSGraph(r, p)
        # rc = get_rc(rc)

        # Convert a SMILES string to graph; parameters are indicative and function should exist
        G = smiles_to_graph(
            smi,
            light_weight=light_weight,
            sanitize=True,
            drop_non_aam=False,
            use_index_as_atom_map=False,
        )

        # Rebuild the ITS graph from the generated graph and the reconstructed ITS
        its = ITSBuilder().ITSGraph(G, rc)

        # Decompose the ITS graph back into modified reactants and products
        r, p = its_decompose(its)

        # Convert the modified reactants and products back into a reaction SMILES string
        return graph_to_rsmi(r, p, its, True, False, True)


if __name__ == "__main__":
    rsmi = "[CH3][CH:1]=[CH2:2].[H:3][H:4]>>[CH3][CH:1]([H:3])[CH2:2][H:4]"
    rsmi = "CC[CH2:3][Cl:1].[NH2:2][H:4]>>CC[CH2:3][NH2:2].[Cl:1][H:4]"
    print(PartialExpand.expand(rsmi))
# self.rsmi = "BrCc1ccc(Br)cc1.COCCO>>Br.COCCOCc1ccc(Br)cc1"
#         self.gml = smart_to_gml("[Br:1][CH3:2].[OH:3][H:4]>>[Br:1][H:4].[CH3:2][OH:3]")
