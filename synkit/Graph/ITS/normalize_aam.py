import networkx as nx
from rdkit import Chem
from typing import List

from synkit.IO.chem_converter import rsmi_to_graph
from synkit.IO.graph_to_mol import GraphToMol
from synkit.Chem.Reaction.fix_aam import FixAAM
from synkit.Graph.Hyrogen._misc import implicit_hydrogen
from synkit.Graph.ITS.its_construction import ITSConstruction
from synkit.Graph.ITS.its_decompose import get_rc


class NormalizeAAM:
    """
    Provides functionalities to normalize atom mappings in SMILES representations,
    extract and process reaction centers from ITS graphs, and convert between
    graph representations and molecular models.
    """

    def __init__(self) -> None:
        """
        Initializes the NormalizeAAM class.
        """
        pass

    @staticmethod
    def fix_rsmi_kekulize(rsmi: str) -> str:
        """
        Filters the reactants and products of a reaction SMILES string.

        Parameters:
        - rsmi (str): A string representing the reaction SMILES in the form of "reactants >> products".

        Returns:
        - str: A filtered reaction SMILES string where invalid reactants/products are removed.
        """
        # Split the reaction into reactants and products
        reactants, products = rsmi.split(">>")

        # Filter valid reactants and products
        filtered_reactants = NormalizeAAM.fix_kekulize(reactants)
        filtered_products = NormalizeAAM.fix_kekulize(products)

        # Return the filtered reaction SMILES
        return f"{filtered_reactants}>>{filtered_products}"

    @staticmethod
    def fix_kekulize(smiles: str) -> str:
        """
        Filters and returns valid SMILES strings from a string of SMILES, joined by '.'.

        This function processes a string of SMILES separated by periods (e.g., "CCO.CC=O"),
        filters out invalid SMILES, and returns a string of valid SMILES joined by periods.

        Parameters:
        - smiles (str): A string containing SMILES strings separated by periods ('.').

        Returns:
        - str: A string of valid SMILES, joined by periods ('.').
        """
        smiles_list = smiles.split(".")  # Split SMILES by period
        valid_smiles = []  # List to store valid SMILES strings

        for smile in smiles_list:
            mol = Chem.MolFromSmiles(smile, sanitize=False)
            if mol:  # Check if molecule is valid
                valid_smiles.append(
                    Chem.MolToSmiles(
                        mol, canonical=True, kekuleSmiles=True, allHsExplicit=True
                    )
                )
        return ".".join(valid_smiles)  # Return valid SMILES joined by '.'

    @staticmethod
    def extract_subgraph(graph: nx.Graph, indices: List[int]) -> nx.Graph:
        """
        Extracts a subgraph from a given graph based on a list of node indices.

        Parameters:
        graph (nx.Graph): The original graph from which to extract the subgraph.
        indices (List[int]): A list of node indices that define the subgraph.

        Returns:
        nx.Graph: The extracted subgraph.
        """
        return graph.subgraph(indices).copy()

    def reset_indices_and_atom_map(
        self, subgraph: nx.Graph, aam_key: str = "atom_map"
    ) -> nx.Graph:
        """
        Resets the node indices and the atom_map of the subgraph to be continuous from 1 onwards.

        Parameters:
        subgraph (nx.Graph): The subgraph with possibly non-continuous indices.
        aam_key (str): The attribute key for atom mapping. Defaults to 'atom_map'.

        Returns:
        nx.Graph: A new subgraph with continuous indices and adjusted atom_map.
        """
        new_graph = nx.Graph()
        node_id_mapping = {
            old_id: new_id for new_id, old_id in enumerate(subgraph.nodes(), 1)
        }
        for old_id, new_id in node_id_mapping.items():
            node_data = subgraph.nodes[old_id].copy()
            node_data[aam_key] = new_id
            new_graph.add_node(new_id, **node_data)
            for u, v, data in subgraph.edges(data=True):
                new_graph.add_edge(node_id_mapping[u], node_id_mapping[v], **data)
        return new_graph

    def fit(self, rsmi: str, fix_aam_indice: bool = True) -> str:
        """
        Processes a reaction SMILES (RSMI) to adjust atom mappings, extract reaction centers,
        decompose into separate reactant and product graphs, and generate the corresponding SMILES.

        Parameters:
        - rsmi (str): The reaction SMILES string to be processed.
        - fix_aam_indice (bool): Whether to fix the atom mapping numbers.
        Defaults to True.

        Returns:
        str: The resulting reaction SMILES string with updated atom mappings.
        """
        rsmi = self.fix_rsmi_kekulize(rsmi)
        if fix_aam_indice:
            rsmi = FixAAM().fix_aam_rsmi(rsmi)
        r_graph, p_graph = rsmi_to_graph(
            rsmi,
            sanitize=True,
            use_index_as_atom_map=True,
            drop_non_aam=True,
        )
        its = ITSConstruction().ITSGraph(r_graph, p_graph)
        rc = get_rc(its)
        list_hydrogen = []
        for _, value in rc.nodes(data=True):
            if value["element"] == "H":
                list_hydrogen.append(value["atom_map"])
        r_graph = implicit_hydrogen(r_graph, list_hydrogen)
        p_graph = implicit_hydrogen(p_graph, list_hydrogen)

        r_mol, p_mol = GraphToMol().graph_to_mol(
            r_graph, sanitize=True, use_h_count=True
        ), GraphToMol().graph_to_mol(p_graph, sanitize=True, use_h_count=True)
        return f"{Chem.MolToSmiles(r_mol)}>>{Chem.MolToSmiles(p_mol)}"
