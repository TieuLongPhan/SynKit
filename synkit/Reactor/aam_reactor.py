import os
import torch
import warnings
from typing import List, Any
from synkit.IO.dg_to_gml import DGToGML
from synkit.IO.debug import setup_logging
from synkit.IO.chem_converter import rsmi_to_its
from synkit.ITS.normalize_aam import NormalizeAAM
from synkit.Chem.Reaction.standardize import Standardize
from synkit.Graph.Cluster.graph_cluster import GraphCluster
from mod import smiles, ruleGMLString, DG, config


std = Standardize()

logger = setup_logging()


class AAMReactor:
    """
    AAMReactor is a class for processing and applying reaction transformations on
    SMILES strings, managing atom-atom mappings (AAM), and performing clustering
    and deduplication operations on molecular graphs. It offers various static methods
    for processing SMILES and applying reaction rules to generate derivation graphs (DGs).

    Methods:
    - get_unique_aam(list_aam: list) -> list: Retrieves unique atom-atom mappings by clustering ITS graphs.
    - deduplicateGraphs(initial) -> list: Removes duplicate molecular graphs based on isomorphism.
    - _apply(smiles_list: List[str], rule: str, verbose: int = 0, print_output: bool = False) -> DG: Applies a reaction rule to a list of SMILES strings and returns the derivation graph.
    - _inference(input: str, gml: Any, check_isomorphic: bool = False) -> List[str]: Infers normalized SMILES from a reaction SMILES string and a graph model (GML).
    """

    def __init__(self) -> None:
        """
        Initializes the AAMReactor instance. This class does not maintain state and all
        methods are static, meaning they do not require an instance of the class to be invoked.
        """
        warnings.warn("This class can only work for connected ITSGraph", UserWarning)
        pass

    @staticmethod
    def get_unique_aam(list_aam: list) -> list:
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
    def deduplicateGraphs(initial) -> list:
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

    @staticmethod
    def _apply(
        smiles_list: List[str], rule: str, verbose: int = 0, print_output: bool = False
    ) -> DG:
        """
        Applies a reaction rule to a list of SMILES strings and optionally prints
        the derivation graph.

        This function first converts the SMILES strings into molecular graphs,
        deduplicates them, sorts them based on the number of vertices, and
        then applies the provided reaction rule in the GML string format.
        The resulting derivation graph (DG) is returned.

        Parameters:
        - smiles_list (List[str]): A list of SMILES strings representing the molecules
        to which the reaction rule will be applied.
        - rule (str): The reaction rule in GML string format. This rule will be applied to the
        molecules represented by the SMILES strings.
        - verbose (int, optional): The verbosity level for logging or debugging. Default is 0 (no verbosity).
        - print_output (bool, optional): If True, the derivation graph will be printed
        to the "out" directory. Default is False.

        Returns:
        - DG: The derivation graph (DG) after applying the reaction rule to the
        initial molecules.

        Raises:
        - Exception: If an error occurs during the process of applying the rule,
        an exception is raised.
        """
        try:
            # Convert SMILES strings to molecular graphs and deduplicate
            initial_molecules = [smiles(smile, add=False) for smile in smiles_list]
            initial_molecules = AAMReactor.deduplicateGraphs(initial_molecules)

            # Sort molecules based on the number of vertices
            initial_molecules = sorted(
                initial_molecules,
                key=lambda molecule: molecule.numVertices,
                reverse=False,
            )

            # Convert the reaction rule from GML string format to a reaction rule object
            reaction_rule = ruleGMLString(rule)

            # Create the derivation graph and apply the reaction rule
            dg = DG(graphDatabase=initial_molecules)
            config.dg.doRuleIsomorphismDuringBinding = False
            dg.build().apply(initial_molecules, reaction_rule, verbosity=verbose)

            # Optionally print the output to a directory
            if print_output:
                os.makedirs("out", exist_ok=True)
                dg.print()

            return dg

        except Exception as e:
            logger.error(f"An error occurred: {e}")
            raise

    @staticmethod
    def _inference(input: str, gml: Any, check_isomorphic: bool = True) -> List[str]:
        """
        Infers a set of normalized SMILES from a reaction SMILES string and a graph model (GML).

        This function takes a reaction SMILES string (rsmi) and a graph model (gml), applies the
        reaction transformation using the graph model, normalizes and standardizes the resulting
        SMILES, and returns a list of SMILES that match the original reaction's structure after
        normalization and standardization.

        Steps:
        1. The reactants in the reaction SMILES string are separated.
        2. The transformation is applied to the reactants using the provided graph model (gml).
        3. The resulting SMILES are transformed to a canonical form.
        4. The resulting SMILES are normalized and standardized.
        5. The function returns the normalized SMILES that match the original reaction SMILES.

        Parameters:
        - rsmi (str): The reaction SMILES string in the form "reactants >> products".
        - gml (Any): A graph model or data structure used for applying the reaction transformation.

        Returns:
        - List[str]: A list of valid, normalized, and standardized SMILES strings that match the original reaction SMILES.
        """
        # Split the input reaction SMILES into reactants and products
        aam_expand = False
        if ">>" in input:
            smiles = input.split(">>")[0].split(".")
            aam_expand = True
        else:
            if isinstance(input, str):
                smiles = input.split(".")
            elif isinstance(input, list):
                smiles = input
            else:
                raise ValueError("Input must be string or list of string")

        # Apply the reaction transformation based on the graph model (GML)
        dg = AAMReactor._apply(smiles, gml)

        # Get the transformed reaction SMILES from the graph
        transformed_rsmi = list(DGToGML.getReactionSmiles(dg).values())
        transformed_rsmi = [value[0] for value in transformed_rsmi]

        # Normalize the transformed SMILES
        normalized_rsmi = []
        for value in transformed_rsmi:
            try:
                value = NormalizeAAM().fit(value)
                normalized_rsmi.append(value)
            except Exception as e:
                print(e)
                continue

        # Standardize the normalized SMILES
        curated_smiles = []
        for value in normalized_rsmi:
            try:
                curated_smiles.append(std.fit(value))
            except Exception as e:
                print(e)
                curated_smiles.append(None)
                continue
        if aam_expand is False:
            final = []
            for key, value in enumerate(curated_smiles):
                if value:
                    final.append(normalized_rsmi[key])
            if check_isomorphic:
                final = AAMReactor.get_unique_aam(final)

        else:
            # Standardize the original SMILES for comparison
            org_smiles = std.fit(input)

            # Filter out the SMILES that match the original reaction SMILES
            final = []
            for key, value in enumerate(curated_smiles):
                if value == org_smiles:
                    final.append(normalized_rsmi[key])
            if check_isomorphic:
                final = AAMReactor.get_unique_aam(final)

        return final
