import os

# import torch
from typing import List, Any
from synkit.IO.dg_to_gml import DGToGML
from synkit.IO.debug import setup_logging
from synkit.Graph.ITS.normalize_aam import NormalizeAAM
from synkit.Chem.Reaction.standardize import Standardize
from synkit.Chem.Reaction.rsmi_utils import reverse_reaction
from synkit.Reactor.reactor_utils import (
    _deduplicateGraphs,
    _get_connected_subgraphs,
    _get_unique_aam,
    _get_reagent,
    _add_reagent,
)
from synkit.Graph.ITS.its_expand import ITSExpand
from mod import smiles, ruleGMLString, DG, config


std = Standardize()

logger = setup_logging()


class ReactorEngine:
    """
    ReactorEngine is a class for processing and applying reaction transformations on
    SMILES strings, managing atom-atom mappings (AAM), and performing clustering
    and deduplication operations on molecular graphs. It offers various static methods
    for processing SMILES and applying reaction rules to generate derivation graphs (DGs).
    In this version, `ReactorEngine` can handle reagents within reactions,
    the complete AAM/ITS can be inferred directly from `ITSExpand`.

    Methods:
    - _apply: Applies a reaction rule to a list of SMILES strings and returns the
    derivation graph.
    - _inference: Infers reaction SMILES with atom-atom mappings from input smiles
    and reaction rules in GML format. If input is reaction smiles, this will become
    AAM expansion task.
    """

    def __init__(self) -> None:
        """
        Initializes the ReactorEngine instance. This class does not maintain state and all
        methods are static, meaning they do not require an instance of the class to be invoked.
        """
        pass

    @staticmethod
    def _apply(
        smiles_list: List[str],
        rule: str,
        invert: bool = False,
        verbose: int = 0,
        print_output: bool = False,
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
        - rule (str): The reaction rule in GML string format. This rule will be applied
        to the molecules represented by the SMILES strings.
        - invert (bool): Flag to indicate forward or backward prediction.
        Default is False, indicating forward prediction
        - verbose (int, optional): The verbosity level for logging or debugging.
        Default is 0 (no verbosity).
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
            ignore_reagent = False
            # Convert SMILES strings to molecular graphs and deduplicate
            initial_molecules = [smiles(smile, add=False) for smile in smiles_list]
            initial_molecules = _deduplicateGraphs(initial_molecules)

            # Sort molecules based on the number of vertices
            initial_molecules = sorted(
                initial_molecules,
                key=lambda molecule: molecule.numVertices,
                reverse=False,
            )

            # Convert the reaction rule from GML string format to a reaction rule object
            reaction_rule = ruleGMLString(rule, invert=invert, add=False)
            _number_subgraphs = _get_connected_subgraphs(rule, invert=invert)
            if len(initial_molecules) <= _number_subgraphs:
                ignore_reagent = True

            # Create the derivation graph and apply the reaction rule
            dg = DG(graphDatabase=initial_molecules)
            config.dg.doRuleIsomorphismDuringBinding = False
            dg.build().apply(
                initial_molecules,
                reaction_rule,
                verbosity=verbose,
                onlyProper=ignore_reagent,
            )

            # Optionally print the output to a directory
            if print_output:
                os.makedirs("out", exist_ok=True)
                dg.print()

            return dg

        except Exception as e:
            logger.error(f"An error occurred: {e}")
            raise

    @staticmethod
    def _inference(
        input: str,
        gml: Any,
        invert: bool = False,
        complete_aam: bool = False,
        check_isomorphic: bool = True,
    ) -> List[str]:
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
        - List[str]: A list of valid, normalized, and standardized SMILES strings
        that match the original reaction SMILES.
        """
        # Split the input reaction SMILES into reactants and products
        aam_expand = False
        if ">>" in input:
            part = input.split(">>")[1 if invert else 0]
            smiles = part.split(".")
            aam_expand = True
        else:
            if isinstance(input, str):
                smiles = input.split(".")
            elif isinstance(input, list):
                smiles = input
            else:
                raise ValueError("Input must be string or list of string")

        # Apply the reaction transformation based on the graph model (GML)
        dg = ReactorEngine._apply(smiles, gml, invert=invert)

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

        # Add reagent
        for key, value in enumerate(normalized_rsmi):
            reagents = _get_reagent(smiles, value)
            new_rsmi = _add_reagent(value, reagents)
            if invert:
                new_rsmi = reverse_reaction(new_rsmi)
            if complete_aam:
                new_rsmi = ITSExpand().expand_aam_with_its(new_rsmi)
            normalized_rsmi[key] = new_rsmi

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
                final = _get_unique_aam(final)

        else:
            # Standardize the original SMILES for comparison
            org_smiles = std.fit(input)

            # Filter out the SMILES that match the original reaction SMILES
            final = []
            for key, value in enumerate(curated_smiles):
                if value == org_smiles:
                    final.append(normalized_rsmi[key])
            if check_isomorphic:
                final = _get_unique_aam(final)

        return final
