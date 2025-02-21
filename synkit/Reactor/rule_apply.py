import os
import torch
from typing import List
from synkit.IO.debug import setup_logging

from mod import smiles, ruleGMLString, DG, config

logger = setup_logging()


def deduplicateGraphs(initial):
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


def rule_apply(
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
        # Convert SMILES strings to molecular graphs and deduplicate
        initial_molecules = [smiles(smile, add=False) for smile in smiles_list]
        initial_molecules = deduplicateGraphs(initial_molecules)

        # Sort molecules based on the number of vertices
        initial_molecules = sorted(
            initial_molecules, key=lambda molecule: molecule.numVertices, reverse=False
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
