import torch
from typing import List, Any
from synkit.IO.dg_to_gml import DGToGML
from synkit.ITS.normalize_aam import NormalizeAAM
from synkit.Chem.Reaction.standardize import Standardize
from synkit.Reactor.rule_apply import rule_apply

std = Standardize()


def aam_infer(rsmi: str, gml: Any) -> List[str]:
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
    smiles = rsmi.split(">>")[0].split(".")

    # Apply the reaction transformation based on the graph model (GML)
    dg = rule_apply(smiles, gml)

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

    # Standardize the original SMILES for comparison
    org_smiles = std.fit(rsmi)

    # Filter out the SMILES that match the original reaction SMILES
    final = []
    for key, value in enumerate(curated_smiles):
        if value == org_smiles:
            final.append(normalized_rsmi[key])

    return final
