from typing import List, Optional, Dict
from synkit.Reactor.reagent import (
    remove_reagent_from_smiles,
    add_catalysis,
)
from synkit.Reactor.multi_step import (
    perform_multi_step_reaction,
    find_all_paths,
)
from synkit.Reactor.inference import aam_infer


def get_aam_reactions(
    list_reactions: List[str],
    rule: Dict[int, str],
    order: List[int],
    cat: Optional[str],
) -> List[Optional[str]]:
    """
    Processes a list of reaction SMILES strings to infer Atom-Atom Mappings (AAM)
    using specified rules and optionally adds a catalyst if no mappings are
    initially found.

    Parameters:
    - list_reactions (List[str]): A list of reaction SMILES strings.
    - rule (Dict[int, str]): A dictionary mapping indices to rules for AAM inference.
    - order (List[int]): A list indicating the order in which rules should be applied
    to reactions.
    - cat (Optional[str]): The SMILES string of the catalyst; can be None or empty
    if no catalyst is to be used.

    Returns:
    - List[Optional[str]]: A list containing the first inferred AAM for each reaction
    or None if AAM could not be inferred.
    """
    aam = []

    for key, entry in enumerate(list_reactions):
        if not entry:
            aam.append(None)  # Handling empty or invalid entries gracefully
            continue

        rsmi = remove_reagent_from_smiles(entry)
        smart = aam_infer(rsmi, rule[order[key]])
        if len(smart) == 0:
            if cat and cat.strip():
                rsmi = add_catalysis(rsmi, cat)
                smart = aam_infer(rsmi, rule[order[key]])
            else:
                aam.append(None)
                continue
        aam.append(smart[0] if smart else None)

    return aam


def get_mechanism(gml: dict, order: List[int], rsmi: str, cat: str = None) -> List[str]:
    """
    Computes the mechanism of a chemical reaction given the reaction SMILES,
    order of reactants, and GML graph.

    Parameters:
    - gml (dict): Graph representation of the molecule.
    - order (List[int]): Order of reactants involved in the reaction.
    - rsmi (str): Reaction SMILES string.

    Returns:
    - List[str]: List of Atom-Atom Mappings (AAM) for each step in the reaction.
    """
    try:
        rsmi = add_catalysis(rsmi, cat)
        results, reaction_tree = perform_multi_step_reaction(gml, order, rsmi)

        target_products = sorted(rsmi.split(">>")[1].split("."))
        max_depth = len(results)
        all_paths = find_all_paths(reaction_tree, target_products, rsmi, max_depth)
        real_path = all_paths[0][1:]  # remove the original
        all_steps = get_aam_reactions(real_path, gml, order, cat)
        return all_steps
    except Exception as e:
        print(e)
        return []
