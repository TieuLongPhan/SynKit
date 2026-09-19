from typing import Union, List, Any
from synkit.Chem.Reaction import remove_explicit_H_from_rsmi
from synkit.IO import rsmi_to_graph, setup_logging
from synkit.Graph.ITS import ITSConstruction, get_rc

logger = setup_logging()


def implicit_rule(
    rsmi: Union[str, List[str]], disconnected: bool = True, balance_its: bool = False
) -> Union[Any, List[Any]]:
    """Construct reaction-center objects from reaction SMILES by applying
    implicit‐H rules and ITS graph construction.

    :param rsmi: A reaction SMILES string, or a list thereof.
    :type rsmi: str or list of str
    :param disconnected: Whether to allow disconnected components in the reaction center (default: True).
    :type disconnected: bool, optional
    :param balance_its: Whether to enforce atom‐balance in the ITS graph (default: False).
    :type balance_its: bool, optional

    :return: The reaction‐center object(s) extracted from the ITS graph.
    :rtype: RC or list of RC

    :raises ValueError: If an empty SMILES string is provided.
    :raises TypeError: If `rsmi` is not a string or list of strings.
    :raises RuntimeError: If graph conversion or ITS construction fails.
    """

    def _process(smiles: str) -> Any:
        if not smiles:
            raise ValueError("Empty reaction SMILES provided.")
        try:
            # 1. Remove explicit hydrogens
            sanitized = remove_explicit_H_from_rsmi(smiles)
            # 2. Convert to reactant/product graphs
            react_graph, prod_graph = rsmi_to_graph(sanitized)
            # 3. Build ITS graph
            its_graph = ITSConstruction().ITSGraph(
                react_graph, prod_graph, balance_its=balance_its
            )
            # 4. Extract reaction center
            return get_rc(its_graph, disconnected=disconnected)
        except Exception as e:
            logger.error("Error processing RSMI '%s': %s", smiles, e)
            raise RuntimeError(f"Failed to construct ITS for '{smiles}': {e}")

    # Handle single string or list of strings
    if isinstance(rsmi, str):
        return _process(rsmi)
    elif isinstance(rsmi, list):
        return [_process(s) for s in rsmi]
    else:
        raise TypeError(
            f"`rsmi` must be a str or list of str, not {type(rsmi).__name__}"
        )
