from typing import List, Optional, Tuple
import networkx as nx
from rdkit import Chem
from rdkit.Chem import rdChemReactions


from synkit.IO.debug import setup_logging
from synkit.IO.mol_to_graph import MolToGraph
from synkit.IO.graph_to_mol import GraphToMol
from synkit.Graph.ITS.its_construction import ITSConstruction
from synkit.IO.nx_to_gml import NXToGML
from synkit.IO.gml_to_nx import GMLToNX
from synkit.Graph.ITS.its_decompose import get_rc, its_decompose
from synkit.Graph.Hyrogen._misc import implicit_hydrogen


logger = setup_logging()


def smiles_to_graph(
    smiles: str,
    drop_non_aam: bool = False,
    light_weight: bool = True,
    sanitize: bool = True,
    use_index_as_atom_map: bool = False,
) -> Optional[nx.Graph]:
    """
    Helper function to convert SMILES string to a graph using MolToGraph class.

    Parameters:
    - smiles (str): SMILES representation of the molecule.
    - drop_non_aam (bool): Whether to drop nodes without atom mapping.
    - light_weight (bool): Whether to create a light-weight graph.
    - sanitize (bool): Whether to sanitize the molecule during conversion.
    - use_index_as_atom_map (bool): Whether to use the index of atoms as atom map numbers

    Returns:
    - nx.Graph or None: The networkx graph representation of the molecule,
    or None if conversion fails.
    """

    try:
        # Parse SMILES to a molecule object, without sanitizing initially
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            logger.warning(f"Failed to parse SMILES: {smiles}")
            return None

        # Perform sanitization if requested
        if sanitize:
            try:
                Chem.SanitizeMol(mol)
            except Exception as sanitize_error:
                logger.error(
                    f"Sanitization failed for SMILES {smiles}: {sanitize_error}"
                )
                return None

        # Convert molecule to graph
        graph_converter = MolToGraph()
        graph = graph_converter.mol_to_graph(
            mol, drop_non_aam, light_weight, use_index_as_atom_map
        )
        if graph is None:
            logger.warning(f"Failed to convert molecule to graph for SMILES: {smiles}")
        return graph

    except Exception as e:
        logger.error(
            "Unhandled exception in converting SMILES to graph"
            + f": {smiles}, Error: {str(e)}"
        )
        return None


def rsmi_to_graph(
    rsmi: str,
    drop_non_aam: bool = True,
    light_weight: bool = True,
    sanitize: bool = True,
    use_index_as_atom_map: bool = True,
) -> Tuple[Optional[nx.Graph], Optional[nx.Graph]]:
    """
    Converts reactant and product SMILES strings from a reaction SMILES (RSMI) format
    to graph representations.

    Parameters:
    - rsmi (str): Reaction SMILES string in "reactants>>products" format.
    - drop_non_aam (bool, optional): If True, nodes without atom mapping numbers
    will be dropped.
    - light_weight (bool, optional): If True, creates a light-weight graph.
    - sanitize (bool, optional): If True, sanitizes molecules during conversion.

    Returns:
    - Tuple[Optional[nx.Graph], Optional[nx.Graph]]: A tuple containing t
    he graph representations of the reactants and products.
    """
    try:
        reactants_smiles, products_smiles = rsmi.split(">>")
        r_graph = smiles_to_graph(
            reactants_smiles,
            drop_non_aam,
            light_weight,
            sanitize,
            use_index_as_atom_map,
        )
        p_graph = smiles_to_graph(
            products_smiles, drop_non_aam, light_weight, sanitize, use_index_as_atom_map
        )
        return (r_graph, p_graph)
    except ValueError:
        logger.error(f"Invalid RSMI format: {rsmi}")
        return (None, None)


def graph_to_smi(
    graph: nx.Graph,
    sanitize: bool = True,
    preserve_atom_maps: Optional[List[int]] = None,
) -> Optional[str]:
    """
    Convert a NetworkX molecular graph to a SMILES string.

    Parameters
    ----------
    graph : nx.Graph
        Graph representation of the molecule. Nodes must have chemical attributes like 'element'.
    sanitize : bool
        Whether to perform RDKit sanitization on the resulting molecule.
    preserve_atom_maps : list of int, optional
        List of atom map numbers for which hydrogens should be preserved explicitly.

    Returns
    -------
    str or None
        SMILES string representation of the molecule, or None if conversion fails.
    """
    try:
        if preserve_atom_maps is None or len(preserve_atom_maps) == 0:
            mol = GraphToMol().graph_to_mol(graph, sanitize=sanitize, use_h_count=True)
        else:
            graph_imp = implicit_hydrogen(graph, set(preserve_atom_maps))
            mol = GraphToMol().graph_to_mol(
                graph_imp, sanitize=sanitize, use_h_count=True
            )

        return Chem.MolToSmiles(mol)
    except Exception as e:
        logger.debug(f"Error in generating SMILES: {str(e)}")
        return None


def graph_to_rsmi(
    r: nx.Graph,
    p: nx.Graph,
    its: Optional[nx.Graph] = None,
    sanitize: bool = True,
    explicit_hydrogen: bool = False,
) -> Optional[str]:
    """
    Convert reactant and product graphs into a reaction SMILES string.

    Parameters
    ----------
    r : nx.Graph
        Graph representing the reactants.
    p : nx.Graph
        Graph representing the products.
    its : nx.Graph, optional
        Imaginary transition state (ITS) graph. If None, it will be constructed.
    sanitize : bool
        Whether to sanitize molecules during conversion.
    explicit_hydrogen : bool
        Whether to preserve all hydrogen atoms explicitly in the SMILES output.

    Returns
    -------
    str or None
        Reaction SMILES string in the format 'reactants >> products', or None if conversion fails.
    """
    try:
        if explicit_hydrogen:
            r_smiles = graph_to_smi(r, sanitize=sanitize)
            p_smiles = graph_to_smi(p, sanitize=sanitize)
        else:
            if its is None:
                its = ITSConstruction().ITSGraph(r, p)
            rc = get_rc(its)
            list_hydrogen = [
                d["atom_map"] for _, d in rc.nodes(data=True) if d.get("element") == "H"
            ]
            r_smiles = graph_to_smi(
                r, sanitize=sanitize, preserve_atom_maps=list_hydrogen
            )
            p_smiles = graph_to_smi(
                p, sanitize=sanitize, preserve_atom_maps=list_hydrogen
            )

        if r_smiles is None or p_smiles is None:
            return None

        return f"{r_smiles}>>{p_smiles}"
    except Exception as e:
        logger.debug(f"Error in generating reaction SMILES: {str(e)}")
        return None


def smart_to_gml(
    smart: str,
    core: bool = True,
    sanitize: bool = False,
    rule_name: str = "rule",
    reindex: bool = True,
    explicit_hydrogen: bool = False,
    useSmiles: bool = True,
) -> str:
    """
    Converts a SMARTS string to GML format, optionally focusing on the reaction core.

    Parameters:
    - smart (str): The SMARTS string representing the reaction.
    - core (bool): Whether to extract and focus on the reaction core. Defaults to True.
    - sanitize (bool): Specifies whether the molecule should be sanitized upon conversion.
    - rule_name (str): The name of the reaction rule. Defaults to "rule".
    - reindex (bool): Whether to reindex the graph nodes. Defaults to True.
    - explicit_hydrogen (bool): Controls whether hydrogens are explicitly represented
    in the output.
    - useSmiles (bool): Controls whether input is SMILES or SMARTS. Defaults to True.


    Returns:
    - str: The GML representation of the reaction.
    """
    if useSmiles is False:
        smart = rsmarts_to_rsmi(smart)
    r, p = rsmi_to_graph(smart, sanitize=sanitize)
    its = ITSConstruction.ITSGraph(r, p)
    if core:
        its = get_rc(its)
        r, p = its_decompose(its)
    gml = NXToGML().transform(
        (r, p, its),
        reindex=reindex,
        rule_name=rule_name,
        explicit_hydrogen=explicit_hydrogen,
    )
    return gml


def gml_to_smart(
    gml: str,
    sanitize: bool = True,
    explicit_hydrogen: bool = False,
    useSmiles: bool = True,
) -> Tuple[str, nx.Graph]:
    """
    Converts a GML string back to a SMARTS string by interpreting the graph structures.

    Parameters:
    - gml (str): The GML string to convert.
    - sanitize (bool): Specifies whether the molecule should be sanitized upon conversion.
    - explicit_hydrogen (bool): Controls whether hydrogens are explicitly represented
    in the output.
    - useSmiles (bool): Controls whether output is SMILES or SMARTS. Defaults to True.


    Returns:
    - str: The corresponding SMARTS string.
    """
    r, p, rc = GMLToNX(gml).transform()
    rsmi = graph_to_rsmi(r, p, rc, sanitize, explicit_hydrogen)
    if useSmiles is False:
        rsmi = rsmi_to_rsmarts(rsmi)
    # return (
    #     smart,
    #     rc,
    # )
    return rsmi


def its_to_gml(
    its: nx.Graph,
    core: bool = True,
    rule_name: str = "rule",
    reindex: bool = True,
    explicit_hydrogen: bool = False,
) -> str:
    """
    Converts an ITS graph (reaction graph) to GML format, optionally focusing on the reaction core.

    Parameters:
    - its (nx.Graph): The input ITS graph representing the reaction.
    - core (bool, optional): If True, focuses on the reaction core. Defaults to True.
    - rule_name (str, optional): The name of the reaction rule. Defaults to "rule".
    - reindex (bool, optional): If True, reindexes the graph nodes. Defaults to True.
    - explicit_hydrogen (bool, optional): If True, includes explicit hydrogens in the output. Defaults to False.

    Returns:
    - str: The GML representation of the ITS graph.
    """

    # Decompose the ITS graph based on whether to focus on the core or not
    r, p = its_decompose(get_rc(its)) if core else its_decompose(its)

    # Convert the decomposed graph to GML format
    gml = NXToGML().transform(
        (r, p, its),
        reindex=reindex,
        rule_name=rule_name,
        explicit_hydrogen=explicit_hydrogen,
    )

    return gml


def gml_to_its(gml: str) -> nx.Graph:
    """
    Converts a GML string representation of a reaction back into an ITS graph.

    Parameters:
    - gml (str): The GML string representing the reaction.

    Returns:
    - nx.Graph: The resulting ITS graph.
    """

    # Convert GML back to the ITS graph using the appropriate GML to NX conversion
    _, _, its = GMLToNX(gml).transform()

    return its


def rsmi_to_its(
    rsmi: str,
    drop_non_aam: bool = True,
    light_weight: bool = True,
    sanitize: bool = True,
    use_index_as_atom_map: bool = True,
    core: bool = False,
) -> nx.Graph:
    """
    Converts a reaction SMILES (rSMI) string to an ITS graph representation using specified processing parameters.

    This function processes the input rSMI string into a graph representation of the reaction,
    considering atom-atom mappings and optionally sanitizing the molecules. It then constructs
    an Intermediate Transition State (ITS) graph based on the provided parameters.

    Parameters:
    - rsmi (str): The reaction SMILES string to be converted.
    - drop_non_aam (bool, optional): If True, non-atom-atom mapped components are dropped. Default is True.
    - light_weight (bool, optional): If True, reduces the complexity of the graph representation. Default is True.
    - sanitize (bool, optional): If True, sanitizes the molecules during conversion. Default is True.
    - use_index_as_atom_map (bool, optional): If True, uses indices as atom mappings. Default is True.

    Returns:
    - nx.Graph: The ITS graph representing the reaction.

    Raises:
    - Exception: If an error occurs during the conversion of rSMI to graph or ITS construction, an exception is raised.
    """
    r, p = rsmi_to_graph(
        rsmi, drop_non_aam, light_weight, sanitize, use_index_as_atom_map
    )
    its = ITSConstruction.ITSGraph(r, p)
    if core:
        its = get_rc(its)
    return its


def its_to_rsmi(
    its: nx.Graph,
    sanitize: bool = True,
    explicit_hydrogen: bool = False,
) -> str:
    """
    Convert an **I**ntermediate/**T**ransition‑**S**tate (ITS) `networkx.Graph`
    into a reaction‑SMILES (rSMI) string.

    Parameters
    ----------
    its : nx.Graph
        A fully annotated ITS graph.
        The nodes should carry atom‑map indices and chemical attributes
        (``element``, ``charge``, etc.), while the edges encode bond orders.
    sanitize : bool, optional
        If *True* (default), the reactant and product sub‑graphs are passed
        through RDKit’s sanitisation pipeline prior to SMILES generation
        (valence checks, kekulisation, aromaticity perception …).
        Set to *False* to skip sanitisation when you know the graph is already
        consistent.
    explicit_hydrogen : bool, optional
        When *True* the generated rSMI will contain explicit hydrogens;
        otherwise (default) hydrogens are implicit.

    Returns
    -------
    str
        A reaction‑SMILES string in the canonical form
        ``reactants > agents > products``
        (the *agents* part is left empty by this function).

    Raises
    ------
    ValueError
        If the ITS graph cannot be decomposed into a valid reactant‑product
        pair or if sanitisation fails.

    Notes
    -----
    This is a convenience wrapper around :pyfunc:`its_decompose` and
    :pyfunc:`graph_to_rsmi`:

    1. **Decompose** the ITS into reactant (*r*) and product (*p*) graphs.
    2. **Serialise** the pair back to rSMI via :pyfunc:`graph_to_rsmi`.

    Both helper functions must be available in the current namespace.

    Examples
    --------
    ```python
    rsmi = its_to_rsmi(its_graph)                 # default behaviour
    rsmi_h = its_to_rsmi(its_graph, explicit_hydrogen=True)
    rsmi_raw = its_to_rsmi(its_graph, sanitize=False)
    ```
    """
    r, p = its_decompose(its)
    return graph_to_rsmi(r, p, its, sanitize, explicit_hydrogen)


def rsmi_to_rsmarts(rsmi: str) -> str:
    """
    Convert a mapped reaction SMILES to a reaction SMARTS string.
    """
    try:
        rxn = rdChemReactions.ReactionFromSmarts(rsmi, useSmiles=True)
        return rdChemReactions.ReactionToSmarts(rxn)
    except Exception as e:
        raise ValueError(f"Failed to convert RSMI to RSMARTS: {e}")


def rsmarts_to_rsmi(rsmarts: str) -> str:
    """
    Convert a reaction SMARTS to a reaction SMILES string.
    """
    try:
        rxn = rdChemReactions.ReactionFromSmarts(rsmarts, useSmiles=False)
        return rdChemReactions.ReactionToSmiles(rxn)
    except Exception as e:
        raise ValueError(f"Failed to convert RSMARTS to RSMI: {e}")
