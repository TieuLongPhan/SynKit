import networkx as nx
from rdkit import Chem
from typing import Optional, Tuple

from synkit.IO.debug import setup_logging
from synkit.IO.mol_to_graph import MolToGraph
from synkit.IO.graph_to_mol import GraphToMol
from synkit.ITS.its_construction import ITSConstruction
from synkit.IO.nx_to_gml import NXToGML
from synkit.IO.gml_to_nx import GMLToNX
from synkit.ITS._misc import get_rc, its_decompose
from synkit._misc import remove_explicit_hydrogen

logger = setup_logging()


def smiles_to_graph(
    smiles: str,
    drop_non_aam: bool,
    light_weight: bool,
    sanitize: bool,
    use_index_as_atom_map: bool,
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


def graph_to_rsmi(
    r: nx.Graph,
    p: nx.Graph,
    its: nx.Graph,
    sanitize: bool = True,
    explicit_hydrogen: bool = False,
    ignore_hcount_inference: bool = False,
) -> str:
    """
    Converts graph representations of reactants and products into a
    reaction SMILES string.

    Parameters:
    - r (nx.Graph): Graph of the reactants.
    - p (nx.Graph): Graph of the products.
    - its (nx.Graph): Intermediate transition state graph, relevant for hydrogen count
    inference.
    - sanitize (bool): Specifies whether the molecule should be sanitized upon conversion.
    - explicit_hydrogen (bool): Controls whether hydrogens are explicitly represented in
    the output.
    - ignore_hcount_inference (bool): If false, hydrogens counts are inferred from
    the ITS graph.

    Returns:
    - str: Reaction SMILES string representing the conversion from reactants to products.
    """
    # Initialize a GraphToMol converter
    converter = GraphToMol()

    if not explicit_hydrogen:
        # Decide whether to infer hydrogen count based on the ITS graph
        if ignore_hcount_inference:
            r_mol = converter.graph_to_mol(r, sanitize=sanitize, use_h_count=True)
            p_mol = converter.graph_to_mol(p, sanitize=sanitize, use_h_count=True)
        else:
            rc = get_rc(its)
            r = remove_explicit_hydrogen(r, rc.nodes())
            p = remove_explicit_hydrogen(p, rc.nodes())
            r_mol = converter.graph_to_mol(r, sanitize=sanitize, use_h_count=True)
            p_mol = converter.graph_to_mol(p, sanitize=sanitize, use_h_count=True)
    else:
        r_mol = converter.graph_to_mol(r, sanitize=sanitize)
        p_mol = converter.graph_to_mol(p, sanitize=sanitize)

    # Convert RDKit Mol objects to SMILES and format them into a reaction SMILES string
    try:
        r_smiles = Chem.MolToSmiles(r_mol)
        p_smiles = Chem.MolToSmiles(p_mol)
        reaction_smiles = f"{r_smiles}>>{p_smiles}"
    except Exception as e:
        # Handle errors gracefully
        reaction_smiles = "Error in generating SMILES: " + str(e)

    return reaction_smiles


def smart_to_gml(
    smart: str,
    core: bool = True,
    sanitize: bool = False,
    rule_name: str = "rule",
    reindex: bool = True,
    explicit_hydrogen: bool = False,
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


    Returns:
    - str: The GML representation of the reaction.
    """
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
    ignore_hcount_inference: bool = False,
) -> str:
    """
    Converts a GML string back to a SMARTS string by interpreting the graph structures.

    Parameters:
    - gml (str): The GML string to convert.
    - sanitize (bool): Specifies whether the molecule should be sanitized upon conversion.
    - explicit_hydrogen (bool): Controls whether hydrogens are explicitly represented
    in the output.
    - ignore_hcount_inference (bool): If false, hydrogens counts are inferred
    from the ITS graph.

    Returns:
    - str: The corresponding SMARTS string.
    """
    r, p, rc = GMLToNX(gml).transform()
    return (
        graph_to_rsmi(r, p, rc, sanitize, explicit_hydrogen, ignore_hcount_inference),
        rc,
    )
