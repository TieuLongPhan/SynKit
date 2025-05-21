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
from synkit.Graph.Hyrogen._misc import implicit_hydrogen, h_to_explicit


logger = setup_logging()


def smiles_to_graph(
    smiles: str,
    drop_non_aam: bool = False,
    sanitize: bool = True,
    use_index_as_atom_map: bool = False,
    node_attrs: Optional[List[str]] = [
        "element",
        "aromatic",
        "hcount",
        "charge",
        "neighbors",
        "atom_map",
    ],
    edge_attrs: Optional[List[str]] = ["order"],
) -> Optional[nx.Graph]:
    """
    Helper function to convert a SMILES string to a NetworkX graph.

    :param smiles: SMILES representation of the molecule.
    :type smiles: str
    :param drop_non_aam: Whether to drop nodes without atom mapping numbers.
    :type drop_non_aam: bool
    :param light_weight: Whether to create a light-weight graph.
    :type light_weight: bool
    :param sanitize: Whether to sanitize the molecule during conversion.
    :type sanitize: bool
    :param use_index_as_atom_map: Whether to use atom indices as atom-map numbers.
    :type use_index_as_atom_map: bool
    :returns: The NetworkX graph representation, or None if conversion fails.
    :rtype: networkx.Graph or None
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
        graph_converter = MolToGraph(node_attrs=node_attrs, edge_attrs=edge_attrs)
        graph = graph_converter.transform(
            mol, drop_non_aam=drop_non_aam, use_index_as_atom_map=use_index_as_atom_map
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
    sanitize: bool = True,
    use_index_as_atom_map: bool = True,
    node_attrs: Optional[List[str]] = [
        "element",
        "aromatic",
        "hcount",
        "charge",
        "neighbors",
        "atom_map",
    ],
    edge_attrs: Optional[List[str]] = ["order"],
    explicit_hydrogen: bool = False,
) -> Tuple[Optional[nx.Graph], Optional[nx.Graph]]:
    """
    Convert a reaction SMILES (RSMI) into reactant and product graphs.

    :param rsmi: Reaction SMILES string in “reactants>>products” format.
    :type rsmi: str
    :param drop_non_aam: If True, drop nodes without atom mapping numbers.
    :type drop_non_aam: bool
    :param light_weight: If True, create a light-weight graph.
    :type light_weight: bool
    :param sanitize: If True, sanitize molecules during conversion.
    :type sanitize: bool
    :param use_index_as_atom_map: Whether to use atom indices as atom-map numbers.
    :type use_index_as_atom_map: bool
    :returns: A tuple `(reactant_graph, product_graph)`, each a NetworkX graph or None.
    :rtype: tuple of (networkx.Graph or None, networkx.Graph or None)
    """
    try:
        reactants_smiles, products_smiles = rsmi.split(">>")
        r_graph = smiles_to_graph(
            reactants_smiles,
            drop_non_aam,
            sanitize,
            use_index_as_atom_map,
            node_attrs,
            edge_attrs,
        )
        p_graph = smiles_to_graph(
            products_smiles,
            drop_non_aam,
            sanitize,
            use_index_as_atom_map,
            node_attrs,
            edge_attrs,
        )
        if explicit_hydrogen:
            r_graph = h_to_explicit(r_graph)
            p_graph = h_to_explicit(p_graph)
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

    :param graph: Graph representation of the molecule.
                  Nodes must carry chemical attributes (e.g. ‘element’, atom maps).
    :type graph: networkx.Graph
    :param sanitize: Whether to perform RDKit sanitization on the resulting molecule.
    :type sanitize: bool
    :param preserve_atom_maps: List of atom-map numbers for which hydrogens remain explicit.
    :type preserve_atom_maps: list of int or None
    :returns: SMILES string, or None if conversion fails.
    :rtype: str or None
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

    :param r: Graph representing the reactants.
    :type r: networkx.Graph
    :param p: Graph representing the products.
    :type p: networkx.Graph
    :param its: Imaginary transition state graph. If None, it will be constructed.
    :type its: networkx.Graph or None
    :param sanitize: Whether to sanitize molecules during conversion.
    :type sanitize: bool
    :param explicit_hydrogen: Whether to preserve explicit hydrogens in the SMILES.
    :type explicit_hydrogen: bool
    :returns: Reaction SMILES string in 'reactants>>products' format or None on failure.
    :rtype: str or None
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
    sanitize: bool = True,
    rule_name: str = "rule",
    reindex: bool = False,
    explicit_hydrogen: bool = False,
    useSmiles: bool = True,
) -> str:
    """
    Convert a reaction SMARTS (or SMILES) template into a GML‐encoded DPO rule.

    :param smart: The reaction SMARTS or SMILES string.
    :type smart: str
    :param core: If True, include only the reaction core in the GML. Defaults to True.
    :type core: bool
    :param sanitize: If True, sanitize molecules during conversion. Defaults to True.
    :type sanitize: bool
    :param rule_name: Identifier for the output rule. Defaults to "rule".
    :type rule_name: str
    :param reindex: If True, reindex graph nodes before exporting. Defaults to False.
    :type reindex: bool
    :param explicit_hydrogen: If True, include explicit hydrogen atoms. Defaults to False.
    :type explicit_hydrogen: bool
    :param useSmiles: If True, treat input as SMILES; if False, as SMARTS. Defaults to True.
    :type useSmiles: bool
    :returns: The GML representation of the reaction rule.
    :rtype: str
    """
    if useSmiles is False:
        smart = rsmarts_to_rsmi(smart)
    r, p = rsmi_to_graph(smart, sanitize=sanitize)
    its = ITSConstruction().ITSGraph(r, p)
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
    Convert a GML string back to a SMARTS string and ITS graph.

    :param gml: The GML string to convert.
    :type gml: str
    :param sanitize: Whether to sanitize molecules upon conversion.
    :type sanitize: bool
    :param explicit_hydrogen: Whether hydrogens are explicitly represented.
    :type explicit_hydrogen: bool
    :param useSmiles: If True, output SMILES; otherwise SMARTS.
    :type useSmiles: bool
    :returns: A tuple of (SMARTS string, ITS graph).
    :rtype: tuple of (str, networkx.Graph)
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
    Convert an ITS graph (reaction graph) to GML format.

    :param its: The input ITS graph representing the reaction.
    :type its: networkx.Graph
    :param core: If True, focus only on the reaction center. Defaults to True.
    :type core: bool
    :param rule_name: Name of the reaction rule. Defaults to "rule".
    :type rule_name: str
    :param reindex: If True, reindex graph nodes. Defaults to True.
    :type reindex: bool
    :param explicit_hydrogen: If True, include explicit hydrogens. Defaults to False.
    :type explicit_hydrogen: bool
    :returns: The GML representation of the ITS graph.
    :rtype: str
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
    Convert a GML string representation of a reaction back into an ITS graph.

    :param gml: The GML string representing the reaction.
    :type gml: str
    :returns: The resulting ITS graph.
    :rtype: networkx.Graph
    """

    # Convert GML back to the ITS graph using the appropriate GML to NX conversion
    _, _, its = GMLToNX(gml).transform()

    return its


def rsmi_to_its(
    rsmi: str,
    drop_non_aam: bool = True,
    sanitize: bool = True,
    use_index_as_atom_map: bool = True,
    core: bool = False,
    node_attrs: Optional[List[str]] = [
        "element",
        "aromatic",
        "hcount",
        "charge",
        "neighbors",
        "atom_map",
    ],
    edge_attrs: Optional[List[str]] = ["order"],
    explicit_hydrogen: bool = False,
) -> nx.Graph:
    """
    Convert a reaction SMILES (rSMI) string to an ITS graph representation.

    :param rsmi: The reaction SMILES string.
    :type rsmi: str
    :param drop_non_aam: If True, drop non-atom-mapped components.
    :type drop_non_aam: bool
    :param light_weight: If True, create a light-weight graph.
    :type light_weight: bool
    :param sanitize: If True, sanitize molecules during conversion.
    :type sanitize: bool
    :param use_index_as_atom_map: If True, use indices as atom-map numbers.
    :type use_index_as_atom_map: bool
    :param core: If True, focus only on the reaction center.
    :type core: bool
    :returns: The ITS graph representing the reaction.
    :rtype: networkx.Graph
    :raises Exception: If conversion fails.
    """
    r, p = rsmi_to_graph(
        rsmi, drop_non_aam, sanitize, use_index_as_atom_map, node_attrs, edge_attrs, explicit_hydrogen
    )
    its = ITSConstruction().ITSGraph(r, p)
    if core:
        its = get_rc(its)
    return its


def its_to_rsmi(
    its: nx.Graph,
    sanitize: bool = True,
    explicit_hydrogen: bool = False,
) -> str:
    """
    Convert an ITS graph into a reaction SMILES (rSMI) string.

    :param its: A fully annotated ITS graph (nodes with atom-map attributes).
    :type its: networkx.Graph
    :param sanitize: If True, sanitize prior to SMILES generation.
    :type sanitize: bool
    :param explicit_hydrogen: If True, include explicit hydrogens.
    :type explicit_hydrogen: bool
    :returns: A canonical reaction-SMILES string ('reactants>agents>products').
    :rtype: str
    :raises ValueError: If graph cannot be decomposed or sanitisation fails.
    """
    r, p = its_decompose(its)
    return graph_to_rsmi(r, p, its, sanitize, explicit_hydrogen)


def rsmi_to_rsmarts(rsmi: str) -> str:
    """
    Convert a mapped reaction SMILES to a reaction SMARTS string.

    :param rsmi: Reaction SMILES input.
    :type rsmi: str
    :returns: Reaction SMARTS string.
    :rtype: str
    :raises ValueError: If conversion fails.
    """
    try:
        rxn = rdChemReactions.ReactionFromSmarts(rsmi, useSmiles=True)
        return rdChemReactions.ReactionToSmarts(rxn)
    except Exception as e:
        raise ValueError(f"Failed to convert RSMI to RSMARTS: {e}")


def rsmarts_to_rsmi(rsmarts: str) -> str:
    """
    Convert a reaction SMARTS to a reaction SMILES string.

    :param rsmarts: Reaction SMARTS input.
    :type rsmarts: str
    :returns: Reaction SMILES string.
    :rtype: str
    :raises ValueError: If conversion fails.
    """
    try:
        rxn = rdChemReactions.ReactionFromSmarts(rsmarts, useSmiles=False)
        return rdChemReactions.ReactionToSmiles(rxn)
    except Exception as e:
        raise ValueError(f"Failed to convert RSMARTS to RSMI: {e}")
