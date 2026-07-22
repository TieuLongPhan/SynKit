from __future__ import annotations

from typing import Literal, Optional, Pattern, Sequence
import re

import networkx as nx
from rdkit import Chem
from rdkit.Chem import rdChemReactions

from synkit.IO.debug import setup_logging
from synkit.IO.mol_to_graph import MolToGraph
from synkit.IO.graph_to_mol import GraphToMol
from synkit.IO.nx_to_gml import NXToGML
from synkit.IO.gml_to_nx import GMLToNX
from synkit.Graph.ITS.its_construction import ITSConstruction
from synkit.Graph.ITS.its_decompose import get_rc, its_decompose
from synkit.Graph.ITS.rc_extractor import RCExtractor
from synkit.Graph.ITS.its_reverter import ITSReverter

_BRACKET_DIGIT_PATTERN: Pattern[str] = re.compile(r"\[([^\]]*?)\](\d+)")
_BRACKET_MAP_PATTERN: Pattern[str] = re.compile(r"\[([^\]]+):(\d+)\]")

ITSFormat = Literal["typesGH", "tuple"]

DEFAULT_NODE_ATTRS = (
    "element",
    "aromatic",
    "hcount",
    "charge",
    "radical",
    "lone_pairs",
    "valence_electrons",
    "neighbors",
    "atom_map",
)

DEFAULT_EDGE_ATTRS = (
    "order",
    "kekule_order",
    "sigma_order",
    "pi_order",
    "aromatic",
)

logger = setup_logging()


def _resolve_graph_attrs(
    node_attrs: Optional[Sequence[str]] = None,
    edge_attrs: Optional[Sequence[str]] = None,
) -> tuple[list[str], list[str]]:
    """
    Resolve node and edge attribute lists.

    :param node_attrs: Optional node attribute names.
    :type node_attrs: Optional[Sequence[str]]
    :param edge_attrs: Optional edge attribute names.
    :type edge_attrs: Optional[Sequence[str]]
    :return: Normalized node and edge attribute lists.
    :rtype: tuple[list[str], list[str]]
    """
    resolved_node_attrs = list(node_attrs or DEFAULT_NODE_ATTRS)
    resolved_edge_attrs = list(edge_attrs or DEFAULT_EDGE_ATTRS)
    return resolved_node_attrs, resolved_edge_attrs


def _validate_its_format(format: str) -> ITSFormat:
    """
    Validate ITS serialization format.

    :param format: Requested ITS format.
    :type format: str
    :return: Validated ITS format.
    :rtype: ITSFormat
    :raises ValueError: If the format is unsupported.
    """
    valid_formats = ("typesGH", "tuple")
    if format not in valid_formats:
        raise ValueError(
            f"Unsupported format: {format!r}. " f"Expected one of {valid_formats}."
        )
    return format


def detect_its_format(graph: nx.Graph) -> ITSFormat:
    """
    Detect the ITS storage representation used by a graph.

    Legacy ITS graphs keep scalar node attributes and store side-specific
    values only in ``typesGH``. Tuple ITS graphs store direct paired node and
    edge attributes such as ``element=("C", "C")`` or
    ``sigma_order=(1.0, 1.0)``.

    :param graph: ITS-like graph to inspect.
    :type graph: nx.Graph
    :return: Detected ITS format.
    :rtype: ITSFormat
    """
    tuple_node_keys = (
        "element",
        "aromatic",
        "hcount",
        "charge",
        "radical",
        "lone_pairs",
        "valence_electrons",
    )
    tuple_edge_keys = ("kekule_order", "sigma_order", "pi_order")

    for _, attrs in graph.nodes(data=True):
        if any(_is_pair(attrs.get(key)) for key in tuple_node_keys):
            return "tuple"

    for _, _, attrs in graph.edges(data=True):
        if any(_is_pair(attrs.get(key)) for key in tuple_edge_keys):
            return "tuple"

    return "typesGH"


def _split_rsmi(rsmi: str) -> tuple[str, str]:
    """
    Split a reaction SMILES string into reactant and product parts.

    :param rsmi: Reaction SMILES string in ``reactants>>products`` format.
    :type rsmi: str
    :return: Tuple of reactant SMILES and product SMILES.
    :rtype: tuple[str, str]
    :raises ValueError: If the string is not a valid reaction SMILES.
    """
    parts = rsmi.split(">>")
    if len(parts) != 2:
        raise ValueError(f"Invalid RSMI format: {rsmi}")
    return parts[0], parts[1]


def _require_graph(graph: Optional[nx.Graph], label: str) -> nx.Graph:
    """
    Require a graph object to be present.

    :param graph: Graph to validate.
    :type graph: Optional[nx.Graph]
    :param label: Human-readable graph label.
    :type label: str
    :return: Validated graph.
    :rtype: nx.Graph
    :raises ValueError: If the graph is ``None``.
    """
    if graph is None:
        raise ValueError(f"Failed to build {label} graph.")
    return graph


def _is_pair(value: object) -> bool:
    """
    Check whether a value is a 2-item pair.

    :param value: Value to inspect.
    :type value: object
    :return: ``True`` if the value is a tuple or list of length 2.
    :rtype: bool
    """
    return isinstance(value, (tuple, list)) and len(value) == 2


def _extract_scalar(value: object) -> object:
    """
    Extract a representative scalar value.

    If the input is a paired value, the first entry is returned.

    :param value: Input value.
    :type value: object
    :return: Scalar-like representative value.
    :rtype: object
    """
    if _is_pair(value):
        return value[0]
    return value


def _extract_atom_map(attrs: dict) -> Optional[int]:
    """
    Extract atom-map value from a node attribute dictionary.

    :param attrs: Node attribute dictionary.
    :type attrs: dict
    :return: Atom-map number if available.
    :rtype: Optional[int]
    """
    value = _extract_scalar(attrs.get("atom_map"))
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_hydrogen_node(attrs: dict) -> bool:
    """
    Determine whether a node represents hydrogen.

    Supports both scalar and paired ``element`` attributes.

    :param attrs: Node attribute dictionary.
    :type attrs: dict
    :return: ``True`` if the node corresponds to hydrogen.
    :rtype: bool
    """
    value = attrs.get("element")
    if _is_pair(value):
        return any(item == "H" for item in value)
    return value == "H"


def _build_its_for_rsmi(
    reactant_graph: nx.Graph,
    product_graph: nx.Graph,
    format: ITSFormat,
) -> nx.Graph:
    """
    Build an ITS graph for downstream RSMI generation.

    :param reactant_graph: Reactant graph.
    :type reactant_graph: nx.Graph
    :param product_graph: Product graph.
    :type product_graph: nx.Graph
    :param format: ITS format.
    :type format: ITSFormat
    :return: Constructed ITS graph.
    :rtype: nx.Graph
    """
    if format == "typesGH":
        return ITSConstruction().ITSGraph(reactant_graph, product_graph)

    return ITSConstruction().construct(
        reactant_graph,
        product_graph,
        node_attrs=list(DEFAULT_NODE_ATTRS),
        edge_attrs=list(DEFAULT_EDGE_ATTRS),
    )


def _extract_rc_graph(its: nx.Graph, format: ITSFormat) -> nx.Graph:
    """
    Extract the reaction-center graph from an ITS graph.

    :param its: ITS graph.
    :type its: nx.Graph
    :param format: ITS format.
    :type format: ITSFormat
    :return: Reaction-center graph.
    :rtype: nx.Graph
    """
    if format == "typesGH":
        return get_rc(its)

    return RCExtractor(preserve_full_attrs=True).extract(its)


def _get_preserved_hydrogen_maps(
    its: nx.Graph,
    format: ITSFormat,
) -> list[int]:
    """
    Collect atom maps of RC hydrogens that should remain explicit.

    :param its: ITS graph.
    :type its: nx.Graph
    :param format: ITS format.
    :type format: ITSFormat
    :return: Sorted unique hydrogen atom-map numbers.
    :rtype: list[int]
    """
    rc_graph = _extract_rc_graph(its, format)
    atom_maps: set[int] = set()

    for _, attrs in rc_graph.nodes(data=True):
        if not _is_hydrogen_node(attrs):
            continue
        atom_map = _extract_atom_map(attrs)
        if atom_map is not None:
            atom_maps.add(atom_map)

    return sorted(atom_maps)


def _decompose_its(
    its: nx.Graph,
    format: ITSFormat,
) -> tuple[nx.Graph, nx.Graph]:
    """
    Convert an ITS graph back to reactant and product graphs.

    :param its: ITS graph.
    :type its: nx.Graph
    :param format: ITS format.
    :type format: ITSFormat
    :return: Reactant and product graphs.
    :rtype: tuple[nx.Graph, nx.Graph]
    """
    if format == "typesGH":
        return its_decompose(its)

    reverter = ITSReverter(its)
    return reverter.to_reactant_graph(), reverter.to_product_graph()


def smiles_to_graph(
    smiles: str,
    drop_non_aam: bool = False,
    sanitize: bool = True,
    use_index_as_atom_map: bool = False,
    node_attrs: Optional[Sequence[str]] = None,
    edge_attrs: Optional[Sequence[str]] = None,
    include_stereo_descriptors: bool = True,
) -> Optional[nx.Graph]:
    """
    Convert a SMILES string to a molecular graph.

    :param smiles: SMILES representation of the molecule.
    :type smiles: str
    :param drop_non_aam: If ``True``, drop atoms without atom-map labels.
    :type drop_non_aam: bool
    :param sanitize: If ``True``, sanitize the RDKit molecule.
    :type sanitize: bool
    :param use_index_as_atom_map: If ``True``, overwrite atom-map labels
        using atom indices.
    :type use_index_as_atom_map: bool
    :param node_attrs: Node attributes to export into the graph.
    :type node_attrs: Optional[Sequence[str]]
    :param edge_attrs: Edge attributes to export into the graph.
    :type edge_attrs: Optional[Sequence[str]]
    :param include_stereo_descriptors: Whether to construct the graph-level
        relative-stereochemistry registry.
    :type include_stereo_descriptors: bool
    :return: Molecular graph or ``None`` on failure.
    :rtype: Optional[nx.Graph]
    """
    resolved_node_attrs, resolved_edge_attrs = _resolve_graph_attrs(
        node_attrs=node_attrs,
        edge_attrs=edge_attrs,
    )

    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            logger.warning("Failed to parse SMILES: %s", smiles)
            return None

        if sanitize:
            try:
                Chem.SanitizeMol(mol)
            except Exception as exc:
                logger.error(
                    "Sanitization failed for SMILES %s: %s",
                    smiles,
                    exc,
                )
                return None

        graph = MolToGraph(
            node_attrs=resolved_node_attrs,
            edge_attrs=resolved_edge_attrs,
            include_stereo_descriptors=include_stereo_descriptors,
        ).transform(
            mol,
            drop_non_aam=drop_non_aam,
            use_index_as_atom_map=use_index_as_atom_map,
        )
        if graph is None:
            logger.warning(
                "Failed to convert molecule to graph for SMILES: %s",
                smiles,
            )
        return graph
    except Exception as exc:
        logger.error(
            "Unhandled exception while converting SMILES to graph: %s; %s",
            smiles,
            exc,
        )
        return None


def rsmi_to_graph(
    rsmi: str,
    drop_non_aam: bool = True,
    sanitize: bool = True,
    use_index_as_atom_map: bool = True,
    node_attrs: Optional[Sequence[str]] = None,
    edge_attrs: Optional[Sequence[str]] = None,
    include_stereo_descriptors: bool = True,
) -> tuple[Optional[nx.Graph], Optional[nx.Graph]]:
    """
    Convert a reaction SMILES into reactant and product graphs.

    :param rsmi: Reaction SMILES string in ``reactants>>products`` format.
    :type rsmi: str
    :param drop_non_aam: If ``True``, drop atoms lacking atom maps.
    :type drop_non_aam: bool
    :param sanitize: If ``True``, sanitize molecules during conversion.
    :type sanitize: bool
    :param use_index_as_atom_map: If ``True``, overwrite atom-map labels
        using atom indices.
    :type use_index_as_atom_map: bool
    :param node_attrs: Node attributes to export into the graphs.
    :type node_attrs: Optional[Sequence[str]]
    :param edge_attrs: Edge attributes to export into the graphs.
    :type edge_attrs: Optional[Sequence[str]]
    :param include_stereo_descriptors: Whether to construct graph-level
        relative-stereochemistry registries.
    :type include_stereo_descriptors: bool
    :return: Tuple of reactant and product graphs.
    :rtype: tuple[Optional[nx.Graph], Optional[nx.Graph]]
    """
    resolved_node_attrs, resolved_edge_attrs = _resolve_graph_attrs(
        node_attrs=node_attrs,
        edge_attrs=edge_attrs,
    )

    try:
        reactants_smiles, products_smiles = _split_rsmi(rsmi)
    except ValueError:
        logger.error("Invalid RSMI format: %s", rsmi)
        return None, None

    reactant_graph = smiles_to_graph(
        smiles=reactants_smiles,
        drop_non_aam=drop_non_aam,
        sanitize=sanitize,
        use_index_as_atom_map=use_index_as_atom_map,
        node_attrs=resolved_node_attrs,
        edge_attrs=resolved_edge_attrs,
        include_stereo_descriptors=include_stereo_descriptors,
    )
    product_graph = smiles_to_graph(
        smiles=products_smiles,
        drop_non_aam=drop_non_aam,
        sanitize=sanitize,
        use_index_as_atom_map=use_index_as_atom_map,
        node_attrs=resolved_node_attrs,
        edge_attrs=resolved_edge_attrs,
        include_stereo_descriptors=include_stereo_descriptors,
    )
    return reactant_graph, product_graph


def graph_to_smi(
    graph: nx.Graph,
    sanitize: bool = True,
    preserve_atom_maps: Optional[Sequence[int]] = None,
    prefer_kekule_order: bool = True,
) -> Optional[str]:
    """
    Convert a molecular graph to a SMILES string.

    :param graph: Molecular graph.
    :type graph: nx.Graph
    :param sanitize: If ``True``, sanitize the generated molecule.
    :type sanitize: bool
    :param preserve_atom_maps: Atom-map numbers whose hydrogens should
        remain explicit.
    :type preserve_atom_maps: Optional[Sequence[int]]
    :param prefer_kekule_order: Whether retained Kekule/sigma-pi edge fields
        should take precedence over aromatic presentation order.
    :type prefer_kekule_order: bool
    :return: SMILES string or ``None`` on failure.
    :rtype: Optional[str]
    """
    try:
        preserved = list(preserve_atom_maps or [])
        if not preserved:
            mol = GraphToMol().graph_to_mol(
                graph,
                sanitize=sanitize,
                use_h_count=True,
                prefer_kekule_order=prefer_kekule_order,
            )
        else:
            from synkit.Graph.Hyrogen._misc import implicit_hydrogen

            graph_imp = implicit_hydrogen(graph, set(preserved))
            mol = GraphToMol().graph_to_mol(
                graph_imp,
                sanitize=sanitize,
                use_h_count=True,
                prefer_kekule_order=prefer_kekule_order,
            )

        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception as exc:
        logger.debug("Error generating SMILES: %s", exc)
        return None


def graph_to_rsmi(
    r: nx.Graph,
    p: nx.Graph,
    its: Optional[nx.Graph] = None,
    sanitize: bool = True,
    explicit_hydrogen: bool = False,
    preserve_hydrogen_maps: Optional[Sequence[int]] = None,
) -> Optional[str]:
    """Convert reactant and product graphs into a reaction SMILES string.

    :param r: Graph representing the reactants.
    :type r: networkx.Graph
    :param p: Graph representing the products.
    :type p: networkx.Graph
    :param its: Imaginary transition state graph. If None, it will be
        constructed.
    :type its: networkx.Graph or None
    :param sanitize: Whether to sanitize molecules during conversion.
    :type sanitize: bool
    :param explicit_hydrogen: Whether to preserve explicit hydrogens in
        the SMILES.
    :type explicit_hydrogen: bool
    :param preserve_hydrogen_maps: Precomputed atom maps of reaction-centre
        hydrogens that must remain explicit. When omitted, derive them from
        ``its`` for backward compatibility.
    :type preserve_hydrogen_maps: Optional[Sequence[int]]
    :returns: Reaction SMILES string in 'reactants>>products' format or
        None on failure.
    :rtype: str or None
    """
    try:
        if explicit_hydrogen:
            r_smiles = graph_to_smi(r, sanitize=sanitize)
            p_smiles = graph_to_smi(p, sanitize=sanitize)
        else:
            if preserve_hydrogen_maps is None:
                if its is None:
                    its = ITSConstruction().ITSGraph(r, p)
                rc = get_rc(its)
                list_hydrogen = [
                    d["atom_map"]
                    for _, d in rc.nodes(data=True)
                    if d.get("element") == "H"
                ]
            else:
                list_hydrogen = list(preserve_hydrogen_maps)
            r_smiles = graph_to_smi(
                r,
                sanitize=sanitize,
                preserve_atom_maps=list_hydrogen,
            )
            p_smiles = graph_to_smi(
                p,
                sanitize=sanitize,
                preserve_atom_maps=list_hydrogen,
            )

        if r_smiles is None or p_smiles is None:
            return None

        return f"{r_smiles}>>{p_smiles}"
    except Exception as e:
        logger.debug(f"Error in generating reaction SMILES: {str(e)}")
        return None


def rsmi_to_its(
    rsmi: str,
    drop_non_aam: bool = True,
    sanitize: bool = True,
    use_index_as_atom_map: bool = True,
    core: bool = False,
    node_attrs: Optional[Sequence[str]] = None,
    edge_attrs: Optional[Sequence[str]] = None,
    explicit_hydrogen: bool = False,
    format: ITSFormat = "typesGH",
) -> nx.Graph:
    """
    Convert a reaction SMILES into an ITS graph.

    Supported formats:

    - ``"typesGH"``: legacy ITS representation
    - ``"tuple"``: paired-attribute ITS representation

    :param rsmi: Reaction SMILES string.
    :type rsmi: str
    :param drop_non_aam: If ``True``, discard fragments lacking atom maps.
    :type drop_non_aam: bool
    :param sanitize: If ``True``, sanitize molecules during conversion.
    :type sanitize: bool
    :param use_index_as_atom_map: If ``True``, overwrite atom maps using
        atom indices.
    :type use_index_as_atom_map: bool
    :param core: If ``True``, return only the reaction-center graph.
    :type core: bool
    :param node_attrs: Node attributes to include in graph construction.
    :type node_attrs: Optional[Sequence[str]]
    :param edge_attrs: Edge attributes to include in graph construction.
    :type edge_attrs: Optional[Sequence[str]]
    :param explicit_hydrogen: If ``True``, convert implicit hydrogens to
        explicit nodes for the selected ITS format.
    :type explicit_hydrogen: bool
    :param format: ITS format.
    :type format: ITSFormat
    :return: ITS graph or RC graph.
    :rtype: nx.Graph
    :raises ValueError: If graph construction fails.
    """
    validated_format = _validate_its_format(format)
    resolved_node_attrs, resolved_edge_attrs = _resolve_graph_attrs(
        node_attrs=node_attrs,
        edge_attrs=edge_attrs,
    )

    reactant_graph, product_graph = rsmi_to_graph(
        rsmi=rsmi,
        drop_non_aam=drop_non_aam,
        sanitize=sanitize,
        use_index_as_atom_map=use_index_as_atom_map,
        node_attrs=resolved_node_attrs,
        edge_attrs=resolved_edge_attrs,
    )

    reactant_graph = _require_graph(reactant_graph, "reactant")
    product_graph = _require_graph(product_graph, "product")

    if validated_format == "typesGH":
        its_graph = ITSConstruction().ITSGraph(reactant_graph, product_graph)
        if explicit_hydrogen:
            from synkit.Graph.Hyrogen._misc import h_to_explicit

            its_graph = h_to_explicit(its_graph, None, True)
        if core:
            its_graph = get_rc(its_graph)
        return its_graph

    its_graph = ITSConstruction().construct(
        reactant_graph,
        product_graph,
        node_attrs=resolved_node_attrs,
        edge_attrs=resolved_edge_attrs,
    )
    if explicit_hydrogen:
        from synkit.Graph.Hyrogen._misc import h_to_explicit

        its_graph = h_to_explicit(its_graph, None, True)
    if core:
        its_graph = RCExtractor(
            node_attrs=resolved_node_attrs,
            edge_attrs=resolved_edge_attrs,
            preserve_full_attrs=False,
        ).extract(its_graph)
    return its_graph


def its_to_rsmi(
    its: nx.Graph,
    sanitize: bool = True,
    explicit_hydrogen: bool = False,
    clean_wildcards: bool = False,
    format: ITSFormat = "typesGH",
) -> str:
    """
    Convert an ITS graph into a reaction SMILES (rSMI) string.

    This function decomposes or reverts the ITS graph into reactant and
    product graphs depending on the selected ITS format, then serializes
    them into a reaction SMILES string.

    :param its: ITS graph to convert back into reaction SMILES.
    :type its: nx.Graph
    :param sanitize: If ``True``, sanitize graphs before SMILES generation.
    :type sanitize: bool
    :param explicit_hydrogen: If ``True``, include explicit hydrogens in
        the generated SMILES.
    :type explicit_hydrogen: bool
    :param clean_wildcards: If ``True``, clean wildcard radicals in the
        generated reaction SMILES.
    :type clean_wildcards: bool
    :param format: ITS format. Supported values are ``"typesGH"`` and
        ``"tuple"``.
    :type format: ITSFormat
    :return: Reaction SMILES string.
    :rtype: str
    :raises ValueError: If the ITS format is unsupported.
    """
    validated_format = _validate_its_format(format)
    reactant_graph, product_graph = _decompose_its(its, validated_format)

    if validated_format == "tuple":
        preserved_hydrogens = (
            []
            if explicit_hydrogen
            else _get_preserved_hydrogen_maps(its, validated_format)
        )
        reactant_smiles = graph_to_smi(
            reactant_graph,
            sanitize=sanitize,
            preserve_atom_maps=preserved_hydrogens,
        )
        try:
            if explicit_hydrogen:
                product = product_graph
            else:
                from synkit.Graph.Hyrogen._misc import implicit_hydrogen

                product = implicit_hydrogen(
                    product_graph,
                    set(preserved_hydrogens),
                )
            product_smiles = graph_to_smi(
                product,
                sanitize=sanitize,
                preserve_atom_maps=preserved_hydrogens,
            )
        except Exception as exc:
            logger.debug("Error generating tuple product SMILES: %s", exc)
            product_smiles = None
        rsmi = (
            f"{reactant_smiles}>>{product_smiles}"
            if reactant_smiles is not None and product_smiles is not None
            else None
        )
    else:
        rsmi = graph_to_rsmi(
            reactant_graph,
            product_graph,
            its,
            sanitize,
            explicit_hydrogen,
        )

    if rsmi is None:
        raise ValueError("Failed to convert ITS graph to reaction SMILES.")

    if clean_wildcards:
        from synkit.Chem.Reaction.radical_wildcard import clean_wc

        rsmi = clean_wc(rsmi)

    return rsmi


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
    Convert a reaction SMARTS or SMILES string into GML.

    This function uses the legacy ITS/GML pipeline.

    :param smart: Reaction SMARTS or SMILES string.
    :type smart: str
    :param core: If ``True``, export only the reaction core.
    :type core: bool
    :param sanitize: If ``True``, sanitize molecules during conversion.
    :type sanitize: bool
    :param rule_name: Rule name stored in the GML output.
    :type rule_name: str
    :param reindex: If ``True``, reindex graph nodes before export.
    :type reindex: bool
    :param explicit_hydrogen: If ``True``, include explicit hydrogens.
    :type explicit_hydrogen: bool
    :param useSmiles: If ``True``, treat input as reaction SMILES.
        Otherwise, treat it as reaction SMARTS.
    :type useSmiles: bool
    :return: GML representation of the reaction rule.
    :rtype: str
    :raises ValueError: If graph construction fails.
    """
    rsmi = smart if useSmiles else rsmarts_to_rsmi(smart)

    reactant_graph, product_graph = rsmi_to_graph(
        rsmi=rsmi,
        sanitize=sanitize,
    )
    reactant_graph = _require_graph(reactant_graph, "reactant")
    product_graph = _require_graph(product_graph, "product")

    its_graph = ITSConstruction().ITSGraph(reactant_graph, product_graph)
    if core:
        its_graph = get_rc(its_graph)
        reactant_graph, product_graph = its_decompose(its_graph)

    return NXToGML().transform(
        (reactant_graph, product_graph, its_graph),
        reindex=reindex,
        rule_name=rule_name,
        explicit_hydrogen=explicit_hydrogen,
    )


def gml_to_smart(
    gml: str,
    sanitize: bool = True,
    explicit_hydrogen: bool = False,
    useSmiles: bool = True,
) -> str:
    """
    Convert a GML reaction rule back to reaction SMILES or SMARTS.

    :param gml: GML string.
    :type gml: str
    :param sanitize: If ``True``, sanitize during SMILES generation.
    :type sanitize: bool
    :param explicit_hydrogen: If ``True``, keep explicit hydrogens.
    :type explicit_hydrogen: bool
    :param useSmiles: If ``True``, return reaction SMILES. Otherwise,
        return reaction SMARTS.
    :type useSmiles: bool
    :return: Reaction SMILES or SMARTS string.
    :rtype: str
    :raises ValueError: If conversion fails.
    """
    reactant_graph, product_graph, rc_graph = GMLToNX(gml).transform()
    rsmi = graph_to_rsmi(
        r=reactant_graph,
        p=product_graph,
        its=rc_graph,
        sanitize=sanitize,
        explicit_hydrogen=explicit_hydrogen,
    )
    if rsmi is None:
        raise ValueError("Failed to convert GML to reaction SMILES.")

    if useSmiles:
        return rsmi
    return rsmi_to_rsmarts(rsmi)


def its_to_gml(
    its: nx.Graph,
    core: bool = True,
    rule_name: str = "rule",
    reindex: bool = True,
    explicit_hydrogen: bool = False,
    format: ITSFormat = "typesGH",
) -> str:
    """
    Convert an ITS graph to GML format.

    :param its: ITS graph.
    :type its: nx.Graph
    :param core: If ``True``, export the reaction-center graph.
    :type core: bool
    :param rule_name: Rule name stored in the GML output.
    :type rule_name: str
    :param reindex: If ``True``, reindex graph nodes before export.
    :type reindex: bool
    :param explicit_hydrogen: If ``True``, include explicit hydrogens.
    :type explicit_hydrogen: bool
    :param format: ITS format.
    :type format: ITSFormat
    :return: GML representation of the reaction.
    :rtype: str
    """
    validated_format = _validate_its_format(format)
    exported_its = _extract_rc_graph(its, validated_format) if core else its
    reactant_graph, product_graph = _decompose_its(exported_its, validated_format)

    return NXToGML().transform(
        (reactant_graph, product_graph, exported_its),
        reindex=reindex,
        rule_name=rule_name,
        explicit_hydrogen=explicit_hydrogen,
    )


def gml_to_its(gml: str) -> nx.Graph:
    """
    Convert a GML reaction rule back into an ITS graph.

    :param gml: GML string.
    :type gml: str
    :return: ITS graph.
    :rtype: nx.Graph
    """
    _, _, its_graph = GMLToNX(gml).transform()
    return its_graph


def rsmi_to_rsmarts(rsmi: str) -> str:
    """
    Convert mapped reaction SMILES to reaction SMARTS.

    :param rsmi: Reaction SMILES input.
    :type rsmi: str
    :return: Reaction SMARTS string.
    :rtype: str
    :raises ValueError: If conversion fails.
    """
    try:
        rxn = rdChemReactions.ReactionFromSmarts(rsmi, useSmiles=True)
        return rdChemReactions.ReactionToSmarts(rxn)
    except Exception as exc:
        raise ValueError(f"Failed to convert RSMI to RSMARTS: {exc}") from exc


def rsmarts_to_rsmi(rsmarts: str) -> str:
    """
    Convert reaction SMARTS to reaction SMILES.

    :param rsmarts: Reaction SMARTS input.
    :type rsmarts: str
    :return: Reaction SMILES string.
    :rtype: str
    :raises ValueError: If conversion fails.
    """
    try:
        rxn = rdChemReactions.ReactionFromSmarts(rsmarts, useSmiles=False)
        return rdChemReactions.ReactionToSmiles(rxn)
    except Exception as exc:
        raise ValueError(f"Failed to convert RSMARTS to RSMI: {exc}") from exc


def dfs_to_smiles(dfs: str, keep_map: bool = True) -> str:
    """
    Convert DFS-style annotated SMILES to normal SMILES form.

    Rules:
    - Replace ``[]`` with ``[*]``.
    - Convert bracketed tokens followed by digits, such as ``[H]12``,
      into atom-mapped tokens ``[H:12]`` when ``keep_map=True``.
    - If ``keep_map=False``, remove trailing digits instead.
    - Tokens already containing ``:`` inside brackets are preserved.

    :param dfs: DFS-style SMILES or reaction SMILES.
    :type dfs: str
    :param keep_map: Whether to keep atom maps.
    :type keep_map: bool
    :return: Converted SMILES string.
    :rtype: str
    """
    if not isinstance(dfs, str):
        raise ValueError("dfs must be a string")

    normalized = dfs.replace("[]", "[*]")

    def _replace(match: re.Match[str]) -> str:
        inner = match.group(1)
        digits = match.group(2)

        if ":" in inner:
            return match.group(0)
        if inner == "":
            inner = "*"

        if keep_map:
            return f"[{inner}:{digits}]"
        return f"[{inner}]"

    return _BRACKET_DIGIT_PATTERN.sub(_replace, normalized)


def smiles_to_dfs(smiles: str) -> str:
    """
    Convert SMILES with atom maps into DFS-style notation.

    Rules:
    - ``[X:123]`` becomes ``[X]123``
    - ``[*:3]`` becomes ``[]3``
    - unmapped tokens remain unchanged
    - remaining ``[*]`` is normalized back to ``[]``

    :param smiles: SMILES or reaction SMILES.
    :type smiles: str
    :return: DFS-style string.
    :rtype: str
    """
    if not isinstance(smiles, str):
        raise ValueError("smiles must be a string")

    def _replace(match: re.Match[str]) -> str:
        inner = match.group(1)
        number = match.group(2)
        if inner == "*":
            return f"[]{number}"
        return f"[{inner}]{number}"

    dfs = _BRACKET_MAP_PATTERN.sub(_replace, smiles)
    return dfs.replace("[*]", "[]")


def normalize_dfs_for_compare(dfs: str) -> str:
    """
    Normalize DFS-style strings for comparison.

    :param dfs: DFS-style string.
    :type dfs: str
    :return: Normalized comparison string.
    :rtype: str
    """
    if not isinstance(dfs, str):
        raise ValueError("dfs must be a string")

    normalized = dfs.replace("[*]", "[]")
    return re.sub(r"\s+", "", normalized)
