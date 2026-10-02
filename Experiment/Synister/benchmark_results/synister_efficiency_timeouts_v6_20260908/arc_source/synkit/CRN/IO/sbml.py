"""SBML import and export for :class:`~synkit.CRN.Structure.syncrn.SynCRN`.

SBML (Systems Biology Markup Language) is the interchange format of the
existing CRN ecosystem — ``crnpy``, CRNT4SBML, CoNtRol, COPASI and the BioModels
repository all read or write it. This module is the adapter, so a network built
by rule expansion here can be analysed there and vice versa.

The reader and writer are self-contained: they use only
:mod:`xml.etree.ElementTree` and target **SBML Level 3 Version 2 core**, which
is all a reaction network needs (compartments, species, reactions with
stoichiometry). No kinetic law is written, because a ``SynCRN`` carries none;
structural analysis does not need one. Files that *do* carry kinetic laws are
read fine — the laws are simply ignored.

Two conversion choices are worth knowing:

- **Reversible reactions.** SBML marks a reaction ``reversible="true"``; a
  ``SynCRN`` reaction is always directed. On import, a reversible reaction is
  split into a forward and a backward reaction by default (``expand_reversible``),
  which is what Chemical Reaction Network Theory expects.
- **Identifiers.** SBML ids must match ``[A-Za-z_][A-Za-z0-9_]*``. Species
  labels are chemistry — SMILES strings, KEGG ids — and usually do not. The
  writer therefore uses the internal ``SynCRN`` ids (``s_1``, ``r_1``) as SBML
  ids and keeps the chemistry in the ``name`` attribute and a SynKit
  annotation, so nothing is lost on a round-trip.

.. rubric:: Example

.. code-block:: python

    from synkit.CRN import SynCRN
    from synkit.CRN.IO import crn_from_sbml, crn_to_sbml

    crn = SynCRN.from_reaction_strings(["2A>>B", "B>>2A"])
    xml = crn_to_sbml(crn)
    back = crn_from_sbml(xml)
    print(back.to_equations(species="label", include_id=False))
"""

from __future__ import annotations

import re
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx

from ..Structure.syncrn import SynCRN

__all__ = [
    "SBML_NS",
    "SYNKIT_NS",
    "crn_from_sbml",
    "crn_to_sbml",
    "read_sbml",
    "write_sbml",
]

#: Namespace of the SBML Level 3 Version 2 core specification.
SBML_NS = "http://www.sbml.org/sbml/level3/version2/core"

#: Namespace used for SynKit-specific annotations (SMILES, source ids).
SYNKIT_NS = "https://github.com/TieuLongPhan/SynKit"

_SID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SID_INVALID_RE = re.compile(r"[^A-Za-z0-9_]")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _localname(tag: str) -> str:
    """Strip the XML namespace from a tag.

    SBML files in the wild use several namespace URIs across levels and
    versions; matching on the local name keeps the reader version-tolerant.

    :param tag:
        Fully qualified tag such as ``"{ns}species"``.
    :type tag: str

    :return:
        Local tag name.
    :rtype: str
    """
    return tag.rsplit("}", 1)[-1]


def _find_children(element: ET.Element, name: str) -> List[ET.Element]:
    """Find the direct children with a given local name.

    Direct children only: an ``<annotation>`` block may legitimately contain
    elements whose local name collides with an SBML one, and those must not be
    mistaken for model content.

    :param element:
        Parent element.
    :type element: xml.etree.ElementTree.Element

    :param name:
        Local tag name to match.
    :type name: str

    :return:
        Matching children in document order.
    :rtype: List[xml.etree.ElementTree.Element]
    """
    return [child for child in element if _localname(child.tag) == name]


def _find_descendants(element: ET.Element, name: str) -> List[ET.Element]:
    """Find all descendants with a given local name.

    :param element:
        Element to search below.
    :type element: xml.etree.ElementTree.Element

    :param name:
        Local tag name to match.
    :type name: str

    :return:
        Matching descendants in document order.
    :rtype: List[xml.etree.ElementTree.Element]
    """
    return [child for child in element.iter() if _localname(child.tag) == name]


def _find_child(element: ET.Element, name: str) -> Optional[ET.Element]:
    """Find the first direct child with a given local name.

    :param element:
        Parent element.
    :type element: xml.etree.ElementTree.Element

    :param name:
        Local tag name to match.
    :type name: str

    :return:
        The child, or ``None`` when absent.
    :rtype: Optional[xml.etree.ElementTree.Element]
    """
    for child in element:
        if _localname(child.tag) == name:
            return child
    return None


def _as_sid(value: Any, *, fallback: str) -> str:
    """Coerce a value into a valid SBML ``SId``.

    :param value:
        Candidate identifier.
    :type value: Any

    :param fallback:
        Identifier used when ``value`` cannot be sanitized into a valid SId.
    :type fallback: str

    :return:
        Valid SBML identifier.
    :rtype: str
    """
    text = str(value)
    if _SID_RE.match(text):
        return text

    cleaned = _SID_INVALID_RE.sub("_", text)
    if cleaned and not cleaned[0].isalpha() and cleaned[0] != "_":
        cleaned = f"_{cleaned}"
    return cleaned if _SID_RE.match(cleaned) else fallback


def _format_stoich(value: Any) -> str:
    """Render a stoichiometric coefficient for an SBML attribute.

    :param value:
        Coefficient value.
    :type value: Any

    :return:
        String form without a redundant ``.0`` suffix.
    :rtype: str
    """
    number = float(value)
    return str(int(number)) if number.is_integer() else repr(number)


def _parse_stoich(value: Optional[str]) -> float:
    """Parse an SBML stoichiometry attribute, defaulting to 1.

    :param value:
        Raw attribute text, possibly ``None``.
    :type value: Optional[str]

    :return:
        Stoichiometric coefficient.
    :rtype: float
    """
    if value is None or value == "":
        return 1.0
    try:
        return float(value)
    except ValueError:
        return 1.0


def _is_true(value: Optional[str]) -> bool:
    """Interpret an SBML boolean attribute.

    :param value:
        Raw attribute text, possibly ``None``.
    :type value: Optional[str]

    :return:
        ``True`` when the attribute spells a true value.
    :rtype: bool
    """
    return str(value).strip().lower() in {"true", "1"}


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def _species_element(
    parent: ET.Element,
    *,
    sid: str,
    species: Any,
    compartment_id: str,
) -> None:
    """Append one ``<species>`` element for a SynCRN species record.

    :param parent:
        The ``<listOfSpecies>`` element.
    :type parent: xml.etree.ElementTree.Element

    :param sid:
        SBML species id.
    :type sid: str

    :param species:
        SynCRN species record.
    :type species: Any

    :param compartment_id:
        Id of the compartment the species belongs to.
    :type compartment_id: str

    :return:
        ``None``. The element is appended in place.
    :rtype: None
    """
    element = ET.SubElement(
        parent,
        "species",
        {
            "id": sid,
            "compartment": compartment_id,
            "hasOnlySubstanceUnits": "true",
            "boundaryCondition": "false",
            "constant": "false",
        },
    )
    if species.label is not None:
        element.set("name", str(species.label))

    extra = {
        key: value
        for key, value in (
            ("smiles", species.smiles),
            ("sourceNodeId", species.source_node_id),
            ("syncrnId", species.id),
        )
        if value is not None
    }
    if extra:
        annotation = ET.SubElement(element, "annotation")
        ET.SubElement(
            annotation,
            f"{{{SYNKIT_NS}}}speciesInfo",
            {key: str(value) for key, value in extra.items()},
        )


def _reaction_element(
    parent: ET.Element,
    *,
    rid: str,
    reaction: Any,
    species_sids: Dict[str, str],
) -> None:
    """Append one ``<reaction>`` element for a SynCRN reaction record.

    :param parent:
        The ``<listOfReactions>`` element.
    :type parent: xml.etree.ElementTree.Element

    :param rid:
        SBML reaction id.
    :type rid: str

    :param reaction:
        SynCRN reaction record.
    :type reaction: Any

    :param species_sids:
        Mapping from internal species id to SBML species id.
    :type species_sids: Dict[str, str]

    :return:
        ``None``. The element is appended in place.
    :rtype: None
    """
    element = ET.SubElement(
        parent,
        "reaction",
        {"id": rid, "reversible": "false"},
    )
    if reaction.label is not None:
        element.set("name", str(reaction.label))

    for side, container, tag in (
        (reaction.lhs, "listOfReactants", "speciesReference"),
        (reaction.rhs, "listOfProducts", "speciesReference"),
    ):
        items = list(side.items())
        if not items:
            continue
        listing = ET.SubElement(element, container)
        for internal_sid, coeff in items:
            ET.SubElement(
                listing,
                tag,
                {
                    "species": species_sids[internal_sid],
                    "stoichiometry": _format_stoich(coeff),
                    "constant": "true",
                },
            )

    extra = {
        key: value
        for key, value in (
            ("syncrnId", reaction.id),
            ("sourceNodeId", reaction.source_node_id),
            ("sourceKind", reaction.source_kind),
            ("ruleId", reaction.rule_id),
            ("ruleRepr", reaction.rule_repr),
        )
        if value is not None
    }
    if extra:
        annotation = ET.SubElement(element, "annotation")
        ET.SubElement(
            annotation,
            f"{{{SYNKIT_NS}}}reactionInfo",
            {key: str(value) for key, value in extra.items()},
        )


def crn_to_sbml(
    crn: SynCRN,
    *,
    model_id: str = "synkit_crn",
    model_name: Optional[str] = None,
    compartment_id: str = "cell",
    pretty: bool = True,
) -> str:
    """Serialize a :class:`SynCRN` to an SBML Level 3 Version 2 document.

    Species and reactions keep their internal ids as SBML ids and their labels
    as SBML names; SMILES and provenance travel in a SynKit annotation, so
    :func:`crn_from_sbml` reproduces the network exactly. No kinetic law is
    emitted — a ``SynCRN`` carries none, and structural analysis needs none.

    :param crn:
        Network to serialize.
    :type crn: SynCRN

    :param model_id:
        SBML model identifier.
    :type model_id: str

    :param model_name:
        Optional human-readable model name.
    :type model_name: Optional[str]

    :param compartment_id:
        Identifier of the single compartment every species is placed in.
    :type compartment_id: str

    :param pretty:
        Whether to indent the output for readability.
    :type pretty: bool

    :return:
        SBML document as a UTF-8 string.
    :rtype: str

    :raises TypeError:
        If ``crn`` is not a :class:`SynCRN`.

    .. rubric:: Example

    .. code-block:: python

        xml = crn_to_sbml(SynCRN.from_reaction_strings(["2A>>B"]))
        print("<reaction" in xml)
        # True
    """
    if not isinstance(crn, SynCRN):
        raise TypeError(f"crn must be a SynCRN, got {type(crn).__name__}")

    root = ET.Element(
        f"{{{SBML_NS}}}sbml",
        {"level": "3", "version": "2"},
    )
    model_attrs = {"id": _as_sid(model_id, fallback="synkit_crn")}
    if model_name is not None:
        model_attrs["name"] = model_name
    model = ET.SubElement(root, "model", model_attrs)

    compartment_sid = _as_sid(compartment_id, fallback="cell")
    compartments = ET.SubElement(model, "listOfCompartments")
    ET.SubElement(
        compartments,
        "compartment",
        {"id": compartment_sid, "spatialDimensions": "3", "constant": "true"},
    )

    species_sids: Dict[str, str] = {}
    if crn.species:
        listing = ET.SubElement(model, "listOfSpecies")
        for i, (internal_sid, species) in enumerate(crn.species.items(), start=1):
            sid = _as_sid(internal_sid, fallback=f"s_{i}")
            species_sids[internal_sid] = sid
            _species_element(
                listing,
                sid=sid,
                species=species,
                compartment_id=compartment_sid,
            )

    if crn.reactions:
        listing = ET.SubElement(model, "listOfReactions")
        for j, (internal_rid, reaction) in enumerate(crn.reactions.items(), start=1):
            _reaction_element(
                listing,
                rid=_as_sid(internal_rid, fallback=f"r_{j}"),
                reaction=reaction,
                species_sids=species_sids,
            )

    if pretty:
        ET.indent(root, space="  ")

    ET.register_namespace("", SBML_NS)
    ET.register_namespace("synkit", SYNKIT_NS)
    body = ET.tostring(root, encoding="unicode")
    return f'<?xml version="1.0" encoding="UTF-8"?>\n{body}\n'


def write_sbml(crn: SynCRN, path: Union[str, Path], **kwargs: Any) -> Path:
    """Write a :class:`SynCRN` to an SBML file.

    :param crn:
        Network to serialize.
    :type crn: SynCRN

    :param path:
        Destination file path.
    :type path: Union[str, Path]

    :param kwargs:
        Extra keyword arguments forwarded to :func:`crn_to_sbml`.
    :type kwargs: Any

    :return:
        The path written to.
    :rtype: pathlib.Path

    .. rubric:: Example

    .. code-block:: python

        write_sbml(crn, "network.xml")
    """
    destination = Path(path)
    destination.write_text(crn_to_sbml(crn, **kwargs), encoding="utf-8")
    return destination


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


def _annotation_values(element: ET.Element, name: str) -> Dict[str, str]:
    """Read the attributes of a SynKit annotation child, if present.

    :param element:
        Species or reaction element.
    :type element: xml.etree.ElementTree.Element

    :param name:
        Local name of the SynKit annotation element.
    :type name: str

    :return:
        Annotation attributes, empty when the annotation is absent.
    :rtype: Dict[str, str]
    """
    annotation = _find_child(element, "annotation")
    if annotation is None:
        return {}
    for child in annotation.iter():
        if _localname(child.tag) == name:
            return dict(child.attrib)
    return {}


def _read_species(
    model: ET.Element,
    *,
    drop_boundary_species: bool,
) -> Tuple[Dict[str, Dict[str, Any]], set]:
    """Collect species definitions from an SBML model element.

    :param model:
        The ``<model>`` element.
    :type model: xml.etree.ElementTree.Element

    :param drop_boundary_species:
        Whether species with ``boundaryCondition="true"`` should be excluded.
    :type drop_boundary_species: bool

    :return:
        Pair ``(species_attrs, dropped_ids)``.
    :rtype: Tuple[Dict[str, Dict[str, Any]], set]
    """
    listing = _find_child(model, "listOfSpecies")
    species: Dict[str, Dict[str, Any]] = {}
    dropped: set = set()

    for element in [] if listing is None else _find_children(listing, "species"):
        sid = element.get("id")
        if sid is None:
            continue
        if drop_boundary_species and _is_true(element.get("boundaryCondition")):
            dropped.add(sid)
            continue

        annotation = _annotation_values(element, "speciesInfo")
        species[sid] = {
            "kind": "species",
            "label": element.get("name") or sid,
            "smiles": annotation.get("smiles"),
            "sbml_id": sid,
        }

    return species, dropped


def _side_references(
    reaction: ET.Element,
    container: str,
) -> List[Tuple[str, float]]:
    """Collect ``(species id, stoichiometry)`` pairs from one reaction side.

    Repeated references to the same species are summed, as SBML allows.

    :param reaction:
        The ``<reaction>`` element.
    :type reaction: xml.etree.ElementTree.Element

    :param container:
        Either ``"listOfReactants"`` or ``"listOfProducts"``.
    :type container: str

    :return:
        Species references with their coefficients.
    :rtype: List[Tuple[str, float]]
    """
    listing = _find_child(reaction, container)
    if listing is None:
        return []

    totals: Dict[str, float] = {}
    for reference in _find_children(listing, "speciesReference"):
        target = reference.get("species")
        if target is None:
            continue
        totals[target] = totals.get(target, 0.0) + _parse_stoich(
            reference.get("stoichiometry")
        )
    return list(totals.items())


def _add_reaction_node(
    graph: nx.DiGraph,
    *,
    node_id: str,
    label: str,
    reactants: List[Tuple[str, float]],
    products: List[Tuple[str, float]],
    known_species: Dict[str, Dict[str, Any]],
    extra: Dict[str, Any],
) -> None:
    """Add one reaction node and its incidence edges to the working graph.

    :param graph:
        Bipartite graph under construction.
    :type graph: networkx.DiGraph

    :param node_id:
        Node id for the reaction.
    :type node_id: str

    :param label:
        Reaction label.
    :type label: str

    :param reactants:
        Reactant references.
    :type reactants: List[Tuple[str, float]]

    :param products:
        Product references.
    :type products: List[Tuple[str, float]]

    :param known_species:
        Species definitions keyed by SBML id; references outside this mapping
        are skipped.
    :type known_species: Dict[str, Dict[str, Any]]

    :param extra:
        Additional node attributes.
    :type extra: Dict[str, Any]

    :return:
        ``None``. The graph is modified in place.
    :rtype: None
    """
    graph.add_node(node_id, kind="reaction", label=label, **extra)

    for sid, coeff in reactants:
        if sid in known_species:
            graph.add_edge(sid, node_id, role="reactant", stoich=_normalize(coeff))
    for sid, coeff in products:
        if sid in known_species:
            graph.add_edge(node_id, sid, role="product", stoich=_normalize(coeff))


def _normalize(coeff: float) -> Union[int, float]:
    """Return an integer coefficient when the value is integral.

    :param coeff:
        Coefficient value.
    :type coeff: float

    :return:
        Integer or float coefficient.
    :rtype: Union[int, float]
    """
    return int(coeff) if float(coeff).is_integer() else coeff


def _warn_no_reactions(model: ET.Element) -> None:
    """Warn that an SBML model declares no reactions, and say why if possible.

    An empty network is a valid but almost always unintended result. The usual
    cause is a model that is not a reaction network at all: SBML-``qual``
    logical models carry ``<listOfTransitions>`` over ``<listOfQualitativeSpecies>``
    and have no ``<listOfReactions>``, and returning ``0`` species silently
    looks like a parse failure to the caller.

    :param model:
        The ``<model>`` element.
    :type model: xml.etree.ElementTree.Element

    :return:
        ``None``. A :class:`UserWarning` is issued.
    :rtype: None
    """
    hints = []
    if _find_child(model, "listOfQualitativeSpecies") is not None or (
        _find_child(model, "listOfTransitions") is not None
    ):
        hints.append(
            "the model uses the SBML qual package (qualitative species and "
            "transitions), which describes a logical model rather than a "
            "reaction network"
        )
    if _find_child(model, "listOfSpecies") is not None:
        hints.append("species are declared but no reactions connect them")

    detail = f" ({'; '.join(hints)})" if hints else ""
    warnings.warn(
        f"SBML model {model.get('id')!r} declares no reactions{detail}; "
        "the resulting network is empty.",
        UserWarning,
        stacklevel=3,
    )


def crn_from_sbml(
    source: Union[str, Path, ET.Element],
    *,
    expand_reversible: bool = True,
    drop_boundary_species: bool = False,
    strict: bool = False,
    **syncrn_kwargs: Any,
) -> SynCRN:
    """Build a :class:`SynCRN` from an SBML document.

    The reader matches elements by local name, so SBML Level 2 and Level 3 files
    both parse. Kinetic laws, units, parameters, events and rules are ignored:
    they carry no structural information.

    :param source:
        SBML document text, a path to an SBML file, or a parsed XML element.
    :type source: Union[str, pathlib.Path, xml.etree.ElementTree.Element]

    :param expand_reversible:
        Whether a reaction marked ``reversible="true"`` is split into a forward
        and a backward reaction. Leave enabled for CRNT analysis.
    :type expand_reversible: bool

    :param drop_boundary_species:
        Whether species with ``boundaryCondition="true"`` are excluded together
        with their incidences. Useful when a model uses boundary species as
        constant sources or sinks.
    :type drop_boundary_species: bool

    :param strict:
        Whether malformed network structure should raise; forwarded to
        :meth:`SynCRN.from_digraph`. Defaults to ``False`` because published
        SBML models routinely contain reactions with an empty side.
    :type strict: bool

    :param syncrn_kwargs:
        Extra keyword arguments forwarded to :meth:`SynCRN.from_digraph`, such
        as ``id_style``.
    :type syncrn_kwargs: Any

    :return:
        Canonical network object.
    :rtype: SynCRN

    :raises ValueError:
        If the document contains no ``<model>`` element.

    .. rubric:: Example

    .. code-block:: python

        crn = crn_from_sbml("model.xml")
        print(crn.n_species, crn.n_reactions)
    """
    root = _resolve_root(source)
    model = root if _localname(root.tag) == "model" else _find_child(root, "model")
    if model is None:
        found = _find_descendants(root, "model")
        model = found[0] if found else None
    if model is None:
        raise ValueError("SBML document contains no <model> element")

    species, _dropped = _read_species(
        model, drop_boundary_species=drop_boundary_species
    )

    graph = nx.DiGraph()
    graph.graph["sbml_model_id"] = model.get("id")
    if model.get("name"):
        graph.graph["sbml_model_name"] = model.get("name")

    for sid, attrs in species.items():
        graph.add_node(sid, **attrs)

    listing = _find_child(model, "listOfReactions")
    if listing is None or not _find_children(listing, "reaction"):
        _warn_no_reactions(model)

    for index, element in enumerate(
        [] if listing is None else _find_children(listing, "reaction"), start=1
    ):
        rid = element.get("id") or f"reaction_{index}"
        label = element.get("name") or rid
        reactants = _side_references(element, "listOfReactants")
        products = _side_references(element, "listOfProducts")
        reversible = _is_true(element.get("reversible"))
        annotation = _annotation_values(element, "reactionInfo")

        extra: Dict[str, Any] = {"sbml_id": rid}
        if annotation.get("ruleRepr"):
            extra["rule_repr"] = annotation["ruleRepr"]

        _add_reaction_node(
            graph,
            node_id=rid,
            label=label,
            reactants=reactants,
            products=products,
            known_species=species,
            extra=extra,
        )

        if reversible and expand_reversible:
            _add_reaction_node(
                graph,
                node_id=f"{rid}_rev",
                label=f"{label}_rev",
                reactants=products,
                products=reactants,
                known_species=species,
                extra={**extra, "sbml_reverse_of": rid},
            )

    return SynCRN.from_digraph(graph, strict=strict, **syncrn_kwargs)


def _resolve_root(source: Union[str, Path, ET.Element]) -> ET.Element:
    """Resolve the input into a parsed XML root element.

    A string is treated as a file path when it names an existing file and as
    document text otherwise.

    :param source:
        SBML document text, a path, or an already parsed element.
    :type source: Union[str, pathlib.Path, xml.etree.ElementTree.Element]

    :return:
        Root element.
    :rtype: xml.etree.ElementTree.Element

    :raises FileNotFoundError:
        If a :class:`~pathlib.Path` is given that does not exist.
    """
    if isinstance(source, ET.Element):
        return source

    if isinstance(source, Path):
        return ET.parse(source).getroot()

    text = str(source)
    stripped = text.lstrip()
    if stripped.startswith("<"):
        return ET.fromstring(text)
    return ET.parse(Path(text)).getroot()


def read_sbml(path: Union[str, Path], **kwargs: Any) -> SynCRN:
    """Read a :class:`SynCRN` from an SBML file.

    :param path:
        Path to the SBML file.
    :type path: Union[str, pathlib.Path]

    :param kwargs:
        Extra keyword arguments forwarded to :func:`crn_from_sbml`.
    :type kwargs: Any

    :return:
        Canonical network object.
    :rtype: SynCRN

    .. rubric:: Example

    .. code-block:: python

        crn = read_sbml("BIOMD0000000001.xml")
    """
    return crn_from_sbml(Path(path), **kwargs)
