"""Bridge from KEGG equations to :class:`~synkit.CRN.Structure.syncrn.SynCRN`.

:mod:`synkit.CRN.Query` retrieves and parses KEGG data; this module turns that
data into the canonical network object the rest of :mod:`synkit.CRN` analyses.
It is the join between the *curated-pathway* entry point and the analysis stack,
so a KEGG module can be pushed straight through conservation laws, semiflows,
siphons, persistence and realizability.

Two conversion choices matter for the result:

- **Reversible equations.** KEGG writes ``<=>`` for a reversible reaction. A
  ``SynCRN`` reaction is directed, so a reversible equation becomes two
  reactions by default (``expand_reversible``). This matters: weak
  reversibility, persistence and siphon verdicts all change if the reverse
  direction is dropped.
- **Currency metabolites.** ATP, ADP, NAD+, water and phosphate participate in
  most reactions of a pathway and dominate its structural analysis — every
  siphon and most semiflows end up describing cofactor recycling rather than
  the pathway. Passing ``drop_compounds=CURRENCY_COMPOUNDS`` removes them, which
  is the usual convention when analysing a pathway's carbon skeleton.

.. rubric:: Example

.. code-block:: python

    from synkit.CRN.Query import syncrn_from_kegg_equations

    crn = syncrn_from_kegg_equations(
        {"R01786": "C00002 + C00267 => C00008 + C00668"}
    )
    print(crn.to_equations(species="label"))
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Set

import networkx as nx

from ..Structure.syncrn import SynCRN
from .kegg_parse import parse_equation

__all__ = [
    "CURRENCY_COMPOUNDS",
    "syncrn_from_kegg_equations",
    "syncrn_from_kegg_module",
]

#: KEGG compound ids for the usual currency metabolites.
#:
#: These appear in most reactions of any pathway, so leaving them in makes the
#: structural analysis describe cofactor turnover instead of the pathway.
CURRENCY_COMPOUNDS: Set[str] = {
    "C00001",  # H2O
    "C00002",  # ATP
    "C00003",  # NAD+
    "C00004",  # NADH
    "C00005",  # NADPH
    "C00006",  # NADP+
    "C00008",  # ADP
    "C00009",  # orthophosphate
    "C00013",  # diphosphate
    "C00020",  # AMP
    "C00080",  # H+
}


def _side_counts(
    items: Iterable[Any],
    *,
    drop: Set[str],
) -> Dict[str, int]:
    """Aggregate one parsed equation side into ``compound -> coefficient``.

    :param items:
        ``(compound_id, coefficient)`` pairs from a parsed KEGG equation.
    :type items: Iterable[Any]

    :param drop:
        Compound ids to exclude.
    :type drop: Set[str]

    :return:
        Coefficients keyed by compound id, with dropped compounds removed.
    :rtype: Dict[str, int]
    """
    counts: Dict[str, int] = {}
    for compound_id, coefficient in items:
        if compound_id in drop:
            continue
        counts[compound_id] = counts.get(compound_id, 0) + int(coefficient)
    return counts


def syncrn_from_kegg_equations(
    equations: Mapping[str, Optional[str]],
    *,
    names: Optional[Mapping[str, str]] = None,
    expand_reversible: bool = True,
    drop_compounds: Iterable[str] = (),
    skip_empty: bool = True,
    strict: bool = False,
) -> SynCRN:
    """Build a :class:`SynCRN` from KEGG reaction equations.

    :param equations:
        Mapping from KEGG reaction id to equation string, as returned by
        :meth:`~synkit.CRN.Query.kegg_extract.KEGGExtractor.get_module_equations`.
        Entries with no equation are skipped.
    :type equations: Mapping[str, Optional[str]]

    :param names:
        Optional mapping from compound id to a display name, used as the species
        label. Compound ids are used when a name is missing.
    :type names: Optional[Mapping[str, str]]

    :param expand_reversible:
        Whether a reversible KEGG equation (``<=>``) becomes two directed
        reactions.
    :type expand_reversible: bool

    :param drop_compounds:
        Compound ids to exclude from every reaction, typically
        :data:`CURRENCY_COMPOUNDS`.
    :type drop_compounds: Iterable[str]

    :param skip_empty:
        Whether reactions left with no reactant *and* no product after dropping
        compounds are omitted rather than kept as empty reactions.
    :type skip_empty: bool

    :param strict:
        Whether malformed structure should raise; forwarded to
        :meth:`SynCRN.from_digraph`.
    :type strict: bool

    :return:
        Canonical network object whose reaction labels are the KEGG reaction
        ids and whose species labels are compound names (or ids).
    :rtype: SynCRN

    .. rubric:: Example

    .. code-block:: python

        crn = syncrn_from_kegg_equations(
            {"R01015": "C00111 <=> C00118"},
            names={"C00111": "DHAP", "C00118": "GAP"},
        )
        print(crn.n_reactions)
        # 2
    """
    drop = set(drop_compounds)
    label_of = dict(names or {})

    graph = nx.DiGraph()
    graph.graph["source"] = "kegg"
    seen_species: Set[str] = set()

    def _ensure_species(compound_id: str) -> None:
        if compound_id in seen_species:
            return
        seen_species.add(compound_id)
        graph.add_node(
            compound_id,
            kind="species",
            label=label_of.get(compound_id, compound_id),
            kegg_compound=compound_id,
        )

    def _add_reaction(
        node_id: str,
        *,
        label: str,
        lhs: Dict[str, int],
        rhs: Dict[str, int],
        kegg_id: str,
        direction: str,
    ) -> None:
        if skip_empty and not lhs and not rhs:
            return
        for compound_id in list(lhs) + list(rhs):
            _ensure_species(compound_id)
        graph.add_node(
            node_id,
            kind="reaction",
            label=label,
            kegg_reaction=kegg_id,
            direction=direction,
        )
        for compound_id, coefficient in lhs.items():
            graph.add_edge(
                compound_id, node_id, role="reactant", stoich=coefficient
            )
        for compound_id, coefficient in rhs.items():
            graph.add_edge(
                node_id, compound_id, role="product", stoich=coefficient
            )

    for kegg_id, equation in equations.items():
        if not equation:
            continue
        parsed = parse_equation(equation)
        lhs = _side_counts(parsed.reactants, drop=drop)
        rhs = _side_counts(parsed.products, drop=drop)

        reverse = parsed.reversible and expand_reversible
        _add_reaction(
            kegg_id,
            label=kegg_id,
            lhs=lhs,
            rhs=rhs,
            kegg_id=kegg_id,
            direction="forward",
        )
        if reverse:
            _add_reaction(
                f"{kegg_id}_rev",
                label=f"{kegg_id}_rev",
                lhs=rhs,
                rhs=lhs,
                kegg_id=kegg_id,
                direction="reverse",
            )

    return SynCRN.from_digraph(graph, strict=strict)


def syncrn_from_kegg_module(
    module_id: str,
    *,
    extractor: Any = None,
    with_names: bool = True,
    **kwargs: Any,
) -> SynCRN:
    """Fetch a KEGG module and build a :class:`SynCRN` from it.

    This performs live KEGG REST requests. For a reproducible, offline network
    use :func:`syncrn_from_kegg_equations` with cached equations — see
    :mod:`synkit.CRN.Benchmark.kegg`.

    :param module_id:
        KEGG module identifier such as ``"M00001"``.
    :type module_id: str

    :param extractor:
        Optional :class:`~synkit.CRN.Query.kegg_extract.KEGGExtractor` instance.
        A default one is created when omitted.
    :type extractor: Any

    :param with_names:
        Whether compound names should be fetched and used as species labels.
        This costs one request per compound.
    :type with_names: bool

    :param kwargs:
        Extra keyword arguments forwarded to
        :func:`syncrn_from_kegg_equations`.
    :type kwargs: Any

    :return:
        Canonical network object for the module.
    :rtype: SynCRN

    .. rubric:: Example

    .. code-block:: python

        crn = syncrn_from_kegg_module("M00001", drop_compounds=CURRENCY_COMPOUNDS)
        print(crn.n_species, crn.n_reactions)
    """
    if extractor is None:
        from .kegg_extract import KEGGExtractor

        extractor = KEGGExtractor()

    equations = extractor.get_module_equations(module_id)

    names: Dict[str, str] = {}
    if with_names:
        compound_ids: Set[str] = set()
        for equation in equations.values():
            if not equation:
                continue
            parsed = parse_equation(equation)
            for compound_id, _ in list(parsed.reactants) + list(parsed.products):
                compound_ids.add(compound_id)
        for compound_id in sorted(compound_ids):
            name = extractor.get_compound_name(compound_id)
            if name:
                names[compound_id] = name

    kwargs.setdefault("names", names or None)
    return syncrn_from_kegg_equations(equations, **kwargs)
