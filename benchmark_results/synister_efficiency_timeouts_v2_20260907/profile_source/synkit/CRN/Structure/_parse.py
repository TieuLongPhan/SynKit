"""Reaction-string parsing and internal-id minting for :class:`SynCRN`.

These helpers back :meth:`SynCRN.from_reaction_strings`: they parse one side of
a reaction (``"2A + B"``) into a coefficient mapping, normalize whatever a
custom parser returns, and mint the internal ids under the policy named by
``id_style``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional
import re

from .reaction import RXNSide, Reaction
from .rule import Rule
from .species import Species

_REACTION_SPLIT_RE = re.compile(r"\s*>>\s*")


#: Recognised internal-id policies shared by every :class:`SynCRN` constructor.
#:
#: ``"prefixed"`` keeps species, reactions and rules in three disjoint
#: namespaces (``s_1`` / ``r_1`` / ``rule_1``); ``"numeric"`` is the legacy
#: single-namespace scheme where species are ``1..n`` and reactions continue at
#: ``n+1``.
ID_STYLES = ("prefixed", "numeric")


def _default_parse_side_text(side_text: str) -> Dict[str, int]:
    """Parse one reaction side into a ``species -> coefficient`` mapping.

    Supported examples include plain species lists, integer stoichiometric
    prefixes, and dot-separated species tokens.

    Supported examples
    ------------------
    - ``A + B``
    - ``2A + B``
    - ``A.A``
    - ``B.3C``
    - ``∅``

    :param side_text:
        One side of a reaction string.
    :type side_text: str

    :return:
        Mapping from species label to stoichiometric coefficient.
    :rtype: Dict[str, int]

    .. rubric:: Example

    .. code-block:: python

        _default_parse_side_text("2A + B")
        # {"A": 2, "B": 1}
    """
    text = str(side_text).strip()
    if text in {"", "∅", "0", "None", "null"}:
        return {}

    parts = re.split(r"\s*(?:\+|\.)\s*", text)
    out: Dict[str, int] = {}

    for part in parts:
        token = part.strip()
        if not token:
            continue

        m = re.match(r"^\s*(\d+)\s*\*?\s*(.+?)\s*$", token)
        if m:
            coeff = int(m.group(1))
            species = m.group(2).strip()
        else:
            m2 = re.match(r"^\s*(\d+)([A-Za-z].*?)\s*$", token)
            if m2:
                coeff = int(m2.group(1))
                species = m2.group(2).strip()
            else:
                coeff = 1
                species = token

        if species and coeff > 0:
            out[species] = out.get(species, 0) + coeff

    return out


def _coerce_side_counts(obj: Any) -> Dict[str, int]:
    """Coerce a parsed side object into a plain dictionary.

    This helper accepts plain mappings, objects exposing ``to_dict()``, and
    RXNSide-like objects exposing ``items()``.

    :param obj:
        Parsed side object.
    :type obj: Any

    :return:
        Mapping from species label to stoichiometric coefficient.
    :rtype: Dict[str, int]

    :raises TypeError:
        If the object cannot be interpreted as a side-count mapping.

    .. rubric:: Example

    .. code-block:: python

        _coerce_side_counts({"A": 2, "B": 1})
        # {"A": 2, "B": 1}
    """
    if obj is None:
        return {}

    if isinstance(obj, Mapping):
        return {str(k): int(v) for k, v in obj.items()}

    if hasattr(obj, "to_dict"):
        d = obj.to_dict()
        return {str(k): int(v) for k, v in d.items()}

    if hasattr(obj, "items"):
        return {str(k): int(v) for k, v in obj.items()}

    raise TypeError(
        "Parsed side must be a mapping, or expose to_dict(), or expose items()."
    )


def _normalize_side_counts(
    counts: Any,
    *,
    rxn_index: int,
    side_name: str,
    strict: bool,
) -> Dict[str, int]:
    """Normalize a parsed reaction side into a validated label-to-coefficient mapping.

    :param counts:
        Parsed side object.
    :type counts: Any

    :param rxn_index:
        Reaction index in the input list.
    :type rxn_index: int

    :param side_name:
        Name of the side, typically ``"lhs"`` or ``"rhs"``.
    :type side_name: str

    :param strict:
        Whether invalid labels or coefficients should raise an error.
    :type strict: bool

    :return:
        Cleaned mapping from species label to coefficient.
    :rtype: Dict[str, int]
    """
    raw = _coerce_side_counts(counts)
    out: Dict[str, int] = {}

    for sp, coeff in raw.items():
        label = str(sp).strip()
        try:
            n = int(coeff)
        except Exception as exc:
            raise TypeError(
                f"Reaction at index {rxn_index} has non-integer coefficient on "
                f"{side_name}: {sp!r} -> {coeff!r}"
            ) from exc

        if not label:
            if strict:
                raise ValueError(
                    f"Reaction at index {rxn_index} has blank species label on {side_name}"
                )
            continue

        if n <= 0:
            if strict:
                raise ValueError(
                    f"Reaction at index {rxn_index} has non-positive coefficient on "
                    f"{side_name}: {sp!r} -> {n}"
                )
            continue

        out[label] = out.get(label, 0) + n

    return out


def _parse_reaction_string_entry(
    rxn_text: str,
    *,
    rxn_index: int,
    rule_repr: Optional[str],
    has_rules: bool,
    parse_side: Callable[[str], Any],
    strict: bool,
) -> Dict[str, Any]:
    """Parse one reaction string into normalized lhs/rhs count dictionaries.

    :param rxn_text:
        Raw reaction string such as ``"2A>>B+3C"``.
    :type rxn_text: str

    :param rxn_index:
        Index of the reaction in the input list.
    :type rxn_index: int

    :param rule_repr:
        Optional rule string paired with the reaction.
    :type rule_repr: Optional[str]

    :param has_rules:
        Whether pairwise rules are being used.
    :type has_rules: bool

    :param parse_side:
        Side parser function.
    :type parse_side: Callable[[str], Any]

    :param strict:
        Whether malformed input should raise an error.
    :type strict: bool

    :return:
        Parsed reaction record.
    :rtype: Dict[str, Any]
    """
    text = str(rxn_text).strip()
    pieces = _REACTION_SPLIT_RE.split(text, maxsplit=1)

    if len(pieces) != 2:
        raise ValueError(
            f"Reaction string at index {rxn_index} must contain exactly one '>>': {rxn_text!r}"
        )

    lhs_text, rhs_text = pieces[0].strip(), pieces[1].strip()

    lhs_counts = _normalize_side_counts(
        parse_side(lhs_text),
        rxn_index=rxn_index,
        side_name="lhs",
        strict=strict,
    )
    rhs_counts = _normalize_side_counts(
        parse_side(rhs_text),
        rxn_index=rxn_index,
        side_name="rhs",
        strict=strict,
    )

    if strict and not lhs_counts:
        raise ValueError(f"Reaction at index {rxn_index} has empty lhs: {rxn_text!r}")
    if strict and not rhs_counts:
        raise ValueError(f"Reaction at index {rxn_index} has empty rhs: {rxn_text!r}")

    return {
        "rxn_text": text,
        "lhs_counts": lhs_counts,
        "rhs_counts": rhs_counts,
        "rule_repr": rule_repr,
        "rule_index": rxn_index if has_rules else None,
    }


def _species_order_from_parsed_reactions(
    parsed_rxns: List[Dict[str, Any]],
) -> List[str]:
    """Derive species order by first appearance in parsed reactions.

    :param parsed_rxns:
        Parsed reaction entries.
    :type parsed_rxns: List[Dict[str, Any]]

    :return:
        Ordered species labels.
    :rtype: List[str]
    """
    order: List[str] = []
    seen = set()

    for item in parsed_rxns:
        for sp in list(item["lhs_counts"].keys()) + list(item["rhs_counts"].keys()):
            if sp not in seen:
                seen.add(sp)
                order.append(sp)

    return order


def _mint_reaction_string_ids(
    *,
    n_species: int,
    n_reactions: int,
    id_style: str,
    species_prefix: str,
    reaction_prefix: str,
    rule_prefix: str,
) -> Dict[str, List[Any]]:
    """Mint internal ids and source node ids for reaction-string input.

    Two policies are supported. ``"prefixed"`` matches
    :meth:`SynCRN.from_digraph` — species, reactions and rules live in three
    disjoint namespaces (``s_1``, ``r_1``, ``rule_1``), so a species id can
    never be mistaken for a reaction id. ``"numeric"`` reproduces the legacy
    scheme where species are ``1..n`` and reactions continue at ``n+1`` in one
    shared numeric namespace.

    :param n_species:
        Number of species.
    :type n_species: int

    :param n_reactions:
        Number of reactions.
    :type n_reactions: int

    :param id_style:
        Either ``"prefixed"`` or ``"numeric"``.
    :type id_style: str

    :param species_prefix:
        Prefix for generated species ids under ``"prefixed"``.
    :type species_prefix: str

    :param reaction_prefix:
        Prefix for generated reaction ids under ``"prefixed"``.
    :type reaction_prefix: str

    :param rule_prefix:
        Prefix for generated rule ids under ``"prefixed"``.
    :type rule_prefix: str

    :return:
        Mapping with ``species_ids``, ``species_source_ids``, ``reaction_ids``,
        ``reaction_source_ids`` and ``rule_ids``.
    :rtype: Dict[str, List[Any]]

    :raises ValueError:
        If ``id_style`` is not a recognised policy.
    """
    if id_style not in ID_STYLES:
        raise ValueError(f"id_style must be one of {sorted(ID_STYLES)}, got {id_style!r}")

    if id_style == "numeric":
        species_ids = [str(i) for i in range(1, n_species + 1)]
        reaction_ids = [
            str(i) for i in range(n_species + 1, n_species + n_reactions + 1)
        ]
        rule_ids = [str(i) for i in range(1, n_reactions + 1)]
        return {
            "species_ids": species_ids,
            "species_source_ids": [int(s) for s in species_ids],
            "reaction_ids": reaction_ids,
            "reaction_source_ids": [int(r) for r in reaction_ids],
            "rule_ids": rule_ids,
        }

    species_ids = [f"{species_prefix}{i}" for i in range(1, n_species + 1)]
    reaction_ids = [f"{reaction_prefix}{i}" for i in range(1, n_reactions + 1)]
    rule_ids = [f"{rule_prefix}{i}" for i in range(1, n_reactions + 1)]
    return {
        "species_ids": species_ids,
        "species_source_ids": list(species_ids),
        "reaction_ids": reaction_ids,
        "reaction_source_ids": list(reaction_ids),
        "rule_ids": rule_ids,
    }


def _build_species_table_from_labels(
    species_order: List[str],
    *,
    species_ids: List[str],
    species_source_ids: List[Any],
) -> Dict[str, Species]:
    """Build the canonical species table from an ordered list of species labels.

    :param species_order:
        Species labels in canonical order.
    :type species_order: List[str]

    :param species_ids:
        Internal species ids, aligned with ``species_order``.
    :type species_ids: List[str]

    :param species_source_ids:
        Source node ids, aligned with ``species_order``.
    :type species_source_ids: List[Any]

    :return:
        Species table keyed by canonical string ids.
    :rtype: Dict[str, Species]
    """
    species: Dict[str, Species] = {}
    for label, sid, source_id in zip(species_order, species_ids, species_source_ids):
        species[sid] = Species(
            id=sid,
            source_node_id=source_id,
            label=label,
            smiles=None,
            source_attrs={
                "kind": "species",
                "label": label,
            },
            metadata={},
        )
    return species


def _build_rules_table_from_reaction_strings(
    parsed_rxns: List[Dict[str, Any]],
    *,
    rule_ids: List[str],
    has_rules: bool,
) -> Dict[str, Rule]:
    """Build the abstract rules table for reaction-string input.

    :param parsed_rxns:
        Parsed reaction entries.
    :type parsed_rxns: List[Dict[str, Any]]

    :param rule_ids:
        Internal rule ids, aligned with ``parsed_rxns``.
    :type rule_ids: List[str]

    :param has_rules:
        Whether rules were supplied pairwise.
    :type has_rules: bool

    :return:
        Rule table keyed by canonical rule ids.
    :rtype: Dict[str, Rule]
    """
    if not has_rules:
        return {}

    rules_table: Dict[str, Rule] = {}
    for item, rule_id in zip(parsed_rxns, rule_ids):
        rule_index = item["rule_index"]
        rule_repr = item["rule_repr"]
        rules_table[rule_id] = Rule(
            id=rule_id,
            rule_index=rule_index,
            rule_repr=rule_repr,
            label=f"r{rule_index}",
            metadata={},
        )
    return rules_table


def _build_reactions_from_parsed_strings(
    parsed_rxns: List[Dict[str, Any]],
    *,
    label_to_sid: Dict[str, str],
    reaction_ids: List[str],
    reaction_source_ids: List[Any],
    rule_ids: List[str],
    has_rules: bool,
) -> Dict[str, Reaction]:
    """Build canonical Reaction objects from parsed reaction-string entries.

    :param parsed_rxns:
        Parsed reaction entries.
    :type parsed_rxns: List[Dict[str, Any]]

    :param label_to_sid:
        Mapping from species label to canonical species id.
    :type label_to_sid: Dict[str, str]

    :param reaction_ids:
        Internal reaction ids, aligned with ``parsed_rxns``.
    :type reaction_ids: List[str]

    :param reaction_source_ids:
        Source node ids, aligned with ``parsed_rxns``.
    :type reaction_source_ids: List[Any]

    :param rule_ids:
        Internal rule ids, aligned with ``parsed_rxns``.
    :type rule_ids: List[str]

    :param has_rules:
        Whether rules were supplied pairwise.
    :type has_rules: bool

    :return:
        Reaction table keyed by canonical reaction ids.
    :rtype: Dict[str, Reaction]
    """
    reactions: Dict[str, Reaction] = {}

    for i, item in enumerate(parsed_rxns):
        rid = reaction_ids[i]
        rid_source = reaction_source_ids[i]

        rule_id: Optional[str] = None
        rule_index = item["rule_index"]
        rule_repr = item["rule_repr"]

        if has_rules:
            rule_id = rule_ids[i]

        lhs = RXNSide(
            {label_to_sid[sp]: coeff for sp, coeff in item["lhs_counts"].items()}
        )
        rhs = RXNSide(
            {label_to_sid[sp]: coeff for sp, coeff in item["rhs_counts"].items()}
        )

        reactions[rid] = Reaction(
            id=rid,
            source_node_id=rid_source,
            source_kind="rule",
            lhs=lhs,
            rhs=rhs,
            label=rule_repr if has_rules and rule_repr is not None else rid,
            step=None,
            rule_index=rule_index,
            app_index=None,
            rule_repr=rule_repr,
            rule_id=rule_id,
            source_attrs={
                "kind": "rule",
                "label": rid,
                "rxn_repr": item["rxn_text"],
                "rule_index": rule_index,
                "rule_repr": rule_repr,
            },
            metadata={},
            reactant_edge_attrs={
                label_to_sid[sp]: {"role": "reactant", "stoich": coeff}
                for sp, coeff in item["lhs_counts"].items()
            },
            product_edge_attrs={
                label_to_sid[sp]: {"role": "product", "stoich": coeff}
                for sp, coeff in item["rhs_counts"].items()
            },
        )

    return reactions
