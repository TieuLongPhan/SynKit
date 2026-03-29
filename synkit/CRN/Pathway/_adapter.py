from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

from ..Petrinet.net import SynCRNIncidence, extract_syncrn_incidence


def _species_token(incidence: SynCRNIncidence, sid: str, mode: str) -> str:
    """
    Convert an internal species identifier into an external token.

    Supported tokenization modes are:

    - ``"id"``: use the internal species identifier
    - ``"label"``: use the species label if available
    - ``"source"``: use the original source-node identifier if available

    :param incidence:
        Canonical SynCRN incidence object.
    :type incidence: SynCRNIncidence

    :param sid:
        Internal species identifier.
    :type sid: str

    :param mode:
        Tokenization mode. Must be one of ``"id"``, ``"label"``, or ``"source"``.
    :type mode: str

    :returns:
        Token representing the species.
    :rtype: str

    :raises ValueError:
        If ``mode`` is not one of the supported values.

    Examples
    --------
    .. code-block:: python

        tok = _species_token(incidence, "s1", mode="label")
        print(tok)
    """
    if mode == "id":
        return sid
    if mode == "label":
        return incidence.species_labels.get(sid, sid)
    if mode == "source":
        return str(incidence.species_source_node_ids.get(sid, sid))
    raise ValueError("species mode must be one of: id, label, source")


def _reaction_token(incidence: SynCRNIncidence, rid: str, mode: str) -> str:
    """
    Convert an internal reaction identifier into an external token.

    Supported tokenization modes are:

    - ``"id"``: use the internal reaction identifier
    - ``"label"``: use the reaction label if available
    - ``"source"``: use the original source-node identifier if available

    :param incidence:
        Canonical SynCRN incidence object.
    :type incidence: SynCRNIncidence

    :param rid:
        Internal reaction identifier.
    :type rid: str

    :param mode:
        Tokenization mode. Must be one of ``"id"``, ``"label"``, or ``"source"``.
    :type mode: str

    :returns:
        Token representing the reaction.
    :rtype: str

    :raises ValueError:
        If ``mode`` is not one of the supported values.

    Examples
    --------
    .. code-block:: python

        tok = _reaction_token(incidence, "r1", mode="id")
        print(tok)
    """
    if mode == "id":
        return rid
    if mode == "label":
        return incidence.reaction_labels.get(rid, rid)
    if mode == "source":
        return str(incidence.reaction_source_node_ids.get(rid, rid))
    raise ValueError("reaction mode must be one of: id, label, source")


def _assert_unique_tokens(tokens: Mapping[str, str], *, kind: str) -> None:
    """
    Validate that exported tokens are unique.

    This prevents ambiguous reverse mapping from public tokens back to internal
    identifiers.

    :param tokens:
        Mapping from internal identifiers to exported tokens.
    :type tokens: Mapping[str, str]

    :param kind:
        Entity kind used in error messages, typically ``"species"`` or
        ``"reaction"``.
    :type kind: str

    :returns:
        ``None``.
    :rtype: None

    :raises ValueError:
        If two distinct internal identifiers map to the same exported token.
    """
    seen: Dict[str, str] = {}
    for internal_id, token in tokens.items():
        if token in seen and seen[token] != internal_id:
            raise ValueError(
                f"Non-unique {kind} token {token!r} encountered for internal ids "
                f"{seen[token]!r} and {internal_id!r}. Use {kind}='id' instead."
            )
        seen[token] = internal_id


def _build_species_token_map(
    incidence: SynCRNIncidence,
    *,
    species: str,
) -> Dict[str, str]:
    """
    Build the mapping from internal species ids to exported species tokens.

    :param incidence:
        Canonical SynCRN incidence object.
    :type incidence: SynCRNIncidence

    :param species:
        Species tokenization mode.
    :type species: str

    :returns:
        Mapping ``internal_species_id -> exported_species_token``.
    :rtype: Dict[str, str]
    """
    return {
        sid: _species_token(incidence, sid, species) for sid in incidence.species_order
    }


def _build_reaction_token_map(
    incidence: SynCRNIncidence,
    *,
    reaction: str,
) -> Dict[str, str]:
    """
    Build the mapping from internal reaction ids to exported reaction tokens.

    :param incidence:
        Canonical SynCRN incidence object.
    :type incidence: SynCRNIncidence

    :param reaction:
        Reaction tokenization mode.
    :type reaction: str

    :returns:
        Mapping ``internal_reaction_id -> exported_reaction_token``.
    :rtype: Dict[str, str]
    """
    return {
        rid: _reaction_token(incidence, rid, reaction)
        for rid in incidence.reaction_order
    }


def _tokenize_reaction_multiset(
    coeffs: Mapping[str, int],
    species_map: Mapping[str, str],
) -> Dict[str, int]:
    """
    Convert one incidence-side multiset from internal species ids to tokens.

    :param coeffs:
        Mapping ``internal_species_id -> stoichiometric_coefficient``.
    :type coeffs: Mapping[str, int]

    :param species_map:
        Mapping ``internal_species_id -> exported_species_token``.
    :type species_map: Mapping[str, str]

    :returns:
        Mapping ``exported_species_token -> stoichiometric_coefficient``.
    :rtype: Dict[str, int]
    """
    return {species_map[sid]: int(coeff) for sid, coeff in coeffs.items()}


def _build_tokenized_edges(
    incidence: SynCRNIncidence,
    species_map: Mapping[str, str],
    reaction_map: Mapping[str, str],
) -> Dict[str, Tuple[Dict[str, int], Dict[str, int]]]:
    """
    Build tokenized reaction edges.

    Each reaction token maps to a pair ``(tail, head)``, where ``tail`` and
    ``head`` are tokenized stoichiometric multisets for reactants and products.

    :param incidence:
        Canonical SynCRN incidence object.
    :type incidence: SynCRNIncidence

    :param species_map:
        Mapping ``internal_species_id -> exported_species_token``.
    :type species_map: Mapping[str, str]

    :param reaction_map:
        Mapping ``internal_reaction_id -> exported_reaction_token``.
    :type reaction_map: Mapping[str, str]

    :returns:
        Mapping
        ``reaction_token -> (reactant_multiset, product_multiset)``.
    :rtype: Dict[str, Tuple[Dict[str, int], Dict[str, int]]]
    """
    edges: Dict[str, Tuple[Dict[str, int], Dict[str, int]]] = {}
    for rid in incidence.reaction_order:
        rtoken = reaction_map[rid]
        tail = _tokenize_reaction_multiset(incidence.pre.get(rid, {}), species_map)
        head = _tokenize_reaction_multiset(incidence.post.get(rid, {}), species_map)
        edges[rtoken] = (tail, head)
    return edges


def _invert_token_map(token_map: Mapping[str, str]) -> Dict[str, str]:
    """
    Invert an internal-id to token mapping.

    :param token_map:
        Mapping ``internal_id -> token``.
    :type token_map: Mapping[str, str]

    :returns:
        Mapping ``token -> internal_id``.
    :rtype: Dict[str, str]
    """
    return {tok: internal_id for internal_id, tok in token_map.items()}


def tokenize_syncrn_incidence(
    crn: Any,
    *,
    species: str = "label",
    reaction: str = "id",
) -> Tuple[
    List[str],
    Dict[str, Tuple[Dict[str, int], Dict[str, int]]],
    Dict[str, str],
    Dict[str, str],
    SynCRNIncidence,
]:
    """
    Convert SynCRN incidence into tokenized vertex and edge dictionaries.

    The returned objects are designed for downstream algorithms that prefer
    stable public-facing tokens over internal SynCRN identifiers.

    :param crn:
        SynCRN-like object accepted by :func:`extract_syncrn_incidence`.
    :type crn: Any

    :param species:
        Species tokenization mode. Must be one of ``"id"``, ``"label"``, or
        ``"source"``.
    :type species: str

    :param reaction:
        Reaction tokenization mode. Must be one of ``"id"``, ``"label"``, or
        ``"source"``.
    :type reaction: str

    :returns:
        Tuple containing:

        - ``vertices``:
          ordered list of tokenized species vertices
        - ``edges``:
          mapping ``reaction_token -> (reactant_multiset, product_multiset)``
        - ``species_token_to_id``:
          reverse mapping from exported species tokens to internal ids
        - ``reaction_token_to_id``:
          reverse mapping from exported reaction tokens to internal ids
        - ``incidence``:
          original extracted canonical incidence object

    :rtype:
        Tuple[
            List[str],
            Dict[str, Tuple[Dict[str, int], Dict[str, int]]],
            Dict[str, str],
            Dict[str, str],
            SynCRNIncidence,
        ]

    :raises ValueError:
        If species or reaction tokenization produces non-unique exported tokens.

    Examples
    --------
    .. code-block:: python

        vertices, edges, s_tok2id, r_tok2id, incidence = tokenize_syncrn_incidence(
            crn,
            species="label",
            reaction="id",
        )

        print(vertices)
        print(edges)
        print(s_tok2id)
        print(r_tok2id)
    """
    incidence = extract_syncrn_incidence(crn)

    species_map = _build_species_token_map(incidence, species=species)
    reaction_map = _build_reaction_token_map(incidence, reaction=reaction)

    _assert_unique_tokens(species_map, kind="species")
    _assert_unique_tokens(reaction_map, kind="reaction")

    vertices = [species_map[sid] for sid in incidence.species_order]
    edges = _build_tokenized_edges(incidence, species_map, reaction_map)

    species_token_to_id = _invert_token_map(species_map)
    reaction_token_to_id = _invert_token_map(reaction_map)
    return vertices, edges, species_token_to_id, reaction_token_to_id, incidence
