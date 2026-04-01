from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from rdkit import Chem


_CID_PATTERN = re.compile(r"C\d{5}")
_MODULE_PATTERN = re.compile(r"M\d{5}")
_TERM_PATTERN = re.compile(r"^(?:(\d+)\s+)?(C\d{5})$")

CompoundStoich = tuple[str, int]
CompoundRecord = Mapping[str, Any]
ReactionSmilesMissing = dict[str, list[str]]


@dataclass(frozen=True, slots=True)
class KEGGEquation:
    """
    Structured representation of a parsed KEGG reaction equation.

    :param reactants:
        Left-hand side of the equation as ``(compound_id, stoichiometry)``
        pairs.
    :type reactants: list[tuple[str, int]]
    :param products:
        Right-hand side of the equation as ``(compound_id, stoichiometry)``
        pairs.
    :type products: list[tuple[str, int]]
    :param reversible:
        Whether the original equation used the reversible arrow ``<=>``.
    :type reversible: bool

    Example
    -------
    .. code-block:: python

        equation = KEGGEquation(
            reactants=[("C00001", 1), ("C00002", 2)],
            products=[("C00008", 1)],
            reversible=False,
        )
    """

    reactants: list[CompoundStoich]
    products: list[CompoundStoich]
    reversible: bool


def parse_kegg_field_blocks(text: str, field: str) -> list[str]:
    """
    Extract payloads from a KEGG flatfile field, including continuation lines.

    Continuation lines are recognized as lines beginning with spaces or tabs and
    are concatenated to the payload of the preceding field occurrence.

    :param text:
        Raw KEGG flatfile text.
    :type text: str
    :param field:
        Flatfile field name such as ``"MODULE"``, ``"REACTION"``,
        ``"EQUATION"``, or ``"NAME"``.
    :type field: str

    :returns:
        One payload string per matching field occurrence.
    :rtype: list[str]

    Example
    -------
    .. code-block:: python

        text = (
            "MODULE      M00001 Glycolysis\n"
            "            continuation line\n"
        )
        payloads = parse_kegg_field_blocks(text, "MODULE")
    """
    payloads: list[str] = []
    lines = text.splitlines()
    i = 0

    while i < len(lines):
        line = lines[i]
        if line.startswith(field):
            payload = line[len(field):].strip()
            j = i + 1
            continuation: list[str] = []

            while j < len(lines) and (
                lines[j].startswith(" ") or lines[j].startswith("\t")
            ):
                continuation.append(lines[j].strip())
                j += 1

            if continuation:
                payload = (payload + " " + " ".join(continuation)).strip()

            payloads.append(payload)
            i = j
        else:
            i += 1

    return payloads


def normalize_module_id(module_id: str) -> Optional[str]:
    """
    Normalize a token to canonical KEGG module form.

    Supported examples include strings such as ``"hsa_M00001"`` and
    ``"M00001"``, both of which normalize to ``"M00001"``.

    :param module_id:
        Raw module token or containing text.
    :type module_id: str

    :returns:
        Canonical KEGG module identifier, or ``None`` when no module identifier
        is present.
    :rtype: Optional[str]

    Example
    -------
    .. code-block:: python

        canonical = normalize_module_id("hsa_M00001")
    """
    match = _MODULE_PATTERN.search(module_id)
    return match.group(0) if match else None


def parse_side(side: str) -> list[CompoundStoich]:
    """
    Parse one side of a KEGG equation into compound/stoichiometry pairs.

    For example, ``"2 C00139 + C00001"`` becomes
    ``[("C00139", 2), ("C00001", 1)]``.

    :param side:
        One side of a KEGG equation.
    :type side: str

    :returns:
        Parsed ``(compound_id, coefficient)`` pairs.
    :rtype: list[tuple[str, int]]

    :raises ValueError:
        Raised when any term does not match KEGG compound-stoichiometry syntax.

    Example
    -------
    .. code-block:: python

        items = parse_side("2 C00139 + C00001")
    """
    side = side.strip()
    if not side:
        return []

    terms = [term.strip() for term in side.split("+")]
    parsed: list[CompoundStoich] = []

    for term in terms:
        match = _TERM_PATTERN.match(term)
        if not match:
            raise ValueError(f"Unparsed KEGG equation term: {term!r}")

        coefficient = int(match.group(1)) if match.group(1) else 1
        compound_id = match.group(2)
        parsed.append((compound_id, coefficient))

    return parsed


def parse_equation(equation: str) -> KEGGEquation:
    """
    Parse a KEGG equation string into reactants, products, and arrow type.

    Supported arrows are ``<=>``, ``=>``, and ``<=``.

    :param equation:
        KEGG equation string.
    :type equation: str

    :returns:
        Parsed equation object.
    :rtype: KEGGEquation

    :raises ValueError:
        Raised when the equation does not contain a supported KEGG arrow.

    Example
    -------
    .. code-block:: python

        parsed = parse_equation("C00001 + C00002 <=> C00003")
    """
    if "<=>" in equation:
        lhs, rhs = equation.split("<=>")
        reversible = True
    elif "=>" in equation:
        lhs, rhs = equation.split("=>")
        reversible = False
    elif "<=" in equation:
        rhs, lhs = equation.split("<=")
        reversible = False
    else:
        raise ValueError(f"Unknown KEGG equation arrow in: {equation!r}")

    return KEGGEquation(
        reactants=parse_side(lhs),
        products=parse_side(rhs),
        reversible=reversible,
    )


def get_compound_ids_from_equations(
    equations_by_rid: Mapping[str, Optional[str]],
) -> tuple[list[str], dict[str, KEGGEquation]]:
    """
    Collect all compound identifiers appearing across KEGG reaction equations.

    Empty or missing equation strings are skipped.

    :param equations_by_rid:
        Mapping from reaction identifier to KEGG equation string.
    :type equations_by_rid: Mapping[str, Optional[str]]

    :returns:
        A tuple containing the sorted unique compound identifiers and the parsed
        equations keyed by reaction identifier.
    :rtype: tuple[list[str], dict[str, KEGGEquation]]

    Example
    -------
    .. code-block:: python

        compound_ids, parsed = get_compound_ids_from_equations(
            {"R00001": "C00001 + C00002 => C00003"}
        )
    """
    compound_ids: set[str] = set()
    parsed_by_rid: dict[str, KEGGEquation] = {}

    for rid, equation in equations_by_rid.items():
        if not equation:
            continue

        parsed = parse_equation(equation)
        parsed_by_rid[rid] = parsed

        for cid, _ in parsed.reactants + parsed.products:
            compound_ids.add(cid)

    return sorted(compound_ids), parsed_by_rid


def get_compound_ids_from_text(text: str) -> list[str]:
    """
    Extract sorted unique KEGG compound identifiers from free text.

    :param text:
        Source text that may contain KEGG compound identifiers.
    :type text: str

    :returns:
        Sorted unique KEGG compound identifiers.
    :rtype: list[str]

    Example
    -------
    .. code-block:: python

        ids = get_compound_ids_from_text("C00001 and C00002 appear here")
    """
    return sorted(set(_CID_PATTERN.findall(text or "")))


def molblock_to_smiles(molblock: Optional[str]) -> Optional[str]:
    """
    Convert a MOL block into canonical RDKit SMILES.

    :param molblock:
        MOL block text, typically retrieved from a KEGG compound record.
    :type molblock: Optional[str]

    :returns:
        Canonical RDKit SMILES when parsing succeeds, otherwise ``None``.
    :rtype: Optional[str]

    Example
    -------
    .. code-block:: python

        smiles = molblock_to_smiles(molblock_text)
    """
    if not molblock:
        return None

    molecule = Chem.MolFromMolBlock(molblock, sanitize=True)
    if molecule is None:
        return None

    return Chem.MolToSmiles(molecule)


def expand_stoichiometry(items: Sequence[CompoundStoich]) -> list[str]:
    """
    Expand stoichiometric pairs into repeated KEGG compound identifiers.

    For example, ``[("C00001", 2), ("C00002", 1)]`` becomes
    ``["C00001", "C00001", "C00002"]``.

    :param items:
        Compound/coefficient pairs.
    :type items: Sequence[tuple[str, int]]

    :returns:
        Expanded compound identifier list.
    :rtype: list[str]

    Example
    -------
    .. code-block:: python

        expanded = expand_stoichiometry([("C00001", 2), ("C00002", 1)])
    """
    expanded: list[str] = []
    for compound_id, coefficient in items:
        expanded.extend([compound_id] * coefficient)
    return expanded


def reaction_smiles_from_equation(
    parsed_equation: KEGGEquation,
    compounds_by_cid: Mapping[str, CompoundRecord],
) -> tuple[str, ReactionSmilesMissing]:
    """
    Build reaction SMILES from a parsed KEGG equation and compound table.

    Stoichiometric multiplicities are expanded into repeated SMILES fragments.
    Missing compounds are reported separately for reactants and products.

    :param parsed_equation:
        Parsed KEGG equation object.
    :type parsed_equation: KEGGEquation
    :param compounds_by_cid:
        Compound table keyed by KEGG compound identifier. Each record should
        provide a ``"smiles"`` entry.
    :type compounds_by_cid: Mapping[str, Mapping[str, Any]]

    :returns:
        Tuple ``(reaction_smiles, missing)`` where ``missing`` contains lists of
        unresolved reactant and product KEGG compound identifiers.
    :rtype: tuple[str, dict[str, list[str]]]

    Example
    -------
    .. code-block:: python

        parsed = parse_equation("C00001 + C00002 => C00003")
        reaction_smiles, missing = reaction_smiles_from_equation(
            parsed,
            {
                "C00001": {"smiles": "O"},
                "C00002": {"smiles": "CCO"},
                "C00003": {"smiles": "CC(=O)O"},
            },
        )
    """
    reactant_ids = expand_stoichiometry(parsed_equation.reactants)
    product_ids = expand_stoichiometry(parsed_equation.products)

    reactant_smiles: list[str] = []
    product_smiles: list[str] = []
    missing: ReactionSmilesMissing = {"reactants": [], "products": []}

    for compound_id in reactant_ids:
        smiles = compounds_by_cid.get(compound_id, {}).get("smiles")
        if smiles:
            reactant_smiles.append(smiles)
        else:
            missing["reactants"].append(compound_id)

    for compound_id in product_ids:
        smiles = compounds_by_cid.get(compound_id, {}).get("smiles")
        if smiles:
            product_smiles.append(smiles)
        else:
            missing["products"].append(compound_id)

    return ".".join(reactant_smiles) + ">>" + ".".join(product_smiles), missing
