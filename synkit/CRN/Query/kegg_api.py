from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rdkit import Chem


_CID_PATTERN = re.compile(r"C\d{5}")
_MODULE_PATTERN = re.compile(r"M\d{5}")
_TERM_PATTERN = re.compile(r"^(?:(\d+)\s+)?(C\d{5})$")


@dataclass(frozen=True)
class KEGGEquation:
    """
    Parsed KEGG reaction equation.

    :param reactants:
        Left-hand side as ``[(compound_id, stoich), ...]``.
    :type reactants: List[Tuple[str, int]]

    :param products:
        Right-hand side as ``[(compound_id, stoich), ...]``.
    :type products: List[Tuple[str, int]]

    :param reversible:
        Whether the equation uses a reversible arrow.
    :type reversible: bool
    """

    reactants: List[Tuple[str, int]]
    products: List[Tuple[str, int]]
    reversible: bool


def parse_kegg_field_blocks(text: str, field: str) -> List[str]:
    """
    Extract KEGG flatfile field payloads, including continuation lines.

    :param text:
        Raw KEGG flatfile text.
    :type text: str

    :param field:
        Flatfile field name, e.g. ``"MODULE"``, ``"REACTION"``, ``"EQUATION"``,
        or ``"NAME"``.
    :type field: str

    :returns:
        List of payload strings, one per field occurrence.
    :rtype: List[str]
    """
    payloads: List[str] = []
    lines = text.splitlines()
    i = 0

    while i < len(lines):
        line = lines[i]
        if line.startswith(field):
            payload = line[len(field) :].strip()
            j = i + 1
            continuation: List[str] = []

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
    Normalize a module token to canonical KEGG module form.

    Examples:
        - ``"hsa_M00001" -> "M00001"``
        - ``"M00001" -> "M00001"``

    :param module_id:
        Raw module token.
    :type module_id: str

    :returns:
        Canonical module identifier or ``None`` if not found.
    :rtype: Optional[str]
    """
    match = _MODULE_PATTERN.search(module_id)
    return match.group(0) if match else None


def parse_equation_side(side: str) -> List[Tuple[str, int]]:
    """
    Parse one side of a KEGG equation.

    Example:
        ``"2 C00139 + C00001"`` becomes
        ``[("C00139", 2), ("C00001", 1)]``.

    :param side:
        One side of the equation.
    :type side: str

    :returns:
        Parsed terms as ``(compound_id, coefficient)`` pairs.
    :rtype: List[Tuple[str, int]]

    :raises ValueError:
        If a term cannot be parsed.
    """
    side = side.strip()
    if not side:
        return []

    terms = [term.strip() for term in side.split("+")]
    parsed: List[Tuple[str, int]] = []

    for term in terms:
        match = _TERM_PATTERN.match(term)
        if not match:
            raise ValueError(f"Unparsed KEGG equation term: {term!r}")

        coefficient = int(match.group(1)) if match.group(1) else 1
        compound_id = match.group(2)
        parsed.append((compound_id, coefficient))

    return parsed


def parse_kegg_equation(equation: str) -> KEGGEquation:
    """
    Parse a KEGG equation string into structured reactants/products.

    Supported arrows:
        - ``<=>``
        - ``=>``
        - ``<=``

    :param equation:
        KEGG equation string.
    :type equation: str

    :returns:
        Parsed equation object.
    :rtype: KEGGEquation

    :raises ValueError:
        If the equation arrow is not recognized.
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
        reactants=parse_equation_side(lhs),
        products=parse_equation_side(rhs),
        reversible=reversible,
    )


def collect_compound_ids_from_equations(
    equations_by_rid: Dict[str, Optional[str]],
) -> Tuple[List[str], Dict[str, KEGGEquation]]:
    """
    Collect all compound IDs appearing in a reaction dictionary.

    :param equations_by_rid:
        Mapping from reaction ID to KEGG equation string.
    :type equations_by_rid: Dict[str, Optional[str]]

    :returns:
        Tuple ``(sorted_compound_ids, parsed_by_reaction_id)``.
    :rtype: Tuple[List[str], Dict[str, KEGGEquation]]
    """
    compound_ids = set()
    parsed_by_rid: Dict[str, KEGGEquation] = {}

    for rid, equation in equations_by_rid.items():
        if not equation:
            continue

        parsed = parse_kegg_equation(equation)
        parsed_by_rid[rid] = parsed

        for cid, _ in parsed.reactants + parsed.products:
            compound_ids.add(cid)

    return sorted(compound_ids), parsed_by_rid


def extract_compound_ids_from_text(text: str) -> List[str]:
    """
    Extract all KEGG compound IDs from free text.

    :param text:
        Source text.
    :type text: str

    :returns:
        Sorted unique KEGG compound IDs.
    :rtype: List[str]
    """
    return sorted(set(_CID_PATTERN.findall(text or "")))


def molblock_to_smiles(molblock: Optional[str]) -> Optional[str]:
    """
    Convert a MOL block into SMILES using RDKit.

    :param molblock:
        KEGG MOL block text.
    :type molblock: Optional[str]

    :returns:
        Canonical RDKit SMILES if parsing succeeds, otherwise ``None``.
    :rtype: Optional[str]
    """
    if not molblock:
        return None

    molecule = Chem.MolFromMolBlock(molblock, sanitize=True)
    if molecule is None:
        return None

    return Chem.MolToSmiles(molecule)


def expand_stoichiometry(items: Sequence[Tuple[str, int]]) -> List[str]:
    """
    Expand ``[(cid, coeff), ...]`` into repeated IDs.

    Example:
        ``[("C00001", 2), ("C00002", 1)]`` becomes
        ``["C00001", "C00001", "C00002"]``.

    :param items:
        Compound/coefficient pairs.
    :type items: Sequence[Tuple[str, int]]

    :returns:
        Expanded compound ID list.
    :rtype: List[str]
    """
    expanded: List[str] = []
    for compound_id, coefficient in items:
        expanded.extend([compound_id] * coefficient)
    return expanded


def reaction_smiles_from_equation(
    parsed_equation: KEGGEquation,
    compounds_by_cid: Dict[str, Dict[str, Any]],
) -> Tuple[str, Dict[str, List[str]]]:
    """
    Build reaction SMILES from a parsed KEGG equation and a compound table.

    Stoichiometric multiplicities are expanded into repeated SMILES entries.

    :param parsed_equation:
        Parsed equation object.
    :type parsed_equation: KEGGEquation

    :param compounds_by_cid:
        Compound table keyed by KEGG compound ID. Each entry should provide
        ``"smiles"``.
    :type compounds_by_cid: Dict[str, Dict[str, Any]]

    :returns:
        Tuple ``(reaction_smiles, missing)`` where ``missing`` contains
        missing reactant/product compound IDs.
    :rtype: Tuple[str, Dict[str, List[str]]]
    """
    reactant_ids = expand_stoichiometry(parsed_equation.reactants)
    product_ids = expand_stoichiometry(parsed_equation.products)

    reactant_smiles: List[str] = []
    product_smiles: List[str] = []
    missing = {"reactants": [], "products": []}

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