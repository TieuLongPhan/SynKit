from __future__ import annotations

import re
from collections import Counter
from typing import Any, Literal, Optional

# ============================================================
# Optional SynKit imports
# ============================================================

try:
    from synkit.Chem.Reaction.canon_rsmi import CanonRSMI
    from synkit.Graph.ITS.its_expand import ITSExpand
    from synkit.IO import rsmi_to_its
except ImportError:
    CanonRSMI = None
    rsmi_to_its = None
    ITSExpand = None


# ============================================================
# Regex helpers
# ============================================================

BRACKET_ATOM_RE = re.compile(r"\[[^\[\]]+\]")
ATOM_MAP_RE = re.compile(r":(\d+)(?=\])")


# ============================================================
# Arrow-code parsing
# ============================================================


def parse_atom_list(text: str) -> list[int]:
    """Parse a comma-separated atom-map list.

    :param text: Atom-map text such as ``"10"`` or ``"10,11"``.
    :type text: str
    :returns: Parsed atom-map numbers.
    :rtype: list[int]
    """
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_arrow_step(step: str) -> tuple[list[int], list[int]]:
    """
    Convert one arrow-code step.

    For example, ``"10=20"`` becomes ``([10], [20])`` and
    ``"12=11,12"`` becomes ``([12], [11, 12])``.

    :param step: One arrow-code step containing a left and right side.
    :type step: str
    :returns: Parsed left-side and right-side atom-map lists.
    :rtype: tuple[list[int], list[int]]
    :raises ValueError: If ``step`` does not contain ``"="``.
    """
    if "=" not in step:
        raise ValueError(f"Invalid arrow step without '=': {step}")

    lhs, rhs = step.split("=", 1)
    return parse_atom_list(lhs), parse_atom_list(rhs)


def split_arrow_code(arrow_code: str) -> list[str]:
    """Split an arrow code into non-empty steps.

    :param arrow_code: Semicolon-separated arrow code.
    :type arrow_code: str
    :returns: Individual stripped arrow-code steps.
    :rtype: list[str]
    """
    return [s.strip() for s in arrow_code.split(";") if s.strip()]


def ef_arrow_code_to_arrow_code(ef_arrow_code: str) -> str:
    """Normalize an EF-SMIRKS hyphen-form flow code for SynKit conversion.

    EF-SMIRKS stores one electron-flow step as ``source-target``; SynKit's
    internal arrow-code helpers use ``source=target``.  For example,
    ``"10,20-20,21"`` becomes ``"10,20=20,21"``.

    :param ef_arrow_code: Semicolon-separated hyphen-form electron-flow code.
    :type ef_arrow_code: str
    :returns: Equivalent equals-form SynKit arrow code.
    :rtype: str
    :raises ValueError: If a step does not contain exactly one ``"-"``.
    """
    normalized_steps = []
    for step in split_arrow_code(ef_arrow_code):
        if step.count("-") != 1:
            raise ValueError(
                "An EF-SMIRKS electron-flow step must contain exactly one '-': "
                f"{step!r}"
            )
        lhs, rhs = step.split("-", 1)
        normalized_steps.append(f"{lhs.strip()}={rhs.strip()}")
    if not normalized_steps:
        raise ValueError("EF-SMIRKS electron-flow code is empty.")
    return ";".join(normalized_steps)


def arrow_code_to_ef_arrow_code(arrow_code: str) -> str:
    """Convert internal equals-form arrow code to EF-SMIRKS hyphen form.

    :param arrow_code: Semicolon-separated SynKit arrow code.
    :type arrow_code: str
    :returns: Equivalent hyphen-form electron-flow code.
    :rtype: str
    """
    return ";".join(step.replace("=", "-", 1) for step in split_arrow_code(arrow_code))


def split_ef_smirks(ef_smirks: str) -> tuple[str, str]:
    """Split an EF-SMIRKS string into reaction SMILES and flow code.

    An EF-SMIRKS is an atom-mapped reaction SMILES followed by whitespace and
    a hyphen-form electron-flow code, e.g. ``"[CH3:1][Cl:2]>>[CH3:1].[Cl-:2]
    1-1,2"``.

    :param ef_smirks: RSMI followed by whitespace and electron-flow code.
    :type ef_smirks: str
    :returns: Reaction SMILES and hyphen-form electron-flow code.
    :rtype: tuple[str, str]
    :raises ValueError: If the input lacks an RSMI arrow or flow code.
    """
    try:
        reaction_smiles, ef_arrow_code = ef_smirks.strip().rsplit(None, 1)
    except ValueError as exc:
        raise ValueError(
            "EF-SMIRKS must contain reaction SMILES followed by whitespace and "
            "an electron-flow code."
        ) from exc
    if ">>" not in reaction_smiles:
        raise ValueError("EF-SMIRKS reaction component must contain '>>'.")
    return reaction_smiles, ef_arrow_code


def arrow_atom_maps(arrow_code: str) -> set[int]:
    """Return all atom maps used in an arrow code.

    :param arrow_code: Semicolon-separated arrow code.
    :type arrow_code: str
    :returns: Atom-map numbers referenced by the code.
    :rtype: set[int]
    """
    maps: set[int] = set()

    for step in split_arrow_code(arrow_code):
        lhs, rhs = parse_arrow_step(step)
        maps.update(lhs)
        maps.update(rhs)

    return maps


def classify_arrow_shape(step: str) -> str:
    """Classify one arrow-code step.

    :param step: One arrow-code step.
    :type step: str
    :returns: Shape label such as ``"a=b"`` or ``"a,b=c,d"``.
    :rtype: str
    """
    lhs, rhs = parse_arrow_step(step)

    if len(lhs) == 1 and len(rhs) == 1:
        return "a=b"

    if len(lhs) == 1 and len(rhs) == 2:
        return "a=b,c"

    if len(lhs) == 2 and len(rhs) == 1:
        return "a,b=c"

    if len(lhs) == 2 and len(rhs) == 2:
        return "a,b=c,d"

    return f"unsupported:{len(lhs)},{len(rhs)}"


def check_arrow_code_coverage(arrow_codes: list[str]) -> dict[str, Any]:
    """Check which arrow-code shapes appear in a dataset.

    :param arrow_codes: Arrow codes to inspect.
    :type arrow_codes: list[str]
    :returns: Shape counts, unsupported steps, and an all-supported flag.
    :rtype: dict[str, Any]
    """
    shape_counter = Counter()
    unsupported = []

    for row_idx, arrow_code in enumerate(arrow_codes, start=1):
        for step in split_arrow_code(arrow_code):
            shape = classify_arrow_shape(step)
            shape_counter[shape] += 1

            if shape.startswith("unsupported"):
                unsupported.append(
                    {
                        "row_index": row_idx,
                        "arrow_code": arrow_code,
                        "step": step,
                        "shape": shape,
                    }
                )

    return {
        "shape_counts": dict(shape_counter),
        "unsupported": unsupported,
        "all_supported": len(unsupported) == 0,
    }


# ============================================================
# Atom-map preprocessing
# ============================================================


def extract_atom_maps_from_smiles(smiles: str) -> list[int]:
    """
    Extract atom-map numbers from bracket atoms.

    For example, ``"[CH:10]"`` yields ``10`` and ``"[N+:61]"`` yields ``61``.

    :param smiles: SMILES or SMIRKS fragment.
    :type smiles: str
    :returns: Atom-map numbers found in bracket atoms.
    :rtype: list[int]
    """
    return [int(x) for x in ATOM_MAP_RE.findall(smiles)]


def duplicate_atom_maps_in_side(smiles: str) -> dict[int, int]:
    """Find duplicated atom maps in one side of a reaction.

    :param smiles: Reactant-side or product-side SMILES text.
    :type smiles: str
    :returns: Mapping of duplicated atom-map numbers to occurrence counts.
    :rtype: dict[int, int]
    """
    counts = Counter(extract_atom_maps_from_smiles(smiles))
    return {atom_map: count for atom_map, count in counts.items() if count > 1}


def validate_arrow_maps(
    rsmi: str,
    arrow_code: str,
    raise_on_arrow_duplicates: bool = True,
    raise_on_missing_arrow_maps: bool = True,
) -> dict[str, Any]:
    """
    Validate atom maps before SynKit.

    Rules
    -----
    1. Duplicated atom maps used by arrow_code are fatal.
    2. Missing atom maps used by arrow_code are fatal.
    3. Duplicated non-arrow atom maps are warnings only, because they
       can be removed before SynKit expansion.

    :param rsmi: Reaction SMILES in ``reactants>>products`` format.
    :type rsmi: str
    :param arrow_code: Arrow code whose atom maps must be validated.
    :type arrow_code: str
    :param raise_on_arrow_duplicates: Whether duplicated arrow atom maps are fatal.
    :type raise_on_arrow_duplicates: bool
    :param raise_on_missing_arrow_maps: Whether missing arrow atom maps are fatal.
    :type raise_on_missing_arrow_maps: bool
    :returns: Validation diagnostics.
    :rtype: dict[str, Any]
    :raises ValueError: If the RSMI is malformed or enabled validation fails.
    """
    if ">>" not in rsmi:
        raise ValueError("RSMI must contain '>>'")

    reactants, products = rsmi.split(">>", 1)

    arrow_maps = arrow_atom_maps(arrow_code)

    reactant_maps = extract_atom_maps_from_smiles(reactants)
    product_maps = extract_atom_maps_from_smiles(products)

    all_raw_maps = set(reactant_maps) | set(product_maps)

    missing_arrow_maps = sorted(m for m in arrow_maps if m not in all_raw_maps)

    r_dupes = duplicate_atom_maps_in_side(reactants)
    p_dupes = duplicate_atom_maps_in_side(products)

    arrow_dupes = {
        "reactants": {m: c for m, c in r_dupes.items() if m in arrow_maps},
        "products": {m: c for m, c in p_dupes.items() if m in arrow_maps},
    }

    non_arrow_dupes = {
        "reactants": {m: c for m, c in r_dupes.items() if m not in arrow_maps},
        "products": {m: c for m, c in p_dupes.items() if m not in arrow_maps},
    }

    diagnostics = {
        "arrow_maps": sorted(arrow_maps),
        "missing_arrow_maps": missing_arrow_maps,
        "arrow_duplicate_maps": arrow_dupes,
        "non_arrow_duplicate_maps": non_arrow_dupes,
        "has_missing_arrow_maps": bool(missing_arrow_maps),
        "has_arrow_duplicate_maps": bool(
            arrow_dupes["reactants"] or arrow_dupes["products"]
        ),
        "has_non_arrow_duplicate_maps": bool(
            non_arrow_dupes["reactants"] or non_arrow_dupes["products"]
        ),
    }

    if raise_on_missing_arrow_maps and diagnostics["has_missing_arrow_maps"]:
        raise ValueError(
            "Some atom maps used in arrow_code are missing from the reaction SMILES. "
            f"Diagnostics: {diagnostics}"
        )

    if raise_on_arrow_duplicates and diagnostics["has_arrow_duplicate_maps"]:
        raise ValueError(
            "Some atom maps used in arrow_code are duplicated in the reaction SMILES. "
            f"Diagnostics: {diagnostics}"
        )

    return diagnostics


def remove_non_arrow_atom_maps(rsmi: str, arrow_code: str) -> str:
    """
    Keep only atom maps involved in arrow_code.
    Remove every other atom map.

    This is important because some source SMIRKS have duplicated
    non-arrow atom maps, e.g.

        [N+:61]2=[CH:61]

    If 61 is not used by arrow_code, we remove it and let SynKit
    CanonRSMI().expand_aam(...) generate clean full maps.

    :param rsmi: Reaction SMILES to clean.
    :type rsmi: str
    :param arrow_code: Arrow code whose atom maps should be preserved.
    :type arrow_code: str
    :returns: Reaction SMILES with non-arrow atom maps removed.
    :rtype: str
    """
    keep_maps = arrow_atom_maps(arrow_code)

    def clean_bracket_atom(match: re.Match) -> str:
        token = match.group(0)

        map_match = ATOM_MAP_RE.search(token)
        if map_match is None:
            return token

        atom_map = int(map_match.group(1))

        if atom_map in keep_maps:
            return token

        return ATOM_MAP_RE.sub("", token)

    return BRACKET_ATOM_RE.sub(clean_bracket_atom, rsmi)


# ============================================================
# Generic LP/B conversion
# ============================================================


def generic_convert_step(step: str) -> list[Any]:
    """
    Generic graph-independent conversion.

    Supported grammar
    -----------------
    a=b
        LP(a) forms bond a-b
        -> ["LP-/B+", [a], [a, b]]

    a=b,c
        LP(a) forms/increases bond b-c
        -> ["LP-/B+", [a], [b, c]]

    a,b=c
        bond a-b breaks; electrons end as LP on c
        -> ["B-/LP+", [a, b], [c]]

    a,b=c,d
        bond a-b becomes bond c-d
        -> ["B-/B+", [a, b], [c, d]]

    :param step: One arrow-code step.
    :type step: str
    :returns: Generic LP/B conversion record.
    :rtype: list[Any]
    :raises ValueError: If the step shape is unsupported.
    """
    lhs, rhs = parse_arrow_step(step)

    # a=b
    if len(lhs) == 1 and len(rhs) == 1:
        a = lhs[0]
        b = rhs[0]
        return ["LP-/B+", [a], [a, b]]

    # a=b,c
    if len(lhs) == 1 and len(rhs) == 2:
        return ["LP-/B+", lhs, rhs]

    # a,b=c
    if len(lhs) == 2 and len(rhs) == 1:
        return ["B-/LP+", lhs, rhs]

    # a,b=c,d
    if len(lhs) == 2 and len(rhs) == 2:
        return ["B-/B+", lhs, rhs]

    raise ValueError(f"Unsupported arrow step: {step}")


def generic_convert_arrow_code(arrow_code: str) -> list[list[Any]]:
    """Convert every step in an arrow code to generic LP/B form.

    :param arrow_code: Semicolon-separated arrow code.
    :type arrow_code: str
    :returns: Generic conversion records for each step.
    :rtype: list[list[Any]]
    """
    return [generic_convert_step(step) for step in split_arrow_code(arrow_code)]


# ============================================================
# SynKit ITS construction
# ============================================================


def build_its_from_rsmi(
    rsmi: str,
    arrow_code: str,
    expand_aam: bool = True,
    remove_non_arrow_maps: bool = True,
    aam_fallback_to_other_side: bool = False,
    require_constitution_preservation: bool = False,
    fold_unmapped_explicit_hydrogens: bool = False,
    ignore_stereochemistry: bool = False,
    explicit_hydrogen: bool = False,
    preserve_radical_state: bool = False,
):
    """
    Build SynKit ITS graph from reaction SMILES.

    Pipeline
    --------
    raw SMIRKS
        -> validate arrow atom maps
        -> remove non-arrow atom maps
        -> CanonRSMI().expand_aam(...)
        -> rsmi_to_its(...)

    :param rsmi: Reaction SMILES in ``reactants>>products`` format.
    :type rsmi: str
    :param arrow_code: Arrow code used to preserve relevant atom maps.
    :type arrow_code: str
    :param expand_aam: Whether to expand atom mapping before ITS construction.
    :type expand_aam: bool
    :param remove_non_arrow_maps: Whether to remove atom maps not used by the arrow code.
    :type remove_non_arrow_maps: bool
    :param aam_fallback_to_other_side: Retry partial-AAM reconstruction from
        the opposite endpoint if the preferred side fails.
    :type aam_fallback_to_other_side: bool
    :param require_constitution_preservation: Reject AAM expansion if either
        endpoint constitution changes.
    :type require_constitution_preservation: bool
    :param fold_unmapped_explicit_hydrogens: Fold removable unmapped explicit
        H atoms before partial-AAM reconstruction while retaining mapped H.
    :type fold_unmapped_explicit_hydrogens: bool
    :param ignore_stereochemistry: Remove stereo from the working partial-AAM
        RSMI before ITS construction while retaining the original endpoint
        constitution as the guard reference.
    :type ignore_stereochemistry: bool
    :param explicit_hydrogen: Preserve explicit H graph nodes during expanded
        RSMI serialization.
    :type explicit_hydrogen: bool
    :param preserve_radical_state: Preserve supplied endpoint radical counts
        during partial-AAM reconstruction.
    :type preserve_radical_state: bool
    :returns: ITS graph, expanded RSMI, cleaned RSMI, and validation diagnostics.
    :rtype: tuple
    :raises ImportError: If required SynKit conversion helpers are unavailable.
    """
    if CanonRSMI is None or rsmi_to_its is None:
        raise ImportError(
            "SynKit is not available. Run this code inside your SynKit environment."
        )

    diagnostics = validate_arrow_maps(
        rsmi=rsmi,
        arrow_code=arrow_code,
        raise_on_arrow_duplicates=True,
        raise_on_missing_arrow_maps=True,
    )

    if remove_non_arrow_maps:
        rsmi_for_its = remove_non_arrow_atom_maps(rsmi, arrow_code)
    else:
        rsmi_for_its = rsmi

    if expand_aam:
        expansion = ITSExpand().expand_aam_with_its_report(
            rsmi_for_its,
            relabel=False,
            preserve_older_map=True,
            fallback_to_other_side=aam_fallback_to_other_side,
            require_constitution_preservation=require_constitution_preservation,
            fold_unmapped_explicit_hydrogens=fold_unmapped_explicit_hydrogens,
            ignore_stereochemistry=ignore_stereochemistry,
            explicit_hydrogen=explicit_hydrogen,
            preserve_radical_state=preserve_radical_state,
        )
        expanded_rsmi = expansion.rsmi
        diagnostics["aam_expansion"] = expansion.to_dict()
    else:
        expanded_rsmi = rsmi_for_its
        diagnostics["aam_expansion"] = {
            "performed": False,
            "constitution_checked": False,
        }

    its = rsmi_to_its(expanded_rsmi)
    endpoint_states = _endpoint_electron_states(expanded_rsmi)
    if endpoint_states is not None:
        # ``typesGH`` retains endpoint charge/H presentation but not the
        # endpoint-local lone-pair/radical partition. Keep that partition as
        # conversion evidence on this transient ITS graph so compact arrow
        # annotations can be typed from the actual pre/post resources.
        its.graph["endpoint_electron_states"] = endpoint_states
        diagnostics["endpoint_electron_states_available"] = True
    else:
        diagnostics["endpoint_electron_states_available"] = False

    return its, expanded_rsmi, rsmi_for_its, diagnostics


def _endpoint_electron_states(
    mapped_reaction: str,
) -> dict[int, tuple[dict[str, Any], dict[str, Any]]] | None:
    """Return mapped pre/post nonbonding state for state-aware arrow typing."""
    try:
        from rdkit import Chem

        from synkit.IO.mol_to_graph import MolToGraph

        reactants, products = mapped_reaction.split(">>", 1)
        parser = Chem.SmilesParserParams()
        parser.removeHs = False
        sides: list[dict[int, dict[str, Any]]] = []
        for smiles in (reactants, products):
            mol = Chem.MolFromSmiles(smiles, parser)
            if mol is None:
                return None
            states: dict[int, dict[str, Any]] = {}
            for atom in mol.GetAtoms():
                atom_map = int(atom.GetAtomMapNum())
                if atom_map <= 0:
                    continue
                if atom_map in states:
                    return None
                states[atom_map] = {
                    "element": atom.GetSymbol(),
                    "charge": int(atom.GetFormalCharge()),
                    "radical": int(atom.GetNumRadicalElectrons()),
                    "lone_pairs": int(MolToGraph.estimate_lone_pairs(atom)),
                }
            sides.append(states)
        if set(sides[0]) != set(sides[1]):
            return None
        return {
            atom_map: (sides[0][atom_map], sides[1][atom_map]) for atom_map in sides[0]
        }
    except (ImportError, TypeError, ValueError):
        return None


# ============================================================
# ITS graph helpers
# ============================================================


def atom_map_to_nodes(its) -> dict[int, list[Any]]:
    """
    Build atom-map-number -> list of ITS node ids.

    This catches ambiguous duplicated atom maps after ITS construction.

    :param its: ITS graph.
    :type its: networkx.Graph
    :returns: Mapping from atom-map number to ITS node IDs.
    :rtype: dict[int, list[Any]]
    """
    mapping: dict[int, list[Any]] = {}

    for node, data in its.nodes(data=True):
        atom_map = int(data.get("atom_map", node))
        mapping.setdefault(atom_map, []).append(node)

    return mapping


def get_unique_node_for_atom_map(
    its,
    atom_map: int,
    strict: bool = True,
    atom_map_nodes: Optional[dict[int, list[Any]]] = None,
) -> Optional[Any]:
    """Get the unique ITS node corresponding to an atom map.

    :param its: ITS graph.
    :type its: networkx.Graph
    :param atom_map: Atom-map number to resolve.
    :type atom_map: int
    :param strict: Whether a missing atom map should raise.
    :type strict: bool
    :param atom_map_nodes: Optional precomputed atom-map to node index.
    :type atom_map_nodes: Optional[dict[int, list[Any]]]
    :returns: Unique ITS node ID, or ``None`` when missing and ``strict`` is false.
    :rtype: Optional[Any]
    :raises ValueError: If the atom map is missing in strict mode or is ambiguous.
    """
    mapping = atom_map_nodes if atom_map_nodes is not None else atom_map_to_nodes(its)
    nodes = mapping.get(int(atom_map), [])

    if len(nodes) == 0:
        if strict:
            raise ValueError(f"Atom map {atom_map} is missing from ITS graph.")
        return None

    if len(nodes) == 1:
        return nodes[0]

    raise ValueError(
        f"Atom map {atom_map} maps to multiple ITS nodes: {nodes}. "
        "This means atom mapping is ambiguous."
    )


def extract_order_from_edge_data(edge_data: Any) -> tuple[float, float]:
    """
    Extract SynKit ITS edge order.

    Expected normal edge format:
        {"order": (reactant_order, product_order)}

    MultiGraph-like fallback:
        {0: {"order": (reactant_order, product_order)}}

    :param edge_data: ITS edge attributes.
    :type edge_data: Any
    :returns: Reactant-side and product-side bond orders.
    :rtype: tuple[float, float]
    """
    if edge_data is None:
        return 0.0, 0.0

    if isinstance(edge_data, dict) and "order" in edge_data:
        order = edge_data["order"]
        return float(order[0]), float(order[1])

    if isinstance(edge_data, dict):
        for value in edge_data.values():
            if isinstance(value, dict) and "order" in value:
                order = value["order"]
                return float(order[0]), float(order[1])

    return 0.0, 0.0


def get_its_bond_order(
    its,
    atom_a: int,
    atom_b: int,
    strict: bool = True,
    context: str = "",
    atom_map_nodes: Optional[dict[int, list[Any]]] = None,
) -> tuple[float, float]:
    """
    Return ITS bond order for atom-map pair.

    For example, an edge with order ``(0.0, 1.0)`` represents new bond
    formation from reactants to products.

    :param its: ITS graph.
    :type its: networkx.Graph
    :param atom_a: First atom-map number.
    :type atom_a: int
    :param atom_b: Second atom-map number.
    :type atom_b: int
    :param strict: Whether missing nodes or edges should raise.
    :type strict: bool
    :param context: Optional context appended to strict-mode edge errors.
    :type context: str
    :param atom_map_nodes: Optional precomputed atom-map to node index.
    :type atom_map_nodes: Optional[dict[int, list[Any]]]
    :returns: Reactant-side and product-side bond orders.
    :rtype: tuple[float, float]
    :raises ValueError: If strict lookup fails.
    """
    node_a = get_unique_node_for_atom_map(
        its,
        atom_a,
        strict=strict,
        atom_map_nodes=atom_map_nodes,
    )
    node_b = get_unique_node_for_atom_map(
        its,
        atom_b,
        strict=strict,
        atom_map_nodes=atom_map_nodes,
    )

    if node_a is None or node_b is None:
        return 0.0, 0.0

    if not its.has_edge(node_a, node_b):
        if strict:
            extra = f" Context: {context}" if context else ""
            raise ValueError(
                f"ITS graph has no edge for atom maps {atom_a}-{atom_b}." f"{extra}"
            )
        return 0.0, 0.0

    edge_data = its.get_edge_data(node_a, node_b)
    return extract_order_from_edge_data(edge_data)


# ============================================================
# Sigma/Pi typing
# ============================================================


def is_zero(x: float, tol: float = 1e-6) -> bool:
    """Return whether a value is approximately zero.

    :param x: Value to compare.
    :type x: float
    :param tol: Absolute tolerance.
    :type tol: float
    :returns: Whether ``x`` is within tolerance of zero.
    :rtype: bool
    """
    return abs(x) < tol


def is_one(x: float, tol: float = 1e-6) -> bool:
    """Return whether a value is approximately one.

    :param x: Value to compare.
    :type x: float
    :param tol: Absolute tolerance.
    :type tol: float
    :returns: Whether ``x`` is within tolerance of one.
    :rtype: bool
    """
    return abs(x - 1.0) < tol


def bond_minus_type(reactant_order: float) -> str:
    """
    Type consumed bond/electron-pair source.

    Rules
    -----
    reactant_order == 1.0  -> Sigma-
    reactant_order >  1.0  -> Pi-
        includes double, triple, aromatic 1.5

    unknown                -> B-

    :param reactant_order: Bond order on the reactant side.
    :type reactant_order: float
    :returns: Typed consumed-bond label.
    :rtype: str
    """
    if is_one(reactant_order):
        return "Sigma-"

    if reactant_order > 1.0:
        return "Pi-"

    return "B-"


def bond_plus_type(
    reactant_order: float,
    product_order: float,
) -> str:
    """
    Type formed/increased bond destination.

    Rules
    -----
    0 -> 1      : Sigma+
    0 -> 1.5    : Sigma+, because new connectivity starts as sigma
    0 -> 2      : Sigma+, because new connectivity starts as sigma
    1 -> 2      : Pi+
    1.5 -> 2    : Pi+
    2 -> 3      : Pi+

    :param reactant_order: Bond order on the reactant side.
    :type reactant_order: float
    :param product_order: Bond order on the product side.
    :type product_order: float
    :returns: Typed formed-bond label.
    :rtype: str
    """
    if product_order <= 0:
        return "B+"

    # New bond formation. First new connectivity is sigma.
    if is_zero(reactant_order) and product_order > 0:
        return "Sigma+"

    # Existing bond order increases. Added component is pi.
    if product_order > reactant_order:
        return "Pi+"

    # Fallbacks.
    if is_one(product_order):
        return "Sigma+"

    if product_order > 1.0:
        return "Pi+"

    return "B+"


def _infer_atom_locus_kind(
    its,
    atom_map: int,
    *,
    role: Literal["source", "target"],
    electron_count: Literal[1, 2],
) -> str | None:
    """Infer an atom locus from its complete endpoint resource transition.

    Compact EPD/EF-SMIRKS identifies atom-map endpoints but does not say
    whether a one-atom fishhook touches a lone pair or a radical orbital. The
    pre/post lone-pair and radical deltas disambiguate the supported cases.
    """
    endpoint_states = its.graph.get("endpoint_electron_states")
    if not isinstance(endpoint_states, dict) or atom_map not in endpoint_states:
        return None
    reactant, product = endpoint_states[atom_map]
    lp_delta = int(product["lone_pairs"]) - int(reactant["lone_pairs"])
    radical_delta = int(product["radical"]) - int(reactant["radical"])
    electron_delta = 2 * lp_delta + radical_delta
    expected_delta = -electron_count if role == "source" else electron_count
    if electron_delta != expected_delta:
        return None

    if electron_count == 2:
        if radical_delta == 0 and lp_delta == (-1 if role == "source" else 1):
            return "LP"
        return None

    if role == "source":
        if radical_delta == -1 and lp_delta == 0:
            return "Rad"
        if radical_delta == 1 and lp_delta == -1:
            return "LP"
    else:
        if radical_delta == 1 and lp_delta == 0:
            return "Rad"
        if radical_delta == -1 and lp_delta == 1:
            return "LP"
    return None


def _infer_atom_to_atom_step(
    its,
    source_map: int,
    target_map: int,
    *,
    electron_count: Literal[1, 2],
) -> list[Any] | None:
    """Resolve a compact ``a=b`` step as a nonbonding atom transfer."""
    source_kind = _infer_atom_locus_kind(
        its, source_map, role="source", electron_count=electron_count
    )
    target_kind = _infer_atom_locus_kind(
        its, target_map, role="target", electron_count=electron_count
    )
    if source_kind is None or target_kind is None:
        return None
    return [f"{source_kind}-/{target_kind}+", [source_map], [target_map]]


# ============================================================
# Typed LP/Sigma/Pi conversion
# ============================================================


def typed_convert_step(
    step: str,
    its,
    strict_bond_lookup: bool = True,
    atom_map_nodes: Optional[dict[int, list[Any]]] = None,
    electron_count: Literal[1, 2] = 2,
) -> list[Any]:
    """
    Convert one arrow-code step into typed LP/Sigma/Pi format.

    Important
    ---------
    This function does NOT globally force Sigma/Pi from orbital_class.
    Each step is typed from local ITS bond-order changes.

    :param step: One arrow-code step.
    :type step: str
    :param its: ITS graph used for local bond-order lookup.
    :type its: networkx.Graph
    :param strict_bond_lookup: Whether missing bond lookups should raise.
    :type strict_bond_lookup: bool
    :param atom_map_nodes: Optional precomputed atom-map to node index.
    :type atom_map_nodes: Optional[dict[int, list[Any]]]
    :param electron_count: Electrons carried by the annotated arrow.
    :type electron_count: Literal[1, 2]
    :returns: Typed LP/Sigma/Pi conversion record.
    :rtype: list[Any]
    :raises ValueError: If the step shape is unsupported or strict lookup fails.
    """
    lhs, rhs = parse_arrow_step(step)

    # --------------------------------------------------------
    # Case 1: a=b
    # LP(a) forms bond a-b
    # --------------------------------------------------------
    if len(lhs) == 1 and len(rhs) == 1:
        a = lhs[0]
        b = rhs[0]

        node_a = get_unique_node_for_atom_map(
            its,
            a,
            strict=strict_bond_lookup,
            atom_map_nodes=atom_map_nodes,
        )
        node_b = get_unique_node_for_atom_map(
            its,
            b,
            strict=strict_bond_lookup,
            atom_map_nodes=atom_map_nodes,
        )
        has_bond = (
            node_a is not None and node_b is not None and its.has_edge(node_a, node_b)
        )

        r_order, p_order = get_its_bond_order(
            its,
            a,
            b,
            strict=False,
            context=step,
            atom_map_nodes=atom_map_nodes,
        )
        if is_zero(p_order - r_order):
            atom_transfer = _infer_atom_to_atom_step(
                its, a, b, electron_count=electron_count
            )
            if atom_transfer is not None:
                return atom_transfer
            if strict_bond_lookup and not has_bond:
                raise ValueError(
                    f"ITS graph has no edge for atom maps {a}-{b}. "
                    "Endpoint states do not support an atom-to-atom transfer."
                )
            if strict_bond_lookup and "endpoint_electron_states" in its.graph:
                raise ValueError(
                    "ATOM_TRANSFER_STATE_MISMATCH: one-atom endpoints neither change "
                    f"bond {a}-{b} nor match a supported nonbonding resource "
                    "transfer."
                )
        elif p_order < r_order and strict_bond_lookup:
            raise ValueError(
                f"ARROW_TARGET_BOND_NOT_FORMED: bond {a}-{b} decreases from "
                f"{r_order} to {p_order}."
            )
        plus = bond_plus_type(r_order, p_order)
        source_kind = (
            _infer_atom_locus_kind(its, a, role="source", electron_count=electron_count)
            or "LP"
        )

        return [f"{source_kind}-/{plus}", [a], [a, b]]

    # --------------------------------------------------------
    # Case 2: a=b,c
    # LP(a) forms/increases bond b-c
    #
    # Example:
    #   12=11,12
    #   LP on 12 forms/increases 11-12 bond
    # --------------------------------------------------------
    if len(lhs) == 1 and len(rhs) == 2:
        a = lhs[0]
        b, c = rhs

        r_order, p_order = get_its_bond_order(
            its,
            b,
            c,
            strict=strict_bond_lookup,
            context=step,
            atom_map_nodes=atom_map_nodes,
        )
        plus = bond_plus_type(r_order, p_order)
        source_kind = (
            _infer_atom_locus_kind(its, a, role="source", electron_count=electron_count)
            or "LP"
        )

        return [f"{source_kind}-/{plus}", [a], [b, c]]

    # --------------------------------------------------------
    # Case 3: a,b=c
    # bond a-b breaks; electrons become LP on c
    # --------------------------------------------------------
    if len(lhs) == 2 and len(rhs) == 1:
        a, b = lhs
        c = rhs[0]

        r_order, _p_order = get_its_bond_order(
            its,
            a,
            b,
            strict=strict_bond_lookup,
            context=step,
            atom_map_nodes=atom_map_nodes,
        )
        minus = bond_minus_type(r_order)
        target_kind = (
            _infer_atom_locus_kind(its, c, role="target", electron_count=electron_count)
            or "LP"
        )

        return [f"{minus}/{target_kind}+", [a, b], [c]]

    # --------------------------------------------------------
    # Case 4: a,b=c,d
    # bond a-b becomes bond c-d
    # --------------------------------------------------------
    if len(lhs) == 2 and len(rhs) == 2:
        a, b = lhs
        c, d = rhs

        src_r_order, _src_p_order = get_its_bond_order(
            its,
            a,
            b,
            strict=strict_bond_lookup,
            context=f"source of {step}",
            atom_map_nodes=atom_map_nodes,
        )

        dst_r_order, dst_p_order = get_its_bond_order(
            its,
            c,
            d,
            strict=strict_bond_lookup,
            context=f"destination of {step}",
            atom_map_nodes=atom_map_nodes,
        )

        minus = bond_minus_type(src_r_order)
        plus = bond_plus_type(dst_r_order, dst_p_order)

        return [f"{minus}/{plus}", [a, b], [c, d]]

    raise ValueError(f"Unsupported arrow step: {step}")


def typed_convert_arrow_code(
    arrow_code: str,
    its,
    strict_bond_lookup: bool = True,
    electron_count: Literal[1, 2] = 2,
) -> list[list[Any]]:
    """Convert every step in an arrow code to typed LP/Sigma/Pi form.

    :param arrow_code: Semicolon-separated arrow code.
    :type arrow_code: str
    :param its: ITS graph used for local bond-order lookup.
    :type its: networkx.Graph
    :param strict_bond_lookup: Whether missing bond lookups should raise.
    :type strict_bond_lookup: bool
    :param electron_count: Electrons carried by every supplied arrow.
    :type electron_count: Literal[1, 2]
    :returns: Typed conversion records for each step.
    :rtype: list[list[Any]]
    """
    atom_map_nodes = atom_map_to_nodes(its)

    return [
        typed_convert_step(
            step=step,
            its=its,
            strict_bond_lookup=strict_bond_lookup,
            atom_map_nodes=atom_map_nodes,
            electron_count=electron_count,
        )
        for step in split_arrow_code(arrow_code)
    ]


# ============================================================
# Main public conversion functions
# ============================================================


def convert_arrow_code(
    arrow_code: str,
    its=None,
    strict_bond_lookup: bool = True,
) -> dict[str, Any]:
    """
    Convert arrow code into generic and typed formats.

    If ``its`` is ``None``, ``typed_converted`` is ``None``.

    :param arrow_code: Semicolon-separated arrow code.
    :type arrow_code: str
    :param its: Optional ITS graph for typed conversion.
    :type its: Optional[networkx.Graph]
    :param strict_bond_lookup: Whether missing typed bond lookups should raise.
    :type strict_bond_lookup: bool
    :returns: Arrow code with generic and optional typed conversions.
    :rtype: dict[str, Any]
    """
    converted = generic_convert_arrow_code(arrow_code)

    if its is None:
        typed_converted = None
    else:
        typed_converted = typed_convert_arrow_code(
            arrow_code=arrow_code,
            its=its,
            strict_bond_lookup=strict_bond_lookup,
        )

    return {
        "arrow_code": arrow_code,
        "converted": converted,
        "typed_converted": typed_converted,
    }


def convert_reaction_arrow(
    reaction_smiles: str,
    arrow_code: str,
    orbital_class: Optional[str] = None,
    expand_aam: bool = True,
    remove_non_arrow_maps: bool = True,
    strict_bond_lookup: bool = True,
) -> dict[str, Any]:
    """
    Complete wrapper.

    reaction SMILES + arrow code
        -> clean non-arrow maps
        -> expand AAM with SynKit
        -> ITS graph
        -> generic converted
        -> typed converted

    orbital_class is stored as metadata only.
    It is not used to force Sigma/Pi typing.

    :param reaction_smiles: Reaction SMILES in ``reactants>>products`` format.
    :type reaction_smiles: str
    :param arrow_code: Semicolon-separated arrow code.
    :type arrow_code: str
    :param orbital_class: Optional source-dataset orbital classification metadata.
    :type orbital_class: Optional[str]
    :param expand_aam: Whether to expand atom mapping before ITS construction.
    :type expand_aam: bool
    :param remove_non_arrow_maps: Whether to remove atom maps not used by the arrow code.
    :type remove_non_arrow_maps: bool
    :param strict_bond_lookup: Whether missing typed bond lookups should raise.
    :type strict_bond_lookup: bool
    :returns: Conversion result and ITS preparation metadata.
    :rtype: dict[str, Any]
    """
    its, expanded_rsmi, rsmi_for_its, diagnostics = build_its_from_rsmi(
        rsmi=reaction_smiles,
        arrow_code=arrow_code,
        expand_aam=expand_aam,
        remove_non_arrow_maps=remove_non_arrow_maps,
    )

    result = convert_arrow_code(
        arrow_code=arrow_code,
        its=its,
        strict_bond_lookup=strict_bond_lookup,
    )

    result["reaction_smiles"] = reaction_smiles
    result["rsmi_for_its"] = rsmi_for_its
    result["expanded_rsmi"] = expanded_rsmi
    result["orbital_class"] = orbital_class
    result["diagnostics"] = diagnostics

    return result


def ef_smirks_to_epd(
    ef_smirks: str,
    orbital_class: Optional[str] = None,
    strict_bond_lookup: bool = True,
) -> dict[str, Any]:
    """Complete AAM and convert an EF-SMIRKS string into EPD data.

    The supplied atom maps used by the electron-flow code are preserved. All
    remaining atoms receive maps through :class:`ITSExpand`.  The result
    contains both generic ``epd`` and Lewis-wavefunction-aware ``epd_lw``
    records; the latter uses local ITS bond orders to distinguish sigma and pi
    edits.

    :param ef_smirks: RSMI followed by a hyphen-form electron-flow code.
    :type ef_smirks: str
    :param orbital_class: Optional source-dataset metadata.
    :type orbital_class: Optional[str]
    :param strict_bond_lookup: Whether missing ITS bond lookups should raise.
    :type strict_bond_lookup: bool
    :returns: Completed AAM, generic EPD, typed ``epd_lw``, and diagnostics.
    :rtype: dict[str, Any]
    """
    reaction_smiles, ef_arrow_code = split_ef_smirks(ef_smirks)
    arrow_code = ef_arrow_code_to_arrow_code(ef_arrow_code)
    result = convert_reaction_arrow(
        reaction_smiles=reaction_smiles,
        arrow_code=arrow_code,
        orbital_class=orbital_class,
        expand_aam=True,
        remove_non_arrow_maps=True,
        strict_bond_lookup=strict_bond_lookup,
    )
    result.update(
        {
            "ef_smirks": ef_smirks,
            "ef_arrow_code": ef_arrow_code,
            "complete_aam": result["expanded_rsmi"],
            "epd": result["converted"],
            "epd_lw": result["typed_converted"],
        }
    )
    return result


def epd_to_ef_smirks(complete_aam: str, epd: list[list[Any]]) -> str:
    """Reconstruct an EF-SMIRKS string from complete AAM and EPD records.

    ``epd`` may contain generic action names (``"B-/B+"``) or typed Lewis
    names (``"Sigma-/Pi+"``). Only action direction and source/target map
    lists are needed to reconstruct the electron-flow code.

    :param complete_aam: Fully atom-mapped reaction SMILES.
    :type complete_aam: str
    :param epd: Generic or typed EPD records of ``[action, source, target]``.
    :type epd: list[list[Any]]
    :returns: EF-SMIRKS text with a hyphen-form electron-flow code.
    :rtype: str
    :raises ValueError: If AAM or EPD records are malformed or unsupported.
    """
    if complete_aam.count(">>") != 1:
        raise ValueError("complete_aam must contain exactly one '>>'.")
    if not epd:
        raise ValueError("epd must contain at least one electron-pushing step.")

    def format_maps(atom_maps: list[int]) -> str:
        """Format an atom-map sequence for one EF-SMIRKS endpoint."""
        return ",".join(str(atom_map) for atom_map in atom_maps)

    steps = []
    for record in epd:
        if len(record) < 3:
            raise ValueError(f"Invalid EPD record: {record!r}")
        action, source, target = record[:3]
        action_parts = str(action).split("/", 1)
        if len(action_parts) != 2:
            raise ValueError(f"Invalid EPD action: {action!r}")
        consumed, produced = action_parts
        source = [int(atom_map) for atom_map in source]
        target = [int(atom_map) for atom_map in target]

        if consumed == "LP-" and len(source) == 1 and len(target) in {1, 2}:
            lhs, rhs = source, target
        elif produced == "LP+" and len(source) == 2 and len(target) == 1:
            lhs, rhs = source, target
        elif (
            consumed.endswith("-")
            and produced.endswith("+")
            and len(source) == 2
            and len(target) == 2
        ):
            lhs, rhs = source, target
        else:
            raise ValueError(f"Unsupported EPD record for EF-SMIRKS: {record!r}")

        steps.append(f"{format_maps(lhs)}-{format_maps(rhs)}")

    return f"{complete_aam} {';'.join(steps)}"


def convert_record(
    record: dict[str, Any],
    reaction_key: str = "SMIRKS",
    arrow_key: str = "arrow_code",
    orbital_key: str = "orbital pair classification",
    expand_aam: bool = True,
    remove_non_arrow_maps: bool = True,
    strict_bond_lookup: bool = True,
) -> dict[str, Any]:
    """
    Convert one dictionary record.

    Expected input keys
    -------------------
    {
        "SMIRKS": "...>>...",
        "arrow_code": "...",
        "orbital pair classification": "pi_empty"
    }

    :param record: Source record to convert.
    :type record: dict[str, Any]
    :param reaction_key: Key containing reaction SMILES.
    :type reaction_key: str
    :param arrow_key: Key containing arrow code.
    :type arrow_key: str
    :param orbital_key: Key containing optional orbital classification metadata.
    :type orbital_key: str
    :param expand_aam: Whether to expand atom mapping before ITS construction.
    :type expand_aam: bool
    :param remove_non_arrow_maps: Whether to remove atom maps not used by the arrow code.
    :type remove_non_arrow_maps: bool
    :param strict_bond_lookup: Whether missing typed bond lookups should raise.
    :type strict_bond_lookup: bool
    :returns: Converted record with original metadata preserved.
    :rtype: dict[str, Any]
    """
    reaction_smiles = record[reaction_key]
    arrow_code = record[arrow_key]
    orbital_class = record.get(orbital_key)

    result = convert_reaction_arrow(
        reaction_smiles=reaction_smiles,
        arrow_code=arrow_code,
        orbital_class=orbital_class,
        expand_aam=expand_aam,
        remove_non_arrow_maps=remove_non_arrow_maps,
        strict_bond_lookup=strict_bond_lookup,
    )

    # Preserve original metadata.
    for key, value in record.items():
        if key not in result:
            result[key] = value

    return result


def convert_records(
    records: list[dict[str, Any]],
    reaction_key: str = "SMIRKS",
    arrow_key: str = "arrow_code",
    orbital_key: str = "orbital pair classification",
    expand_aam: bool = True,
    remove_non_arrow_maps: bool = True,
    strict_bond_lookup: bool = True,
    keep_errors: bool = False,
) -> list[dict[str, Any]]:
    """
    Batch conversion.

    keep_errors=False:
        raise immediately on first error.

    keep_errors=True:
        collect errors into result dictionaries.

    :param records: Source records to convert.
    :type records: list[dict[str, Any]]
    :param reaction_key: Key containing reaction SMILES.
    :type reaction_key: str
    :param arrow_key: Key containing arrow code.
    :type arrow_key: str
    :param orbital_key: Key containing optional orbital classification metadata.
    :type orbital_key: str
    :param expand_aam: Whether to expand atom mapping before ITS construction.
    :type expand_aam: bool
    :param remove_non_arrow_maps: Whether to remove atom maps not used by the arrow code.
    :type remove_non_arrow_maps: bool
    :param strict_bond_lookup: Whether missing typed bond lookups should raise.
    :type strict_bond_lookup: bool
    :param keep_errors: Whether to collect conversion failures instead of raising.
    :type keep_errors: bool
    :returns: Converted records, including failures when ``keep_errors`` is true.
    :rtype: list[dict[str, Any]]
    """
    results = []

    for idx, record in enumerate(records, start=1):
        try:
            result = convert_record(
                record=record,
                reaction_key=reaction_key,
                arrow_key=arrow_key,
                orbital_key=orbital_key,
                expand_aam=expand_aam,
                remove_non_arrow_maps=remove_non_arrow_maps,
                strict_bond_lookup=strict_bond_lookup,
            )
            result["row_index"] = idx
            results.append(result)

        except Exception as e:
            if not keep_errors:
                print("=" * 100)
                print(f"FAILED ROW {idx}")
                print("=" * 100)
                print("orbital_class:", record.get(orbital_key))
                print("arrow_code:", record.get(arrow_key))
                print("error:", repr(e))
                print("=" * 100)
                raise

            failed = dict(record)
            failed["row_index"] = idx
            failed["error"] = repr(e)
            results.append(failed)

    return results


# ============================================================
# Debug helpers
# ============================================================


def debug_arrow_bond_orders(
    reaction_smiles: str,
    arrow_code: str,
    expand_aam: bool = True,
    remove_non_arrow_maps: bool = True,
    strict_bond_lookup: bool = True,
) -> None:
    """Print the ITS bond orders used by each arrow step.

    :param reaction_smiles: Reaction SMILES in ``reactants>>products`` format.
    :type reaction_smiles: str
    :param arrow_code: Semicolon-separated arrow code.
    :type arrow_code: str
    :param expand_aam: Whether to expand atom mapping before ITS construction.
    :type expand_aam: bool
    :param remove_non_arrow_maps: Whether to remove atom maps not used by the arrow code.
    :type remove_non_arrow_maps: bool
    :param strict_bond_lookup: Whether missing typed bond lookups should raise.
    :type strict_bond_lookup: bool
    :returns: ``None``.
    :rtype: None
    """
    its, expanded_rsmi, rsmi_for_its, diagnostics = build_its_from_rsmi(
        rsmi=reaction_smiles,
        arrow_code=arrow_code,
        expand_aam=expand_aam,
        remove_non_arrow_maps=remove_non_arrow_maps,
    )

    print("Diagnostics:")
    print(diagnostics)
    print()

    print("RSMI used for ITS:")
    print(rsmi_for_its)
    print()

    print("Expanded RSMI:")
    print(expanded_rsmi)
    print()

    for step in split_arrow_code(arrow_code):
        lhs, rhs = parse_arrow_step(step)

        print(f"Step: {step}")
        print(f"  shape: {classify_arrow_shape(step)}")

        if len(lhs) == 1 and len(rhs) == 1:
            a = lhs[0]
            b = rhs[0]
            print(
                f"  destination bond {a}-{b}: "
                f"{get_its_bond_order(its, a, b, strict=strict_bond_lookup, context=step)}"
            )

        elif len(lhs) == 1 and len(rhs) == 2:
            a = lhs[0]
            b, c = rhs
            print(f"  LP source atom    {a}")
            print(
                f"  destination bond {b}-{c}: "
                f"{get_its_bond_order(its, b, c, strict=strict_bond_lookup, context=step)}"
            )

        elif len(lhs) == 2 and len(rhs) == 1:
            a, b = lhs
            c = rhs[0]
            print(
                f"  source bond      {a}-{b}: "
                f"{get_its_bond_order(its, a, b, strict=strict_bond_lookup, context=step)}"
            )
            print(f"  LP destination atom {c}")

        elif len(lhs) == 2 and len(rhs) == 2:
            a, b = lhs
            c, d = rhs
            print(
                f"  source bond      {a}-{b}: "
                f"{get_its_bond_order(its, a, b, strict=strict_bond_lookup, context=step)}"
            )
            print(
                f"  destination bond {c}-{d}: "
                f"{get_its_bond_order(its, c, d, strict=strict_bond_lookup, context=step)}"
            )

        print()


def debug_record(
    record: dict[str, Any],
    reaction_key: str = "SMIRKS",
    arrow_key: str = "arrow_code",
    orbital_key: str = "orbital pair classification",
) -> dict[str, Any]:
    """Print full debug output for one record.

    :param record: Source record to inspect.
    :type record: dict[str, Any]
    :param reaction_key: Key containing reaction SMILES.
    :type reaction_key: str
    :param arrow_key: Key containing arrow code.
    :type arrow_key: str
    :param orbital_key: Key containing optional orbital classification metadata.
    :type orbital_key: str
    :returns: Converted record.
    :rtype: dict[str, Any]
    """
    from pprint import pprint

    rsmi = record[reaction_key]
    arrow_code = record[arrow_key]
    orbital_class = record.get(orbital_key)

    print("=" * 100)
    print("DEBUG RECORD")
    print("=" * 100)

    print("orbital_class:")
    print(orbital_class)
    print()

    print("arrow_code:")
    print(arrow_code)
    print()

    print("arrow shapes:")
    for step in split_arrow_code(arrow_code):
        print(f"  {step:25s} -> {classify_arrow_shape(step)}")
    print()

    print("raw map diagnostics:")
    diagnostics = validate_arrow_maps(
        rsmi=rsmi,
        arrow_code=arrow_code,
        raise_on_arrow_duplicates=False,
        raise_on_missing_arrow_maps=False,
    )
    pprint(diagnostics, width=160)
    print()

    print("RSMI after removing non-arrow maps:")
    cleaned = remove_non_arrow_atom_maps(rsmi, arrow_code)
    print(cleaned)
    print()

    print("Arrow bond orders:")
    debug_arrow_bond_orders(
        reaction_smiles=rsmi,
        arrow_code=arrow_code,
        expand_aam=True,
        remove_non_arrow_maps=True,
        strict_bond_lookup=True,
    )

    result = convert_reaction_arrow(
        reaction_smiles=rsmi,
        arrow_code=arrow_code,
        orbital_class=orbital_class,
        expand_aam=True,
        remove_non_arrow_maps=True,
        strict_bond_lookup=True,
    )

    print("converted:")
    pprint(result["converted"], width=120)
    print()

    print("typed_converted:")
    pprint(result["typed_converted"], width=120)
    print()

    return result


def check_typed_conversion_quality(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Check whether typed conversions still contain generic B-/B+ labels.

    :param results: Conversion results to inspect.
    :type results: list[dict[str, Any]]
    :returns: Error and untyped-step diagnostics.
    :rtype: dict[str, Any]
    """
    errors = []
    untyped = []

    for result in results:
        if "error" in result:
            errors.append(result)
            continue

        typed = result.get("typed_converted")

        if typed is None:
            untyped.append(
                {
                    "row_index": result.get("row_index"),
                    "reason": "typed_converted is None",
                }
            )
            continue

        for step in typed:
            label = step[0]
            if "B-" in label or "B+" in label:
                untyped.append(
                    {
                        "row_index": result.get("row_index"),
                        "orbital_class": result.get("orbital_class"),
                        "arrow_code": result.get("arrow_code"),
                        "step": step,
                    }
                )

    return {
        "n_results": len(results),
        "n_errors": len(errors),
        "n_untyped_steps": len(untyped),
        "all_fully_typed": len(errors) == 0 and len(untyped) == 0,
        "errors": errors,
        "untyped": untyped,
    }
