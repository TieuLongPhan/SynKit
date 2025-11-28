from __future__ import annotations

from typing import Dict, List

import pandas as pd

from .core import CRNSpecies, CRNReaction, CRNNetwork


# Tokens that denote the empty complex (no species)
_EMPTY_COMPLEX_TOKENS = {"0", "Ø", "ø", "∅"}


def _parse_side(
    side: str,
    species_index: Dict[str, int],
    species_list: List[CRNSpecies],
) -> Dict[int, float]:
    """
    Parse a reaction side like ``"2A + B"`` into a stoichiometric mapping.

    Special handling:

    * If the trimmed side is one of ``{"0", "Ø", "ø", "∅"}``, an empty
      mapping is returned, representing the zero complex.

    :param side: String representation of reactants or products.
    :type side: str
    :param species_index: Mapping from species name to index (updated in-place).
    :type species_index: Dict[str, int]
    :param species_list: List of species objects (updated in-place).
    :type species_list: List[CRNSpecies]
    :returns: Mapping from species index to stoichiometric coefficient.
    :rtype: Dict[int, float]
    """
    side = side.strip()
    if not side:
        return {}

    # Treat "0", "Ø", "∅" as the empty complex, not a real species
    if side in _EMPTY_COMPLEX_TOKENS:
        return {}

    mapping: Dict[int, float] = {}
    terms = [t.strip() for t in side.split("+") if t.strip()]

    for term in terms:
        # Allow forms: "A", "2A", "3 * A"
        coeff = 1.0
        name = term

        if " " in term:
            # e.g. "2 A" or "2 * A"
            parts = term.replace("*", " ").split()
            if len(parts) == 2:
                try:
                    coeff = float(parts[0])
                    name = parts[1]
                except ValueError:
                    coeff = 1.0
                    name = term
        else:
            # e.g. "2A"
            i = 0
            while i < len(term) and term[i].isdigit():
                i += 1
            if i > 0:
                try:
                    coeff = float(term[:i])
                    name = term[i:]
                except ValueError:
                    coeff = 1.0
                    name = term

        name = name.strip()
        if not name:
            continue

        if name not in species_index:
            species_index[name] = len(species_list)
            species_list.append(CRNSpecies(name=name))

        idx = species_index[name]
        mapping[idx] = mapping.get(idx, 0.0) + coeff

    return mapping


def crn_from_rxn_table(df: pd.DataFrame) -> CRNNetwork:
    """
    Build a :class:`CRNNetwork` from a pandas table of reactions.

    Expected columns (minimal):

    * ``reactants`` – string, e.g. ``"A + 2B"``
    * ``products`` – string, e.g. ``"C"``
    * ``reversible`` – optional bool, default False

    :param df: Reaction table.
    :type df: pandas.DataFrame
    :returns: Constructed CRN network.
    :rtype: CRNNetwork
    """
    species_index: Dict[str, int] = {}
    species_list: List[CRNSpecies] = []
    reactions: List[CRNReaction] = []

    if "reactants" not in df.columns or "products" not in df.columns:
        raise ValueError("DataFrame must contain 'reactants' and 'products' columns.")

    for _, row in df.iterrows():
        reactants_str = str(row["reactants"])
        products_str = str(row["products"])
        rev = bool(row["reversible"]) if "reversible" in df.columns else False

        reactants = _parse_side(reactants_str, species_index, species_list)
        products = _parse_side(products_str, species_index, species_list)

        reactions.append(
            CRNReaction(
                reactants=reactants,
                products=products,
                reversible=rev,
                metadata={},
            )
        )

    return CRNNetwork(species=species_list, reactions=reactions)


def crn_from_sbml(path: str) -> CRNNetwork:
    """
    Build a :class:`CRNNetwork` from an SBML file.

    This function requires the :mod:`python-libsbml` package to be installed.

    :param path: Path to the SBML file.
    :type path: str
    :returns: Constructed CRN network.
    :rtype: CRNNetwork
    :raises ImportError: If :mod:`libsbml` is not installed.
    :raises RuntimeError: If the SBML file cannot be parsed.
    """
    try:
        import libsbml  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "crn_from_sbml requires the 'python-libsbml' package."
        ) from exc

    reader = libsbml.SBMLReader()
    document = reader.readSBML(path)
    if document.getNumErrors() > 0:
        raise RuntimeError(
            f"Error reading SBML file '{path}': {document.getErrorLog().toString()}"
        )

    model = document.getModel()
    if model is None:
        raise RuntimeError(f"SBML file '{path}' does not contain a valid model.")

    species_list: List[CRNSpecies] = []
    species_index: Dict[str, int] = {}
    for s in model.getListOfSpecies():
        name = s.getId()
        species_index[name] = len(species_list)
        species_list.append(CRNSpecies(name=name))

    reactions: List[CRNReaction] = []
    for r in model.getListOfReactions():
        reactants: Dict[int, float] = {}
        products: Dict[int, float] = {}

        for sr in r.getListOfReactants():
            sid = sr.getSpecies()
            stoich = float(sr.getStoichiometry())
            idx = species_index[sid]
            reactants[idx] = reactants.get(idx, 0.0) + stoich

        for sp in r.getListOfProducts():
            sid = sp.getSpecies()
            stoich = float(sp.getStoichiometry())
            idx = species_index[sid]
            products[idx] = products.get(idx, 0.0) + stoich

        reversible = bool(r.getReversible())
        reactions.append(
            CRNReaction(
                reactants=reactants,
                products=products,
                reversible=reversible,
                metadata={"sbml_id": r.getId()},
            )
        )

    return CRNNetwork(species=species_list, reactions=reactions)
