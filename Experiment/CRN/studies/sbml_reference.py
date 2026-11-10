"""Independent exact CRNT quantities from static SBML reaction lists.

This module uses only the Python standard library and can be executed directly
with ``python -I sbml_reference.py model.xml``. It does not import SynKit or use
its graph conversion, complex partitioning, or rank implementation. Level 2's
omitted ``reversible`` flag means true; Level 3 requires an explicit flag.
Modifiers, rate laws, rules, and events do not enter this structural reading.
"""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path


def local_name(element):
    """Return an XML element's unqualified name."""
    return element.tag.rsplit("}", 1)[-1]


def child(element, name):
    """Return a direct child by local name, excluding package extensions."""
    namespace = element.tag.partition("}")[0] + "}" if "}" in element.tag else ""
    return next((item for item in element if item.tag == namespace + name), None)


def document_root(source):
    """Accept a parsed element, XML text or bytes, or an SBML file path."""
    if isinstance(source, ET.Element):
        return source
    if isinstance(source, bytes):
        return ET.fromstring(source)
    if isinstance(source, str) and source.lstrip().startswith("<"):
        return ET.fromstring(source)
    return ET.parse(source).getroot()


def model_element(root):
    """Return the document's Core model element."""
    model = root if local_name(root) == "model" else child(root, "model")
    if model is None:
        raise ValueError("SBML document contains no Core model")
    return model


def sbml_level(root):
    """Determine the SBML level without using a production-reader helper."""
    value = root.get("level")
    if value is None:
        match = re.search(r"/level([123])(?:/|\})", root.tag)
        value = match.group(1) if match else None
    if value not in {"1", "2", "3"}:
        raise ValueError("Independent SBML audit requires a declared level")
    return int(value)


def reaction_reversible(root, reaction):
    """Read the SBML boolean and the standard's level-dependent default."""
    value = reaction.get("reversible")
    if value is None:
        if sbml_level(root) in {1, 2}:
            return True
        raise ValueError("Level 3 reaction requires an explicit reversible attribute")
    if value not in {"true", "false", "1", "0"}:
        raise ValueError(f"Invalid SBML reversible value {value!r}")
    return value in {"true", "1"}


def reaction_side(reaction, name):
    """Sum repeated static references with exact rational coefficients."""
    listing = child(reaction, name)
    result = {}
    if listing is None:
        return result
    for reference in listing:
        if local_name(reference) != "speciesReference":
            continue
        if child(reference, "stoichiometryMath") is not None:
            raise ValueError("stoichiometryMath has no fixed coefficient for this audit")
        species = reference.get("species")
        if not species:
            raise ValueError("Species reference has no species identifier")
        try:
            coefficient = Fraction(reference.get("stoichiometry", "1"))
        except (ValueError, ZeroDivisionError) as error:
            raise ValueError("Malformed static stoichiometric coefficient") from error
        if coefficient < 0:
            raise ValueError("Stoichiometric coefficients must be non-negative")
        result[species] = result.get(species, Fraction(0)) + coefficient
    return {species: value for species, value in result.items() if value}


def read_structure(source):
    """Return species and directed reaction sides from the XML alone."""
    root = document_root(source)
    model = model_element(root)
    level = sbml_level(root)
    listing = child(model, "listOfSpecies")
    species = []
    for item in [] if listing is None else listing:
        if local_name(item) == "species":
            identifier = item.get("id")
            if not identifier or identifier in species:
                raise ValueError("Species identifiers must be present and unique")
            species.append(identifier)
    known = set(species)
    listing = child(model, "listOfReactions")
    reactions = []
    original = reversible = omitted = 0
    for reaction in [] if listing is None else listing:
        if local_name(reaction) != "reaction":
            continue
        left = reaction_side(reaction, "listOfReactants")
        right = reaction_side(reaction, "listOfProducts")
        if (set(left) | set(right)) - known:
            raise ValueError("Reaction refers to an undeclared species")
        original += 1
        omitted += reaction.get("reversible") is None
        reactions.append((left, right))
        if reaction_reversible(root, reaction):
            reversible += 1
            reactions.append((right, left))
    return {
        "species": tuple(species),
        "reactions": reactions,
        "level": level,
        "n_original_reactions": original,
        "n_reversible_reactions": reversible,
        "n_omitted_reversible": omitted,
    }


def stoichiometric_columns(source):
    """Return exact net columns in XML reaction order, including reverses."""
    structure = read_structure(source)
    return tuple(
        tuple(
            sorted(
                (species, right.get(species, 0) - left.get(species, 0))
                for species in set(left) | set(right)
                if right.get(species, 0) != left.get(species, 0)
            )
        )
        for left, right in structure["reactions"]
    )


def _rank(columns):
    """Find rank by sparse rational column elimination, ignoring duplicates."""
    pivots = {}
    seen = set()
    for entries in columns:
        column = dict(entries)
        if not column:
            continue
        scale = column[min(column)]
        column = {key: value / scale for key, value in column.items()}
        signature = tuple(sorted(column.items()))
        if signature in seen:
            continue
        seen.add(signature)
        while column:
            index = min(column)
            pivot = pivots.get(index)
            if pivot is None:
                scale = column[index]
                pivots[index] = {key: value / scale for key, value in column.items()}
                break
            factor = column[index]
            for key, value in pivot.items():
                remaining = column.get(key, 0) - factor * value
                if remaining:
                    column[key] = remaining
                else:
                    column.pop(key, None)
    return len(pivots)


def crnt_from_sbml(source):
    """Calculate exact CRNT quantities without SynKit, NumPy, or NetworkX."""
    structure = read_structure(source)
    reactions = structure["reactions"]

    def key(side):
        return tuple(sorted(side.items()))

    complexes = {key(side) for reaction in reactions for side in reaction}
    forward = {node: set() for node in complexes}
    backward = {node: set() for node in complexes}
    for left, right in reactions:
        a, b = key(left), key(right)
        forward[a].add(b)
        backward[b].add(a)

    def reachable(start, adjacency, reverse=None):
        reached, pending = set(), [start]
        while pending:
            node = pending.pop()
            if node not in reached:
                reached.add(node)
                pending.extend(adjacency[node] - reached)
                if reverse is not None:
                    pending.extend(reverse[node] - reached)
        return reached

    remaining = set(complexes)
    linkage = 0
    weakly_reversible = True
    while remaining:
        start = next(iter(remaining))
        component = reachable(start, forward, backward)
        remaining -= component
        linkage += 1
        weakly_reversible &= reachable(start, forward) == component and reachable(start, backward) == component
    columns = [
        tuple(
            (species, right.get(species, 0) - left.get(species, 0))
            for species in set(left) | set(right)
            if right.get(species, 0) != left.get(species, 0)
        )
        for left, right in reactions
    ]
    rank = _rank(columns)
    return {
        "n_species": len(structure["species"]),
        "n_reactions": len(reactions),
        "n_complexes": len(complexes),
        "n_linkage_classes": linkage,
        "rank": rank,
        "deficiency": len(complexes) - linkage - rank,
        "n_conservation_laws": len(structure["species"]) - rank,
        "weakly_reversible": weakly_reversible,
        **{
            name: structure[name]
            for name in (
                "level",
                "n_original_reactions",
                "n_reversible_reactions",
                "n_omitted_reversible",
            )
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(crnt_from_sbml(arguments.model), indent=2, sort_keys=True))
