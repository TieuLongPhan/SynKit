"""Strict heavy-atom endpoints and joint reaction labels for evaluation.

This study interface deliberately does not balance, repair, or discard atoms.
Stereochemistry is outside its connectivity-only contract; isotope inputs and
hydrogens that cannot be folded into heavy atoms are rejected.
"""

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from numbers import Integral

from rdkit import Chem

from .graph.labeled_graph import LabeledGraph


@dataclass(frozen=True)
class Endpoint:
    atomic_numbers: tuple[int, ...]
    charges: tuple[int, ...]
    hcounts: tuple[int, ...]
    # Bond orders are represented as exact twice-order integers.
    bonds: tuple[tuple[int, int, int], ...]

    def __post_init__(self):
        n = len(self.atomic_numbers)
        if len(self.charges) != n or len(self.hcounts) != n:
            raise ValueError("Endpoint property lengths differ")
        if not n or any(z <= 1 for z in self.atomic_numbers):
            raise ValueError("A nonempty heavy-atom inventory is required")
        if any(not isinstance(x, Integral) for values in
               (self.atomic_numbers, self.charges, self.hcounts) for x in values):
            raise ValueError("Atom properties must be integers")
        if any(h < 0 for h in self.hcounts):
            raise ValueError("Hydrogen counts must be nonnegative")
        pairs = set()
        for i, j, order in self.bonds:
            if not all(isinstance(x, Integral) for x in (i, j, order)):
                raise ValueError("Bond records must use integers")
            if not 0 <= i < j < n or order not in (2, 3, 4, 6):
                raise ValueError("Unsupported bond coordinate or order")
            if (i, j) in pairs:
                raise ValueError("Duplicate bond")
            pairs.add((i, j))

    def graph(self):
        adjacency = {i: {} for i in range(len(self.atomic_numbers))}
        for i, j, order in self.bonds:
            adjacency[i][j] = adjacency[j][i] = order / 2
        result = LabeledGraph(adjacency, self.atomic_numbers)
        for name, values in (("atomic numbers", self.atomic_numbers),
                             ("charges", self.charges), ("hcounts", self.hcounts)):
            result.set_prop(name, list(values))
        return result


def parse_endpoint(smiles: str) -> Endpoint:
    """Parse every component, discard map labels, fold ordinary explicit H.

    Coordinates follow RDKit's retained heavy-atom order, not atom-map numbers.
    The caller must save the exact input string with exported labels.
    """
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(smiles, params)
    if mol is None:
        raise ValueError("Invalid endpoint SMILES")
    for atom in mol.GetAtoms():
        if atom.GetIsotope():
            raise ValueError("Isotope-specific endpoints are unsupported")
        if atom.GetNumRadicalElectrons():
            raise ValueError("Radical endpoints are unsupported")
        atom.SetAtomMapNum(0)
    mol = Chem.RemoveHs(mol)
    bonds = []
    for bond in mol.GetBonds():
        order = 2 * bond.GetBondTypeAsDouble()
        if order not in (2, 3, 4, 6):
            raise ValueError("Unsupported bond type")
        i, j = sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        bonds.append((i, j, int(order)))
    return Endpoint(
        tuple(a.GetAtomicNum() for a in mol.GetAtoms()),
        tuple(a.GetFormalCharge() for a in mol.GetAtoms()),
        tuple(a.GetTotalNumHs() for a in mol.GetAtoms()),
        tuple(sorted(bonds)),
    )


def parse_reaction(reaction: str) -> tuple[Endpoint, Endpoint]:
    """Accept balanced ``reactants>>products`` without a hidden agent policy."""
    fields = reaction.split(">")
    if len(fields) != 3 or fields[1]:
        raise ValueError("Expected reactants>>products with no agent field")
    reactant, product = map(parse_endpoint, (fields[0], fields[2]))
    if Counter(reactant.atomic_numbers) != Counter(product.atomic_numbers):
        raise ValueError("Heavy-atom inventories are not element balanced")
    return reactant, product


@dataclass(frozen=True)
class JointLabel:
    typed_bond_edits: tuple[tuple[int, int, int, int], ...]
    unary_changes: tuple[tuple[int, str, int, int], ...]

    @property
    def changed_bonds(self):
        return frozenset((i, j) for i, j, _, _ in self.typed_bond_edits)

    @property
    def centre_atoms(self):
        return frozenset(
            [i for pair in self.changed_bonds for i in pair]
            + [i for i, _, _, _ in self.unary_changes]
        )

    @property
    def weighted_distance(self):
        return Fraction(sum(abs(after - before)
                            for _, _, before, after in self.typed_bond_edits), 2)


def extract_label(reactant: Endpoint, product: Endpoint, mapping) -> JointLabel:
    """Derive one *joint* label from a complete element-compatible bijection."""
    mapping = tuple(mapping)
    n = len(reactant.atomic_numbers)
    if (len(product.atomic_numbers) != n or len(mapping) != n
            or any(not isinstance(p, Integral) for p in mapping)
            or set(mapping) != set(range(n))):
        raise ValueError("Mapping must be a complete bijection")
    if any(z != product.atomic_numbers[mapping[i]]
           for i, z in enumerate(reactant.atomic_numbers)):
        raise ValueError("Mapping changes an element")
    rb = {(i, j): order for i, j, order in reactant.bonds}
    pb = {(i, j): order for i, j, order in product.bonds}
    edits = []
    for i, j in combinations(range(n), 2):
        before = rb.get((i, j), 0)
        after = pb.get(tuple(sorted((mapping[i], mapping[j]))), 0)
        if before != after:
            edits.append((i, j, before, after))
    unary = []
    for name in ("charges", "hcounts"):
        before_values, after_values = getattr(reactant, name), getattr(product, name)
        for i, p in enumerate(mapping):
            if before_values[i] != after_values[p]:
                unary.append((i, name, before_values[i], after_values[p]))
    return JointLabel(tuple(edits), tuple(sorted(unary)))
