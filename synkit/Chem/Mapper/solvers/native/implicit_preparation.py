"""Reference-blind implicit-H Lewis proofs with exact H-component offsets.

This preparation does not call a mapper or expand H. Product isotope-zero H
may be ordinary neutral terminal H, isolated isotope-zero H, or isolated
ordinary H2. Free positive-electron H is matched by sorted electron counts
when positive-electron isotope-zero source H is also isolated. Source
free/charged/radical/H2/isotopic H is supported by separating its
mapping-independent objective terms. Matched
ordinary H2 bonds contribute a constant alongside attached-H counts. It returns
prepared heavy graphs and an **integer doubled-CD offset**, never H classes.
"""

from collections import Counter

from rdkit import Chem

from synkit.Chem.Mapper.graph.labeled_graph import LabeledGraph
from synkit.IO.mol_to_graph import MolToGraph

from .reaction_text import split_reaction_fields


class UnsupportedImplicitHydrogen(ValueError):
    """The stated constant/count reduction cannot prove this model."""

    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


def _weight2(bond):
    value = bond.GetBondTypeAsDouble() * 2
    encoded = round(value)
    if abs(value - encoded) > 1e-9 or encoded < 0:
        raise UnsupportedImplicitHydrogen("non_half_integer_bond_weight")
    return encoded


def _electrons(atom):
    return 2 * MolToGraph.estimate_lone_pairs(atom) + atom.GetNumRadicalElectrons()


# Model guards and reduction offsets must remain paired with their endpoint data.
def prepare_implicit_problem(reaction, *, balance="dummy"):  # noqa: C901
    """Return ``(heavy_graphs, offset2, metadata)`` for a mapless reaction.

    Native proof costs for these graphs are in CD units; add offset2/2 once to
    minimum/incumbent costs, or offset2 once to their doubled-CD counterparts.
    This reduction is numeric/proof-only: it does not infer reference H truth
    or enumerate H/class output. All map tags are forbidden at this blind API.

    ``UnsupportedImplicitHydrogen.reason`` gives an explicit unsupported-model
    code. Invalid SMILES, maps, wildcards and strict inventory mismatch raise
    ordinary ValueError. A caller may then use a separately labeled full-H
    native proof with the remaining original budget.
    """
    if not isinstance(reaction, str):
        raise TypeError("reaction must be mapless reaction SMILES")
    if balance not in {"dummy", "strict"}:
        raise ValueError("balance must be 'dummy' or 'strict'")
    parts = split_reaction_fields(
        reaction,
        error_message="Expected reactants>agents>products or reactants>>products",
    )
    if not parts[0] or not parts[2]:
        raise ValueError("Expected reactants>agents>products or reactants>>products")
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mols = [Chem.MolFromSmiles(parts[i], params) for i in (0, 2)]
    if any(mol is None for mol in mols):
        raise ValueError("Could not parse reaction endpoint")
    if any(atom.GetAtomMapNum() for mol in mols for atom in mol.GetAtoms()):
        raise ValueError("Reference-blind PMCD preparation requires mapless endpoints")
    if any(atom.GetAtomicNum() == 0 for mol in mols for atom in mol.GetAtoms()):
        raise ValueError("Wildcard atoms require a concrete element")

    heavy = [[atom for atom in mol.GetAtoms() if atom.GetAtomicNum() != 1] for mol in mols]
    counts = [[MolToGraph._non_neighbor_h_count(atom) for atom in atoms] for atoms in heavy]
    positions = [{atom.GetIdx(): i for i, atom in enumerate(atoms)} for atoms in heavy]
    hydrogen_inventories = []
    for side, mol in enumerate(mols):
        inventory = Counter({0: sum(counts[side])})
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 1:
                inventory[atom.GetIsotope()] += 1
        hydrogen_inventories.append(+inventory)

    # Positive-electron isotope-zero H must be free on both endpoints.
    # Sorted electron matching is disjoint from attached-H and H2 matches.
    product_positive_free_ids = set()
    product_free_electrons = []
    for atom in mols[1].GetAtoms():
        if atom.GetAtomicNum() != 1 or atom.GetIsotope() or not _electrons(atom):
            continue
        if atom.GetDegree() or MolToGraph._non_neighbor_h_count(atom):
            raise UnsupportedImplicitHydrogen("product_nonfree_positive_electron_hydrogen")
        product_positive_free_ids.add(atom.GetIdx())
        product_free_electrons.append(_electrons(atom))
    source_free_electrons = []
    if product_free_electrons:
        for atom in mols[0].GetAtoms():
            if atom.GetAtomicNum() != 1 or atom.GetIsotope() or not _electrons(atom):
                continue
            if atom.GetDegree() or MolToGraph._non_neighbor_h_count(atom):
                raise UnsupportedImplicitHydrogen(
                    "source_nonfree_positive_electron_hydrogen_for_product_free_H_extension"
                )
            source_free_electrons.append(_electrons(atom))
    source_free_electrons.sort(reverse=True)
    product_free_electrons.sort(reverse=True)
    preserved_free_electrons = sum(min(a, b) for a, b in zip(source_free_electrons, product_free_electrons))
    # H2 components and heavy-attached H use disjoint vertices on both sides.
    # Thus maximum preserved H-H bonds and maximum preserved heavy-H bonds
    # can be attained simultaneously for every fixed typed heavy bijection.
    # Isotopic source H-H bonds cannot match isotope-zero product dimers.
    product_h2_atoms = set()
    product_h2 = source_h2 = 0
    for bond in mols[1].GetBonds():
        pair = (bond.GetBeginAtom(), bond.GetEndAtom())
        if not all(atom.GetAtomicNum() == 1 for atom in pair):
            continue
        if any(atom.GetIsotope() for atom in pair):
            raise UnsupportedImplicitHydrogen("product_isotopic_hydrogen")
        if _weight2(bond) != 2 or any(
            atom.GetFormalCharge()
            or atom.GetNumRadicalElectrons()
            or atom.GetDegree() != 1
            or _electrons(atom)
            or MolToGraph._non_neighbor_h_count(atom)
            for atom in pair
        ):
            raise UnsupportedImplicitHydrogen("product_nonordinary_hydrogen_H2_component")
        product_h2 += 1
        product_h2_atoms.update(atom.GetIdx() for atom in pair)
    if product_h2:
        for bond in mols[0].GetBonds():
            pair = (bond.GetBeginAtom(), bond.GetEndAtom())
            if not all(atom.GetAtomicNum() == 1 for atom in pair):
                continue
            if _weight2(bond) != 2 or any(atom.GetDegree() != 1 for atom in pair):
                raise UnsupportedImplicitHydrogen("source_nonisolated_or_nonunit_HH_for_H2_extension")
            if all(atom.GetIsotope() == 0 for atom in pair):
                if any(atom.GetFormalCharge() or atom.GetNumRadicalElectrons() or _electrons(atom) for atom in pair):
                    raise UnsupportedImplicitHydrogen("source_nonordinary_H2_for_product_H2_extension")
                source_h2 += 1

    offset_hh2 = offset_electrons2 = offset_isotope_edges2 = 0
    source_free0 = source_isotope = source_attached0 = product_free0 = 0
    for side, mol in enumerate(mols):
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 1:
                continue
            if MolToGraph._non_neighbor_h_count(atom):
                raise UnsupportedImplicitHydrogen("non_neighbor_hydrogen_on_hydrogen")
            neighbors = [
                (bond.GetOtherAtom(atom), bond)
                for bond in atom.GetBonds()
                if bond.GetOtherAtom(atom).GetAtomicNum() != 1
            ]
            if side == 1:
                if atom.GetIdx() in product_positive_free_ids:
                    product_free0 += 1
                    continue
                if atom.GetIdx() in product_h2_atoms:
                    continue
                if atom.GetIsotope():
                    raise UnsupportedImplicitHydrogen("product_isotopic_hydrogen")
                # Isolated isotope-zero product H with zero modeled electron
                # weight is a free reservoir, like a zero-incidence typed H
                # dummy. In particular H+ needs no vertex expansion. Its
                # endpoint charge/state remains in the caller's raw molecule.
                if not atom.GetBonds() and not atom.GetNumRadicalElectrons() and not _electrons(atom):
                    product_free0 += 1
                    continue
                if (
                    atom.GetFormalCharge()
                    or atom.GetNumRadicalElectrons()
                    or len(atom.GetBonds()) != 1
                    or len(neighbors) != 1
                    or _weight2(neighbors[0][1]) != 2
                ):
                    raise UnsupportedImplicitHydrogen("product_nonordinary_hydrogen")
                if _electrons(atom):
                    raise UnsupportedImplicitHydrogen("product_nonzero_hydrogen_electrons")
                counts[side][positions[side][neighbors[0][0].GetIdx()]] += 1
                continue

            # Begin with source H electron loss. When positive-electron
            # product H is supported, subtract the optimal free-H overlap
            # after both endpoints have been counted.
            offset_electrons2 += _electrons(atom)
            if atom.GetIsotope():
                # Product has no real isotope-H: all such source atoms map to
                # typed zero-state dummies. Their heavy edges are all lost.
                source_isotope += 1
                offset_isotope_edges2 += sum(_weight2(bond) for _, bond in neighbors)
            elif not neighbors:
                source_free0 += 1
            elif len(neighbors) == 1 and _weight2(neighbors[0][1]) == 2:
                counts[side][positions[side][neighbors[0][0].GetIdx()]] += 1
                source_attached0 += 1
            else:
                raise UnsupportedImplicitHydrogen("source_nonunit_or_multiple_heavy_hydrogen_incidence")
        if side == 0:
            offset_hh2 = sum(
                _weight2(bond)
                for bond in mol.GetBonds()
                if bond.GetBeginAtom().GetAtomicNum() == bond.GetEndAtom().GetAtomicNum() == 1
            )

    source_hh2 = offset_hh2
    offset_hh2 += 2 * product_h2 - 4 * min(source_h2, product_h2)
    if sum(counts[1]) + product_free0 + 2 * product_h2 != hydrogen_inventories[1].get(0, 0):
        raise UnsupportedImplicitHydrogen("product_hydrogen_count_model_mismatch")
    inventories = [Counter((atom.GetAtomicNum(), atom.GetIsotope()) for atom in atoms) for atoms in heavy]
    if balance == "strict" and (inventories[0] != inventories[1] or hydrogen_inventories[0] != hydrogen_inventories[1]):
        raise ValueError("Unbalanced atom inventories; select balance='dummy' explicitly")
    universe = inventories[0] | inventories[1]
    graphs = []
    for side, atoms in enumerate(heavy):
        missing = sorted((universe - inventories[side]).elements())
        labels = [(atom.GetAtomicNum(), atom.GetIsotope()) for atom in atoms] + missing
        hcounts = counts[side] + [0] * len(missing)
        electrons = [_electrons(atom) for atom in atoms] + [0] * len(missing)
        n = len(labels)
        adjacency = {i: {} for i in range(n + 2)}
        for bond in mols[side].GetBonds():
            left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if left in positions[side] and right in positions[side]:
                i, j = positions[side][left], positions[side][right]
                adjacency[i][j] = adjacency[j][i] = _weight2(bond) / 2
        for i, (hydrogen, electron) in enumerate(zip(hcounts, electrons)):
            if hydrogen:
                adjacency[i][n + 1] = adjacency[n + 1][i] = hydrogen
            if electron:
                adjacency[i][n] = adjacency[n][i] = electron / 2
        graph = LabeledGraph(adjacency, labels + [(0, -1), (0, -2)])
        graph.props["hcounts"] = hcounts + [0, 0]
        graph.props["electrons"] = electrons + [0, 0]
        graphs.append(graph)

    offset_electrons2 += sum(product_free_electrons) - 2 * preserved_free_electrons
    offset2 = offset_hh2 + offset_electrons2 + offset_isotope_edges2
    metadata = {
        "model": (
            "full_lewis_pmcd_source_H_constant_and_product_H_counts_with_free_zero_electron_reservoir"
            if product_free0
            else "full_lewis_pmcd_source_H_constant_and_ordinary_product_H_counts"
        ),
        "objective": "normalized_full_lewis_distance",
        "balance": balance,
        "references_used": False,
        "offset_units": "doubled_chemical_distance",
        "offset2": offset2,
        "offset_HH2": offset_hh2,
        "offset_H_electrons2": offset_electrons2,
        "offset_isotope_H_heavy2": offset_isotope_edges2,
        "heavy_atoms_r": len(heavy[0]),
        "heavy_atoms_p": len(heavy[1]),
        "heavy_original_indices_r": [atom.GetIdx() for atom in heavy[0]],
        "heavy_original_indices_p": [atom.GetIdx() for atom in heavy[1]],
        "hydrogens_r": sum(hydrogen_inventories[0].values()),
        "hydrogens_p": sum(hydrogen_inventories[1].values()),
        "source_explicit_isotope0_attached": source_attached0,
        "source_explicit_isotope0_zero_heavy_incidence": source_free0,
        "source_explicit_isotopic_H": source_isotope,
        "product_explicit_isotope0_zero_heavy_incidence": product_free0,
        "product_H_ordinary": product_free0 == 0,
        "product_H_supported_domain": ("ordinary_terminal_or_isolated_zero_electron_isotope0"),
        "search_atoms": len(graphs[0].labels),
        "hydrogen_nodes_added": 0,
        "hydrogen_vertices_searched": 0,
        "scope": "numeric_PMCD_proof_only_no_H_or_class_enumeration",
    }
    if product_h2:
        metadata.update(
            model="full_lewis_pmcd_attached_H_counts_and_isolated_ordinary_H2_v1",
            product_H_supported_domain="ordinary_terminal_or_isolated_zero_electron_isotope0_or_ordinary_H2",
            product_H_ordinary=False,
            product_explicit_isotope0_zero_heavy_incidence=product_free0 + 2 * product_h2,
            product_isolated_zero_electron_reservoir=product_free0,
            product_H2_molecules=product_h2,
            source_ordinary_H2_molecules=source_h2,
            source_HH_bond_weight2=source_hh2,
            preserved_H2_bonds=min(source_h2, product_h2),
            H2_reduction="disjoint_attached_H_and_ordinary_H2_matching_v1",
        )
    if product_free_electrons:
        metadata.update(
            model="full_lewis_pmcd_free_H_electron_reservoir_with_attached_H_and_H2_v1",
            product_H_supported_domain=metadata["product_H_supported_domain"] + "_or_free_positive_electron_isotope0_H",
            product_H_ordinary=False,
            source_free_H_electrons=source_free_electrons,
            product_free_H_electrons=product_free_electrons,
            preserved_free_H_electron_overlap=preserved_free_electrons,
            product_isolated_zero_electron_reservoir=product_free0 - len(product_free_electrons),
            free_H_electron_reduction="descending_positive_free_H_electron_matching_v1",
        )
    return tuple(graphs), offset2, metadata


# The experiment API retains the same callable and exception identities.
prepare_implicit_pmcd = prepare_implicit_problem

__all__ = [
    "UnsupportedImplicitHydrogen",
    "prepare_implicit_pmcd",
    "prepare_implicit_problem",
]
