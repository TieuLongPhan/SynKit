try:
    from rdkit import Chem

    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
from collections import Counter, defaultdict
from ..graph.labeled_graph import LabeledGraph


def _mol_from_smiles(smi, remove_hs=True):
    params = Chem.SmilesParserParams()
    params.removeHs = remove_hs
    return Chem.MolFromSmiles(smi, params)


def count_elements_by_atomic_num(mol):

    element_count = Counter()
    for atom in mol.GetAtoms():
        element_count[atom.GetAtomicNum()] += 1
    return element_count


def balance_elements(mol1, mol2):

    diff = count_elements_by_atomic_num(mol2)
    diff.subtract(count_elements_by_atomic_num(mol1))

    # `any(diff.values())` is True for any non-zero (positive or negative) count.
    if not any(diff.values()):
        return Chem.Mol(mol1), Chem.Mol(mol2)

    rwmol1 = Chem.RWMol(mol1)
    rwmol2 = Chem.RWMol(mol2)

    for atomic_num, count in diff.items():
        if count > 0:
            for _ in range(abs(count)):
                rwmol1.AddAtom(Chem.Atom(atomic_num))
        elif count < 0:
            for _ in range(abs(count)):
                rwmol2.AddAtom(Chem.Atom(atomic_num))

    return rwmol1.GetMol(), rwmol2.GetMol()


def get_labeled_graph_from_mol(mol):

    labels = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    graph = {}
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()

        others = {}
        for bond in atom.GetBonds():
            other_idx = bond.GetOtherAtomIdx(idx)
            bond_order = int(2 * bond.GetBondTypeAsDouble() + 0.1)
            others[other_idx] = bond_order
        graph[idx] = others

    return LabeledGraph(graph, labels)


def smiles2lgp(rxn_smiles, add_Hs=True):

    natoms_pair = [[], []]
    sources_pair = [[], []]

    ss = rxn_smiles.split(">")
    r = _mol_from_smiles(ss[0], remove_hs=False)
    p = _mol_from_smiles(ss[2], remove_hs=False)

    if r is None or p is None:
        raise ValueError(f"Could not parse reaction SMILES: {rxn_smiles!r}")

    sources_pair[0].append(r)
    sources_pair[1].append(p)
    natoms_pair[0].append(r.GetNumAtoms())
    natoms_pair[1].append(p.GetNumAtoms())

    if add_Hs:
        r = Chem.AddHs(r)
        p = Chem.AddHs(p)

    sources_pair[0].append(r)
    sources_pair[1].append(p)
    natoms_pair[0].append(r.GetNumAtoms())
    natoms_pair[1].append(p.GetNumAtoms())

    r, p = balance_elements(r, p)

    sources_pair[0].append(r)
    sources_pair[1].append(p)
    natoms_pair[0].append(r.GetNumAtoms())
    natoms_pair[1].append(p.GetNumAtoms())

    atomic_nums_pair = [
        [atom.GetAtomicNum() for atom in mol.GetAtoms()] for mol in [r, p]
    ]

    lgp = []
    for mol in [r, p]:
        lgp.append(get_labeled_graph_from_mol(mol))

    ini_l2i_pair = [defaultdict(list), defaultdict(list)]
    for mol, ini_l2i in zip([r, p], ini_l2i_pair):
        for atom in mol.GetAtoms():
            label = atom.GetAtomMapNum()
            if label > 0:
                ini_l2i[label * 1000].append(atom.GetIdx())

    for label, idxs0 in ini_l2i_pair[0].items():
        if label in ini_l2i_pair[1]:
            idxs1 = ini_l2i_pair[1][label]

            if len(idxs0) != len(idxs1):
                import warnings

                warnings.warn(
                    f"Ignoring unbalanced atom-map constraint: "
                    f"{len(idxs0)} reactant atom(s) vs {len(idxs1)} product atom(s) "
                    f"share map number {label // 1000}",
                    stacklevel=4,
                )
                continue

            for lg, idxs in zip(lgp, [idxs0, idxs1]):
                for idx in idxs:
                    lg.labels[idx] = label
                lg.build_label2idxs()

    for lg, atomic_nums, natoms_slices, sources in zip(
        lgp, atomic_nums_pair, natoms_pair, sources_pair
    ):
        lg.set_prop("atomic numbers", atomic_nums)
        lg.set_prop("natoms slices", natoms_slices)
        lg.set_prop("sources", sources)

    return lgp


# elg = extended labeled graph
def smiles2elg(rxn_smiles, add_Hs=True, binarize=True, weight=1000):

    natoms = []
    sources = []

    ss = rxn_smiles.split(">")
    s = ss[0] + "." + ss[2]

    _r = _mol_from_smiles(ss[0], remove_hs=False)
    _p = _mol_from_smiles(ss[2], remove_hs=False)
    mol = _mol_from_smiles(s, remove_hs=False)
    if _r is None or _p is None or mol is None:
        raise ValueError(f"Could not parse reaction SMILES: {rxn_smiles!r}")
    natoms_r = _r.GetNumAtoms()
    natoms_p = _p.GetNumAtoms()

    sources.append(mol)
    natoms.append(mol.GetNumAtoms())

    if add_Hs:
        mol = Chem.AddHs(mol)

    sources.append(mol)
    natoms.append(mol.GetNumAtoms())

    atomic_nums = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    atom_map_nums = [atom.GetAtomMapNum() for atom in mol.GetAtoms()]

    elg = get_labeled_graph_from_mol(mol)
    if binarize:
        elg.binarize_graph()

    amn2i_r = defaultdict(list)
    for i in range(natoms_r):
        if atom_map_nums[i] > 0 and atomic_nums[i] > 1:
            amn2i_r[atom_map_nums[i]].append(i)

    amn2i_p = defaultdict(list)
    for i in range(natoms_r, natoms_p + natoms_r):
        if atom_map_nums[i] > 0 and atomic_nums[i] > 1:
            amn2i_p[atom_map_nums[i]].append(i)

    amns = set(amn for amn in amn2i_r.keys() if len(amn2i_r[amn]) == 1) & set(
        amn for amn in amn2i_p.keys() if len(amn2i_p[amn]) == 1
    )

    for amn in amns:
        i = amn2i_r[amn][0]
        j = amn2i_p[amn][0]
        elg.graph[i][j] = weight
        elg.graph[j][i] = weight

    elg.set_prop("atomic numbers", atomic_nums)
    elg.set_prop("natoms slices", natoms)
    elg.set_prop("sources", sources)

    return elg


def _normalise_explicit_h_atoms(explicit_h_atoms_pair):
    if explicit_h_atoms_pair is None:
        return [None, None]
    return [
        None if atoms is None else sorted(set(atoms)) for atoms in explicit_h_atoms_pair
    ]


def _assign_selected_hydrogen_maps(mols, base_map_nums_pair):
    """Locally match selected explicit hydrogens by mapped heavy-atom parent."""
    next_map = 1 + max((m for nums in base_map_nums_pair for m in nums), default=0)
    grouped_pair = []
    for mol, base_map_nums in zip(mols, base_map_nums_pair):
        grouped = defaultdict(list)
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 1:
                continue
            nbrs = atom.GetNeighbors()
            if not nbrs:
                continue
            parent_idx = nbrs[0].GetIdx()
            if parent_idx >= len(base_map_nums):
                continue
            parent_map = base_map_nums[parent_idx]
            if parent_map:
                grouped[parent_map].append(atom.GetIdx())
        grouped_pair.append(grouped)

    h_maps_pair = [{}, {}]
    parent_maps = set(grouped_pair[0]) | set(grouped_pair[1])
    for parent_map in sorted(parent_maps):
        left = grouped_pair[0].get(parent_map, [])
        right = grouped_pair[1].get(parent_map, [])
        n_shared = min(len(left), len(right))
        for k in range(n_shared):
            h_maps_pair[0][left[k]] = next_map
            h_maps_pair[1][right[k]] = next_map
            next_map += 1
        for atom_idx in left[n_shared:]:
            h_maps_pair[0][atom_idx] = next_map
            next_map += 1
        for atom_idx in right[n_shared:]:
            h_maps_pair[1][atom_idx] = next_map
            next_map += 1

    return h_maps_pair


def _hydrogen_counts_by_parent(mol):
    grouped = defaultdict(list)
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1:
            continue
        nbrs = atom.GetNeighbors()
        if nbrs:
            grouped[nbrs[0].GetIdx()].append(atom.GetIdx())
    return grouped


def _add_exact_parent_hydrogens(mol, h_counts):
    if not h_counts:
        return mol

    parents = sorted(h_counts)
    mol = Chem.RWMol(Chem.AddHs(mol, onlyOnAtoms=parents))
    grouped = _hydrogen_counts_by_parent(mol)
    remove = []
    for parent_idx, desired_count in h_counts.items():
        h_idxs = grouped.get(parent_idx, [])
        remove.extend(h_idxs[desired_count:])

    for atom_idx in sorted(remove, reverse=True):
        mol.RemoveAtom(atom_idx)
    return mol


def get_numbered_rxn_smiles(
    rxn_smiles,
    map_nums_pair,
    explicit_hs=False,
    explicit_h_atoms_pair=None,
    explicit_h_counts_pair=None,
    map_selected_hs=True,
    all_hs_explicit=None,
):
    new_smiles_pair = []
    explicit_h_atoms_pair = _normalise_explicit_h_atoms(explicit_h_atoms_pair)
    if explicit_h_counts_pair is None:
        explicit_h_counts_pair = [None, None]
    mols = []
    adjusted_pair = []
    for smi, map_nums, h_atoms, h_counts in zip(
        rxn_smiles.split(">>"),
        map_nums_pair,
        explicit_h_atoms_pair,
        explicit_h_counts_pair,
    ):
        mol = _mol_from_smiles(smi, remove_hs=False)
        base_natoms = mol.GetNumAtoms()
        if h_atoms is not None:
            h_atoms = [i for i in h_atoms if i < base_natoms]
        if explicit_hs and h_counts is not None:
            h_counts = {
                parent_idx: count
                for parent_idx, count in h_counts.items()
                if parent_idx < base_natoms and count > 0
            }
            mol = _add_exact_parent_hydrogens(mol, h_counts)
        elif explicit_hs and h_atoms is None:
            mol = Chem.AddHs(mol)
        elif explicit_hs and h_atoms:
            mol = Chem.AddHs(mol, onlyOnAtoms=h_atoms)
        natoms = mol.GetNumAtoms()
        nnums = len(map_nums)
        if nnums < natoms:
            adjusted_map_nums = map_nums + [0] * (natoms - nnums)
        else:
            adjusted_map_nums = map_nums[:natoms]
        mols.append(Chem.RWMol(mol))
        adjusted_pair.append(adjusted_map_nums)

    h_maps_pair = (
        _assign_selected_hydrogen_maps(mols, adjusted_pair)
        if (
            explicit_hs
            and map_selected_hs
            and any(atoms is not None for atoms in explicit_h_atoms_pair)
        )
        else [{}, {}]
    )
    if all_hs_explicit is None:
        all_hs_explicit = explicit_hs

    for side, (mol, adjusted_map_nums) in enumerate(zip(mols, adjusted_pair)):
        for atom, map_num in zip(mol.GetAtoms(), adjusted_map_nums):
            atom.SetAtomMapNum(map_num)
        for atom_idx, map_num in h_maps_pair[side].items():
            mol.GetAtomWithIdx(atom_idx).SetAtomMapNum(map_num)
        new_smiles_pair.append(
            Chem.MolToSmiles(
                mol,
                canonical=False,
                allHsExplicit=all_hs_explicit,
            )
        )
    return ">>".join(new_smiles_pair)


def selected_atom_indices_from_maps(map_nums_pair, selected_maps):
    return [
        [i for i, map_num in enumerate(map_nums) if map_num in selected_maps]
        for map_nums in map_nums_pair
    ]


def reaction_center_signature_from_mapped_smiles(mapped_rxn_smiles):
    """Cheap reaction-center signature from mapped heavy-atom SMILES.

    The signature captures changed heavy-heavy bonds and mapped-heavy-atom H-count
    deltas. It is enough to decide which hydrogens should be expanded for
    reaction-center display, without constructing a full SynKit ITS graph.
    """
    mols = [
        _mol_from_smiles(smi, remove_hs=False) for smi in mapped_rxn_smiles.split(">>")
    ]
    edge_pair = []
    hcount_pair = []
    for mol in mols:
        edges = {}
        hcounts = {}
        for atom in mol.GetAtoms():
            atom_map = atom.GetAtomMapNum()
            if atom_map and atom.GetAtomicNum() != 1:
                hcounts[int(atom_map)] = atom.GetTotalNumHs(includeNeighbors=True)
        for bond in mol.GetBonds():
            a0 = bond.GetBeginAtom()
            a1 = bond.GetEndAtom()
            if a0.GetAtomicNum() == 1 or a1.GetAtomicNum() == 1:
                continue
            m0 = a0.GetAtomMapNum()
            m1 = a1.GetAtomMapNum()
            if not m0 or not m1:
                continue
            key = tuple(sorted((int(m0), int(m1))))
            edges[key] = int(2 * bond.GetBondTypeAsDouble() + 0.1)
        edge_pair.append(edges)
        hcount_pair.append(hcounts)

    changed_edges = []
    for key in sorted(set(edge_pair[0]) | set(edge_pair[1])):
        left = edge_pair[0].get(key, 0)
        right = edge_pair[1].get(key, 0)
        if left != right:
            changed_edges.append((key[0], key[1], left, right))

    hcount_deltas = []
    for atom_map in sorted(set(hcount_pair[0]) | set(hcount_pair[1])):
        left = hcount_pair[0].get(atom_map, 0)
        right = hcount_pair[1].get(atom_map, 0)
        if left != right:
            hcount_deltas.append((atom_map, left, right))

    return (tuple(changed_edges), tuple(hcount_deltas))


def reaction_center_atom_maps_from_signature(signature):
    maps = set()
    changed_edges, hcount_deltas = signature
    for a, b, _, _ in changed_edges:
        maps.add(a)
        maps.add(b)
    for atom_map, _, _ in hcount_deltas:
        maps.add(atom_map)
    return maps


def selected_hydrogen_counts_from_hcount_deltas(
    rxn_smiles, map_nums_pair, selected_maps
):
    mols = [_mol_from_smiles(smi, remove_hs=False) for smi in rxn_smiles.split(">>")]
    atom_by_map_pair = []
    hcount_by_map_pair = []
    for mol, map_nums in zip(mols, map_nums_pair):
        atom_by_map = {}
        hcount_by_map = {}
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            if idx >= len(map_nums) or atom.GetAtomicNum() == 1:
                continue
            map_num = map_nums[idx]
            if map_num in selected_maps:
                atom_by_map[map_num] = idx
                hcount_by_map[map_num] = atom.GetTotalNumHs(includeNeighbors=True)
        atom_by_map_pair.append(atom_by_map)
        hcount_by_map_pair.append(hcount_by_map)

    counts_pair = [{}, {}]
    for map_num in sorted(set(atom_by_map_pair[0]) & set(atom_by_map_pair[1])):
        left = hcount_by_map_pair[0].get(map_num, 0)
        right = hcount_by_map_pair[1].get(map_num, 0)
        if left > right:
            counts_pair[0][atom_by_map_pair[0][map_num]] = left - right
        elif right > left:
            counts_pair[1][atom_by_map_pair[1][map_num]] = right - left

    if sum(counts_pair[0].values()) != sum(counts_pair[1].values()):
        return [{}, {}]
    if not counts_pair[0] and not counts_pair[1]:
        return None
    return counts_pair


def expand_reaction_center_hydrogens(
    rxn_smiles,
    map_nums_pair,
    selected_maps,
):
    """Expand, but do not map, hydrogens attached to selected heavy maps."""
    explicit_h_counts_pair = selected_hydrogen_counts_from_hcount_deltas(
        rxn_smiles,
        map_nums_pair,
        selected_maps,
    )
    explicit_h_atoms_pair = (
        selected_atom_indices_from_maps(
            map_nums_pair,
            selected_maps,
        )
        if explicit_h_counts_pair is None
        else [None, None]
    )
    return get_numbered_rxn_smiles(
        rxn_smiles,
        map_nums_pair,
        explicit_hs=True,
        explicit_h_atoms_pair=explicit_h_atoms_pair,
        explicit_h_counts_pair=explicit_h_counts_pair,
        map_selected_hs=False,
        all_hs_explicit=True,
    )


def remap_reaction_center_hydrogens(
    rxn_smiles,
    map_nums_pair,
    selected_maps,
    binary=True,
):
    """Backward-compatible local expansion helper.

    The mapper now performs the real second-pass hydrogen remapping itself so
    the newly explicit H atoms are optimized, not assigned locally.
    """
    expanded_rxn = expand_reaction_center_hydrogens(
        rxn_smiles,
        map_nums_pair,
        selected_maps,
    )
    return expanded_rxn


def canonicalize_rxn_smiles(rxn_smiles):
    components_cano = []
    ss = rxn_smiles.split(">")
    ss.pop(1)
    for s in ss:
        mol = Chem.MolFromSmiles(s)
        mol_cano = Chem.RWMol(mol)
        mapnums = []
        for atom in mol_cano.GetAtoms():
            mapnums.append(atom.GetAtomMapNum())
            atom.SetAtomMapNum(0)
        mol_cano = Chem.RWMol(Chem.MolFromSmiles(Chem.MolToSmiles(mol_cano)))
        matches = mol.GetSubstructMatches(mol_cano)
        if matches:
            for atom, idx in zip(mol_cano.GetAtoms(), matches[0]):
                atom.SetAtomMapNum(mapnums[idx])
            s_cano = Chem.MolToSmiles(mol_cano, canonical=False, allHsExplicit=True)
        else:
            s_cano = Chem.MolToSmiles(mol, canonical=False, allHsExplicit=True)
        components_cano.append(s_cano)
    return ">>".join(components_cano)
