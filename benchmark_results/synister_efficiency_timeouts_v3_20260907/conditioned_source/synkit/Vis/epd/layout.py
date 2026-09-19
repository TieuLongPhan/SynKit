from __future__ import annotations

"""RDKit-driven coordinate generation and shared-frame alignment."""

from typing import Dict, Sequence, Tuple

import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDepictor

from .chem import edge_order
from .mapping import amap_to_node
from .utils import normalize_scalar_attr


def edge_to_rdkit_bond_type(edata: Dict[str, object]) -> Chem.BondType:
    """Map graph edge annotations to an RDKit bond type."""
    order = edge_order(edata)
    bond_type = edata.get("bond_type")

    if bond_type == "AROMATIC" or order == 1.5:
        return Chem.BondType.AROMATIC
    if bond_type == "TRIPLE" or (order is not None and order >= 2.9):
        return Chem.BondType.TRIPLE
    if bond_type == "DOUBLE" or (order is not None and order >= 1.9):
        return Chem.BondType.DOUBLE
    return Chem.BondType.SINGLE


def _build_rdkit_atom(
    node: int,
    data: Dict[str, object],
    atom_map_key: str,
) -> Chem.Atom:
    """Create one RDKit atom from graph node data."""
    elem = str(normalize_scalar_attr(data.get("element", "C"), "C"))
    charge = int(normalize_scalar_attr(data.get("charge", 0), 0))
    aromatic = bool(normalize_scalar_attr(data.get("aromatic", False), False))
    atom_map = int(normalize_scalar_attr(data.get(atom_map_key, node), node))

    atom = Chem.Atom(elem)
    atom.SetFormalCharge(charge)
    atom.SetAtomMapNum(atom_map)
    if aromatic:
        atom.SetIsAromatic(True)
    return atom


def _add_rdkit_bond(
    rw: Chem.RWMol,
    node_to_idx: Dict[int, int],
    u: int,
    v: int,
    edata: Dict[str, object],
) -> None:
    """Add one RDKit bond from graph edge data."""
    bond_type = edge_to_rdkit_bond_type(edata)
    rw.AddBond(node_to_idx[u], node_to_idx[v], bond_type)
    bond = rw.GetBondBetweenAtoms(node_to_idx[u], node_to_idx[v])
    if bond_type == Chem.BondType.AROMATIC:
        bond.SetIsAromatic(True)


def _sanitize_or_cache(mol: Chem.Mol) -> None:
    """Sanitize molecule, falling back to relaxed property-cache update."""
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        mol.UpdatePropertyCache(strict=False)


def graph_to_rdkit_mol(
    graph: nx.Graph,
    atom_map_key: str = "atom_map",
) -> Tuple[Chem.Mol, Dict[int, int], Dict[int, int]]:
    """Convert a NetworkX molecular graph to an RDKit molecule."""
    rw = Chem.RWMol()
    node_to_idx: Dict[int, int] = {}

    for node, data in graph.nodes(data=True):
        atom = _build_rdkit_atom(node, data, atom_map_key=atom_map_key)
        node_to_idx[node] = rw.AddAtom(atom)

    for u, v, edata in graph.edges(data=True):
        _add_rdkit_bond(rw, node_to_idx, u, v, edata)

    mol = rw.GetMol()
    _sanitize_or_cache(mol)

    idx_to_node = {idx: node for node, idx in node_to_idx.items()}
    return mol, node_to_idx, idx_to_node


def _conformer_to_pos_dict(
    mol: Chem.Mol,
    idx_to_node: Dict[int, int],
) -> Dict[int, np.ndarray]:
    """Convert an RDKit conformer into node-position mapping."""
    conf = mol.GetConformer()
    pos: Dict[int, np.ndarray] = {}

    for idx in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(idx)
        pos[idx_to_node[idx]] = np.array([p.x, p.y], dtype=float)

    return pos


def rdkit_positions(
    graph: nx.Graph,
    atom_map_key: str = "atom_map",
) -> Dict[int, np.ndarray]:
    """Compute 2D coordinates from RDKit depiction."""
    mol, _, idx_to_node = graph_to_rdkit_mol(graph, atom_map_key=atom_map_key)
    rdDepictor.SetPreferCoordGen(True)
    rdDepictor.Compute2DCoords(mol)
    return _conformer_to_pos_dict(mol, idx_to_node)


def its_reference_positions(
    its_graph: nx.Graph,
    atom_map_key: str = "atom_map",
) -> Dict[int, np.ndarray]:
    """Compute ITS positions using a single-bond scaffold graph."""
    ref = nx.Graph()

    for node, data in its_graph.nodes(data=True):
        elem = str(normalize_scalar_attr(data.get("element", "C"), "C"))
        charge = int(normalize_scalar_attr(data.get("charge", 0), 0))
        ref.add_node(
            node,
            element=elem,
            charge=charge,
            aromatic=False,
            atom_map=int(normalize_scalar_attr(data.get(atom_map_key, node), node)),
        )

    for u, v in its_graph.edges():
        ref.add_edge(u, v, order=1.0, kekule_order=1.0, bond_type="SINGLE")

    return rdkit_positions(ref, atom_map_key=atom_map_key)


def rigid_align_points(
    moving: np.ndarray,
    target: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return rotation/translation aligning *moving* onto *target*."""
    mc = moving.mean(axis=0)
    tc = target.mean(axis=0)

    X = moving - mc
    Y = target - tc

    H = X.T @ Y
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = tc - mc @ R
    return R, t


def apply_rigid_transform(
    pos: Dict[int, np.ndarray],
    R: np.ndarray,
    t: np.ndarray,
) -> Dict[int, np.ndarray]:
    """Apply rigid transform to all coordinates."""
    return {k: v @ R + t for k, v in pos.items()}


def align_pos_dict_to_reference(
    moving_pos: Dict[int, np.ndarray],
    moving_nodes: Sequence[int],
    target_points: Sequence[np.ndarray],
) -> Dict[int, np.ndarray]:
    """Align a coordinate dictionary using corresponding anchor points."""
    if len(moving_nodes) == 0:
        return {k: v.copy() for k, v in moving_pos.items()}

    if len(moving_nodes) == 1:
        shift = np.asarray(target_points[0], dtype=float) - moving_pos[moving_nodes[0]]
        return {k: v + shift for k, v in moving_pos.items()}

    moving = np.array([moving_pos[n] for n in moving_nodes], dtype=float)
    target = np.array(target_points, dtype=float)
    R, t = rigid_align_points(moving, target)
    return apply_rigid_transform(moving_pos, R, t)


def _master_amap_positions(
    pos_r_raw: Dict[int, np.ndarray],
    pos_its_raw: Dict[int, np.ndarray],
    r_amap_to_node: Dict[int, int],
    its_amap_to_node: Dict[int, int],
    reference_layout: str,
) -> Dict[int, np.ndarray]:
    """Choose the master atom-map frame used for shared alignment."""
    if reference_layout == "reactant":
        return {amap: pos_r_raw[node].copy() for amap, node in r_amap_to_node.items()}

    if reference_layout == "its":
        return {
            amap: pos_its_raw[node].copy() for amap, node in its_amap_to_node.items()
        }

    raise ValueError("reference_layout must be 'reactant' or 'its'.")


def _align_graph_to_master(
    raw_pos: Dict[int, np.ndarray],
    amap_to_node_dict: Dict[int, int],
    master_amap_pos: Dict[int, np.ndarray],
) -> Dict[int, np.ndarray]:
    """Align one graph position dictionary to the shared atom-map frame."""
    common_amaps = sorted(set(amap_to_node_dict) & set(master_amap_pos))
    aligned = align_pos_dict_to_reference(
        raw_pos,
        moving_nodes=[amap_to_node_dict[a] for a in common_amaps],
        target_points=[master_amap_pos[a] for a in common_amaps],
    )

    for amap in common_amaps:
        aligned[amap_to_node_dict[amap]] = master_amap_pos[amap].copy()

    return aligned


def build_shared_layout(
    reactant_graph: nx.Graph,
    its_graph: nx.Graph,
    atom_map_key: str = "atom_map",
    reference_layout: str = "its",
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Build a shared layout for reactant and ITS using one master frame."""
    pos_r_raw = rdkit_positions(reactant_graph, atom_map_key=atom_map_key)
    pos_its_raw = its_reference_positions(its_graph, atom_map_key=atom_map_key)

    r_amap_to_node = amap_to_node(reactant_graph, atom_map_key=atom_map_key)
    its_amap_to_node = amap_to_node(its_graph, atom_map_key=atom_map_key)

    master_amap_pos = _master_amap_positions(
        pos_r_raw=pos_r_raw,
        pos_its_raw=pos_its_raw,
        r_amap_to_node=r_amap_to_node,
        its_amap_to_node=its_amap_to_node,
        reference_layout=reference_layout,
    )

    pos_r = _align_graph_to_master(pos_r_raw, r_amap_to_node, master_amap_pos)
    pos_its = _align_graph_to_master(pos_its_raw, its_amap_to_node, master_amap_pos)
    return pos_r, pos_its


def separate_left_right(
    pos_left: Dict[int, np.ndarray],
    pos_right: Dict[int, np.ndarray],
    gap: float = 3.0,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Translate two layouts side-by-side while keeping their orientation."""
    left = {k: v.copy() for k, v in pos_left.items()}
    right = {k: v.copy() for k, v in pos_right.items()}

    lx = [p[0] for p in left.values()]
    ly = [p[1] for p in left.values()]
    rx = [p[0] for p in right.values()]
    ry = [p[1] for p in right.values()]

    shift_x = (max(lx) - min(lx)) + gap + (min(lx) - min(rx))
    right = {k: v + np.array([shift_x, 0.0], dtype=float) for k, v in right.items()}

    cy_left = 0.5 * (min(ly) + max(ly))
    cy_right = 0.5 * (min(ry) + max(ry))
    right = {
        k: v + np.array([0.0, cy_left - cy_right], dtype=float)
        for k, v in right.items()
    }
    return left, right
