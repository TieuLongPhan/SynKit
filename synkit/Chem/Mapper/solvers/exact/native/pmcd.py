"""Prove normalized full Lewis PMCD without adding or enumerating H atoms.

Ordinary terminal H is represented by counts on heavy atoms. For a proposed
heavy bijection f its exact minimum H cost is sum_i |h_R[i]-h_P[f(i)]|, also
with unequal H inventories under zero-state dummy padding. This term enters
the native objective before optimization. No H graph, transport enumeration,
ITS classification or mapped-reaction rendering is performed.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math
from numbers import Real
import time

from rdkit import Chem

from synkit.IO.mol_to_graph import MolToGraph
from ....graph.labeled_graph import LabeledGraph
from ...slap.lap import _adjacency_and_elements, chemical_distance
from .candidates import minimize_native_candidates, prepare_native_candidates
from .search import _profile_seed


class _UnsupportedHydrogen(ValueError):
    pass


@dataclass(frozen=True)
class PMCDProofResult:
    """An optimum proof, separate from complete mapping-class enumeration.

    heavy_mapping uses the padded heavy-atom order (real atoms, then dummies).
    minimum_cost is set only after a completed proof. The feasible incumbent
    can remain available if the proof is interrupted. Unsupported H states
    are explicit and never silently drop out of the objective.
    """

    minimum_cost: float | None
    incumbent_cost: float | None
    optimality_proven: bool
    status: str
    reason: str | None
    heavy_mapping: tuple[int, ...] | None
    heavy_atom_counts: tuple[int, int] | None
    hydrogen_counts: tuple[int, int] | None
    search_atoms: int
    visited_nodes: int
    elapsed_seconds: float
    search_result: dict | None = None
    objective: str = "normalized_full_lewis_distance"
    hydrogen_nodes_added: int = 0
    hydrogen_vertices_searched: int = 0
    enumeration_complete: bool = False

    def as_dict(self):
        return asdict(self)


# Keep hydrogen eligibility, balancing and anchor construction in one reduction.
def _implicit_quotient_graphs(reaction, balance):  # noqa: C901
    """Read endpoint H counts and electron states without Chem.AddHs."""
    if not isinstance(reaction, str):
        raise TypeError("reaction must be reaction SMILES")
    parts = reaction.split(">")
    if len(parts) != 3 or not parts[0] or not parts[2]:
        raise ValueError("Expected reactants>agents>products or reactants>>products")
    endpoints = []
    for side in (parts[0], parts[2]):
        params = Chem.SmilesParserParams()
        params.removeHs = False
        mol = Chem.MolFromSmiles(side, params)
        if mol is None:
            raise ValueError("Could not parse reaction endpoint")
        heavy = []
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
            if atom.GetAtomicNum() == 0:
                raise ValueError("Wildcard atoms require a concrete element")
            if atom.GetAtomicNum() != 1:
                heavy.append(atom)
                continue
            if atom.GetIsotope():
                raise _UnsupportedHydrogen("isotopic_hydrogen")
            if atom.GetFormalCharge() or atom.GetNumRadicalElectrons():
                raise _UnsupportedHydrogen("nonneutral_or_radical_hydrogen")
            bonds = list(atom.GetBonds())
            if (len(bonds) != 1 or bonds[0].GetBondTypeAsDouble() != 1
                    or bonds[0].GetOtherAtom(atom).GetAtomicNum() <= 1):
                raise _UnsupportedHydrogen("nonordinary_terminal_hydrogen")
        labels = [(atom.GetAtomicNum(), atom.GetIsotope()) for atom in heavy]
        hcounts = [MolToGraph._total_h_count(atom) for atom in heavy]
        electrons = [2 * MolToGraph.estimate_lone_pairs(atom)
                     + atom.GetNumRadicalElectrons() for atom in heavy]
        endpoints.append((mol, heavy, labels, hcounts, electrons))
    inventories = [Counter(value[2]) for value in endpoints]
    totals = tuple(sum(value[3]) for value in endpoints)
    if balance == "strict" and (inventories[0] != inventories[1] or totals[0] != totals[1]):
        raise ValueError("Unbalanced atom inventories; select balance='dummy' explicitly")
    universe = inventories[0] | inventories[1]
    graphs = []
    for (mol, heavy, labels, hcounts, electrons), inventory in zip(endpoints, inventories):
        positions = {atom.GetIdx(): i for i, atom in enumerate(heavy)}
        missing = sorted((universe - inventory).elements())
        labels = labels + missing
        hcounts = hcounts + [0] * len(missing)
        electrons = electrons + [0] * len(missing)
        size = len(labels)
        adjacency = {i: {} for i in range(size + 2)}
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if i in positions and j in positions:
                x, y = positions[i], positions[j]
                adjacency[x][y] = adjacency[y][x] = bond.GetBondTypeAsDouble()
        # Separate forced anchors encode the electron and minimum-H unary terms.
        for i, (count, electron) in enumerate(zip(hcounts, electrons)):
            if electron:
                adjacency[i][size] = adjacency[size][i] = electron / 2
            if count:
                adjacency[i][size + 1] = adjacency[size + 1][i] = count
        graph = LabeledGraph(adjacency, labels + [(0, -1), (0, -2)])
        graph.props["hcounts"] = hcounts + [0, 0]
        graph.props["electrons"] = electrons + [0, 0]
        graphs.append(graph)
    return tuple(graphs), tuple(len(value[1]) for value in endpoints), totals


def prove_reaction_pmcd(
    reaction,
    *,
    library_path,
    balance="strict",
    time_limit_seconds=10.0,
    symmetry=True,
):
    """Prove full Lewis PMCD using only heavy vertices and H counts.

    Supported H atoms are neutral, nonradical, nonisotopic terminal single-bond
    H. Other H states return status='unsupported' without expanding anything.
    Stereo does not need expansion for the existing numerical CD objective:
    it remains an output-class distinction, and this API returns no classes.
    The ordinary map_reaction/Mapper minimum query still enumerates all minima.
    """
    started = time.perf_counter()
    if balance not in {"strict", "dummy"}:
        raise ValueError("balance must be 'strict' or 'dummy'")
    if not isinstance(symmetry, bool):
        raise TypeError("symmetry must be boolean")
    if time_limit_seconds is not None and (
        isinstance(time_limit_seconds, bool) or not isinstance(time_limit_seconds, Real)
        or not math.isfinite(time_limit_seconds) or time_limit_seconds < 0
    ):
        raise ValueError("time limit must be finite and nonnegative")
    deadline = None if time_limit_seconds is None else started + time_limit_seconds

    def outcome(record=None, *, reason=None, status=None, graphs=(), heavy=None, hydrogens=None):
        proven = record is not None and record["complete"]
        mapping = tuple(record["mapping"][:-2]) if record else None
        return PMCDProofResult(
            minimum_cost=record["minimum_cost"] if proven else None,
            incumbent_cost=record["incumbent_cost"] if record else None,
            optimality_proven=proven,
            status=status or ("proven" if proven else "time_limit"),
            reason=reason if reason is not None else record["reason"] if record else None,
            heavy_mapping=mapping, heavy_atom_counts=heavy, hydrogen_counts=hydrogens,
            search_atoms=len(graphs[0].labels) if graphs else 0,
            visited_nodes=record["visited_nodes"] if record else 0,
            elapsed_seconds=time.perf_counter() - started, search_result=record,
        )

    try:
        graphs, heavy, hydrogens = _implicit_quotient_graphs(reaction, balance)
    except _UnsupportedHydrogen as error:
        return outcome(status="unsupported", reason=str(error))
    if len(graphs[0].labels) > 256:
        return outcome(status="unsupported", reason="native_node_limit", graphs=graphs,
                       heavy=heavy, hydrogens=hydrogens)
    if deadline is not None and time.perf_counter() >= deadline:
        return outcome(status="time_limit", reason="time_limit", graphs=graphs,
                       heavy=heavy, hydrogens=hydrogens)
    a, labels = _adjacency_and_elements(graphs[0], False)
    b, product_labels = _adjacency_and_elements(graphs[1], False)
    if len(set(a.ravel()) | set(b.ravel())) > 16:
        return outcome(status="unsupported", reason="native_bond_level_limit", graphs=graphs,
                       heavy=heavy, hydrogens=hydrogens)
    seed = _profile_seed(a, b, labels, product_labels)
    prepared = prepare_native_candidates(
        graphs, 0, library_path=library_path, initial_mapping=seed,
        binary=False, symmetry=symmetry, node_properties=(), allow_partial_symmetry=True,
    )
    seconds = 1e100 if deadline is None else max(0, deadline - time.perf_counter())
    record = minimize_native_candidates(
        graphs, library_path=library_path, initial_mapping=seed,
        time_limit_seconds=seconds, _prepared=prepared,
    )
    if chemical_distance(graphs, record["mapping"], False) != record["incumbent_cost"]:
        raise RuntimeError("Native proof witness does not match its reduced Lewis cost")
    return outcome(record, graphs=graphs, heavy=heavy, hydrogens=hydrogens)


__all__ = ["PMCDProofResult", "prove_reaction_pmcd"]
