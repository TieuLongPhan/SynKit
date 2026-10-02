"""Reaction-SMILES interface for exact PABS mapping and double-orbit classes."""

from __future__ import annotations

import hashlib
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass
from numbers import Real

import numpy as np
from rdkit import Chem

from .exact.hydrogen import enumerate_minimal_hydrogen_transfers
from .exact.propagation import PropagationConfig
from .exact.search import enumerate_pabs_mappings
from .graph.labeled_graph import LabeledGraph
from .slap.lap import _adjacency_and_elements
from .spectrum import _attributed_its_graph, _exact_code


@dataclass(frozen=True)
class ReactionMapping:
    """One exact symmetry class representative, including any missing atoms."""

    mapped_reaction: str
    mapping: tuple[int, ...]
    class_id: str
    distance: float
    hydrogen_distance: int | None
    reactant_only_maps: tuple[int, ...]
    product_only_maps: tuple[int, ...]


@dataclass(frozen=True)
class ReactionMappingResult:
    """User-facing mappings with independent search/classification completion."""

    reactions: tuple[ReactionMapping, ...]
    complete: bool
    search_complete: bool
    classification_complete: bool
    hydrogen_complete: bool
    minimum_cost: float | None
    incomplete_reason: str | None
    backend: str
    hydrogen: str
    objective: str
    balance: str
    agents: str
    labeled_mapping_count: int
    labeled_mapping_count_scope: str
    minimum_cost_scope: str
    elapsed_seconds: float
    search_result: object

    def as_dict(self):
        """Export public metadata and mapped reactions without solver internals."""
        from dataclasses import fields

        return {
            field.name: (
                [asdict(item) for item in self.reactions]
                if field.name == "reactions"
                else getattr(self, field.name)
            )
            for field in fields(self)
            if field.name != "search_result"
        }

    @property
    def mapped_reactions(self):
        return tuple(item.mapped_reaction for item in self.reactions)


def _parse(reaction, hydrogen):
    if not isinstance(reaction, str):
        raise TypeError("reaction must be reaction SMILES")
    parts = reaction.split(">")
    if len(parts) != 3 or not parts[0] or not parts[2]:
        raise ValueError("Expected reactants>agents>products or reactants>>products")
    params = Chem.SmilesParserParams()
    params.removeHs = False
    endpoints = []
    for side in (parts[0], parts[2]):
        mol = Chem.MolFromSmiles(side, params)
        if mol is None:
            raise ValueError("Could not parse reaction endpoint")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                raise ValueError("Wildcard atoms require a concrete element")
            atom.SetAtomMapNum(0)
        if hydrogen == "explicit":
            mol = Chem.AddHs(mol)
        else:
            mol = Chem.RemoveHs(mol)
            if any(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms()):
                raise ValueError("Free or isotopic H requires hydrogen='explicit'")
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        endpoints.append(mol)
    return endpoints, parts[1]


def _identity(atom):
    return atom.GetAtomicNum(), atom.GetIsotope()


def _graphs(mols, balance, objective):
    inventories = [Counter(_identity(atom) for atom in mol.GetAtoms()) for mol in mols]
    if balance == "strict" and inventories[0] != inventories[1]:
        raise ValueError(
            "Unbalanced atom inventories; select balance='dummy' explicitly"
        )
    universe = inventories[0] | inventories[1]
    graphs = []
    for mol, inventory in zip(mols, inventories):
        atoms = list(mol.GetAtoms())
        labels = [_identity(atom) for atom in atoms]
        dummy_labels = sorted((universe - inventory).elements())
        labels.extend(dummy_labels)
        adjacency = {i: {} for i in range(len(labels))}
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            adjacency[i][j] = adjacency[j][i] = bond.GetBondTypeAsDouble()
        properties = {
            "charge": [atom.GetFormalCharge() for atom in atoms],
            "radical": [atom.GetNumRadicalElectrons() for atom in atoms],
            "hcount": [atom.GetTotalNumHs() for atom in atoms],
            "stereo": [
                atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else ""
                for atom in atoms
            ],
            "dummy": [False] * len(atoms),
        }
        for name, values in properties.items():
            values.extend(
                [True if name == "dummy" else "" if name == "stereo" else 0]
                * len(dummy_labels)
            )
        if objective == "lewis":
            from synkit.IO.mol_to_graph import MolToGraph

            # A uniquely colored anchor converts unary electron costs into
            # ordinary edges. Its bijection is forced by atom compatibility.
            electrons = [
                2 * MolToGraph.estimate_lone_pairs(atom) + atom.GetNumRadicalElectrons()
                for atom in atoms
            ]
            electrons.extend([0] * len(dummy_labels))
            anchor = len(labels)
            labels.append((0, -1))
            adjacency[anchor] = {}
            for i, count in enumerate(electrons):
                if count:
                    adjacency[i][anchor] = adjacency[anchor][i] = count / 2
            for name, values in properties.items():
                values.append(
                    True if name == "dummy" else "" if name == "stereo" else 0
                )
        graph = LabeledGraph(adjacency, labels)
        graph.props.update(properties)
        graphs.append(graph)
    return graphs


def _render(mols, mapping):
    mapped = [Chem.Mol(mol) for mol in mols]
    product_numbers = {image: atom + 1 for atom, image in enumerate(mapping)}
    for atom in mapped[0].GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    for atom in mapped[1].GetAtoms():
        atom.SetAtomMapNum(product_numbers[atom.GetIdx()])
    nr, np = (mol.GetNumAtoms() for mol in mols)
    reactant_only = tuple(i + 1 for i, j in enumerate(mapping[:nr]) if j >= np)
    product_only = tuple(i + 1 for i, j in enumerate(mapping) if i >= nr and j < np)
    return (
        ">>".join(Chem.MolToSmiles(mol, canonical=True) for mol in mapped),
        reactant_only,
        product_only,
    )


def _lift_mapping(mols, heavy_mapping, plan):
    explicit = [Chem.AddHs(mol) for mol in mols]
    children = []
    for mol in explicit:
        children.append(
            {
                atom.GetIdx(): [
                    neighbor.GetIdx()
                    for neighbor in atom.GetNeighbors()
                    if neighbor.GetAtomicNum() == 1
                ]
                for atom in mol.GetAtoms()
                if atom.GetAtomicNum() != 1
            }
        )
    mapping = [-1] * explicit[0].GetNumAtoms()
    mapping[: len(heavy_mapping)] = heavy_mapping

    def transfer(source, target, count):
        r = children[0][source]
        p = children[1][heavy_mapping[target]]
        for i, j in zip(r[:count], p[:count]):
            mapping[i] = j
        del r[:count]
        del p[:count]

    for i, count in enumerate(plan.preserved):
        transfer(i, i, count)
    for source, target, count in plan.transfers:
        transfer(source, target, count)
    if sorted(mapping) != list(range(len(mapping))):
        raise RuntimeError("Hydrogen flow did not produce a full bijection")
    return explicit, tuple(mapping)


def map_reaction(
    reaction,
    *,
    backend="python",
    library_path=None,
    CD="minimal",
    hydrogen="heavy",
    objective="bond",
    balance="strict",
    binary=False,
    time_limit_seconds=10.0,
    max_mappings=100_000,
    max_bijections=1_000_000,
    max_hydrogen_plans=10_000,
    classification_max_search_nodes=100_000,
    search_config=None,
):
    """Return symmetry-distinct mapped reactions from unmapped reaction SMILES.

    Hydrogen modes: ``heavy`` searches heavy bonds; ``explicit`` searches all
    H bonds jointly; ``compressed`` returns minimum-H lifts conditional on each
    selected heavy mapping (not a global full-H optimum). ``objective='lewis'``
    requires explicit H and weighted bonds, adding half the L1 difference of
    estimated nonbonding electron counts to bond-order distance.

    ``balance='dummy'`` pads missing element/isotope inventories with isolated
    zero-state placeholders. Output contains only real endpoint atoms; map IDs
    present on one side are reported explicitly. Agents are excluded from search
    and returned separately. Existing atom-map numbers are discarded, not fixed.
    ``search_config`` optionally supplies ``PropagationConfig`` controls for the
    Python PABS backend, including experimental separator and factor-spectrum
    bounds.

    Classes use exact colored ITS canonicalization, equivalent to the two-sided
    automorphism action beta*f*alpha^-1. Incomplete canonicalization never merges
    unverified classes. The deadline is shared by search, H lifting and dedup.
    """
    started = time.perf_counter()
    if hydrogen not in {"heavy", "explicit", "compressed"}:
        raise ValueError("hydrogen must be 'heavy', 'explicit', or 'compressed'")
    if objective not in {"bond", "lewis"} or balance not in {"strict", "dummy"}:
        raise ValueError(
            "objective must be 'bond' or 'lewis'; balance must be 'strict' or 'dummy'"
        )
    if objective == "lewis" and (hydrogen != "explicit" or binary):
        raise ValueError(
            "Lewis objective requires hydrogen='explicit' and binary=False"
        )
    if search_config is not None and not isinstance(search_config, PropagationConfig):
        raise TypeError("search_config must be a PropagationConfig or None")
    if search_config is not None and backend != "python":
        raise ValueError("search_config is currently supported by backend='python'")
    if hydrogen == "compressed" and balance != "strict":
        raise ValueError(
            "Compressed H requires strict balance; use explicit H with dummy balance"
        )
    if time_limit_seconds is not None and (
        isinstance(time_limit_seconds, bool)
        or not isinstance(time_limit_seconds, Real)
        or not math.isfinite(time_limit_seconds)
        or time_limit_seconds < 0
    ):
        raise ValueError("time_limit_seconds must be finite and non-negative")
    for name, value in (
        ("max_hydrogen_plans", max_hydrogen_plans),
        ("classification_max_search_nodes", classification_max_search_nodes),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    deadline = None if time_limit_seconds is None else started + time_limit_seconds

    def remaining():
        return None if deadline is None else max(0, deadline - time.perf_counter())

    mols, agents = _parse(reaction, hydrogen)
    graphs = _graphs(mols, balance, objective)
    hcounts = [[atom.GetTotalNumHs() for atom in mol.GetAtoms()] for mol in mols]
    if hydrogen == "compressed" and sum(hcounts[0]) != sum(hcounts[1]):
        raise ValueError(
            "Compressed H requires equal H inventories; use explicit H with dummy balance"
        )
    records = {}
    canonical_cache = {}
    classification_complete = hydrogen_complete = True
    failure = None

    def classify(endpoint_mols, endpoint_graphs, mapping, cost, hcost=None):
        nonlocal classification_complete, failure
        if not classification_complete:
            return
        if deadline is not None and time.perf_counter() >= deadline:
            classification_complete = False
            failure = "classification:time_limit"
            return
        a, labels = _adjacency_and_elements(endpoint_graphs[0], binary)
        b, _ = _adjacency_and_elements(endpoint_graphs[1], binary)
        properties = {
            name: (endpoint_graphs[0].props[name], endpoint_graphs[1].props[name])
            for name in ("charge", "radical", "hcount", "stereo", "dummy")
        }
        # _attributed_its_graph expects product already in reactant coordinates.
        images = np.asarray(mapping, dtype=int)
        its = _attributed_its_graph(
            a, b[images[:, None], images[None, :]], labels, properties, mapping
        )
        # Retain endpoint E/Z annotations in the exact equivalence relation.
        for i, j, attrs in its.edges(data=True):
            stereo = []
            for mol, left, right in (
                (endpoint_mols[0], i, j),
                (endpoint_mols[1], mapping[i], mapping[j]),
            ):
                bond = (
                    mol.GetBondBetweenAtoms(left, right)
                    if max(left, right) < mol.GetNumAtoms()
                    else None
                )
                stereo.append(str(bond.GetStereo()) if bond is not None else "")
            attrs["color"] += tuple(stereo)
        # Identical fully colored indexed graphs need no second proof. This
        # cache never treats a hash or WL fingerprint as class equality.
        signature = (
            tuple((i, attrs["color"]) for i, attrs in its.nodes(data=True)),
            tuple((i, j, attrs["color"]) for i, j, attrs in its.edges(data=True)),
        )
        code = canonical_cache.get(signature)
        reason = None
        if code is None:
            code, reason = _exact_code(
                its,
                timeout_seconds=remaining(),
                max_search_nodes=classification_max_search_nodes,
            )
        if code is None:
            classification_complete = False
            failure = f"classification:{reason}"
            return
        if len(canonical_cache) < 256:
            canonical_cache[signature] = code
        if code not in records:
            rendered, left_only, right_only = _render(endpoint_mols, mapping)
            records[code] = ReactionMapping(
                rendered,
                tuple(mapping),
                hashlib.sha256(repr(code).encode()).hexdigest(),
                float(cost),
                hcost,
                left_only,
                right_only,
            )

    def receive(mapping, cost):
        nonlocal hydrogen_complete, failure
        if hydrogen == "compressed":
            plans = enumerate_minimal_hydrogen_transfers(
                *hcounts,
                mapping,
                max_plans=max_hydrogen_plans,
                time_limit_seconds=remaining(),
            )
            if not plans.complete:
                hydrogen_complete = False
                failure = f"hydrogen:{plans.truncation_reason}"
            for plan in plans.plans:
                explicit, lifted = _lift_mapping(mols, mapping, plan)
                classify(
                    explicit,
                    _graphs(explicit, "strict", "bond"),
                    lifted,
                    cost + plan.distance,
                    plan.distance,
                )
        else:
            classify(mols, graphs, mapping, cost)

    search_options = {} if search_config is None else {"config": search_config}
    search = enumerate_pabs_mappings(
        graphs,
        backend=backend,
        library_path=library_path,
        CD=CD,
        binary=binary,
        max_bijections=max_bijections,
        max_mappings=max_mappings,
        time_limit_seconds=remaining(),
        collect_mappings=False,
        mapping_callback=receive,
        compute_minimum_cost=(CD == "minimal"),
        **search_options,
    )
    objective_name = (
        "heavy_cd_with_conditional_minimum_h_lifts"
        if hydrogen == "compressed"
        else (
            "lewis_bond_plus_nonbonding_electron_distance"
            if objective == "lewis"
            else (
                "explicit_h_bond_distance"
                if hydrogen == "explicit"
                else "heavy_bond_distance"
            )
        )
    )
    return ReactionMappingResult(
        reactions=tuple(sorted(records.values(), key=lambda item: item.class_id)),
        complete=search.complete and classification_complete and hydrogen_complete,
        search_complete=search.complete,
        classification_complete=classification_complete,
        hydrogen_complete=hydrogen_complete,
        minimum_cost=search.minimum_cost,
        incomplete_reason=failure or search.truncation_reason,
        backend=search.backend,
        hydrogen=hydrogen,
        objective=objective_name,
        balance=balance,
        agents=agents,
        labeled_mapping_count=search.selected_mapping_count,
        labeled_mapping_count_scope=(
            "heavy_mapping_shell"
            if hydrogen == "compressed"
            else "augmented_mapping_shell" if balance == "dummy" else "mapping_shell"
        ),
        minimum_cost_scope=(
            "heavy_bond_distance" if hydrogen == "compressed" else objective_name
        ),
        elapsed_seconds=time.perf_counter() - started,
        search_result=search,
    )


__all__ = ["ReactionMapping", "ReactionMappingResult", "map_reaction"]
