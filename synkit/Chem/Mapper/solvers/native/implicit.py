"""Exact heavy-atom classes scored by minimum implicit-H Lewis completion.

This additive interface never expands hydrogen or enumerates H identities. A
numeric shell selects heavy maps whose *minimum* H-completion score equals CD;
it is distinct from projecting an arbitrary full-H exact shell onto heavy atoms.
Literal H atoms remain in rendered endpoints, with no identity map numbers.
"""

from __future__ import annotations

import base64
import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from rdkit import Chem

from synkit.IO.mol_to_graph import MolToGraph

from ..exact.distance_search.records import (
    DistanceEnumerationResult,
    normalize_distance_target,
)
from ...reaction_search import ReactionMapping, ReactionMappingResult
from ..slap.lap import _adjacency_and_elements
from .classification import NativeITSBatchClassifier
from .core import enumerate_core_mappings
from .implicit_preparation import (
    UnsupportedImplicitHydrogen,
    prepare_implicit_problem,
)
from .reaction_text import split_reaction_fields
from .runtime import resolve_library_path
from .stereo import assign_stereo, stereo_smiles

SCOPE = "heavyclasses_minimum_H_completion_score"
_PROPERTIES = ("charge", "radical", "hcount", "stereo", "dummy")


@dataclass(frozen=True)
class ImplicitReactionMapping(ReactionMapping):
    """One heavy ITS class, with its actual native equality certificate."""

    canonical_key_base64: str


@dataclass(frozen=True)
class PreparedImplicitReaction:
    """Heavy graphs, original molecules and a fixed doubled-distance offset.

    Graph order is real heavy atoms, typed heavy dummies, electron anchor, H
    count anchor. ``heavy_indices`` addresses the unexpanded endpoint molecules.
    Anchor vertices participate in scoring and certificates, never in SMILES.
    """

    mols: tuple
    graphs: tuple
    heavy_indices: tuple[tuple[int, ...], tuple[int, ...]]
    offset2: int
    metadata: dict
    edge_stereo: tuple
    matrices2: tuple
    agents: str
    reaction: str

    def classifier(self, library_path, *, identifier_mode="legacy"):
        return make_implicit_classifier(
            self, library_path=library_path, identifier_mode=identifier_mode
        )


def prepare_implicit_reaction(reaction, *, balance="dummy"):
    """Prepare the unexpanded model, discarding all input identity map tags.

    Unsupported H chemistry raises ``UnsupportedImplicitHydrogen``; there is no
    hidden explicit-H fallback. Charge, radicals, total H counts, map-independent
    CIP labels and heavy bond E/Z remain exact class attributes.
    """
    if not isinstance(reaction, str):
        raise TypeError("reaction must be reaction SMILES")
    parts = split_reaction_fields(
        reaction,
        error_message="Expected reactants>agents>products or reactants>>products",
    )
    if not parts[0] or not parts[2]:
        raise ValueError("Expected reactants>agents>products or reactants>>products")
    params = Chem.SmilesParserParams()
    params.removeHs = False
    cleaned = []
    for side in (parts[0], parts[2]):
        mol = Chem.MolFromSmiles(side, params)
        if mol is None:
            raise ValueError("Could not parse reaction endpoint")
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        # Remove only the parsed atom-map field in-place. A SMILES writer can
        # reorder branches even with canonical=False; retaining the input
        # traversal keeps reference heavy indices aligned with this model.
        cleaned.append(re.sub(r":\d+(?=\])", "", side))
    mapless = f"{cleaned[0]}>{parts[1]}>{cleaned[1]}"
    graphs, offset2, metadata = prepare_implicit_problem(mapless, balance=balance)
    mols = tuple(Chem.MolFromSmiles(side, params) for side in cleaned)
    heavy_indices = tuple(
        tuple(metadata[f"heavy_original_indices_{side}"]) for side in ("r", "p")
    )
    n = len(graphs[0].labels)
    stereo_matrices = []
    for mol, graph, indices in zip(mols, graphs, heavy_indices):
        assign_stereo(mol)
        atoms = [mol.GetAtomWithIdx(i) for i in indices]
        pad = n - len(atoms)
        graph.props.update(
            charge=[a.GetFormalCharge() for a in atoms] + [0] * pad,
            radical=[a.GetNumRadicalElectrons() for a in atoms] + [0] * pad,
            hcount=[MolToGraph._total_h_count(a) for a in atoms] + [0] * pad,
            stereo=[
                a.GetProp("_CIPCode") if a.HasProp("_CIPCode") else "" for a in atoms
            ]
            + [""] * pad,
            dummy=[False] * len(atoms) + [True] * pad,
            _stereo_anchor=[0] * n,
        )
        positions = {original: i for i, original in enumerate(indices)}
        edges = np.full((n, n), "", dtype=object)
        for bond in mol.GetBonds():
            left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if left not in positions or right not in positions:
                continue
            i, j = positions[left], positions[right]
            edges[i, j] = edges[j, i] = str(bond.GetStereo())
            if bond.GetStereo() != Chem.BondStereo.STEREONONE:
                graph.props["_stereo_anchor"][i] = i + 1
                graph.props["_stereo_anchor"][j] = j + 1
        stereo_matrices.append(edges)
    colors = {"": 0}
    for graph, edges in zip(graphs, stereo_matrices):
        graph.props["_native_edge_colors"] = np.asarray(
            [[colors.setdefault(value, len(colors)) for value in row] for row in edges],
            dtype=np.int32,
        )
    matrices2 = tuple(
        np.rint(_adjacency_and_elements(g, False)[0] * 2).astype(np.int64)
        for g in graphs
    )
    return PreparedImplicitReaction(
        mols,
        graphs,
        heavy_indices,
        offset2,
        {
            **metadata,
            "scope": SCOPE,
            "class_scope": SCOPE,
            "stereo_assignment": "modern_CIP_v1; supplied_geometry_preserved",
        },
        tuple(stereo_matrices),
        matrices2,
        parts[1],
        mapless,
    )


def make_implicit_classifier(prepared, *, library_path, identifier_mode="legacy"):
    """Construct the exact heavy ITS palette used by returned certificates."""
    a, elements = _adjacency_and_elements(prepared.graphs[0], False)
    b, _ = _adjacency_and_elements(prepared.graphs[1], False)
    return NativeITSBatchClassifier(
        a,
        b,
        elements,
        {name: tuple(g.props[name] for g in prepared.graphs) for name in _PROPERTIES},
        library_path=library_path,
        edge_properties={"stereo": prepared.edge_stereo},
        identifier_mode=identifier_mode,
    )


def score_implicit_mapping2(prepared, mapping, *, include_offset=True):
    """Independently rescore a complete typed map, returning doubled CD."""
    n = len(prepared.graphs[0].labels)
    mapping = tuple(mapping)
    if (
        len(mapping) != n
        or any(isinstance(j, bool) or not isinstance(j, Integral) for j in mapping)
        or sorted(mapping) != list(range(n))
    ):
        raise ValueError("Expected a complete integer bijection")
    if any(
        prepared.graphs[0].labels[i] != prepared.graphs[1].labels[j]
        for i, j in enumerate(mapping)
    ):
        raise ValueError("Mapping violates atom types")
    a, b = prepared.matrices2
    cost2 = int(np.abs(a - b[np.ix_(mapping, mapping)]).sum() // 2)
    return cost2 + (prepared.offset2 if include_offset else 0)


def _render_reactant(prepared, *, canonical_output=True):
    mol = Chem.Mol(prepared.mols[0])
    for i, original in enumerate(prepared.heavy_indices[0]):
        mol.GetAtomWithIdx(original).SetAtomMapNum(i + 1)
    return stereo_smiles(mol, canonical=canonical_output)


def _render(prepared, mapping, *, canonical_output=True, reactant_text=None):
    product = Chem.Mol(prepared.mols[1])
    left, right = prepared.heavy_indices
    inverse = {j: i for i, j in enumerate(mapping)}
    for j, original in enumerate(right):
        product.GetAtomWithIdx(original).SetAtomMapNum(inverse[j] + 1)
    reactant_only = tuple(i + 1 for i in range(len(left)) if mapping[i] >= len(right))
    product_only = tuple(
        inverse[j] + 1 for j in range(len(right)) if inverse[j] >= len(left)
    )
    if reactant_text is None:
        reactant_text = _render_reactant(prepared, canonical_output=canonical_output)
    product_text = stereo_smiles(product, canonical=canonical_output)
    return (
        f"{reactant_text}>{prepared.agents}>{product_text}",
        reactant_only,
        product_only,
    )


def _empty_search(target, backend, reason=None, *, complete=False, unsupported=False):
    return DistanceEnumerationResult(
        target=target,
        cost=None,
        minimum_cost=None,
        mappings=[],
        distances=[],
        total_bijections=0,
        maximum_cost_upper_bound=0,
        visited_leaves=0,
        pruned_branches=0,
        elapsed_seconds=0,
        status="no_solutions"
        if complete
        else "unsupported"
        if unsupported
        else "timeout",
        complete=complete,
        truncation_reason=reason,
        backend=f"{backend}_cpp",
        backend_statistics={
            "minimum_proven": 0,
            "minimum2": -1,
            "incumbent2": -1,
            "lower_bound2": -1,
            "search_complete": int(complete),
            "classification_complete": int(complete),
            "candidate_count": 0,
            "class_count": 0,
            "backend_executed": False,
        },
    )


def map_reaction_implicit(
    reaction,
    *,
    library_path=None,
    backend="pabs",
    CD="minimal",
    CD_mode="exact",
    balance="dummy",
    time_limit_seconds=None,
    max_mappings=10000,
    max_states=None,
    max_classes=None,
    max_frontier=None,
    memory_limit_bytes=None,
    classification_max_search_nodes=100000,
    symmetry=True,
    proof_only=False,
    cancel_token=None,
    identifier_mode="legacy",
    canonical_output=True,
):
    """Enumerate exact heavy classes with minimum implicit-H Lewis scores.

    Preparation, native search, classification and rendering share one deadline.
    ``CD='minimal'`` proves the minimum and enumerates its heavy classes unless
    ``proof_only=True``. Numeric exact/at-most queries do no extra minimum solve.
    Unsupported H models return an explicit unsupported result, with no H lift.
    ``canonical_key_base64`` is the actual native packed equality key; class IDs
    are presentation identifiers and are never used for deduplication.
    """
    started = time.perf_counter()
    if not isinstance(canonical_output, bool):
        raise TypeError("canonical_output must be boolean")
    if identifier_mode not in {"legacy", "packed"}:
        raise ValueError("identifier_mode must be 'legacy' or 'packed'")
    target = normalize_distance_target(CD)
    if CD_mode not in {"exact", "at_most"} or (
        CD_mode == "at_most" and target == "minimal"
    ):
        raise ValueError("at_most requires numeric CD; otherwise use exact")
    if not isinstance(proof_only, bool) or not isinstance(symmetry, bool):
        raise TypeError("proof_only and symmetry must be boolean")
    if proof_only and target != "minimal":
        raise ValueError("proof_only requires minimal CD")
    if time_limit_seconds is not None and (
        isinstance(time_limit_seconds, bool)
        or not isinstance(time_limit_seconds, Real)
        or not math.isfinite(time_limit_seconds)
        or time_limit_seconds < 0
    ):
        raise ValueError("time_limit_seconds must be finite and nonnegative")
    if classification_max_search_nodes is not None and (
        isinstance(classification_max_search_nodes, bool)
        or not isinstance(classification_max_search_nodes, Integral)
        or classification_max_search_nodes < 1
    ):
        raise ValueError("classification_max_search_nodes must be positive or None")
    library_path = resolve_library_path(library_path)
    deadline = None if time_limit_seconds is None else started + time_limit_seconds

    def remaining():
        return None if deadline is None else max(0, deadline - time.perf_counter())

    records, registry = {}, {}
    reactant_text = None
    rendering_complete = True
    preparation_reason = None
    prepared = None
    try:
        prepared = prepare_implicit_reaction(reaction, balance=balance)
    except UnsupportedImplicitHydrogen as error:
        preparation_reason = f"implicit_H_unsupported:{error.reason}"
    if prepared is None:
        search = _empty_search(target, backend, preparation_reason, unsupported=True)
    else:
        reduced_target = (
            target if target == "minimal" else target - prepared.offset2 / 2
        )
        if reduced_target != "minimal" and reduced_target < 0:
            search = _empty_search(target, backend, complete=True)
        elif remaining() == 0:
            search = _empty_search(target, backend, "time_limit")
        else:
            classifier = None
            if not proof_only:
                classifier = make_implicit_classifier(
                    prepared, library_path=library_path, identifier_mode=identifier_mode
                )
                classifier.frontend_max_canonical_nodes = (
                    classification_max_search_nodes
                )

            def receive(mapping, cost):
                nonlocal rendering_complete, reactant_text
                record = registry.pop(tuple(mapping), None)
                if record is None:
                    raise RuntimeError(
                        "Native callback lacks its exact class certificate"
                    )
                if (
                    score_implicit_mapping2(prepared, mapping, include_offset=False)
                    != record.cost2
                    or abs(cost * 2 - record.cost2) > 1e-9
                ):
                    raise RuntimeError(
                        "Native implicit mapping failed independent rescoring"
                    )
                if record.key.parts in records:
                    return
                if deadline is not None and time.perf_counter() >= deadline:
                    rendering_complete = False
                    return
                if reactant_text is None:
                    reactant_text = _render_reactant(
                        prepared, canonical_output=canonical_output
                    )
                rendered, left_only, right_only = _render(
                    prepared,
                    mapping,
                    canonical_output=canonical_output,
                    reactant_text=reactant_text,
                )
                records[record.key.parts] = ImplicitReactionMapping(
                    rendered,
                    tuple(mapping),
                    record.identifier,
                    (record.cost2 + prepared.offset2) / 2,
                    None,
                    left_only,
                    right_only,
                    base64.b64encode(record.key.parts).decode("ascii"),
                )

            search = enumerate_core_mappings(
                prepared.graphs,
                library_path=library_path,
                backend=backend,
                CD=reduced_target,
                CD_mode=CD_mode,
                symmetry=symmetry,
                node_properties=(*_PROPERTIES, "_stereo_anchor"),
                time_limit_seconds=remaining(),
                max_states=max_states,
                max_mappings=max_mappings,
                max_classes=max_classes,
                max_frontier=max_frontier,
                memory_limit_bytes=memory_limit_bytes,
                collect_mappings=False,
                mapping_callback=None if proof_only else receive,
                proof_only=proof_only,
                cancel_token=cancel_token,
                _native_classifier=classifier,
                _native_class_registry=None if proof_only else registry,
            )
        if not search.backend_statistics.get("backend_executed", False):
            search.total_bijections = math.prod(
                math.factorial(count)
                for count in Counter(prepared.graphs[0].labels).values()
            )
            search.maximum_cost_upper_bound = (
                sum(int(matrix.sum()) for matrix in prepared.matrices2) / 4
            )
        search.target = target
        for field in ("cost", "minimum_cost", "maximum_cost_upper_bound"):
            value = getattr(search, field)
            if value is not None:
                setattr(search, field, value + prepared.offset2 / 2)
    details = search.backend_statistics
    details.update(
        scope=SCOPE,
        class_scope=SCOPE,
        representation="implicit_H_heavy_graph_with_two_score_anchors",
        hydrogen_nodes_added=0,
        hydrogen_vertices_searched=0,
        hydrogen_nodes_enumerated=0,
        hydrogen_enumeration_requested=False,
        H_identity_scope="marginalized; literal H rendered without map tags",
        enumeration_requested=not proof_only,
        CD_mode=CD_mode,
        class_equality="exact_native_packed_key",
        identifier_mode=identifier_mode,
        canonical_output=canonical_output,
        delivered_heavy_classes=len(records),
        rendering_complete=rendering_complete,
        preparation_unsupported_reason=preparation_reason,
    )
    if prepared is not None:
        details["implicit_preparation"] = prepared.metadata
        details["offset2"] = prepared.offset2
        for field in ("minimum2", "incumbent2", "lower_bound2"):
            if field in details:
                details[f"reduced_{field}"] = details[field]
                if details[field] >= 0:
                    details[field] += prepared.offset2
    classification_complete = (
        bool(details.get("classification_complete")) and rendering_complete
    )
    failure = preparation_reason
    if not proof_only and not classification_complete and failure is None:
        failure = "classification:" + (
            details.get("classification_reason")
            or ("rendering_time_limit" if not rendering_complete else None)
            or search.truncation_reason
            or "incomplete"
        )
    count = search.selected_labeled_mapping_count
    return ReactionMappingResult(
        reactions=tuple(
            sorted(records.values(), key=lambda item: item.canonical_key_base64)
        ),
        complete=search.complete and classification_complete and not proof_only,
        search_complete=search.complete,
        classification_complete=classification_complete,
        hydrogen_complete=True,
        minimum_cost=search.minimum_cost,
        incomplete_reason=failure or search.truncation_reason,
        backend=search.backend,
        hydrogen="implicit_minimum_completion",
        objective=SCOPE,
        balance=balance,
        agents=prepared.agents if prepared else split_reaction_fields(reaction)[1],
        labeled_mapping_count=count,
        labeled_mapping_count_scope=(
            "not_computed_symmetry_pruned_search"
            if count is None
            else "padded_heavy_mapping"
        ),
        minimum_cost_scope=SCOPE,
        elapsed_seconds=time.perf_counter() - started,
        search_result=search,
    )


__all__ = [
    "SCOPE",
    "ImplicitReactionMapping",
    "PreparedImplicitReaction",
    "UnsupportedImplicitHydrogen",
    "make_implicit_classifier",
    "map_reaction_implicit",
    "prepare_implicit_reaction",
    "score_implicit_mapping2",
]
