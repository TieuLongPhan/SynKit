from __future__ import annotations

"""Geometry helpers for electron-pushing arrows."""

from typing import Any, Dict, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from .constants import transition_family
from .chem import infer_shared_atom, other_in_bond
from .utils import as_tuple, mid, rot90, tget, unit


def best_free_direction(
    graph: nx.Graph,
    pos: Dict[int, np.ndarray],
    atom: int,
    preferred: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Choose a direction around an atom that avoids existing bonds."""
    p = pos[atom]
    blocked = []

    for nb in graph.neighbors(atom):
        vec = pos[nb] - p
        if np.linalg.norm(vec) > 1e-12:
            blocked.append(unit(vec))

    preferred = (
        np.array([1.0, 0.0], dtype=float) if preferred is None else unit(preferred)
    )

    best = None
    best_score = -1e18
    for deg in range(0, 360, 12):
        th = np.deg2rad(deg)
        cand = np.array([np.cos(th), np.sin(th)], dtype=float)
        sep = min((1.0 - float(np.dot(cand, b)) for b in blocked), default=2.0)
        score = sep + 0.35 * float(np.dot(cand, preferred))
        if score > best_score:
            best_score = score
            best = cand
    return unit(best)


def lp_anchor(
    graph: nx.Graph,
    pos: Dict[int, np.ndarray],
    atom: int,
    toward: Optional[np.ndarray],
    dist: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return lone-pair anchor position and direction."""
    pref = None if toward is None else (np.asarray(toward, dtype=float) - pos[atom])
    d = best_free_direction(graph, pos, atom, preferred=pref)
    return pos[atom] + dist * d, d


def virtual_h_position(
    graph: nx.Graph,
    pos: Dict[int, np.ndarray],
    donor_atom: int,
    acceptor_atom: int,
    dist: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Place a virtual proton near the donor atom."""
    d = best_free_direction(
        graph,
        pos,
        donor_atom,
        preferred=(pos[acceptor_atom] - pos[donor_atom]),
    )
    return pos[donor_atom] + dist * d, d


def bond_lobe_anchor(
    p_shared: np.ndarray,
    p_other: np.ndarray,
    side: float,
    along: float,
    offset: float,
) -> np.ndarray:
    """Anchor an arrow near one side of a bond instead of the midpoint."""
    vec = p_other - p_shared
    u = unit(vec)
    perp = rot90(u)
    return p_shared + along * vec + side * offset * perp


def choose_side_from_turn(
    p_shared: np.ndarray,
    p_src_other: np.ndarray,
    p_dst_other: np.ndarray,
) -> float:
    """Choose curvature sign from the source-to-destination turn direction."""
    a = unit(p_src_other - p_shared)
    b = unit(p_dst_other - p_shared)
    turn = a[0] * b[1] - a[1] * b[0]
    return 1.0 if turn >= 0 else -1.0


def build_proton_pair_context(
    graph: nx.Graph,
    pos: Dict[int, np.ndarray],
    transitions: Sequence[Any],
    h_dist: float,
) -> Dict[int, Dict[str, Any]]:
    """Associate LP-/H+ and H-/LP+ or H-/B+ steps with a shared virtual proton."""
    ctx: Dict[int, Dict[str, Any]] = {}

    for i in range(len(transitions) - 1):
        t1 = transitions[i]
        t2 = transitions[i + 1]

        if tget(t1, "kind") != "LP-/H+":
            continue
        if tget(t2, "kind") not in {"H-/LP+", "H-/B+"}:
            continue

        acceptor = as_tuple(tget(t1, "src"))[0]
        donor = as_tuple(tget(t2, "src"))[0]
        h_pos, _ = virtual_h_position(graph, pos, donor, acceptor, h_dist)

        ctx[i] = {
            "pair_id": (i, i + 1),
            "acceptor_atom": acceptor,
            "donor_atom": donor,
            "h_pos": h_pos,
        }
        ctx[i + 1] = {
            "pair_id": (i, i + 1),
            "acceptor_atom": acceptor,
            "donor_atom": donor,
            "h_pos": h_pos,
        }

    return ctx


def _base_spec(step: int) -> Dict[str, Any]:
    """Create a default arrow spec container."""
    return {
        "step": step,
        "tail": None,
        "head": None,
        "rad": 0.18 if step % 2 else -0.18,
        "lp_tail": None,
        "lp_tail_dir": None,
        "lp_head": None,
        "lp_head_dir": None,
        "virtual_h": None,
        "transition": None,
    }


def _maybe_add_virtual_h(
    spec: Dict[str, Any],
    info: Dict[str, Any],
    pos: Dict[int, np.ndarray],
    drawn_virtual_h: set,
) -> None:
    """Add the virtual proton guide once per proton-transfer pair."""
    if info["pair_id"] not in drawn_virtual_h:
        spec["virtual_h"] = (pos[info["donor_atom"]], info["h_pos"])
        drawn_virtual_h.add(info["pair_id"])


def _handle_lp_to_bond(
    spec: Dict[str, Any],
    graph: nx.Graph,
    pos: Dict[int, np.ndarray],
    src: Tuple[int, ...],
    dst: Tuple[int, ...],
    scale: float,
    lp_dist: float,
) -> None:
    """Handle LP-/B+."""
    donor = src[0]
    target = other_in_bond(dst, donor) if len(dst) == 2 else dst[0]
    if target is None:
        target = dst[-1]

    spec["tail"], spec["lp_tail_dir"] = lp_anchor(
        graph, pos, donor, toward=pos[target], dist=lp_dist
    )
    spec["lp_tail"] = spec["tail"]
    spec["head"] = pos[target] - unit(pos[target] - spec["tail"]) * 0.14 * scale


def _handle_bond_to_lp(
    spec: Dict[str, Any],
    graph: nx.Graph,
    pos: Dict[int, np.ndarray],
    src: Tuple[int, ...],
    dst: Tuple[int, ...],
    lp_dist: float,
) -> None:
    """Handle B-/LP+."""
    u, v = src
    acceptor = dst[0]
    spec["tail"] = mid(pos[u], pos[v])
    spec["head"], spec["lp_head_dir"] = lp_anchor(
        graph, pos, acceptor, toward=spec["tail"], dist=lp_dist
    )
    spec["lp_head"] = spec["head"]


def _handle_lp_to_proton(
    spec: Dict[str, Any],
    graph: nx.Graph,
    pos: Dict[int, np.ndarray],
    src: Tuple[int, ...],
    info: Dict[str, Any],
    drawn_virtual_h: set,
    lp_dist: float,
    scale: float,
) -> None:
    """Handle LP-/H+."""
    _maybe_add_virtual_h(spec, info, pos, drawn_virtual_h)

    acceptor = src[0]
    h_pos = info["h_pos"]
    spec["tail"], spec["lp_tail_dir"] = lp_anchor(
        graph, pos, acceptor, toward=h_pos, dist=lp_dist
    )
    spec["lp_tail"] = spec["tail"]
    spec["head"] = h_pos - unit(h_pos - spec["tail"]) * 0.04 * scale


def _handle_proton_to_lp(
    spec: Dict[str, Any],
    graph: nx.Graph,
    pos: Dict[int, np.ndarray],
    dst: Tuple[int, ...],
    info: Dict[str, Any],
    drawn_virtual_h: set,
    lp_dist: float,
) -> None:
    """Handle H-/LP+."""
    _maybe_add_virtual_h(spec, info, pos, drawn_virtual_h)

    donor = info["donor_atom"]
    h_pos = info["h_pos"]
    acceptor = dst[0]

    spec["tail"] = mid(pos[donor], h_pos)
    spec["head"], spec["lp_head_dir"] = lp_anchor(
        graph, pos, acceptor, toward=spec["tail"], dist=lp_dist
    )
    spec["lp_head"] = spec["head"]


def _handle_proton_to_bond(
    spec: Dict[str, Any],
    pos: Dict[int, np.ndarray],
    dst: Tuple[int, ...],
    info: Dict[str, Any],
    drawn_virtual_h: set,
) -> None:
    """Handle H-/B+."""
    _maybe_add_virtual_h(spec, info, pos, drawn_virtual_h)

    donor = info["donor_atom"]
    h_pos = info["h_pos"]

    spec["tail"] = mid(pos[donor], h_pos)
    spec["head"] = mid(pos[dst[0]], pos[dst[1]]) if len(dst) == 2 else pos[dst[0]]


def _handle_bond_to_bond_shared(
    spec: Dict[str, Any],
    pos: Dict[int, np.ndarray],
    src: Tuple[int, ...],
    dst: Tuple[int, ...],
    shared: int,
    along: float,
    offset: float,
) -> None:
    """Handle B-/B+ when source and destination bonds share an atom."""
    src_other = other_in_bond(src, shared)
    dst_other = other_in_bond(dst, shared)

    if src_other is None or dst_other is None:
        spec["tail"] = mid(pos[src[0]], pos[src[1]])
        spec["head"] = mid(pos[dst[0]], pos[dst[1]])
        return

    p_shared = pos[shared]
    side = choose_side_from_turn(p_shared, pos[src_other], pos[dst_other])
    spec["rad"] = 0.23 * side
    spec["tail"] = bond_lobe_anchor(p_shared, pos[src_other], side, along, offset)
    spec["head"] = bond_lobe_anchor(p_shared, pos[dst_other], side, along, offset)


def _handle_bond_to_bond(
    spec: Dict[str, Any],
    pos: Dict[int, np.ndarray],
    src: Tuple[int, ...],
    dst: Tuple[int, ...],
    data: Dict[str, Any],
    along: float,
    offset: float,
) -> None:
    """Handle B-/B+."""
    if len(src) != 2:
        return

    if len(dst) != 2:
        spec["tail"] = mid(pos[src[0]], pos[src[1]])
        spec["head"] = pos[dst[0]]
        return

    shared = infer_shared_atom(src, dst, data)
    if shared is None:
        spec["tail"] = mid(pos[src[0]], pos[src[1]])
        spec["head"] = mid(pos[dst[0]], pos[dst[1]])
        return

    _handle_bond_to_bond_shared(spec, pos, src, dst, shared, along, offset)


def _populate_spec_for_transition(
    spec: Dict[str, Any],
    graph: nx.Graph,
    pos: Dict[int, np.ndarray],
    kind: str,
    src: Tuple[int, ...],
    dst: Tuple[int, ...],
    data: Dict[str, Any],
    proton_ctx: Dict[int, Dict[str, Any]],
    transition_index0: int,
    drawn_virtual_h: set,
    scale: float,
    lp_dist: float,
    along: float,
    offset: float,
) -> None:
    """Populate one spec according to transition kind."""
    kind = transition_family(kind)
    if kind == "LP-/B+":
        _handle_lp_to_bond(spec, graph, pos, src, dst, scale, lp_dist)
        return

    if kind == "B-/LP+":
        _handle_bond_to_lp(spec, graph, pos, src, dst, lp_dist)
        return

    info = proton_ctx.get(transition_index0)
    if kind == "LP-/H+":
        if info is None:
            return
        _handle_lp_to_proton(
            spec, graph, pos, src, info, drawn_virtual_h, lp_dist, scale
        )
        return

    if kind == "H-/LP+":
        if info is None:
            return
        _handle_proton_to_lp(spec, graph, pos, dst, info, drawn_virtual_h, lp_dist)
        return

    if kind == "H-/B+":
        if info is None:
            return
        _handle_proton_to_bond(spec, pos, dst, info, drawn_virtual_h)
        return

    if kind == "B-/B+":
        _handle_bond_to_bond(spec, pos, src, dst, data, along, offset)


def arrow_specs_from_transitions(
    graph: nx.Graph,
    pos: Dict[int, np.ndarray],
    transitions: Sequence[Any],
    scale: float,
) -> list[dict]:
    """Convert transition records into drawable arrow specifications."""
    proton_ctx = build_proton_pair_context(graph, pos, transitions, h_dist=0.34 * scale)
    drawn_virtual_h = set()

    lp_dist = 0.23 * scale
    lp_sep = 0.034 * scale
    lp_radius = 0.011 * scale
    along = 0.68
    offset = 0.055 * scale

    specs: list[dict] = []

    for idx0, t in enumerate(transitions):
        step = idx0 + 1
        kind = tget(t, "kind")
        src = as_tuple(tget(t, "src"))
        dst = as_tuple(tget(t, "dst"))
        data = tget(t, "data", {}) or {}

        spec = _base_spec(step)
        spec["transition"] = t
        _populate_spec_for_transition(
            spec=spec,
            graph=graph,
            pos=pos,
            kind=kind,
            src=src,
            dst=dst,
            data=data,
            proton_ctx=proton_ctx,
            transition_index0=idx0,
            drawn_virtual_h=drawn_virtual_h,
            scale=scale,
            lp_dist=lp_dist,
            along=along,
            offset=offset,
        )

        if spec["tail"] is None or spec["head"] is None:
            continue

        spec["lp_sep"] = lp_sep
        spec["lp_radius"] = lp_radius
        specs.append(spec)

    return specs
