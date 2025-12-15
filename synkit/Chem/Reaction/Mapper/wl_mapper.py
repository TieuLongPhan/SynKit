from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import (
    Any,
    Dict,
    Hashable,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
    Set,
)
import hashlib
import itertools
import logging
import math
import time

from synkit.IO import rsmi_to_graph, graph_to_smi

try:
    from synkit.Chem.Reaction.canon_rsmi import CanonRSMI  # type: ignore
except Exception:
    CanonRSMI = None  # type: ignore

try:
    from synkit.Graph.ITS.its_construction import ITSConstruction  # type: ignore
    from synkit.Graph.ITS.its_decompose import get_rc  # type: ignore
except Exception:
    ITSConstruction = None  # type: ignore
    get_rc = None  # type: ignore


_NodeLabelKeys = Union[Tuple[str, ...], List[str], str]


def _safe_int(x: Any) -> int:
    try:
        return int(x)
    except Exception:
        return 0


def _norm_pair(u: Hashable, v: Hashable) -> Tuple[Hashable, Hashable]:
    return (u, v) if repr(u) <= repr(v) else (v, u)


def _stable_sort_key(x: Hashable) -> str:
    return f"{type(x).__name__}:{repr(x)}"


def _u64_from_bytes(b: bytes) -> int:
    return int.from_bytes(b[:8], "little", signed=False) if b else 0


def _u64_to_bytes(x: int) -> bytes:
    return int(x & ((1 << 64) - 1)).to_bytes(8, "little", signed=False)


def _token_bytes(tok: Any) -> bytes:
    if tok is None:
        return b""
    if isinstance(tok, bytes):
        return tok
    if isinstance(tok, str):
        return tok.encode("utf-8", errors="ignore")
    if isinstance(tok, bool):
        return b"1" if tok else b"0"
    if isinstance(tok, int):
        return f"i:{tok}".encode("utf-8")
    if isinstance(tok, float):
        return f"f:{tok:.6g}".encode("utf-8")
    return repr(tok).encode("utf-8", errors="ignore")


def _is_double_token(tok: Any) -> bool:
    if tok is None:
        return False
    if isinstance(tok, (int, float)):
        return abs(float(tok) - 2.0) < 1e-6
    if isinstance(tok, str):
        s = tok.strip().lower()
        if s in {"2", "double", "d"}:
            return True
        try:
            return abs(float(s) - 2.0) < 1e-6
        except Exception:
            return False
    return False


def _is_single_token(tok: Any) -> bool:
    if tok is None:
        return False
    if isinstance(tok, (int, float)):
        return abs(float(tok) - 1.0) < 1e-6
    if isinstance(tok, str):
        s = tok.strip().lower()
        if s in {"1", "single", "s"}:
            return True
        try:
            return abs(float(s) - 1.0) < 1e-6
        except Exception:
            return False
    return False


def _multiset_l1_sorted(a: List[int], b: List[int]) -> int:
    i = 0
    j = 0
    da = len(a)
    db = len(b)
    miss = 0
    while i < da and j < db:
        va = a[i]
        vb = b[j]
        if va == vb:
            ca = 1
            cb = 1
            i += 1
            j += 1
            while i < da and a[i] == va:
                ca += 1
                i += 1
            while j < db and b[j] == vb:
                cb += 1
                j += 1
            miss += abs(ca - cb)
            continue
        if va < vb:
            ca = 1
            i += 1
            while i < da and a[i] == va:
                ca += 1
                i += 1
            miss += ca
            continue
        cb = 1
        j += 1
        while j < db and b[j] == vb:
            cb += 1
            j += 1
        miss += cb
    while i < da:
        va = a[i]
        ca = 1
        i += 1
        while i < da and a[i] == va:
            ca += 1
            i += 1
        miss += ca
    while j < db:
        vb = b[j]
        cb = 1
        j += 1
        while j < db and b[j] == vb:
            cb += 1
            j += 1
        miss += cb
    return int(miss)


@dataclass(frozen=True)
class WLMapperConfig:
    # WL hashing
    iterations: int = 4
    digest_size: int = 16
    include_initial: bool = True
    edge_attr: str = "order"
    node_label_keys: _NodeLabelKeys = ("element",)
    progressive_fallback: bool = True
    normalize_aromatic_bonds: bool = True

    # Candidate exploration (edge masks)
    enable_bond_cut: bool = True
    max_bond_cut_size: int = 2
    max_candidates: int = 200
    candidate_edge_pool: int = 16
    time_limit_s: Optional[float] = 2.0

    # Allow deeper masks (still PMCD-first; only used to *generate* candidates)
    enable_heuristic_scoring: bool = True
    heuristic_max_cut_size: int = 4
    heuristic_candidate_budget: int = 120

    # PMCD objective: strict minimization key (lexicographic)
    pmcd_unmapped_weight: int = 1
    pmcd_bond_weight: int = 1
    pmcd_hcount_weight: int = 1

    # Heuristic tie-break objective (ONLY after PMCD)
    heuristic_carbonyl_double_penalty: float = 50.0
    heuristic_aromatic_c_break_penalty: float = 0.75

    bc_cost_acyl_co: float = 0.10
    bc_cost_x_deg3_o: float = 0.35
    bc_cost_x_deg2_o: float = 0.55
    bc_cost_x_deg1_o: float = 0.75
    bc_cost_aromatic_co: float = 1.50
    bc_cost_other: float = 1.00
    bc_cost_order_mismatch_scale: float = 0.60

    # Peracid/peroxyacyl: prefer O-O cleavage; avoid acyl C-O cleavage in C(=O)-O-O
    bc_cost_peroxy_oo: float = 0.05
    bc_cost_acyl_co_peroxy: float = 3.00

    # Reaction-center restriction for PMCD stats (optional)
    rc_only_bond_changes: bool = True
    rc_only_hcount_changes: bool = True
    rc_expand_hops: int = 1

    # Refinements
    enable_rc_refine: bool = True
    rc_distance_weight: float = 0.5

    enable_symmetry_pruning: bool = True
    symmetry_depth: int = 4

    large_bucket_threshold: int = 25
    greedy_topk_per_u: int = 10
    hungarian_max_size: int = 10

    enable_dynamic_wl: bool = True
    enable_swap_refine: bool = True
    swap_refine_max_iter: int = 10
    swap_refine_class_depth: int = 4
    swap_refine_max_group_size: int = 14

    # Output
    multi_solutions: bool = True
    max_solutions: int = (
        6  # returned solutions (PMCD-min set will be trimmed by heuristic ordering)
    )
    solution_score_slack: float = (
        0.0  # not used in PMCD-first mode (kept for compatibility)
    )

    start_atom_map: int = 1
    unmapped_value: int = 0
    assign_maps_to_unmapped: bool = True
    use_its_final: bool = True

    drop_non_aam: bool = False
    use_index_as_atom_map: bool = False

    def validated(self) -> "WLMapperConfig":
        keys = self._normalize_node_keys(self.node_label_keys)
        self._validate(keys)
        if keys != self.node_label_keys:
            d = asdict(self)
            d["node_label_keys"] = keys
            return WLMapperConfig(**d)
        return self

    @staticmethod
    def _normalize_node_keys(keys: _NodeLabelKeys) -> Tuple[str, ...]:
        if isinstance(keys, str):
            return (keys,)
        if isinstance(keys, list):
            return tuple(keys)
        if isinstance(keys, tuple):
            return keys
        raise ValueError("node_label_keys must be str/list/tuple")

    def _validate(self, keys: Tuple[str, ...]) -> None:
        if not keys:
            raise ValueError("node_label_keys must be non-empty")
        if self.iterations < 1:
            raise ValueError("iterations must be >= 1")
        if self.digest_size < 8:
            raise ValueError("digest_size must be >= 8")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be >= 1")
        if self.candidate_edge_pool < 1:
            raise ValueError("candidate_edge_pool must be >= 1")
        if self.max_bond_cut_size < 0:
            raise ValueError("max_bond_cut_size must be >= 0")
        if self.time_limit_s is not None and self.time_limit_s < 0:
            raise ValueError("time_limit_s must be >= 0/None")
        if self.symmetry_depth < 1:
            raise ValueError("symmetry_depth must be >= 1")
        if self.large_bucket_threshold < 2:
            raise ValueError("large_bucket_threshold must be >= 2")
        if self.start_atom_map < 1:
            raise ValueError("start_atom_map must be >= 1")
        if self.hungarian_max_size < 2:
            raise ValueError("hungarian_max_size must be >= 2")
        if self.greedy_topk_per_u < 1:
            raise ValueError("greedy_topk_per_u must be >= 1")
        for v in (
            (
                self.heuristic_extra_penalty
                if hasattr(self, "heuristic_extra_penalty")
                else 0.0
            ),  # backward-compat
            self.heuristic_aromatic_c_break_penalty,
            self.heuristic_carbonyl_double_penalty,
            self.bc_cost_acyl_co,
            self.bc_cost_x_deg3_o,
            self.bc_cost_x_deg2_o,
            self.bc_cost_x_deg1_o,
            self.bc_cost_aromatic_co,
            self.bc_cost_other,
            self.bc_cost_order_mismatch_scale,
            self.bc_cost_peroxy_oo,
            self.bc_cost_acyl_co_peroxy,
        ):
            if v < 0:
                raise ValueError("weights must be >= 0")

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MappingResult:
    mapping: Dict[Hashable, Hashable]
    # NOTE: score is PMCD numeric proxy (for convenience). True minimization uses pmcd_key.
    score: float
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Solution:
    result: MappingResult
    mapped_rsmi: str


@dataclass(frozen=True)
class EdgeMaskCandidate:
    side: str
    removed_pairs: frozenset
    prior_score: float = 0.0  # kept for API-compat; NOT used in PMCD stage
    meta: Dict[str, Any] = field(default_factory=dict)


class _CanonAdapter:
    def canonicalise(self, rsmi: str) -> str:
        if CanonRSMI is None:
            return rsmi
        try:
            return CanonRSMI().canonicalise(rsmi).canonical_rsmi
        except Exception:
            return rsmi


class _SymmetryPruner:
    def __init__(self, cfg: WLMapperConfig) -> None:
        self._cfg = cfg

    def node_classes(
        self, hmap: Dict[Hashable, List[bytes]]
    ) -> Dict[Hashable, Tuple[bytes, ...]]:
        depth = int(self._cfg.symmetry_depth)
        out: Dict[Hashable, Tuple[bytes, ...]] = {}
        for u, seq in hmap.items():
            out[u] = tuple(seq[: min(depth, len(seq))])
        return out

    def unique_edges(
        self,
        edges: List[Tuple[Hashable, Hashable]],
        nclass: Dict[Hashable, Tuple[bytes, ...]],
        edge_key_fn,
    ) -> List[Tuple[Hashable, Hashable]]:
        seen: set = set()
        out: List[Tuple[Hashable, Hashable]] = []
        for u, v in edges:
            cu = nclass.get(u)
            cv = nclass.get(v)
            if cu is None or cv is None:
                out.append((u, v))
                continue
            a, b = (cu, cv) if repr(cu) <= repr(cv) else (cv, cu)
            sig = (a, b, edge_key_fn(u, v))
            if sig in seen:
                continue
            seen.add(sig)
            out.append((u, v))
        return out


class GraphCache:
    def __init__(self, G: Any, cfg: WLMapperConfig) -> None:
        self.G = G
        self.cfg = cfg

        self.nodes: List[Hashable] = list(G.nodes)
        self.node_set: Set[Hashable] = set(self.nodes)

        self._node_data: Dict[Hashable, Dict[str, Any]] = {
            u: d for u, d in G.nodes(data=True)
        }
        self._neighbors: Dict[Hashable, Tuple[Hashable, ...]] = {
            u: tuple(G.neighbors(u)) for u in self.nodes
        }
        self._neighbor_set: Dict[Hashable, Set[Hashable]] = {
            u: set(self._neighbors[u]) for u in self.nodes
        }

        self._element: Dict[Hashable, Any] = {
            u: self._node_data[u].get("element") for u in self.nodes
        }
        self._aromatic: Dict[Hashable, bool] = {
            u: bool(self._node_data[u].get("aromatic")) for u in self.nodes
        }
        self._hcount: Dict[Hashable, int] = {
            u: _safe_int(self._node_data[u].get("hcount", 0)) for u in self.nodes
        }
        self._charge: Dict[Hashable, int] = {
            u: _safe_int(self._node_data[u].get("charge", 0)) for u in self.nodes
        }

        self._edge_orders: Dict[Tuple[Hashable, Hashable], Tuple[Any, ...]] = {}
        self._edge_label_one: Dict[Tuple[Hashable, Hashable], Any] = {}
        self._edge_code_u64: Dict[Tuple[Hashable, Hashable], int] = {}
        self._edge_is_single: Dict[Tuple[Hashable, Hashable], bool] = {}
        self._edge_is_double: Dict[Tuple[Hashable, Hashable], bool] = {}

        self._neigh_codes: Dict[Hashable, List[int]] = {}
        self._carbonyl_c: Set[Hashable] = set()
        self._acyl_oxygen: Set[Hashable] = set()
        self._acyl_oxygen_peroxy: Set[Hashable] = set()

        self._build_edge_caches()
        self._build_carbonyl_cache()
        self._build_acyl_oxygen_cache()
        self._build_neighbor_codes()

    def node_data(self, u: Hashable) -> Dict[str, Any]:
        return self._node_data[u]

    def neighbors(self, u: Hashable) -> Tuple[Hashable, ...]:
        return self._neighbors.get(u, ())

    def neighbors_set(self, u: Hashable) -> Set[Hashable]:
        return self._neighbor_set.get(u, set())

    def element(self, u: Hashable) -> Any:
        return self._element.get(u)

    def aromatic(self, u: Hashable) -> bool:
        return bool(self._aromatic.get(u, False))

    def hcount(self, u: Hashable) -> int:
        return int(self._hcount.get(u, 0))

    def charge(self, u: Hashable) -> int:
        return int(self._charge.get(u, 0))

    def degree(self, u: Hashable) -> int:
        return int(len(self.neighbors(u)))

    def edge_orders(self, u: Hashable, v: Hashable) -> Tuple[Any, ...]:
        a, b = _norm_pair(u, v)
        return self._edge_orders.get((a, b), ())

    def edge_label_one(self, u: Hashable, v: Hashable) -> Any:
        a, b = _norm_pair(u, v)
        return self._edge_label_one.get((a, b))

    def edge_code_u64(self, u: Hashable, v: Hashable) -> int:
        a, b = _norm_pair(u, v)
        return int(self._edge_code_u64.get((a, b), 0))

    def edge_is_single(self, u: Hashable, v: Hashable) -> bool:
        a, b = _norm_pair(u, v)
        return bool(self._edge_is_single.get((a, b), False))

    def edge_is_double(self, u: Hashable, v: Hashable) -> bool:
        a, b = _norm_pair(u, v)
        return bool(self._edge_is_double.get((a, b), False))

    def neigh_codes(self, u: Hashable) -> List[int]:
        return self._neigh_codes.get(u, [])

    def is_carbonyl_c(self, u: Hashable) -> bool:
        return u in self._carbonyl_c

    def is_acyl_oxygen(self, o: Hashable) -> bool:
        return o in self._acyl_oxygen

    def is_acyl_oxygen_peroxy(self, o: Hashable) -> bool:
        return o in self._acyl_oxygen_peroxy

    # ---------- internal builds ----------

    def _build_edge_caches(self) -> None:
        for u, v in self.G.edges():
            a, b = _norm_pair(u, v)
            if (a, b) in self._edge_orders:
                continue
            toks = self._edge_orders_from_get_edge_data(a, b)
            self._edge_orders[(a, b)] = toks
            tok1 = toks[0] if toks else None
            self._edge_label_one[(a, b)] = tok1
            self._edge_code_u64[(a, b)] = self._stable_u64_for_token(tok1)
            self._edge_is_single[(a, b)] = _is_single_token(tok1)
            self._edge_is_double[(a, b)] = _is_double_token(tok1)

    def _edge_orders_from_get_edge_data(
        self, u: Hashable, v: Hashable
    ) -> Tuple[Any, ...]:
        data = self.G.get_edge_data(u, v, default=None)
        if data is None:
            return ()
        if isinstance(data, dict) and any(isinstance(k, int) for k in data.keys()):
            toks: List[Any] = []
            for d in data.values():
                if not isinstance(d, dict):
                    continue
                raw = d.get(self.cfg.edge_attr)
                tok = self._edge_order_token(u, v, raw)
                if tok is not None:
                    toks.append(tok)
            if not toks:
                return ()
            return tuple(sorted(toks, key=repr))
        if isinstance(data, dict):
            raw = data.get(self.cfg.edge_attr)
            tok = self._edge_order_token(u, v, raw)
            return (tok,) if tok is not None else ()
        return ()

    def _edge_order_token(self, u: Hashable, v: Hashable, o: Any) -> Any:
        if o is None:
            return "aromatic" if self._is_aromatic_edge(u, v) else None
        if self._is_aromatic_edge(u, v):
            return "aromatic"
        if isinstance(o, str):
            s = o.lower()
            return "aromatic" if "arom" in s else o
        if isinstance(o, (int, float)):
            val = float(o)
            if abs(val - 1.5) < 1e-3:
                return "aromatic"
            return round(val, 3) if isinstance(o, float) else o
        return o

    def _is_aromatic_edge(self, u: Hashable, v: Hashable) -> bool:
        if not self.cfg.normalize_aromatic_bonds:
            return False
        return self.aromatic(u) and self.aromatic(v)

    def _stable_u64_for_token(self, tok: Any) -> int:
        b = _token_bytes(tok)
        h = hashlib.blake2b(digest_size=8)
        h.update(b"edge:")
        h.update(b)
        return _u64_from_bytes(h.digest())

    def _stable_u32_for_neighbor_key(self, tok: Any, v: Hashable) -> int:
        h = hashlib.blake2b(digest_size=8)
        h.update(b"n:")
        h.update(_token_bytes(tok))
        h.update(b"|")
        h.update(_token_bytes(self.element(v)))
        h.update(b"|")
        h.update(b"1" if self.aromatic(v) else b"0")
        h.update(b"|")
        h.update(str(self.charge(v)).encode("utf-8"))
        h.update(b"|")
        h.update(str(self.hcount(v)).encode("utf-8"))
        return int.from_bytes(h.digest()[:4], "little", signed=False)

    def _build_neighbor_codes(self) -> None:
        out: Dict[Hashable, List[int]] = {}
        for u in self.nodes:
            codes: List[int] = []
            for v in self.neighbors(u):
                tok = self.edge_label_one(u, v)
                codes.append(self._stable_u32_for_neighbor_key(tok, v))
            codes.sort()
            out[u] = codes
        self._neigh_codes = out

    def _build_carbonyl_cache(self) -> None:
        carbonyl: Set[Hashable] = set()
        for u in self.nodes:
            if self.element(u) != "C":
                continue
            for v in self.neighbors(u):
                if self.element(v) != "O":
                    continue
                if self.edge_is_double(u, v):
                    carbonyl.add(u)
                    break
        self._carbonyl_c = carbonyl

    def _build_acyl_oxygen_cache(self) -> None:
        acyl_o: Set[Hashable] = set()
        acyl_o_peroxy: Set[Hashable] = set()

        for c in self._carbonyl_c:
            for o in self.neighbors(c):
                if self.element(o) != "O":
                    continue
                if self.edge_is_double(c, o):
                    continue
                acyl_o.add(o)

        # peroxyacyl oxygen: acyl oxygen bonded to another oxygen (C(=O)-O-O...)
        for o in acyl_o:
            for n in self.neighbors(o):
                if self.element(n) == "O" and n != o and self.edge_orders(o, n):
                    acyl_o_peroxy.add(o)
                    break

        self._acyl_oxygen = acyl_o
        self._acyl_oxygen_peroxy = acyl_o_peroxy


class MaskView:
    def __init__(self, cache: GraphCache, removed_pairs: Optional[frozenset]) -> None:
        self.cache = cache
        self.removed_pairs = removed_pairs
        self._over_neigh_set: Dict[Hashable, Set[Hashable]] = {}
        self._over_deg: Dict[Hashable, int] = {}
        self._over_neigh_codes: Dict[Hashable, List[int]] = {}
        if removed_pairs:
            self._build_overrides(removed_pairs)

    def masked(self, u: Hashable, v: Hashable) -> bool:
        if not self.removed_pairs:
            return False
        a, b = _norm_pair(u, v)
        return (a, b) in self.removed_pairs

    def neighbors(self, u: Hashable) -> Tuple[Hashable, ...]:
        if not self.removed_pairs:
            return self.cache.neighbors(u)
        if u not in self._over_neigh_set:
            return tuple(v for v in self.cache.neighbors(u) if not self.masked(u, v))
        return tuple(self._over_neigh_set[u])

    def neighbors_set(self, u: Hashable) -> Set[Hashable]:
        if not self.removed_pairs:
            return self.cache.neighbors_set(u)
        if u in self._over_neigh_set:
            return self._over_neigh_set[u]
        return self.cache.neighbors_set(u)

    def degree(self, u: Hashable) -> int:
        if not self.removed_pairs:
            return self.cache.degree(u)
        if u in self._over_deg:
            return self._over_deg[u]
        return self.cache.degree(u)

    def edge_orders(self, u: Hashable, v: Hashable) -> Tuple[Any, ...]:
        return () if self.masked(u, v) else self.cache.edge_orders(u, v)

    def edge_label_one(self, u: Hashable, v: Hashable) -> Any:
        return None if self.masked(u, v) else self.cache.edge_label_one(u, v)

    def edge_code_u64(self, u: Hashable, v: Hashable) -> int:
        return 0 if self.masked(u, v) else self.cache.edge_code_u64(u, v)

    def edge_is_single(self, u: Hashable, v: Hashable) -> bool:
        return False if self.masked(u, v) else self.cache.edge_is_single(u, v)

    def edge_is_double(self, u: Hashable, v: Hashable) -> bool:
        return False if self.masked(u, v) else self.cache.edge_is_double(u, v)

    def neigh_codes(self, u: Hashable) -> List[int]:
        if not self.removed_pairs:
            return self.cache.neigh_codes(u)
        if u not in self._over_neigh_set:
            return self.cache.neigh_codes(u)
        if u in self._over_neigh_codes:
            return self._over_neigh_codes[u]
        codes = self._compute_neigh_codes_masked(u)
        self._over_neigh_codes[u] = codes
        return codes

    def _compute_neigh_codes_masked(self, u: Hashable) -> List[int]:
        codes: List[int] = []
        for v in self.neighbors(u):
            tok = self.edge_label_one(u, v)
            codes.append(self.cache._stable_u32_for_neighbor_key(tok, v))
        codes.sort()
        return codes

    def _build_overrides(self, removed_pairs: frozenset) -> None:
        affected: Set[Hashable] = set()
        for a, b in removed_pairs:
            affected.add(a)
            affected.add(b)

        for u in affected:
            if u not in self.cache.node_set:
                continue
            self._over_neigh_set[u] = set(self.cache.neighbors_set(u))

        for a, b in removed_pairs:
            if a in self._over_neigh_set:
                self._over_neigh_set[a].discard(b)
            if b in self._over_neigh_set:
                self._over_neigh_set[b].discard(a)

        for u, ns in self._over_neigh_set.items():
            self._over_deg[u] = len(ns)


class _FastWLHasher:
    def __init__(self, cfg: WLMapperConfig) -> None:
        self.cfg = cfg

    def full_hashes(
        self,
        view: MaskView,
        init_labels: Dict[Hashable, bytes],
        blake_fn,
    ) -> Dict[Hashable, List[bytes]]:
        out: Dict[Hashable, List[bytes]] = {u: [] for u in view.cache.nodes}
        prev: Dict[Hashable, bytes] = {}

        if self.cfg.include_initial:
            for u in view.cache.nodes:
                hu = blake_fn(b"init:" + init_labels.get(u, b""))
                out[u].append(hu)
                prev[u] = hu
        else:
            for u in view.cache.nodes:
                prev[u] = blake_fn(b"init:" + init_labels.get(u, b""))

        for _ in range(int(self.cfg.iterations)):
            prev = self._one_round(view, prev, out)

        return out

    def masked_hashes(
        self,
        view: MaskView,
        init_labels: Dict[Hashable, bytes],
        base_wl: Dict[Hashable, List[bytes]],
        removed_pairs: frozenset,
        blake_fn,
    ) -> Tuple[Dict[Hashable, List[bytes]], Set[Hashable]]:
        if not removed_pairs:
            return base_wl, set()

        affected = self._affected_nodes(view.cache, removed_pairs)
        if not affected:
            return base_wl, set()

        out: Dict[Hashable, List[bytes]] = {}
        for u, seq in base_wl.items():
            out[u] = list(seq) if u in affected else seq

        prev_aff: Dict[Hashable, bytes] = {}
        if self.cfg.include_initial:
            for u in affected:
                hu = blake_fn(b"init:" + init_labels.get(u, b""))
                out[u][0] = hu
                prev_aff[u] = hu
        else:
            for u in affected:
                prev_aff[u] = blake_fn(b"init:" + init_labels.get(u, b""))

        for r in range(1, int(self.cfg.iterations) + 1):
            prev_aff = self._one_round_partial(view, prev_aff, out, affected, r)

        return out, affected

    def _affected_nodes(
        self, cache: GraphCache, removed_pairs: frozenset
    ) -> Set[Hashable]:
        seeds: Set[Hashable] = set()
        for a, b in removed_pairs:
            if a in cache.node_set:
                seeds.add(a)
            if b in cache.node_set:
                seeds.add(b)
        if not seeds:
            return set()

        radius = int(self.cfg.iterations)
        seen = set(seeds)
        frontier = set(seeds)
        for _ in range(radius):
            nxt: Set[Hashable] = set()
            for u in frontier:
                for v in cache.neighbors(u):
                    if v not in seen:
                        seen.add(v)
                        nxt.add(v)
            frontier = nxt
            if not frontier:
                break
        return seen

    def _one_round(
        self,
        view: MaskView,
        prev: Dict[Hashable, bytes],
        out: Dict[Hashable, List[bytes]],
    ) -> Dict[Hashable, bytes]:
        new_prev: Dict[Hashable, bytes] = {}
        for u in view.cache.nodes:
            hu = self._hash_node(view, u, prev)
            out[u].append(hu)
            new_prev[u] = hu
        return new_prev

    def _one_round_partial(
        self,
        view: MaskView,
        prev_aff: Dict[Hashable, bytes],
        out: Dict[Hashable, List[bytes]],
        affected: Set[Hashable],
        round_idx: int,
    ) -> Dict[Hashable, bytes]:
        new_prev: Dict[Hashable, bytes] = {}
        pos = round_idx if self.cfg.include_initial else (round_idx - 1)
        for u in affected:
            hu = self._hash_node_partial(view, u, prev_aff, out, affected, pos)
            out[u][pos] = hu
            new_prev[u] = hu
        return new_prev

    def _hash_node(
        self, view: MaskView, u: Hashable, prev: Dict[Hashable, bytes]
    ) -> bytes:
        xu = _u64_from_bytes(prev.get(u, b""))
        acc_xor = 0
        acc_sum = 0
        for v in view.neighbors(u):
            ev = int(view.edge_code_u64(u, v))
            xv2 = _u64_from_bytes(prev.get(v, b""))
            c = (ev ^ xv2) & ((1 << 64) - 1)
            acc_xor ^= c
            acc_sum = (acc_sum + c) & ((1 << 64) - 1)

        h = hashlib.blake2b(digest_size=int(self.cfg.digest_size))
        h.update(b"wl:")
        h.update(_u64_to_bytes(xu))
        h.update(_u64_to_bytes(acc_xor))
        h.update(_u64_to_bytes(acc_sum))
        h.update(_u64_to_bytes(view.degree(u)))
        return h.digest()

    def _hash_node_partial(
        self,
        view: MaskView,
        u: Hashable,
        prev_aff: Dict[Hashable, bytes],
        out: Dict[Hashable, List[bytes]],
        affected: Set[Hashable],
        prev_pos: int,
    ) -> bytes:
        prev_u = prev_aff.get(u)
        if prev_u is None:
            prev_u = out[u][prev_pos - 1] if prev_pos - 1 >= 0 else b""
        xu = _u64_from_bytes(prev_u)

        acc_xor = 0
        acc_sum = 0
        for v in view.neighbors(u):
            ev = int(view.edge_code_u64(u, v))
            if v in affected:
                pv = prev_aff.get(v)
                if pv is None:
                    pv = out[v][prev_pos - 1] if prev_pos - 1 >= 0 else b""
            else:
                pv = out[v][prev_pos - 1] if prev_pos - 1 >= 0 else b""
            xv2 = _u64_from_bytes(pv)
            c = (ev ^ xv2) & ((1 << 64) - 1)
            acc_xor ^= c
            acc_sum = (acc_sum + c) & ((1 << 64) - 1)

        h = hashlib.blake2b(digest_size=int(self.cfg.digest_size))
        h.update(b"wl:")
        h.update(_u64_to_bytes(xu))
        h.update(_u64_to_bytes(acc_xor))
        h.update(_u64_to_bytes(acc_sum))
        h.update(_u64_to_bytes(view.degree(u)))
        return h.digest()


class WLMapper:
    """
    PMCD-first mapping:
      1) Enumerate candidate masks; for each candidate compute mapping and its PMCD key.
      2) Keep the PMCD-minimal set (can be multiple solutions).
      3) Apply chemical heuristic ONLY to choose optimal among PMCD-minimal.
    """

    def __init__(
        self,
        config: WLMapperConfig = WLMapperConfig(),
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.cfg = config.validated()
        self._log = logger or logging.getLogger(__name__)
        self._canon = _CanonAdapter()
        self._sym = _SymmetryPruner(self.cfg)
        self._hasher = _FastWLHasher(self.cfg)

        self._rG: Any = None
        self._pG: Any = None
        self._r_cache: Optional[GraphCache] = None
        self._p_cache: Optional[GraphCache] = None

        self._mapped_rsmi: Optional[str] = None
        self._mapping: Dict[Hashable, Hashable] = {}
        self._best: Optional[MappingResult] = None
        self._solutions: List[Solution] = []
        self._bond_change_report: List[Dict[str, Any]] = []

        self._r_init: Dict[Hashable, bytes] = {}
        self._p_init: Dict[Hashable, bytes] = {}
        self._r_wl: Dict[Hashable, List[bytes]] = {}
        self._p_wl: Dict[Hashable, List[bytes]] = {}
        self._r_class: Dict[Hashable, Tuple[bytes, ...]] = {}
        self._p_class: Dict[Hashable, Tuple[bytes, ...]] = {}

    # -------- public API --------

    def reset(self) -> "WLMapper":
        self._rG = None
        self._pG = None
        self._r_cache = None
        self._p_cache = None
        self._mapped_rsmi = None
        self._mapping = {}
        self._best = None
        self._solutions = []
        self._bond_change_report = []
        self._r_init = {}
        self._p_init = {}
        self._r_wl = {}
        self._p_wl = {}
        self._r_class = {}
        self._p_class = {}
        return self

    def fit(self, rsmi: str) -> "WLMapper":
        self.reset()
        rG, pG = self._parse_rsmi(rsmi)
        self._rG, self._pG = rG, pG

        self._r_cache = GraphCache(rG, self.cfg)
        self._p_cache = GraphCache(pG, self.cfg)

        self._prime_baseline()

        best, pmcd_min_pool, full_pool = self._solve_pmcd_then_heuristic()
        self._best = best
        self._mapping = dict(best.mapping)

        # expose solutions: PMCD-min pool sorted by heuristic
        self._solutions = self._materialize_solutions(pmcd_min_pool)
        self._mapped_rsmi = (
            self._solutions[0].mapped_rsmi
            if self._solutions
            else self._materialize_one(best)
        )

        # report for chosen best
        self._bond_change_report = self._bond_change_report_for_mapping(best.mapping)
        return self

    @property
    def mapped_rsmi(self) -> str:
        if self._mapped_rsmi is None:
            raise RuntimeError("WLMapper not fitted")
        return self._mapped_rsmi

    @property
    def aam(self) -> Dict[Hashable, Hashable]:
        return dict(self._mapping)

    @property
    def best_score(self) -> float:
        # numeric proxy of PMCD (still minimized by tuple key)
        if self._best is None:
            raise RuntimeError("WLMapper not fitted")
        return float(self._best.score)

    @property
    def best_pmcd_key(self) -> Tuple[int, int, int]:
        if self._best is None:
            raise RuntimeError("WLMapper not fitted")
        return tuple(self._best.meta.get("pmcd_key", (10**9, 10**9, 10**9)))  # type: ignore[return-value]

    @property
    def best_heuristic_cost(self) -> float:
        if self._best is None:
            raise RuntimeError("WLMapper not fitted")
        return float(self._best.meta.get("heuristic_cost", 1e9))

    @property
    def best_meta(self) -> Dict[str, Any]:
        return {} if self._best is None else dict(self._best.meta)

    @property
    def solutions(self) -> List[Solution]:
        return list(self._solutions)

    @property
    def bond_change_report(self) -> List[Dict[str, Any]]:
        return list(self._bond_change_report)

    # -------- parsing + baseline --------

    def _parse_rsmi(self, rsmi: str) -> Tuple[Any, Any]:
        rG, pG = rsmi_to_graph(
            rsmi,
            drop_non_aam=self.cfg.drop_non_aam,
            use_index_as_atom_map=self.cfg.use_index_as_atom_map,
        )
        if rG is None or pG is None:
            raise ValueError("rsmi_to_graph returned None")
        return rG, pG

    def _prime_baseline(self) -> None:
        if self._r_cache is None or self._p_cache is None:
            raise RuntimeError("cache missing")
        self._r_init = self._make_init_bytes(self._r_cache)
        self._p_init = self._make_init_bytes(self._p_cache)

        rv = MaskView(self._r_cache, None)
        pv = MaskView(self._p_cache, None)

        self._r_wl = self._hasher.full_hashes(rv, self._r_init, self._blake_bytes)
        self._p_wl = self._hasher.full_hashes(pv, self._p_init, self._blake_bytes)

        if self.cfg.enable_symmetry_pruning:
            self._r_class = self._sym.node_classes(self._r_wl)
            self._p_class = self._sym.node_classes(self._p_wl)

    def _make_init_bytes(self, cache: GraphCache) -> Dict[Hashable, bytes]:
        keys = WLMapperConfig._normalize_node_keys(self.cfg.node_label_keys)
        out: Dict[Hashable, bytes] = {}
        for u in cache.nodes:
            d = cache.node_data(u)
            s = "|".join(f"{k}={d.get(k, None)}" for k in keys)
            out[u] = s.encode("utf-8", errors="ignore")
        return out

    # -------- PMCD-first solve --------

    def _solve_pmcd_then_heuristic(
        self,
    ) -> Tuple[MappingResult, List[MappingResult], List[MappingResult]]:
        """
        Returns:
          best: chosen by (PMCD key) then heuristic cost
          pmcd_min_pool: all PMCD-min solutions (trimmed & ordered by heuristic)
          full_pool: all evaluated solutions (debug)
        """
        t0 = time.time()
        full_pool: List[MappingResult] = []

        # baseline
        base_map = self._map_two_pass(None)
        base_res = self._score_pmcd_and_heuristic(base_map, meta={"branch": "baseline"})
        full_pool.append(base_res)

        best_pmcd_key = base_res.meta["pmcd_key"]
        pmcd_min: List[MappingResult] = [base_res]

        explored = 0
        for cand in self._candidate_stream():
            if explored >= int(self.cfg.max_candidates):
                break
            if self._time_exceeded(t0):
                break
            explored += 1

            cand_map = self._map_two_pass(cand)
            res = self._score_pmcd_and_heuristic(cand_map, meta=dict(cand.meta))
            full_pool.append(res)

            k = res.meta["pmcd_key"]
            if k < best_pmcd_key:
                best_pmcd_key = k
                pmcd_min = [res]
            elif k == best_pmcd_key:
                pmcd_min.append(res)

        # tie-break among PMCD-min by heuristic cost (and then by mapped_pairs desc)
        for r in pmcd_min:
            r.meta.setdefault("explored_candidates", explored)
            r.meta.setdefault("pmcd_min_set_size", len(pmcd_min))
        pmcd_min.sort(
            key=lambda r: (
                float(r.meta.get("heuristic_cost", 1e9)),
                -int(r.meta.get("mapped_pairs", 0)),
                float(r.score),
            )
        )

        # keep only up to max_solutions
        pmcd_min = pmcd_min[: max(1, int(self.cfg.max_solutions))]

        best = pmcd_min[0]
        self._final_meta_its(best)
        return best, pmcd_min, full_pool

    def _time_exceeded(self, t0: float) -> bool:
        lim = self.cfg.time_limit_s
        if lim is None or lim == 0:
            return False
        return (time.time() - t0) >= float(lim)

    # -------- candidate stream (exploration only; PMCD ignores candidate priors) --------

    def _candidate_stream(self) -> Iterator[EdgeMaskCandidate]:
        if not self.cfg.enable_bond_cut or self.cfg.max_bond_cut_size <= 0:
            return iter(())  # type: ignore[return-value]
        return self._bond_cut_candidates()

    def _bond_cut_candidates(self) -> Iterator[EdgeMaskCandidate]:
        if (
            self._r_cache is None
            or self._p_cache is None
            or self._rG is None
            or self._pG is None
        ):
            return iter(())  # type: ignore[return-value]

        # edges incident to "likely RC" from baseline *graph-only* diff (cheap, no heuristic scoring)
        # here we just use a quick baseline mapping (already computed WL in prime), but we do not rely on heuristic.
        base_map = self._map_wl(None, None, None)
        rc_r, rc_p = self._diff_rc_nodes(base_map)

        edges_r = self._incident_edges_unique(self._r_cache, rc_r)
        edges_p = self._incident_edges_unique(self._p_cache, rc_p)

        # enrich with global priority edges so we can reach correct acyl/peroxy patterns
        edges_r = self._merge_unique(
            edges_r, self._global_priority_edges(self._r_cache, limit=48)
        )
        edges_p = self._merge_unique(
            edges_p, self._global_priority_edges(self._p_cache, limit=48)
        )

        edges_r = self._prioritize_edges(self._r_cache, edges_r)
        edges_p = self._prioritize_edges(self._p_cache, edges_p)

        if self.cfg.enable_symmetry_pruning and self._r_class and self._p_class:
            edges_r = self._sym.unique_edges(
                edges_r, self._r_class, lambda u, v: self._r_cache.edge_orders(u, v)
            )
            edges_p = self._sym.unique_edges(
                edges_p, self._p_class, lambda u, v: self._p_cache.edge_orders(u, v)
            )

        pool_r = edges_r[: int(self.cfg.candidate_edge_pool)]
        pool_p = edges_p[: int(self.cfg.candidate_edge_pool)]

        # Stage A: up to max_bond_cut_size
        yield from self._emit_cut_combos(
            "reactant", pool_r, max_k=int(self.cfg.max_bond_cut_size), tag="bond_cut"
        )
        yield from self._emit_cut_combos(
            "product", pool_p, max_k=int(self.cfg.max_bond_cut_size), tag="bond_cut"
        )

        # Stage B: optional deeper masks (still PMCD-first; used only to discover additional PMCD-min solutions)
        if (
            self.cfg.enable_heuristic_scoring
            and self.cfg.heuristic_max_cut_size > self.cfg.max_bond_cut_size
        ):
            hk = int(self.cfg.heuristic_max_cut_size)
            budget = int(self.cfg.heuristic_candidate_budget)
            yield from self._emit_cut_combos(
                "reactant", pool_r, max_k=hk, tag="deep_cut", combo_cap=budget
            )
            yield from self._emit_cut_combos(
                "product", pool_p, max_k=hk, tag="deep_cut", combo_cap=budget
            )

    @staticmethod
    def _merge_unique(
        a: List[Tuple[Hashable, Hashable]], b: List[Tuple[Hashable, Hashable]]
    ) -> List[Tuple[Hashable, Hashable]]:
        seen = set(_norm_pair(x, y) for x, y in a)
        out = list(a)
        for x, y in b:
            p = _norm_pair(x, y)
            if p in seen:
                continue
            seen.add(p)
            out.append(p)
        return out

    def _emit_cut_combos(
        self,
        side: str,
        pool: List[Tuple[Hashable, Hashable]],
        *,
        max_k: int,
        tag: str,
        combo_cap: int = 200,
    ) -> Iterator[EdgeMaskCandidate]:
        max_k = max(1, int(max_k))
        combo_cap = max(1, int(combo_cap))
        for k in range(1, max_k + 1):
            for subset in itertools.islice(
                itertools.combinations(pool, k), 0, combo_cap
            ):
                removed = frozenset(_norm_pair(a, b) for a, b in subset)
                meta = {
                    "candidate": tag,
                    "branch": tag,
                    "side": side,
                    "k": k,
                    "cut_edges": list(subset),
                }
                yield EdgeMaskCandidate(
                    side=side, removed_pairs=removed, prior_score=0.0, meta=meta
                )

    # -------- edge ranking helpers (search ordering only) --------

    def _xo_nodes(
        self, cache: GraphCache, u: Hashable, v: Hashable
    ) -> Tuple[Optional[Hashable], Optional[Hashable]]:
        eu = cache.element(u)
        ev = cache.element(v)
        if eu == "O" and ev in {"C", "Si"}:
            return v, u
        if ev == "O" and eu in {"C", "Si"}:
            return u, v
        return None, None

    def _is_peroxy_oo(self, cache: GraphCache, u: Hashable, v: Hashable) -> bool:
        if cache.element(u) != "O" or cache.element(v) != "O":
            return False
        return cache.is_acyl_oxygen_peroxy(u) or cache.is_acyl_oxygen_peroxy(v)

    def _edge_rank(self, cache: GraphCache, u: Hashable, v: Hashable) -> int:
        # peroxy O-O: very important to include early
        if cache.element(u) == "O" and cache.element(v) == "O":
            if self._is_peroxy_oo(cache, u, v):
                return -5
            return 95

        x, o = self._xo_nodes(cache, u, v)
        if x is None or o is None:
            return 90

        if cache.element(x) == "C" and cache.is_carbonyl_c(x):
            if cache.edge_is_single(x, o):
                if cache.is_acyl_oxygen_peroxy(o):
                    return 25  # keep available, but not before peroxy O-O
                return 0
            return 999  # C=O

        if cache.element(x) == "C" and cache.aromatic(x):
            return 80

        deg = cache.degree(x)
        if deg >= 3:
            return 10
        if deg == 2:
            return 20
        return 30

    def _global_priority_edges(
        self, cache: GraphCache, limit: int = 48
    ) -> List[Tuple[Hashable, Hashable]]:
        out: List[Tuple[Hashable, Hashable]] = []
        for u, v in cache.G.edges():
            eu = cache.element(u)
            ev = cache.element(v)
            keep = False
            if (eu == "O" and ev in {"C", "Si"}) or (ev == "O" and eu in {"C", "Si"}):
                keep = True
            elif eu == "O" and ev == "O" and self._is_peroxy_oo(cache, u, v):
                keep = True
            if not keep:
                continue
            out.append(_norm_pair(u, v))

        out = list(dict.fromkeys(out))
        out.sort(
            key=lambda e: (self._edge_rank(cache, e[0], e[1]), _stable_sort_key(e))
        )
        return out[: max(0, int(limit))]

    def _prioritize_edges(
        self, cache: GraphCache, edges: List[Tuple[Hashable, Hashable]]
    ) -> List[Tuple[Hashable, Hashable]]:
        def bond_strength_sum(orders: Tuple[Any, ...]) -> float:
            s = 0.0
            for x in orders:
                if isinstance(x, (int, float)):
                    s += float(x)
                elif isinstance(x, str):
                    try:
                        s += float(x)
                    except Exception:
                        s += 0.0
            return s

        def key(e: Tuple[Hashable, Hashable]) -> Tuple[int, float, str]:
            u, v = e
            rank = self._edge_rank(cache, u, v)
            s = bond_strength_sum(cache.edge_orders(u, v))
            return (rank, -s, _stable_sort_key((u, v)))

        return sorted(edges, key=key)

    def _incident_edges_unique(
        self, cache: GraphCache, nodes: List[Hashable]
    ) -> List[Tuple[Hashable, Hashable]]:
        seen: set = set()
        out: List[Tuple[Hashable, Hashable]] = []
        for u in nodes:
            if u not in cache.node_set:
                continue
            for v in cache.neighbors(u):
                a, b = _norm_pair(u, v)
                if (a, b) in seen:
                    continue
                seen.add((a, b))
                out.append((a, b))
        return out

    # -------- mapping core --------

    def _map_two_pass(
        self, cand: Optional[EdgeMaskCandidate]
    ) -> Dict[Hashable, Hashable]:
        base = self._map_wl(cand, None, None)
        if not base:
            return {}

        chosen = base
        if self.cfg.enable_rc_refine:
            rc_r, rc_p = self._diff_rc_nodes(base)
            dist_r = self._dist_to_set(self._rG, rc_r)
            dist_p = self._dist_to_set(self._pG, rc_p)
            ref = self._map_wl(cand, dist_r, dist_p)
            if ref and self._pmcd_key(ref) < self._pmcd_key(base):
                chosen = ref

        if self.cfg.enable_swap_refine:
            chosen = self._swap_refine_mapping(chosen)

        return chosen

    def _candidate_masks(
        self, cand: Optional[EdgeMaskCandidate]
    ) -> Tuple[Optional[frozenset], Optional[frozenset]]:
        if cand is None:
            return None, None
        if cand.side == "reactant":
            return cand.removed_pairs, None
        return None, cand.removed_pairs

    def _masked_or_base_wl(
        self,
        view: MaskView,
        removed: Optional[frozenset],
        base_wl: Dict[Hashable, List[bytes]],
        init_labels: Dict[Hashable, bytes],
    ) -> Dict[Hashable, List[bytes]]:
        if removed is None:
            return base_wl
        if not self.cfg.enable_dynamic_wl:
            return self._hasher.full_hashes(view, init_labels, self._blake_bytes)
        hmap, _ = self._hasher.masked_hashes(
            view, init_labels, base_wl, removed, self._blake_bytes
        )
        return hmap

    def _depth_schedule(self, max_depth: int) -> List[int]:
        if self.cfg.progressive_fallback:
            return list(range(max_depth, 0, -1))
        return [max_depth]

    def _wl_key(
        self, hmap: Dict[Hashable, List[bytes]], u: Hashable, depth: int
    ) -> Optional[bytes]:
        seq = hmap.get(u)
        if not seq:
            return None
        if depth <= 0 or depth > len(seq):
            return None
        return seq[depth - 1]

    def _build_buckets(
        self, nodes: Iterable[Hashable], hmap: Dict[Hashable, List[bytes]], depth: int
    ) -> Dict[bytes, List[Hashable]]:
        buckets: Dict[bytes, List[Hashable]] = {}
        for u in nodes:
            k = self._wl_key(hmap, u, depth)
            if k is None:
                continue
            buckets.setdefault(k, []).append(u)
        return buckets

    def _map_wl(
        self,
        cand: Optional[EdgeMaskCandidate],
        rc_dist_r: Optional[Dict[Hashable, int]],
        rc_dist_p: Optional[Dict[Hashable, int]],
    ) -> Dict[Hashable, Hashable]:
        if (
            self._r_cache is None
            or self._p_cache is None
            or self._rG is None
            or self._pG is None
        ):
            return {}

        rm, pm = self._candidate_masks(cand)
        rv = MaskView(self._r_cache, rm)
        pv = MaskView(self._p_cache, pm)

        r_hash = self._masked_or_base_wl(rv, rm, self._r_wl, self._r_init)
        p_hash = self._masked_or_base_wl(pv, pm, self._p_wl, self._p_init)
        if not r_hash or not p_hash:
            return {}

        r_un = set(self._r_cache.nodes)
        p_un = set(self._p_cache.nodes)
        mapping: Dict[Hashable, Hashable] = {}

        max_depth = len(next(iter(r_hash.values())))
        for depth in self._depth_schedule(max_depth):
            r_b = self._build_buckets(r_un, r_hash, depth)
            p_b = self._build_buckets(p_un, p_hash, depth)
            for key, ru in r_b.items():
                pv_nodes = p_b.get(key)
                if not pv_nodes:
                    continue
                self._match_bucket(
                    rv, pv, ru, pv_nodes, mapping, r_un, p_un, rc_dist_r, rc_dist_p
                )

        return mapping

    def _use_hungarian(self, n: int, m: int) -> bool:
        lim = int(self.cfg.hungarian_max_size)
        return n <= lim and m <= lim

    def _match_bucket(
        self,
        rv: MaskView,
        pv: MaskView,
        ru: List[Hashable],
        pv_nodes: List[Hashable],
        mapping: Dict[Hashable, Hashable],
        r_un: Set[Hashable],
        p_un: Set[Hashable],
        rc_dist_r: Optional[Dict[Hashable, int]],
        rc_dist_p: Optional[Dict[Hashable, int]],
    ) -> None:
        if len(ru) == 1 and len(pv_nodes) == 1:
            u = ru[0]
            v = pv_nodes[0]
            if self._compatible(rv.cache, u, pv.cache, v):
                mapping[u] = v
                r_un.discard(u)
                p_un.discard(v)
            return

        ru_s = sorted(ru, key=_stable_sort_key)
        pv_s = sorted(pv_nodes, key=_stable_sort_key)

        if self._use_hungarian(len(ru_s), len(pv_s)):
            cost = self._cost_matrix(rv, pv, ru_s, pv_s, mapping, rc_dist_r, rc_dist_p)
            pairs = self._hungarian_min_cost_assignment(cost)
            self._accept_pairs_matrix(ru_s, pv_s, cost, pairs, r_un, p_un, mapping)
            return

        pairs = self._greedy_pairs_topk(
            rv, pv, ru_s, pv_s, mapping, rc_dist_r, rc_dist_p
        )
        for i, j in pairs:
            u = ru_s[i]
            v = pv_s[j]
            mapping[u] = v
            r_un.discard(u)
            p_un.discard(v)

    def _mapped_neighbors_per_u(
        self, rv: MaskView, ru_s: List[Hashable], mapping: Dict[Hashable, Hashable]
    ) -> Dict[Hashable, List[Hashable]]:
        out: Dict[Hashable, List[Hashable]] = {}
        for u in ru_s:
            xs = [x for x in rv.neighbors(u) if x in mapping]
            if xs:
                out[u] = xs
        return out

    def _rc_distance_penalty(
        self,
        u: Hashable,
        v: Hashable,
        rc_dist_r: Optional[Dict[Hashable, int]],
        rc_dist_p: Optional[Dict[Hashable, int]],
    ) -> float:
        if rc_dist_r is None or rc_dist_p is None:
            return 0.0
        du = float(rc_dist_r.get(u, 10_000))
        dv = float(rc_dist_p.get(v, 10_000))
        return float(self.cfg.rc_distance_weight) * abs(du - dv)

    @staticmethod
    def _compatible(
        r_cache: GraphCache, u: Hashable, p_cache: GraphCache, v: Hashable
    ) -> bool:
        return r_cache.element(u) == p_cache.element(v)

    def _local_distance(
        self, rv: MaskView, pv: MaskView, u: Hashable, v: Hashable
    ) -> float:
        rC = rv.cache
        pC = pv.cache
        if rC.element(u) != pC.element(v):
            return math.inf

        base = 0.0
        base += 2.0 if rC.aromatic(u) != pC.aromatic(v) else 0.0
        base += abs(rC.hcount(u) - pC.hcount(v)) * 1.0
        base += abs(rC.charge(u) - pC.charge(v)) * 2.0
        base += abs(rv.degree(u) - pv.degree(v)) * 0.5
        base += float(_multiset_l1_sorted(rv.neigh_codes(u), pv.neigh_codes(v)))
        return base

    def _neighbor_consistency_penalty(
        self,
        rv: MaskView,
        pv: MaskView,
        u: Hashable,
        v: Hashable,
        mapping: Dict[Hashable, Hashable],
        mapped_neigh_u: Optional[List[Hashable]],
        neigh_set_v: Set[Hashable],
    ) -> float:
        if not mapped_neigh_u:
            return 0.0
        miss = 0.0
        for x in mapped_neigh_u:
            y = mapping.get(x)
            if y is None:
                continue
            if y not in neigh_set_v:
                miss += 1.0
                continue
            if rv.edge_label_one(u, x) != pv.edge_label_one(v, y):
                miss += 0.5
        return miss

    def _pair_cost(
        self,
        rv: MaskView,
        pv: MaskView,
        u: Hashable,
        v: Hashable,
        mapping: Dict[Hashable, Hashable],
        mapped_neigh_u: Optional[List[Hashable]],
        neigh_set_v: Set[Hashable],
        rc_dist_r: Optional[Dict[Hashable, int]],
        rc_dist_p: Optional[Dict[Hashable, int]],
    ) -> float:
        if not self._compatible(rv.cache, u, pv.cache, v):
            return math.inf
        base = self._local_distance(rv, pv, u, v)
        base += self._neighbor_consistency_penalty(
            rv, pv, u, v, mapping, mapped_neigh_u, neigh_set_v
        )
        base += self._rc_distance_penalty(u, v, rc_dist_r, rc_dist_p)
        return base

    def _cost_matrix(
        self,
        rv: MaskView,
        pv: MaskView,
        ru_s: List[Hashable],
        pv_s: List[Hashable],
        mapping: Dict[Hashable, Hashable],
        rc_dist_r: Optional[Dict[Hashable, int]],
        rc_dist_p: Optional[Dict[Hashable, int]],
    ) -> List[List[float]]:
        mapped_neigh = self._mapped_neighbors_per_u(rv, ru_s, mapping)
        pv_neigh = {v: pv.neighbors_set(v) for v in pv_s}
        out: List[List[float]] = []
        for u in ru_s:
            row: List[float] = []
            mu = mapped_neigh.get(u)
            for v in pv_s:
                row.append(
                    self._pair_cost(
                        rv, pv, u, v, mapping, mu, pv_neigh[v], rc_dist_r, rc_dist_p
                    )
                )
            out.append(row)
        return out

    def _greedy_pairs_topk(
        self,
        rv: MaskView,
        pv: MaskView,
        ru_s: List[Hashable],
        pv_s: List[Hashable],
        mapping: Dict[Hashable, Hashable],
        rc_dist_r: Optional[Dict[Hashable, int]],
        rc_dist_p: Optional[Dict[Hashable, int]],
    ) -> List[Tuple[int, int]]:
        mapped_neigh = self._mapped_neighbors_per_u(rv, ru_s, mapping)
        pv_neigh = {v: pv.neighbors_set(v) for v in pv_s}

        topk = int(self.cfg.greedy_topk_per_u)
        items: List[Tuple[float, int, int]] = []

        for i, u in enumerate(ru_s):
            mu = mapped_neigh.get(u)
            best: List[Tuple[float, int]] = []
            for j, v in enumerate(pv_s):
                c = self._pair_cost(
                    rv, pv, u, v, mapping, mu, pv_neigh[v], rc_dist_r, rc_dist_p
                )
                if math.isinf(c) or c > 1e8:
                    continue
                best.append((float(c), j))
            if not best:
                continue
            best.sort(key=lambda x: x[0])
            for c, j in best[:topk]:
                items.append((c, i, j))

        items.sort(key=lambda x: x[0])

        used_i: set = set()
        used_j: set = set()
        out: List[Tuple[int, int]] = []
        for _, i, j in items:
            if i in used_i or j in used_j:
                continue
            used_i.add(i)
            used_j.add(j)
            out.append((i, j))
        return out

    @staticmethod
    def _hungarian_min_cost_assignment(
        cost: List[List[float]],
    ) -> List[Tuple[int, int]]:
        n = len(cost)
        m = len(cost[0]) if n else 0
        if n == 0 or m == 0:
            return []
        N = max(n, m)
        BIG = 1e9

        a = [[BIG] * N for _ in range(N)]
        for i in range(n):
            for j in range(m):
                c = cost[i][j]
                a[i][j] = BIG if math.isinf(c) else float(c)

        u = [0.0] * (N + 1)
        v = [0.0] * (N + 1)
        p = [0] * (N + 1)
        way = [0] * (N + 1)

        for i in range(1, N + 1):
            p[0] = i
            j0 = 0
            minv = [BIG] * (N + 1)
            used = [False] * (N + 1)

            while True:
                used[j0] = True
                i0 = p[j0]
                delta = BIG
                j1 = 0
                for j in range(1, N + 1):
                    if used[j]:
                        continue
                    cur = a[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j

                for j in range(0, N + 1):
                    if used[j]:
                        u[p[j]] += delta
                        v[j] -= delta
                    else:
                        minv[j] -= delta

                j0 = j1
                if p[j0] == 0:
                    break

            while True:
                j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1
                if j0 == 0:
                    break

        ans: List[Tuple[int, int]] = []
        for j in range(1, N + 1):
            i = p[j]
            if 1 <= i <= n and 1 <= j <= m:
                ans.append((i - 1, j - 1))
        return ans

    @staticmethod
    def _accept_pairs_matrix(
        ru_s: List[Hashable],
        pv_s: List[Hashable],
        cost: List[List[float]],
        pairs: List[Tuple[int, int]],
        r_un: Set[Hashable],
        p_un: Set[Hashable],
        mapping: Dict[Hashable, Hashable],
    ) -> None:
        for i, j in pairs:
            c = cost[i][j]
            if math.isinf(c) or c > 1e8:
                continue
            u = ru_s[i]
            v = pv_s[j]
            mapping[u] = v
            r_un.discard(u)
            p_un.discard(v)

    # -------- swap refine (unchanged) --------

    def _swap_refine_mapping(
        self, mapping: Dict[Hashable, Hashable]
    ) -> Dict[Hashable, Hashable]:
        if not mapping:
            return mapping
        groups = self._swap_groups(mapping)
        if not groups:
            return mapping

        cur = dict(mapping)
        best = self._heuristic_bond_cost(cur)  # heuristic tie-break, not PMCD

        for _ in range(int(self.cfg.swap_refine_max_iter)):
            improved = False
            for nodes in groups:
                cur2, best2, changed = self._swap_refine_group(cur, best, nodes)
                if changed:
                    cur, best = cur2, best2
                    improved = True
            if not improved:
                break
        return cur

    def _swap_groups(self, mapping: Dict[Hashable, Hashable]) -> List[List[Hashable]]:
        depth = int(self.cfg.swap_refine_class_depth)
        max_sz = int(self.cfg.swap_refine_max_group_size)
        buckets: Dict[Tuple[Tuple[bytes, ...], Tuple[bytes, ...]], List[Hashable]] = {}

        for u, v in mapping.items():
            rk = self._wl_prefix(self._r_wl.get(u, []), depth)
            pk = self._wl_prefix(self._p_wl.get(v, []), depth)
            if rk is None or pk is None:
                continue
            buckets.setdefault((rk, pk), []).append(u)

        out: List[List[Hashable]] = []
        for nodes in buckets.values():
            if 2 <= len(nodes) <= max_sz:
                out.append(sorted(nodes, key=_stable_sort_key))
        return out

    @staticmethod
    def _wl_prefix(seq: List[bytes], depth: int) -> Optional[Tuple[bytes, ...]]:
        if not seq:
            return None
        d = min(int(depth), len(seq))
        if d <= 0:
            return None
        return tuple(seq[:d])

    def _swap_refine_group(
        self,
        mapping: Dict[Hashable, Hashable],
        best_score: float,
        nodes: List[Hashable],
    ) -> Tuple[Dict[Hashable, Hashable], float, bool]:
        cur = dict(mapping)
        best = float(best_score)
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                u1 = nodes[i]
                u2 = nodes[j]
                if u1 not in cur or u2 not in cur:
                    continue
                v1 = cur[u1]
                v2 = cur[u2]
                if v1 == v2:
                    continue
                trial = dict(cur)
                trial[u1] = v2
                trial[u2] = v1
                sc = self._heuristic_bond_cost(trial)
                if sc < best:
                    return trial, sc, True
        return cur, best, False

    # -------- PMCD key + scoring --------

    def _pmcd_key(self, mapping: Dict[Hashable, Hashable]) -> Tuple[int, int, int]:
        """
        PMCD key: (unmapped_atoms, bond_changes_count, hcount_changes_count)
        This is the ONLY criterion used in stage-1 selection.
        """
        if self._r_cache is None or self._p_cache is None:
            return (10**9, 10**9, 10**9)

        ur, up = self._unmapped_counts(mapping)
        if not mapping:
            return (ur + up, 10**9, 10**9)

        # RC restriction (optional)
        rc_r, rc_p = self._diff_rc_nodes(mapping)
        rc_r = self._expand_nodes(self._r_cache, rc_r, int(self.cfg.rc_expand_hops))
        rc_p = self._expand_nodes(self._p_cache, rc_p, int(self.cfg.rc_expand_hops))

        if self.cfg.rc_only_bond_changes:
            sub = {u: mapping[u] for u in rc_r if u in mapping}
            bond_changes = self._bond_change_count(sub)
        else:
            bond_changes = self._bond_change_count(mapping)

        if self.cfg.rc_only_hcount_changes:
            hcount_changes = self._hcount_change_count(mapping, rc_r)
        else:
            hcount_changes = self._hcount_change_count(mapping, None)

        return (int(ur + up), int(bond_changes), int(hcount_changes))

    def _pmcd_numeric(self, key: Tuple[int, int, int]) -> float:
        # numeric proxy; comparisons MUST use the tuple key, not this number
        # large bases avoid collisions for typical sizes
        u, b, h = key
        return float(u) * 1e6 + float(b) * 1e3 + float(h)

    def _score_pmcd_and_heuristic(
        self, mapping: Dict[Hashable, Hashable], meta: Dict[str, Any]
    ) -> MappingResult:
        pmcd_key = self._pmcd_key(mapping)
        pmcd_score = self._pmcd_numeric(pmcd_key)

        heur_cost = self._heuristic_cost(mapping, meta)

        out = MappingResult(mapping=mapping, score=float(pmcd_score), meta=dict(meta))
        out.meta["pmcd_key"] = tuple(pmcd_key)
        out.meta["pmcd_score"] = float(pmcd_score)
        out.meta["heuristic_cost"] = float(heur_cost)
        out.meta["mapped_pairs"] = int(len(mapping))
        out.meta["unmapped_atoms"] = int(pmcd_key[0])
        out.meta["bond_changes"] = int(pmcd_key[1])
        out.meta["hcount_changes"] = int(pmcd_key[2])
        return out

    # -------- PMCD components --------

    def _bond_change_count(self, mapping: Dict[Hashable, Hashable]) -> int:
        """
        Unweighted chemical distance: count of bond edits (removed/created/order mismatch) = 1 each.
        """
        if (
            self._r_cache is None
            or self._p_cache is None
            or self._rG is None
            or self._pG is None
        ):
            return 10**9
        inv = {pv: ru for ru, pv in mapping.items()}
        mapped_r = set(mapping.keys())
        mapped_p = set(mapping.values())

        seen_p: set = set()
        changes = 0

        # edges in reactants among mapped nodes -> check product
        for u, v in self._iter_edges_between(self._rG, mapped_r):
            pu = mapping.get(u)
            pv = mapping.get(v)
            if pu is None or pv is None:
                continue
            a, b = _norm_pair(pu, pv)
            if (a, b) in seen_p:
                continue
            seen_p.add((a, b))

            ro = self._r_cache.edge_orders(u, v)
            po = self._p_cache.edge_orders(pu, pv)
            if not po:
                changes += 1
            elif ro != po:
                changes += 1

        # edges created in product among mapped nodes -> check reactant
        for u, v in self._iter_edges_between(self._pG, mapped_p):
            ru = inv.get(u)
            rv = inv.get(v)
            if ru is None or rv is None:
                continue
            if not self._r_cache.edge_orders(ru, rv):
                changes += 1

        return int(changes)

    def _hcount_change_count(
        self, mapping: Dict[Hashable, Hashable], nodes: Optional[List[Hashable]]
    ) -> int:
        if not mapping or self._r_cache is None or self._p_cache is None:
            return 0
        it = mapping.keys() if nodes is None else (u for u in nodes if u in mapping)
        total = 0
        for u in it:
            v = mapping.get(u)
            if v is None:
                continue
            total += abs(self._r_cache.hcount(u) - self._p_cache.hcount(v))
        return int(total)

    # -------- heuristic tie-break (ONLY stage 2) --------

    def _heuristic_cost(
        self, mapping: Dict[Hashable, Hashable], meta: Dict[str, Any]
    ) -> float:
        """
        Tie-break ONLY:
          - type-weighted bond change cost (handles peracid: prefers O-O cleavage)
          - + small candidate cut preference penalty/bonus (based on cut_edges)
        """
        if not mapping:
            return 1e9

        bond_cost = self._heuristic_bond_cost(mapping)

        cut_pen = 0.0
        side = meta.get("side")
        cut_edges = meta.get("cut_edges") or []
        if (
            isinstance(cut_edges, list)
            and cut_edges
            and (side in {"reactant", "product"})
        ):
            cache = self._r_cache if side == "reactant" else self._p_cache
            if cache is not None:
                cut_pen = self._heuristic_cut_penalty(cache, cut_edges)

        # prefer fewer cuts slightly (but only after PMCD)
        k = float(meta.get("k", 0.0) or 0.0)
        return float(bond_cost) + float(cut_pen) + 0.02 * k

    def _heuristic_cut_penalty(
        self, cache: GraphCache, cut_edges: List[Tuple[Hashable, Hashable]]
    ) -> float:
        """
        If peroxyacyl present, strongly prefer cutting O-O (outer oxygen bond),
        and strongly avoid cutting acyl C-O in C(=O)-O-O.
        """
        pen = 0.0
        for u, v in cut_edges:
            eu = cache.element(u)
            ev = cache.element(v)

            # bonus: cutting peroxy O-O
            if eu == "O" and ev == "O" and self._is_peroxy_oo(cache, u, v):
                pen -= 1.5
                continue

            x, o = self._xo_nodes(cache, u, v)
            if x is None or o is None:
                continue

            # penalty: cutting acyl C-O when that oxygen is peroxyacyl oxygen
            if (
                cache.element(x) == "C"
                and cache.is_carbonyl_c(x)
                and cache.is_acyl_oxygen_peroxy(o)
            ):
                pen += 2.0
        return float(pen)

    def _heuristic_bond_cost(self, mapping: Dict[Hashable, Hashable]) -> float:
        cost, _ = self._bond_change_cost_and_report(mapping, want_report=False)
        return float(cost)

    def _bond_change_cost_and_report(
        self, mapping: Dict[Hashable, Hashable], want_report: bool
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Type-weighted cost used ONLY in heuristic tie-break.
        """
        if (
            self._r_cache is None
            or self._p_cache is None
            or self._rG is None
            or self._pG is None
        ):
            return 0.0, []
        inv = {pv: ru for ru, pv in mapping.items()}
        mapped_r = set(mapping.keys())
        mapped_p = set(mapping.values())

        seen_p: set = set()
        total_cost = 0.0
        report: List[Dict[str, Any]] = []

        for u, v in self._iter_edges_between(self._rG, mapped_r):
            pu = mapping.get(u)
            pv = mapping.get(v)
            if pu is None or pv is None:
                continue
            a, b = _norm_pair(pu, pv)
            if (a, b) in seen_p:
                continue
            seen_p.add((a, b))

            ro = self._r_cache.edge_orders(u, v)
            po = self._p_cache.edge_orders(pu, pv)

            if not po:
                c = self._bond_type_cost(self._r_cache, u, v)
                total_cost += c
                if want_report:
                    report.append(
                        {
                            "kind": "removed_in_product",
                            "r_edge": (u, v),
                            "p_edge": (pu, pv),
                            "r_order": ro,
                            "p_order": po,
                            "cost": float(c),
                        }
                    )
                continue

            if ro != po:
                cr = self._bond_type_cost(self._r_cache, u, v)
                cp = self._bond_type_cost(self._p_cache, pu, pv)
                c = float(self.cfg.bc_cost_order_mismatch_scale) * 0.5 * (cr + cp)
                total_cost += c
                if want_report:
                    report.append(
                        {
                            "kind": "order_mismatch",
                            "r_edge": (u, v),
                            "p_edge": (pu, pv),
                            "r_order": ro,
                            "p_order": po,
                            "cost": float(c),
                        }
                    )
                continue

        for u, v in self._iter_edges_between(self._pG, mapped_p):
            ru = inv.get(u)
            rv = inv.get(v)
            if ru is None or rv is None:
                continue
            ro = self._r_cache.edge_orders(ru, rv)
            if not ro:
                c = self._bond_type_cost(self._p_cache, u, v)
                total_cost += c
                if want_report:
                    report.append(
                        {
                            "kind": "created_in_product",
                            "r_edge": (ru, rv),
                            "p_edge": (u, v),
                            "r_order": ro,
                            "p_order": self._p_cache.edge_orders(u, v),
                            "cost": float(c),
                        }
                    )

        return float(total_cost), report

    def _bond_type_cost(self, cache: GraphCache, u: Hashable, v: Hashable) -> float:
        # peroxy O-O is cheap to change (preferred cleavage for peracids)
        if cache.element(u) == "O" and cache.element(v) == "O":
            if self._is_peroxy_oo(cache, u, v):
                return float(self.cfg.bc_cost_peroxy_oo)
            return float(self.cfg.bc_cost_other)

        x, o = self._xo_nodes(cache, u, v)
        if x is not None and o is not None:
            if cache.element(x) == "C" and cache.is_carbonyl_c(x):
                if cache.edge_is_double(x, o):
                    return float(self.cfg.heuristic_carbonyl_double_penalty)
                if cache.is_acyl_oxygen_peroxy(o):
                    return float(self.cfg.bc_cost_acyl_co_peroxy)
                return float(self.cfg.bc_cost_acyl_co)

            if cache.element(x) == "C" and cache.aromatic(x):
                return float(self.cfg.bc_cost_aromatic_co)

            deg = cache.degree(x)
            if deg >= 3:
                return float(self.cfg.bc_cost_x_deg3_o)
            if deg == 2:
                return float(self.cfg.bc_cost_x_deg2_o)
            return float(self.cfg.bc_cost_x_deg1_o)

        if (cache.element(u) == "C" and cache.aromatic(u)) or (
            cache.element(v) == "C" and cache.aromatic(v)
        ):
            return float(max(self.cfg.bc_cost_other, self.cfg.bc_cost_aromatic_co))
        return float(self.cfg.bc_cost_other)

    # -------- RC detection + utilities --------

    @staticmethod
    def _iter_edges_between(
        G: Any, nodes: Set[Hashable]
    ) -> Iterable[Tuple[Hashable, Hashable]]:
        if G is None:
            return ()
        for u, v in G.edges(nodes):
            if u in nodes and v in nodes:
                yield u, v

    def _unmapped_counts(self, mapping: Dict[Hashable, Hashable]) -> Tuple[int, int]:
        if self._r_cache is None or self._p_cache is None:
            return 0, 0
        mr = set(mapping.keys())
        mp = set(mapping.values())
        ur = sum(1 for u in self._r_cache.nodes if u not in mr)
        up = sum(1 for v in self._p_cache.nodes if v not in mp)
        return int(ur), int(up)

    def _diff_rc_nodes(
        self, mapping: Dict[Hashable, Hashable]
    ) -> Tuple[List[Hashable], List[Hashable]]:
        if not mapping:
            return [], []
        changed_r = self._changed_endpoints_r(mapping)
        rc_r = sorted(changed_r, key=repr)
        rc_p = [mapping[u] for u in rc_r if u in mapping]
        return rc_r, rc_p

    def _changed_endpoints_r(self, mapping: Dict[Hashable, Hashable]) -> Set[Hashable]:
        if (
            self._r_cache is None
            or self._p_cache is None
            or self._rG is None
            or self._pG is None
        ):
            return set()

        inv = {pv: ru for ru, pv in mapping.items()}
        mapped_r = set(mapping.keys())
        mapped_p = set(mapping.values())

        out: Set[Hashable] = set()

        for u, v in self._iter_edges_between(self._rG, mapped_r):
            pu = mapping.get(u)
            pv = mapping.get(v)
            if pu is None or pv is None:
                continue
            po = self._p_cache.edge_orders(pu, pv)
            ro = self._r_cache.edge_orders(u, v)
            if not po or ro != po:
                out.add(u)
                out.add(v)

        for u, v in self._iter_edges_between(self._pG, mapped_p):
            ru = inv.get(u)
            rv = inv.get(v)
            if ru is None or rv is None:
                continue
            if not self._r_cache.edge_orders(ru, rv):
                out.add(ru)
                out.add(rv)

        return out

    def _expand_nodes(
        self, cache: GraphCache, nodes: List[Hashable], hops: int
    ) -> List[Hashable]:
        if hops <= 0 or not nodes:
            return list(dict.fromkeys(nodes))
        seen = set(nodes)
        frontier = set(nodes)
        for _ in range(hops):
            nxt: Set[Hashable] = set()
            for u in frontier:
                for v in cache.neighbors(u):
                    if v not in seen:
                        seen.add(v)
                        nxt.add(v)
            frontier = nxt
            if not frontier:
                break
        return sorted(seen, key=repr)

    def _dist_to_set(self, G: Any, sources: List[Hashable]) -> Dict[Hashable, int]:
        if not sources or G is None:
            return {}
        try:
            import networkx as nx
        except Exception:
            return {}
        UG = G.to_undirected() if hasattr(G, "to_undirected") else G
        try:
            return dict(nx.multi_source_shortest_path_length(UG, sources))
        except Exception:
            dist: Dict[Hashable, int] = {}
            for s in sources:
                if s not in UG:
                    continue
                d = nx.single_source_shortest_path_length(UG, s)
                for n, ln in d.items():
                    prev = dist.get(n)
                    dist[n] = ln if prev is None else min(prev, ln)
            return dist

    # -------- ITS meta (optional) --------

    def _final_meta_its(self, best: MappingResult) -> None:
        if not self.cfg.use_its_final:
            return
        if ITSConstruction is None or get_rc is None:
            return
        if not best.mapping or self._rG is None or self._pG is None:
            return
        try:
            rr, pp = self._rG.copy(), self._pG.copy()
            self._write_temp_atom_maps(rr, pp, best.mapping)
            its = ITSConstruction().fit(rr, pp).its  # type: ignore[attr-defined]
            rc = get_rc(its)  # type: ignore[misc]
            best.meta["its_rc_nodes"] = int(len(list(rc.nodes())))
            best.meta["its_rc_edges"] = int(len(list(rc.edges())))
        except Exception:
            best.meta["its_rc_nodes"] = None
            best.meta["its_rc_edges"] = None

    # -------- materialization --------

    def _materialize_solutions(self, pool: List[MappingResult]) -> List[Solution]:
        out: List[Solution] = []
        seen: set = set()
        for res in pool:
            rsmi = self._materialize_one(res)
            if rsmi in seen:
                continue
            seen.add(rsmi)
            out.append(Solution(result=res, mapped_rsmi=rsmi))
        return out

    def _materialize_one(self, res: MappingResult) -> str:
        if self._rG is None or self._pG is None:
            raise RuntimeError("missing graphs")
        rr, pp = self._rG.copy(), self._pG.copy()
        self._apply_atom_map_numbers(rr, pp, res.mapping)
        mapped = self._graphs_to_rsmi(rr, pp)
        return self._canon.canonicalise(mapped)

    def _graphs_to_rsmi(self, rG: Any, pG: Any) -> str:
        r_smi = self._graph_to_smi_strict(rG)
        p_smi = self._graph_to_smi_strict(pG)
        return f"{r_smi}>>{p_smi}"

    def _graph_to_smi_strict(self, G: Any) -> str:
        for kwargs in (
            {"canonical": False, "use_atom_map": True},
            {"canonical": False, "atom_map_key": "atom_map"},
            {"use_atom_map": True},
            {"atom_map_key": "atom_map"},
            {"canonical": False},
            {},
        ):
            try:
                s = graph_to_smi(G, **kwargs)
            except Exception:
                s = None
            if isinstance(s, str) and s.strip():
                return s
        raise ValueError("graph_to_smi failed")

    def _apply_atom_map_numbers(
        self, rG: Any, pG: Any, mapping: Dict[Hashable, Hashable]
    ) -> None:
        for u in rG.nodes:
            rG.nodes[u]["atom_map"] = int(self.cfg.unmapped_value)
        for v in pG.nodes:
            pG.nodes[v]["atom_map"] = int(self.cfg.unmapped_value)

        k = int(self.cfg.start_atom_map)
        used_r: Set[Hashable] = set()
        used_p: Set[Hashable] = set()

        for u in sorted(mapping.keys(), key=_stable_sort_key):
            v = mapping[u]
            rG.nodes[u]["atom_map"] = k
            pG.nodes[v]["atom_map"] = k
            used_r.add(u)
            used_p.add(v)
            k += 1

        if not self.cfg.assign_maps_to_unmapped:
            return

        for u in sorted((x for x in rG.nodes if x not in used_r), key=_stable_sort_key):
            rG.nodes[u]["atom_map"] = k
            k += 1
        for v in sorted((x for x in pG.nodes if x not in used_p), key=_stable_sort_key):
            pG.nodes[v]["atom_map"] = k
            k += 1

    @staticmethod
    def _write_temp_atom_maps(
        rr: Any, pp: Any, mapping: Dict[Hashable, Hashable]
    ) -> None:
        for u in rr.nodes:
            rr.nodes[u]["atom_map"] = 0
        for v in pp.nodes:
            pp.nodes[v]["atom_map"] = 0
        k = 1
        for u in sorted(mapping.keys(), key=repr):
            v = mapping[u]
            rr.nodes[u]["atom_map"] = k
            pp.nodes[v]["atom_map"] = k
            k += 1

    def _bond_change_report_for_mapping(
        self, mapping: Dict[Hashable, Hashable]
    ) -> List[Dict[str, Any]]:
        _, rep = self._bond_change_cost_and_report(mapping, want_report=True)
        return rep

    def _blake_bytes(self, b: bytes) -> bytes:
        h = hashlib.blake2b(digest_size=int(self.cfg.digest_size))
        h.update(b)
        return h.digest()
