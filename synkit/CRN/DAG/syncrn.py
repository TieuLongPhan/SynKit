from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple
import logging
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations

import networkx as nx

from synkit.Synthesis.Reactor.syn_reactor import SynReactor

logger = logging.getLogger(__name__)

try:
    from rdkit import Chem
except ImportError:
    Chem = None


# --------------------------------------------------------------------------- #
# Parallel worker (rule application on a chosen substrate mixture)
# --------------------------------------------------------------------------- #


def _apply_rule_worker(
    args: Tuple[int, Any, str, bool, bool, Optional[str], Tuple[str, ...]],
) -> Tuple[int, Tuple[str, ...], List[str]]:
    idx, rule, substrate, explicit_h, implicit_temp, strategy, reactant_keys = args

    kwargs = dict(
        smiles=substrate,
        template=rule,
        invert=False,
        explicit_h=explicit_h,
        implicit_temp=implicit_temp,
    )
    if strategy is not None:
        kwargs["strategy"] = strategy

    reactor = SynReactor.from_smiles(**kwargs)
    return idx, reactant_keys, list(reactor.smiles_list)


# --------------------------------------------------------------------------- #
# Small helpers to keep methods below C901
# --------------------------------------------------------------------------- #


def _count_lhs_components(text: str) -> Optional[int]:
    if not text:
        return None
    lhs = text.split(">>", 1)[0].strip() if ">>" in text else text.strip()
    parts = [p for p in lhs.split(".") if p.strip()]
    return len(parts) if parts else None


def _canonicalize_nomap_rdkit(smiles: str) -> Optional[str]:
    if Chem is None:
        return smiles if smiles else None
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    for a in mol.GetAtoms():
        a.SetAtomMapNum(0)
    mol = Chem.RemoveAllHs(mol)
    return Chem.MolToSmiles(mol, canonical=True)


def _mol_from_smiles_safe(smiles: str) -> Optional[Any]:
    """Return RDKit Mol or None (no exceptions)."""
    if Chem is None or not smiles:
        return None
    try:
        return Chem.MolFromSmiles(smiles, sanitize=False)
    except Exception:
        return None


def _sanitize_safe(mol: Any) -> bool:
    """Sanitize mol in-place; return True on success."""
    try:
        Chem.SanitizeMol(mol)
        return True
    except Exception:
        return False


def _has_atom_maps(mol: Any) -> bool:
    """Return True if any atom has a non-zero atom-map number."""
    try:
        return any(a.GetAtomMapNum() > 0 for a in mol.GetAtoms())
    except Exception:
        return False


def _strip_maps_and_canonical_from_mol(mol: Any) -> Optional[str]:
    """
    Zero all atom-map numbers, remove Hs and return canonical SMILES.
    Returns None on failure.
    """
    try:
        for a in mol.GetAtoms():
            a.SetAtomMapNum(0)
        mol_nomap = Chem.RemoveAllHs(mol)
        return Chem.MolToSmiles(mol_nomap, canonical=True)
    except Exception:
        return None


def _canonical_from_mol(mol: Any) -> Optional[str]:
    """Return canonical SMILES for mol (may keep maps)."""
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def _assign_deterministic_maps_and_canonical(nomap_smiles: str) -> Optional[str]:
    """
    Given a canonical nomap SMILES:
      - reparse it,
      - sanitize,
      - assign atom-map = atom index + 1,
      - return canonical SMILES for the mapped mol.
    """
    try:
        mol2 = Chem.MolFromSmiles(nomap_smiles, sanitize=False)
        if mol2 is None:
            return None
        if not _sanitize_safe(mol2):
            return None
        for a in mol2.GetAtoms():
            a.SetAtomMapNum(a.GetIdx() + 1)
        return Chem.MolToSmiles(mol2, canonical=True)
    except Exception:
        return None


def _standardize_smiles_rdkit(smiles: str, *, keep_aam: bool) -> Optional[str]:
    """
    Thin wrapper that sequences small helpers to keep cyclomatic complexity low.
    Behavior:
      - If Chem is None: return input SMILES (passthrough) or None for empty input.
      - If keep_aam=False: return canonical SMILES with maps stripped.
      - If keep_aam=True:
          * if input has maps -> return canonical SMILES (keep maps)
          * else deterministically assign maps and return canonical mapped SMILES
    """
    if not smiles:
        return None

    if Chem is None:
        # RDKit not available: passthrough (best-effort)
        return smiles

    mol = _mol_from_smiles_safe(smiles)
    if mol is None:
        return None

    if not _sanitize_safe(mol):
        return None

    if not keep_aam:
        return _strip_maps_and_canonical_from_mol(mol)

    # keep_aam True
    if _has_atom_maps(mol):
        return _canonical_from_mol(mol)

    # no maps: produce deterministic mapped canonical SMILES
    nomap = _strip_maps_and_canonical_from_mol(mol)
    if nomap is None:
        return None
    return _assign_deterministic_maps_and_canonical(nomap)


def _dedup_key(
    *,
    dedup_across_rules: bool,
    rule_index: int,
    r_keep_keys: Tuple[str, ...],
    p_keep_keys: Tuple[str, ...],
) -> Tuple[Optional[int], Tuple[str, ...], Tuple[str, ...]]:
    ridx: Optional[int] = None if dedup_across_rules else int(rule_index)
    return (ridx, r_keep_keys, p_keep_keys)


def _sorted_smiles_from_ids(graph: nx.DiGraph, ids: List[int]) -> str:
    return ".".join(sorted(graph.nodes[i]["smiles"] for i in ids))


# --------------------------------------------------------------------------- #
# Mixture generators split by arity to reduce complexity
# --------------------------------------------------------------------------- #


def _iter_mixtures_arity1(
    pool_keys: List[str],
    frontier_keys: List[str],
    *,
    use_frontier: bool,
    cap: int,
) -> Iterator[Tuple[str, ...]]:
    pool_u = sorted(set(pool_keys))
    if not pool_u:
        return
    frontier_u = sorted(set(frontier_keys))
    if not use_frontier:
        frontier_u = pool_u

    n = 0
    for f in frontier_u:
        yield (f,)
        n += 1
        if n >= cap:
            return


def _iter_mixtures_arity2(
    pool_keys: List[str],
    frontier_keys: List[str],
    *,
    use_frontier: bool,
    cap: int,
) -> Iterator[Tuple[str, ...]]:
    pool_u = sorted(set(pool_keys))
    if not pool_u:
        return
    frontier_u = sorted(set(frontier_keys))
    if not use_frontier:
        frontier_u = pool_u

    n = 0
    for f in frontier_u:
        for x in pool_u:
            if x == f:
                continue
            yield (f, x) if f < x else (x, f)
            n += 1
            if n >= cap:
                return


def _iter_mixtures_arityk(
    pool_keys: List[str],
    frontier_keys: List[str],
    *,
    use_frontier: bool,
    arity: int,
    cap: int,
) -> Iterator[Tuple[str, ...]]:
    pool_u = sorted(set(pool_keys))
    if not pool_u:
        return
    frontier_u = sorted(set(frontier_keys))
    if not use_frontier:
        frontier_u = pool_u

    n = 0
    for f in frontier_u:
        others = [x for x in pool_u if x != f]
        for comb in combinations(others, arity - 1):
            yield tuple(sorted((f, *comb)))
            n += 1
            if n >= cap:
                return


# --------------------------------------------------------------------------- #
# SynCRN
# --------------------------------------------------------------------------- #


@dataclass
class SynCRN:
    rules: List[Any]
    repeats: int = 50

    explicit_h: bool = False
    implicit_temp: bool = False
    strategy: Optional[str] = None
    keep_aam: bool = True

    max_components: int = 3
    use_frontier: bool = True
    max_mixtures_per_rule_step: int = 50_000
    max_tasks_per_step: int = 200_000

    skip_no_change: bool = True
    allow_empty_side: bool = False

    dedup_delta: bool = True
    dedup_across_rules: bool = False

    graph: nx.DiGraph = field(init=False)

    _species_index: Dict[str, int] = field(init=False)  # canonical nomap -> node_id
    _next_node_id: int = field(init=False)
    _smiles_cache: Dict[str, Optional[str]] = field(init=False)
    _nomap_cache: Dict[str, Optional[str]] = field(init=False)
    _app_counter: Dict[Tuple[int, int], int] = field(init=False)

    _seen_attempts: Set[Tuple[int, Tuple[str, ...]]] = field(init=False)
    _seen_delta: Set[Tuple[Optional[int], Tuple[str, ...], Tuple[str, ...]]] = field(
        init=False
    )
    _rule_arity_cache: Dict[int, int] = field(init=False)

    def __post_init__(self) -> None:
        if self.max_components < 1:
            raise ValueError("max_components must be >= 1")

        self.graph = nx.DiGraph()
        self._species_index = {}
        self._next_node_id = 1
        self._smiles_cache = {}
        self._nomap_cache = {}
        self._app_counter = {}

        self._seen_attempts = set()
        self._seen_delta = set()
        self._rule_arity_cache = {}

    # ------------------- IDs ------------------- #

    def _alloc_node_id(self) -> int:
        nid = self._next_node_id
        self._next_node_id += 1
        return nid

    def _next_app_index(self, *, step: int, rule_index: int) -> int:
        k = (step, rule_index)
        cur = self._app_counter.get(k, 0)
        self._app_counter[k] = cur + 1
        return cur

    # ------------------- SMILES processing ------------------- #

    def _canonical_nomap(self, smiles: str) -> Optional[str]:
        if smiles in self._nomap_cache:
            return self._nomap_cache[smiles]
        out = _canonicalize_nomap_rdkit(smiles)
        self._nomap_cache[smiles] = out
        return out

    def _standardize_smiles(self, smiles: str) -> Optional[str]:
        if smiles in self._smiles_cache:
            return self._smiles_cache[smiles]
        out = _standardize_smiles_rdkit(smiles, keep_aam=self.keep_aam)
        self._smiles_cache[smiles] = out
        if Chem is None and out is not None:
            logger.warning(
                "RDKit not available; SMILES standardization is a passthrough."
            )
        return out

    # ------------------- Rule arity ------------------- #

    def _infer_rule_arity(self, rule: Any, rule_index: int) -> int:
        if rule_index in self._rule_arity_cache:
            return self._rule_arity_cache[rule_index]

        ar: Optional[int] = None
        if isinstance(rule, str):
            ar = _count_lhs_components(rule)
        else:
            ar = self._infer_rule_arity_from_attrs(rule)

        if ar is None or ar < 1:
            ar = 2

        self._rule_arity_cache[rule_index] = int(ar)
        return int(ar)

    def _infer_rule_arity_from_attrs(self, rule: Any) -> Optional[int]:
        for attr in ("smarts", "smirks", "template"):
            if not hasattr(rule, attr):
                continue
            try:
                ar = _count_lhs_components(str(getattr(rule, attr)))
                if ar is not None:
                    return ar
            except Exception:
                continue
        try:
            return _count_lhs_components(repr(rule))
        except Exception:
            return None

    # ------------------- Graph nodes ------------------- #

    def _add_species_node(self, smiles: str) -> Optional[int]:
        std = self._standardize_smiles(smiles)
        if std is None:
            return None
        key = self._canonical_nomap(std)
        if key is None:
            return None

        if key in self._species_index:
            return self._species_index[key]

        nid = self._alloc_node_id()
        self._species_index[key] = nid
        self.graph.add_node(
            nid,
            kind="species",
            smiles=std,
            smiles_nomap=key,
            label=std,
        )
        return nid

    def _add_rxn_event_node(self, *, step: int, rule_index: int) -> int:
        rule = self.rules[rule_index]
        rule_name = getattr(rule, "name", f"r{rule_index}")
        app_index = self._next_app_index(step=step, rule_index=rule_index)
        label = f"{rule_name}@{step}@{app_index}"

        eid = self._alloc_node_id()
        self.graph.add_node(
            eid,
            kind="rxn",
            label=label,
            step=step,
            rule_index=rule_index,
            rule_name=rule_name,
            app_index=app_index,
            rule_repr=repr(rule),
        )
        return eid

    # ------------------- Delta / overlap ------------------- #

    def _delta_keep_ids(
        self,
        reactant_ids: List[int],
        products_raw: List[str],
    ) -> Tuple[List[int], List[int], Tuple[str, ...], Tuple[str, ...]]:
        r_keys_all = [
            self.graph.nodes[rid].get(
                "smiles_nomap", self.graph.nodes[rid].get("smiles")
            )
            for rid in reactant_ids
        ]
        r_set_all = set(r_keys_all)

        p_ids_all: List[int] = []
        p_keys_all: List[str] = []
        for p in products_raw:
            pid = self._add_species_node(p)
            if pid is None:
                continue
            p_ids_all.append(pid)
            p_keys_all.append(
                self.graph.nodes[pid].get(
                    "smiles_nomap", self.graph.nodes[pid].get("smiles")
                )
            )

        p_set_all = set(p_keys_all)
        unchanged = r_set_all & p_set_all

        r_keep_ids = [
            rid for rid, rk in zip(reactant_ids, r_keys_all) if rk not in unchanged
        ]
        p_keep_ids = [
            pid for pid, pk in zip(p_ids_all, p_keys_all) if pk not in unchanged
        ]

        r_keep_keys = tuple(
            sorted(
                self.graph.nodes[rid].get(
                    "smiles_nomap", self.graph.nodes[rid].get("smiles")
                )
                for rid in r_keep_ids
            )
        )
        p_keep_keys = tuple(
            sorted(
                self.graph.nodes[pid].get(
                    "smiles_nomap", self.graph.nodes[pid].get("smiles")
                )
                for pid in p_keep_ids
            )
        )
        return r_keep_ids, p_keep_ids, r_keep_keys, p_keep_keys

    # ------------------- Mixture generation ------------------- #

    def _iter_mixtures_for_rule(
        self,
        pool_keys: List[str],
        frontier_keys: List[str],
        *,
        arity: int,
        cap: int,
    ) -> Iterator[Tuple[str, ...]]:
        if arity < 1 or arity > self.max_components:
            return iter(())
        if arity == 1:
            return _iter_mixtures_arity1(
                pool_keys,
                frontier_keys,
                use_frontier=self.use_frontier,
                cap=cap,
            )
        if arity == 2:
            return _iter_mixtures_arity2(
                pool_keys,
                frontier_keys,
                use_frontier=self.use_frontier,
                cap=cap,
            )
        return _iter_mixtures_arityk(
            pool_keys,
            frontier_keys,
            use_frontier=self.use_frontier,
            arity=arity,
            cap=cap,
        )

    # ------------------- Build orchestration split to reduce C901 ------------------- #

    def build(
        self,
        seeds: Iterable[str],
        *,
        parallel: bool = False,
        max_workers: Optional[int] = None,
    ) -> nx.DiGraph:
        pool_keys, frontier_keys = self._init_pool(seeds)
        if not pool_keys:
            return self.graph

        for step in range(1, self.repeats + 1):
            if self.use_frontier and not frontier_keys:
                break

            tasks = self._make_tasks_for_step(pool_keys, frontier_keys, step)
            if not tasks:
                break

            results = self._run_tasks(tasks, parallel=parallel, max_workers=max_workers)
            next_frontier = self._integrate_results(results, pool_keys, step)

            frontier_keys = next_frontier
            if self.use_frontier and not frontier_keys:
                break

        return self.graph

    def _init_pool(self, seeds: Iterable[str]) -> Tuple[Set[str], Set[str]]:
        pool_keys: Set[str] = set()
        frontier_keys: Set[str] = set()

        for s in seeds:
            sid = self._add_species_node(s)
            if sid is None:
                continue
            k = self.graph.nodes[sid].get(
                "smiles_nomap", self.graph.nodes[sid].get("smiles")
            )
            pool_keys.add(k)
            frontier_keys.add(k)

        return pool_keys, frontier_keys

    def _make_tasks_for_step(
        self,
        pool_keys: Set[str],
        frontier_keys: Set[str],
        step: int,
    ) -> List[Tuple[int, Any, str, bool, bool, Optional[str], Tuple[str, ...]]]:
        budget = int(self.max_tasks_per_step)
        tasks: List[
            Tuple[int, Any, str, bool, bool, Optional[str], Tuple[str, ...]]
        ] = []

        # Pre-cache id->smiles for the pool to avoid repeated graph lookups.
        id_to_smiles = self._cache_id_to_smiles(pool_keys)

        for ridx, rule in enumerate(self.rules):
            if budget <= 0:
                break

            arity = self._infer_rule_arity(rule, ridx)
            if arity > self.max_components:
                continue

            cap = min(int(self.max_mixtures_per_rule_step), budget)
            mix_iter = self._iter_mixtures_for_rule(
                list(pool_keys),
                list(frontier_keys),
                arity=arity,
                cap=cap,
            )

            for mix_keys in mix_iter:
                if budget <= 0:
                    break
                task = self._task_from_mix(
                    ridx=ridx,
                    rule=rule,
                    mix_keys=mix_keys,
                    id_to_smiles=id_to_smiles,
                )
                if task is None:
                    continue
                tasks.append(task)
                budget -= 1

        return tasks

    def _cache_id_to_smiles(self, pool_keys: Set[str]) -> Dict[int, str]:
        id_to_smiles: Dict[int, str] = {}
        for k in pool_keys:
            nid = self._species_index.get(k)
            if nid is None:
                continue
            id_to_smiles[nid] = self.graph.nodes[nid]["smiles"]
        return id_to_smiles

    def _task_from_mix(
        self,
        *,
        ridx: int,
        rule: Any,
        mix_keys: Tuple[str, ...],
        id_to_smiles: Dict[int, str],
    ) -> Optional[Tuple[int, Any, str, bool, bool, Optional[str], Tuple[str, ...]]]:
        app_key = (int(ridx), mix_keys)
        if app_key in self._seen_attempts:
            return None
        self._seen_attempts.add(app_key)

        reactant_ids = [self._species_index.get(k) for k in mix_keys]
        if any(nid is None for nid in reactant_ids):
            return None
        ids = [int(nid) for nid in reactant_ids if nid is not None]

        # Build substrate from cached smiles
        try:
            substrate = ".".join(sorted(id_to_smiles[i] for i in ids))
        except KeyError:
            # A reactant not in current pool cache; fallback to graph lookup
            substrate = _sorted_smiles_from_ids(self.graph, ids)

        return (
            int(ridx),
            rule,
            substrate,
            self.explicit_h,
            self.implicit_temp,
            self.strategy,
            mix_keys,
        )

    def _run_tasks(
        self,
        tasks: List[Tuple[int, Any, str, bool, bool, Optional[str], Tuple[str, ...]]],
        *,
        parallel: bool,
        max_workers: Optional[int],
    ) -> List[Tuple[int, Tuple[str, ...], List[str]]]:
        results: List[Tuple[int, Tuple[str, ...], List[str]]] = []
        if parallel and len(tasks) > 1:
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                for idx, mix_keys, products_list in ex.map(_apply_rule_worker, tasks):
                    results.append((idx, mix_keys, products_list))
            return results

        for t in tasks:
            results.append(_apply_rule_worker(t))
        return results

    def _integrate_results(
        self,
        results: List[Tuple[int, Tuple[str, ...], List[str]]],
        pool_keys: Set[str],
        step: int,
    ) -> Set[str]:
        next_frontier: Set[str] = set()

        for rule_index, mix_keys, products_list in results:
            if not products_list:
                continue
            self._integrate_one_result(
                rule_index=rule_index,
                mix_keys=mix_keys,
                products_list=products_list,
                pool_keys=pool_keys,
                next_frontier=next_frontier,
                step=step,
            )

        return next_frontier

    def _integrate_one_result(
        self,
        *,
        rule_index: int,
        mix_keys: Tuple[str, ...],
        products_list: List[str],
        pool_keys: Set[str],
        next_frontier: Set[str],
        step: int,
    ) -> None:
        reactant_ids = [self._species_index.get(k) for k in mix_keys]
        if any(nid is None for nid in reactant_ids):
            return
        reactant_ids_int = [int(nid) for nid in reactant_ids if nid is not None]

        for prod_mix in products_list:
            prod_mix = (prod_mix or "").strip()
            if not prod_mix:
                continue

            products_raw = [s for s in prod_mix.split(".") if s]
            r_keep, p_keep, r_keep_keys, p_keep_keys = self._delta_keep_ids(
                reactant_ids_int,
                products_raw,
            )

            if self.skip_no_change and (not r_keep and not p_keep):
                continue
            if (not self.allow_empty_side) and (not r_keep or not p_keep):
                continue

            if self.dedup_delta:
                dkey = _dedup_key(
                    dedup_across_rules=self.dedup_across_rules,
                    rule_index=int(rule_index),
                    r_keep_keys=r_keep_keys,
                    p_keep_keys=p_keep_keys,
                )
                if dkey in self._seen_delta:
                    continue
                self._seen_delta.add(dkey)

            eid = self._add_rxn_event_node(step=step, rule_index=int(rule_index))

            for rid in r_keep:
                self.graph.add_edge(
                    rid,
                    eid,
                    step=step,
                    rule_index=int(rule_index),
                    rxn_id=eid,
                    role="reactant",
                )

            for pid in p_keep:
                self.graph.add_edge(
                    eid,
                    pid,
                    step=step,
                    rule_index=int(rule_index),
                    rxn_id=eid,
                    role="product",
                )

            self._update_pool_with_products(products_raw, pool_keys, next_frontier)

    def _update_pool_with_products(
        self,
        products_raw: List[str],
        pool_keys: Set[str],
        next_frontier: Set[str],
    ) -> None:
        for p in products_raw:
            pid_all = self._add_species_node(p)
            if pid_all is None:
                continue
            pk = self.graph.nodes[pid_all].get(
                "smiles_nomap", self.graph.nodes[pid_all].get("smiles")
            )
            if pk not in pool_keys:
                pool_keys.add(pk)
                next_frontier.add(pk)

    # ------------------- convenience ------------------- #

    @property
    def species_nodes(self) -> List[int]:
        return [n for n, d in self.graph.nodes(data=True) if d.get("kind") == "species"]

    @property
    def rxn_nodes(self) -> List[int]:
        return [n for n, d in self.graph.nodes(data=True) if d.get("kind") == "rxn"]


# --------------------------------------------------------------------------- #
# Flattening
# --------------------------------------------------------------------------- #


@dataclass
class ReactionDeltaFlattener:
    graph: nx.DiGraph
    skip_no_change: bool = True
    allow_empty_side: bool = False
    deduplicate: bool = True
    _cache: List[Dict[str, Any]] = field(default_factory=list, init=False)

    def build(self) -> "ReactionDeltaFlattener":
        self._cache = self._flatten()
        return self

    @property
    def reactions(self) -> List[Dict[str, Any]]:
        return list(self._cache)

    def _collect_in(self, eid: int) -> List[str]:
        out: List[str] = []
        for u, _, ed in self.graph.in_edges(eid, data=True):
            if ed.get("role") != "reactant":
                continue
            if self.graph.nodes[u].get("kind") != "species":
                continue
            out.append(self.graph.nodes[u].get("smiles", str(u)))
        return out

    def _collect_out(self, eid: int) -> List[str]:
        out: List[str] = []
        for _, v, ed in self.graph.out_edges(eid, data=True):
            if ed.get("role") != "product":
                continue
            if self.graph.nodes[v].get("kind") != "species":
                continue
            out.append(self.graph.nodes[v].get("smiles", str(v)))
        return out

    def _nz(self, x: Optional[int]) -> int:
        return 10**9 if x is None else int(x)

    def _flatten(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen: Set[str] = set()

        for eid, nd in self.graph.nodes(data=True):
            if nd.get("kind") != "rxn":
                continue

            r = sorted(set(self._collect_in(eid)))
            p = sorted(set(self._collect_out(eid)))

            unchanged = set(r) & set(p)
            rchg = [x for x in r if x not in unchanged]
            pchg = [x for x in p if x not in unchanged]

            if self.skip_no_change and not rchg and not pchg:
                continue
            if (not self.allow_empty_side) and (not rchg or not pchg):
                continue

            rxn_smiles = f"{'.'.join(rchg)}>>{'.'.join(pchg)}"
            if self.deduplicate and rxn_smiles in seen:
                continue
            seen.add(rxn_smiles)

            out.append(
                {
                    "rxn_id": eid,
                    "label": nd.get("label"),
                    "step": nd.get("step"),
                    "rule_index": nd.get("rule_index"),
                    "app_index": nd.get("app_index"),
                    "reactants": rchg,
                    "products": pchg,
                    "rxn_smiles": rxn_smiles,
                }
            )

        out.sort(
            key=lambda r: (
                self._nz(r.get("step")),
                self._nz(r.get("rule_index")),
                self._nz(r.get("app_index")),
                r["rxn_id"],
            )
        )
        return out


# --------------------------------------------------------------------------- #
# Convenience wrapper (line-length safe)
# --------------------------------------------------------------------------- #


def build_syncrn_from_smarts(
    rules: List[str],
    seeds: List[str],
    *,
    repeats: int = 50,
    explicit_h: bool = False,
    implicit_temp: bool = False,
    strategy: Optional[str] = None,
    keep_aam: bool = True,
    parallel: bool = False,
    max_workers: Optional[int] = None,
    max_components: int = 3,
    use_frontier: bool = True,
    max_mixtures_per_rule_step: int = 50_000,
    max_tasks_per_step: int = 200_000,
    skip_no_change: bool = True,
    allow_empty_side: bool = False,
    dedup_delta: bool = True,
    dedup_across_rules: bool = False,
) -> nx.DiGraph:
    crn = SynCRN(
        rules=rules,
        repeats=repeats,
        explicit_h=explicit_h,
        implicit_temp=implicit_temp,
        strategy=strategy,
        keep_aam=keep_aam,
        max_components=max_components,
        use_frontier=use_frontier,
        max_mixtures_per_rule_step=max_mixtures_per_rule_step,
        max_tasks_per_step=max_tasks_per_step,
        skip_no_change=skip_no_change,
        allow_empty_side=allow_empty_side,
        dedup_delta=dedup_delta,
        dedup_across_rules=dedup_across_rules,
    )
    return crn.build(seeds, parallel=parallel, max_workers=max_workers)
