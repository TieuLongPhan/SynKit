from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple
import logging
from concurrent.futures import ProcessPoolExecutor

import networkx as nx

from .arity import infer_rule_arity
from .keys import make_dedup_key
from .smiles import Chem, standardize_smiles_rdkit
from .worker import apply_rule_worker
from .state import DerivationState
from .strategy import FrontierStrategy
from .derivation import DerivationLog

logger = logging.getLogger(__name__)


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
    allow_self_mixtures: bool = False

    skip_no_change: bool = True
    allow_empty_side: bool = False

    dedup_delta: bool = True
    dedup_across_rules: bool = False

    graph: nx.DiGraph = field(init=False)

    _species_index: Dict[str, int] = field(init=False)
    _next_node_id: int = field(init=False)
    _smiles_cache: Dict[str, Optional[str]] = field(init=False)
    _app_counter: Dict[int, int] = field(init=False)

    _seen_attempts: Set[Tuple[int, Tuple[str, ...]]] = field(init=False)
    _seen_delta: Set[Tuple[Optional[int], Tuple[str, ...], Tuple[str, ...]]] = field(
        init=False
    )
    _rule_arity_cache: Dict[int, int] = field(init=False)
    _warned_no_rdkit: bool = field(init=False)
    state: DerivationState = field(init=False, repr=False)
    strategy_engine: FrontierStrategy = field(init=False, repr=False)
    derivations: DerivationLog = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_components < 1:
            raise ValueError("max_components must be >= 1")
        self.state = DerivationState()
        self.strategy_engine = FrontierStrategy()
        self.derivations = DerivationLog()
        self.reset()

    def reset(self) -> None:
        self.graph = nx.DiGraph()
        self._species_index = {}
        self._next_node_id = 1
        self._smiles_cache = {}
        self._app_counter = {}

        self._seen_attempts = set()
        self._seen_delta = set()
        self._rule_arity_cache = {}
        self._warned_no_rdkit = False
        self.state.set_initial(pool_keys=set(), frontier_keys=set())
        self.derivations.clear()

    def _alloc_node_id(self) -> int:
        nid = self._next_node_id
        self._next_node_id += 1
        return nid

    def _next_app_index(self, *, rule_index: int) -> int:
        cur = self._app_counter.get(rule_index, 0) + 1
        self._app_counter[rule_index] = cur
        return cur

    def _standardize_smiles(self, smiles: str) -> Optional[str]:
        if smiles in self._smiles_cache:
            return self._smiles_cache[smiles]

        out = standardize_smiles_rdkit(smiles, keep_aam=self.keep_aam)
        self._smiles_cache[smiles] = out

        if Chem is None and out is not None and not self._warned_no_rdkit:
            logger.warning(
                "RDKit not available; SMILES standardization is a passthrough."
            )
            self._warned_no_rdkit = True

        return out

    def _standardize_product_mixture(self, prod_mix: str) -> List[str]:
        out: List[str] = []
        for s in (prod_mix or "").split("."):
            if not s:
                continue
            std = self._standardize_smiles(s)
            if std is None:
                continue
            out.append(std)
        return out

    def _infer_rule_arity(self, rule: Any, rule_index: int) -> int:
        if rule_index in self._rule_arity_cache:
            return self._rule_arity_cache[rule_index]
        arity = infer_rule_arity(rule)
        self._rule_arity_cache[rule_index] = int(arity)
        return int(arity)

    def _add_species_node(self, smiles: str) -> Optional[int]:
        std = self._standardize_smiles(smiles)
        if std is None:
            return None

        if std in self._species_index:
            return self._species_index[std]

        nid = self._alloc_node_id()
        self._species_index[std] = nid
        self.graph.add_node(
            nid,
            kind="species",
            smiles=std,
            label=std,
        )
        return nid

    def _add_rxn_event_node(self, *, step: int, rule_index: int) -> int:
        app_index = self._next_app_index(rule_index=rule_index)
        label = f"r@{rule_index}@{app_index}"

        eid = self._alloc_node_id()
        self.graph.add_node(
            eid,
            kind="rule",
            label=label,
            step=step,
            rule_index=rule_index,
            app_index=app_index,
            rule_repr=repr(self.rules[rule_index]),
        )
        return eid

    def _delta_keep_smiles(
        self,
        reactant_ids: List[int],
        products_std_all: List[str],
    ) -> Tuple[List[int], List[str], Tuple[str, ...], Tuple[str, ...]]:
        r_smiles_all = [self.graph.nodes[rid]["smiles"] for rid in reactant_ids]
        unchanged = set(r_smiles_all) & set(products_std_all)

        r_keep_ids = [
            rid for rid, rs in zip(reactant_ids, r_smiles_all) if rs not in unchanged
        ]
        p_keep_smiles = [ps for ps in products_std_all if ps not in unchanged]

        r_keep_keys = tuple(
            sorted(self.graph.nodes[rid]["smiles"] for rid in r_keep_ids)
        )
        p_keep_keys = tuple(sorted(p_keep_smiles))

        return r_keep_ids, p_keep_smiles, r_keep_keys, p_keep_keys

    def _iter_mixtures_for_rule(
        self,
        pool_keys: List[str],
        frontier_keys: List[str],
        *,
        arity: int,
        cap: int,
    ) -> Iterator[Tuple[str, ...]]:
        return self.strategy_engine.iter_mixtures(
            pool_keys=pool_keys,
            frontier_keys=frontier_keys,
            arity=arity,
            use_frontier=self.use_frontier,
            allow_self_mixtures=self.allow_self_mixtures,
            cap=cap,
            max_components=self.max_components,
        )

    def build(
        self,
        seeds: Iterable[str],
        *,
        parallel: bool = False,
        max_workers: Optional[int] = None,
        reset: bool = True,
    ) -> nx.DiGraph:
        if reset:
            self.reset()

        pool_keys, frontier_keys = self._init_pool(seeds)
        self.state.set_initial(pool_keys=pool_keys, frontier_keys=frontier_keys)
        if not pool_keys:
            return self.graph

        for step in range(1, self.repeats + 1):
            self.state.begin_step(step)
            if self.use_frontier and not frontier_keys:
                break

            tasks = self._make_tasks_for_step(pool_keys, frontier_keys)
            if not tasks:
                break

            results = self._run_tasks(tasks, parallel=parallel, max_workers=max_workers)
            next_frontier = self._integrate_results(results, pool_keys, step)

            frontier_keys = next_frontier
            self.state.advance(next_frontier)
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
            k = self.graph.nodes[sid]["smiles"]
            pool_keys.add(k)
            frontier_keys.add(k)

        return pool_keys, frontier_keys

    def _make_tasks_for_step(
        self,
        pool_keys: Set[str],
        frontier_keys: Set[str],
    ) -> List[Tuple[int, Any, str, bool, bool, Optional[str], Tuple[str, ...]]]:
        budget = int(self.max_tasks_per_step)
        tasks: List[
            Tuple[int, Any, str, bool, bool, Optional[str], Tuple[str, ...]]
        ] = []

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
                )
                if task is None:
                    continue
                tasks.append(task)
                budget -= 1

        return tasks

    def _task_from_mix(
        self,
        *,
        ridx: int,
        rule: Any,
        mix_keys: Tuple[str, ...],
    ) -> Optional[Tuple[int, Any, str, bool, bool, Optional[str], Tuple[str, ...]]]:
        app_key = (int(ridx), mix_keys)
        if app_key in self._seen_attempts:
            return None
        self._seen_attempts.add(app_key)

        substrate = ".".join(mix_keys)
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
                for idx, mix_keys, products_list in ex.map(apply_rule_worker, tasks):
                    results.append((idx, mix_keys, products_list))
            return results

        for t in tasks:
            results.append(apply_rule_worker(t))
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

        seen_prod_mix: Set[str] = set()
        for prod_mix in products_list:
            prod_mix = (prod_mix or "").strip()
            if not prod_mix or prod_mix in seen_prod_mix:
                continue
            seen_prod_mix.add(prod_mix)

            products_std_all = self._standardize_product_mixture(prod_mix)
            if not products_std_all:
                continue

            r_keep, p_keep_smiles, r_keep_keys, p_keep_keys = self._delta_keep_smiles(
                reactant_ids_int,
                products_std_all,
            )

            if self.skip_no_change and (not r_keep and not p_keep_smiles):
                continue
            if (not self.allow_empty_side) and (not r_keep or not p_keep_smiles):
                continue

            if self.dedup_delta:
                dkey = make_dedup_key(
                    dedup_across_rules=self.dedup_across_rules,
                    rule_index=int(rule_index),
                    r_keep_keys=r_keep_keys,
                    p_keep_keys=p_keep_keys,
                )
                if dkey in self._seen_delta:
                    continue
                self._seen_delta.add(dkey)

            eid = self._add_rxn_event_node(step=step, rule_index=int(rule_index))
            nd = self.graph.nodes[eid]
            self._add_reactant_edges(
                reactant_ids=r_keep,
                eid=eid,
                step=step,
                rule_index=int(rule_index),
            )
            self._add_product_edges(
                product_smiles=p_keep_smiles,
                eid=eid,
                step=step,
                rule_index=int(rule_index),
            )

            self.derivations.append(
                event_id=eid,
                label=str(nd.get("label")),
                step=int(step),
                rule_index=int(rule_index),
                reactants=tuple(
                    sorted(self.graph.nodes[rid]["smiles"] for rid in r_keep)
                ),
                products=tuple(sorted(p_keep_smiles)),
            )

            self._update_pool_with_products(
                products_std_all=products_std_all,
                pool_keys=pool_keys,
                next_frontier=next_frontier,
            )

    def _add_reactant_edges(
        self,
        *,
        reactant_ids: List[int],
        eid: int,
        step: int,
        rule_index: int,
    ) -> None:
        for rid, stoich in Counter(reactant_ids).items():
            self.graph.add_edge(
                rid,
                eid,
                step=step,
                rule_index=rule_index,
                rxn_id=eid,
                role="reactant",
                stoich=int(stoich),
            )

    def _add_product_edges(
        self,
        *,
        product_smiles: List[str],
        eid: int,
        step: int,
        rule_index: int,
    ) -> None:
        for ps, stoich in Counter(product_smiles).items():
            pid = self._add_species_node(ps)
            if pid is None:
                continue
            self.graph.add_edge(
                eid,
                pid,
                step=step,
                rule_index=rule_index,
                rxn_id=eid,
                role="product",
                stoich=int(stoich),
            )

    def _update_pool_with_products(
        self,
        *,
        products_std_all: List[str],
        pool_keys: Set[str],
        next_frontier: Set[str],
    ) -> None:
        for ps in set(products_std_all):
            pid = self._add_species_node(ps)
            if pid is None:
                continue
            pk = self.graph.nodes[pid]["smiles"]
            if pk not in pool_keys:
                pool_keys.add(pk)
                next_frontier.add(pk)

    @property
    def species_nodes(self) -> List[int]:
        return [n for n, d in self.graph.nodes(data=True) if d.get("kind") == "species"]

    @property
    def rxn_nodes(self) -> List[int]:
        return [n for n, d in self.graph.nodes(data=True) if d.get("kind") == "rule"]

    @property
    def derivation_records(self) -> List[Dict[str, object]]:
        return self.derivations.as_dicts()


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
    allow_self_mixtures: bool = False,
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
        allow_self_mixtures=allow_self_mixtures,
        skip_no_change=skip_no_change,
        allow_empty_side=allow_empty_side,
        dedup_delta=dedup_delta,
        dedup_across_rules=dedup_across_rules,
    )
    return crn.build(seeds, parallel=parallel, max_workers=max_workers)
