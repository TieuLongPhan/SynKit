from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import heapq
from itertools import count
import logging

import networkx as nx

from synkit.Synthesis.Reactor.syn_reactor import SynReactor
from .dag import DAG

try:
    from rdkit import Chem
except ImportError:
    Chem = None

logger = logging.getLogger(__name__)


@dataclass
class TargetDAG:
    """
    Forward-directed pathway constructor on top of :class:`DAG` with
    heuristic guidance, pruning, and optional A* search.

    Features
    --------
    - Grows a :class:`DAG` from seed molecules using SynReactor rules.
    - Stops when the target is first created (or when saturated / max
      repeats is reached).
    - Extracts a seed→target path using BFS or A*.
    - Heuristic guidance via ``heuristic_fn``.
    - Pruning knobs for speed:
      * ``beam_width``: limit size of active pool after each step.
      * ``max_mappings_per_rule``: truncate SynReactor mappings per rule.
      * ``max_new_species_per_step``: cap globally how many new species
        are allowed in each expansion step.

    Matching against the target is done on canonical, map-free, H-stripped
    SMILES, independent of :attr:`keep_aam`.
    """

    rules: List[Any]
    repeats: int = 50
    explicit_h: bool = False
    implicit_temp: bool = False
    strategy: Optional[str] = None
    keep_aam: bool = True

    # Heuristic / search settings
    beam_width: Optional[int] = None
    heuristic_fn: Optional[Callable[[str, str], float]] = None
    search_mode: str = "bfs"  # or "astar"

    # New pruning knobs
    max_mappings_per_rule: Optional[int] = None
    max_new_species_per_step: Optional[int] = None

    dag: "DAG" = field(init=False)
    _nomap_cache: Dict[str, Optional[str]] = field(init=False)

    def __post_init__(self) -> None:
        # NOTE: assumes DAG class and _apply_rule_worker are defined in this module
        self.dag = DAG(
            rules=self.rules,
            repeats=self.repeats,
            explicit_h=self.explicit_h,
            implicit_temp=self.implicit_temp,
            strategy=self.strategy,
            keep_aam=self.keep_aam,
        )
        self._nomap_cache = {}

    # --------------- internal: map-free canonical SMILES ------------------ #

    def _canonical_nomap(self, smiles: str) -> Optional[str]:
        """
        Canonicalize a SMILES with all atom-maps stripped and H removed.

        Used only for matching + heuristic scoring; independent of keep_aam.
        """
        if smiles in self._nomap_cache:
            return self._nomap_cache[smiles]

        if Chem is None:
            self._nomap_cache[smiles] = smiles
            return smiles

        if not smiles:
            self._nomap_cache[smiles] = None
            return None

        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            self._nomap_cache[smiles] = None
            return None

        try:
            Chem.SanitizeMol(mol)
        except Exception:
            self._nomap_cache[smiles] = None
            return None

        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        mol = Chem.RemoveAllHs(mol)
        norm = Chem.MolToSmiles(mol, canonical=True)

        self._nomap_cache[smiles] = norm
        return norm

    def _score(self, smiles: str, target_norm: Optional[str]) -> float:
        """
        Wrapper around the user-provided heuristic (if any) or a default.
        """
        if target_norm is None:
            return 1e9

        s_norm = self._canonical_nomap(smiles)
        if s_norm is None:
            return 1e9

        if self.heuristic_fn is not None:
            try:
                return float(self.heuristic_fn(s_norm, target_norm))
            except Exception:
                logger.exception(
                    "TargetDAG.heuristic_fn raised an error; "
                    "falling back to default heuristic."
                )

        # Default: SMILES length difference
        return abs(len(s_norm) - len(target_norm))

    # --------------- BFS / A* path reconstruction ------------------------ #

    def _bfs_path(
        self,
        seed_nodes: Iterable[int],
        target_node: int,
    ) -> Optional[List[int]]:
        """
        Multi-source BFS to find a directed path from any seed node to target.
        """
        seed_nodes = list(seed_nodes)
        if not seed_nodes:
            return None

        visited: Dict[int, Optional[int]] = {}
        q = deque()

        for s in seed_nodes:
            visited[s] = None
            q.append(s)

        while q:
            u = q.popleft()
            if u == target_node:
                break
            for v in self.dag.graph.successors(u):
                if v not in visited:
                    visited[v] = u
                    q.append(v)

        if target_node not in visited:
            return None

        path: List[int] = []
        cur = target_node
        while cur is not None:
            path.append(cur)
            cur = visited[cur]
        path.reverse()
        return path

    def _astar_path(
        self,
        seed_nodes: Iterable[int],
        target_node: int,
        target_norm: Optional[str],
    ) -> Optional[List[int]]:
        """
        A* search on the already-constructed DAG.

        Species nodes use the heuristic; rule nodes get h=0.
        """
        if target_norm is None:
            return self._bfs_path(seed_nodes, target_node)

        seed_nodes = list(seed_nodes)
        if not seed_nodes:
            return None

        g_score: Dict[int, float] = {}
        f_score: Dict[int, float] = {}
        came_from: Dict[int, int] = {}
        open_heap: List[Tuple[float, int, int]] = []
        counter = count()
        closed: Set[int] = set()

        # initialise frontier with all seed nodes
        for s in seed_nodes:
            g_score[s] = 0.0
            data = self.dag.graph.nodes[s]
            h = 0.0
            if data.get("kind") == "species":
                smi = data.get("smiles", "")
                h = float(self._score(smi, target_norm))
            f_score[s] = g_score[s] + h
            heapq.heappush(open_heap, (f_score[s], next(counter), s))

        while open_heap:
            _, _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            if current == target_node:
                break
            closed.add(current)

            for nb in self.dag.graph.successors(current):
                tentative_g = g_score.get(current, float("inf")) + 1.0
                if tentative_g < g_score.get(nb, float("inf")):
                    g_score[nb] = tentative_g
                    came_from[nb] = current

                    data_nb = self.dag.graph.nodes[nb]
                    h_nb = 0.0
                    if data_nb.get("kind") == "species":
                        smi_nb = data_nb.get("smiles", "")
                        h_nb = float(self._score(smi_nb, target_norm))
                    f_nb = tentative_g + h_nb
                    f_score[nb] = f_nb
                    heapq.heappush(open_heap, (f_nb, next(counter), nb))

        if target_node not in g_score:
            return None

        # reconstruct path
        path: List[int] = []
        cur = target_node
        while cur in came_from:
            path.append(cur)
            cur = came_from[cur]
        path.append(cur)
        path.reverse()
        return path

    # --------------- main API: build until target ------------------------ #

    def build_to_target(
        self,
        seeds: Iterable[str],
        target_smiles: str,
        parallel: bool = False,
        max_workers: Optional[int] = None,
    ) -> Tuple[nx.DiGraph, Optional[List[int]], Optional[int]]:
        """
        Grow the DAG from the given seeds until the target is reached or
        the maximum number of repeats is exhausted.

        Returns
        -------
        graph :
            Directed DAG with species + rule nodes.
        path :
            List of node ids on a seed→target path, or None if unreachable.
        target_step :
            Expansion round where target first appeared (0 if already a
            seed), or None if unreachable.
        """
        target_norm = self._canonical_nomap(target_smiles)
        if target_norm is None:
            logger.warning(
                "TargetDAG: target %s could not be canonicalised.",
                target_smiles,
            )

        pool: Set[str] = set()
        seed_nodes: Set[int] = set()
        target_node: Optional[int] = None
        target_step: Optional[int] = None

        # ---- initial seeding ----
        for s in seeds:
            sid = self.dag._add_species_node(s)
            if sid is None:
                continue
            smi_std = self.dag.graph.nodes[sid]["smiles"]
            pool.add(smi_std)
            seed_nodes.add(sid)

            if target_norm is not None:
                s_norm = self._canonical_nomap(smi_std)
                if s_norm == target_norm and target_node is None:
                    target_node = sid
                    target_step = 0  # already present in seeds

        if target_node is not None:
            # target already in seeds
            mode = self.search_mode.lower()
            if mode == "astar":
                path = self._astar_path(seed_nodes, target_node, target_norm)
                if path is None:
                    path = self._bfs_path(seed_nodes, target_node)
            else:
                path = self._bfs_path(seed_nodes, target_node)
            return self.dag.graph, path, target_step

        # ---- forward expansion ----
        last_step: Optional[int] = None

        for step in range(1, self.dag.repeats + 1):
            last_step = step
            current_pool = set(pool)
            if not current_pool:
                break

            substrate = ".".join(sorted(current_pool))
            logger.debug("[TargetDAG] Step %d substrate: %s", step, substrate)

            reactant_ids = [self.dag._species_index[s] for s in current_pool]
            any_new_species = False
            new_species_this_step = 0

            # compute products per rule (with optional parallelisation)
            rule_results: Dict[int, List[str]] = {}

            if parallel and len(self.dag.rules) > 1:
                tasks = [
                    (
                        idx,
                        rule,
                        substrate,
                        self.dag.explicit_h,
                        self.dag.implicit_temp,
                        self.dag.strategy,
                    )
                    for idx, rule in enumerate(self.dag.rules)
                ]
                with ProcessPoolExecutor(max_workers=max_workers) as ex:
                    for idx, products_list in ex.map(_apply_rule_worker, tasks):
                        # truncate per rule if requested
                        if (
                            self.max_mappings_per_rule is not None
                            and len(products_list) > self.max_mappings_per_rule
                        ):
                            products_list = products_list[: self.max_mappings_per_rule]
                        rule_results[idx] = products_list
            else:
                for idx, rule in enumerate(self.dag.rules):
                    kwargs = dict(
                        smiles=substrate,
                        template=rule,
                        invert=False,
                        explicit_h=self.dag.explicit_h,
                        implicit_temp=self.dag.implicit_temp,
                    )
                    if self.dag.strategy is not None:
                        kwargs["strategy"] = self.dag.strategy

                    reactor = SynReactor.from_smiles(**kwargs)
                    products_list = list(reactor.smiles_list)

                    if (
                        self.max_mappings_per_rule is not None
                        and len(products_list) > self.max_mappings_per_rule
                    ):
                        products_list = products_list[: self.max_mappings_per_rule]

                    rule_results[idx] = products_list

            # integrate & check for target
            for idx, products_list in rule_results.items():
                rnode = self.dag._get_rule_node(idx)
                if not products_list:
                    continue

                # reactants → rule
                for rid in reactant_ids:
                    if not self.dag.graph.has_edge(rid, rnode):
                        self.dag.graph.add_edge(
                            rid,
                            rnode,
                            step=step,
                            rule_index=idx,
                            role="reactant",
                        )

                # rule → product species
                for prod_mix in products_list:
                    prod_mix = prod_mix.strip()
                    if not prod_mix:
                        continue

                    products_raw = [s for s in prod_mix.split(".") if s]
                    for p_raw in products_raw:
                        # optional global cap on new species per step
                        if (
                            self.max_new_species_per_step is not None
                            and new_species_this_step >= self.max_new_species_per_step
                        ):
                            break

                        pid = self.dag._add_species_node(p_raw)
                        if pid is None:
                            continue

                        p_std = self.dag.graph.nodes[pid]["smiles"]

                        # add edge rule → product
                        if not self.dag.graph.has_edge(rnode, pid):
                            self.dag.graph.add_edge(
                                rnode,
                                pid,
                                step=step,
                                rule_index=idx,
                                role="product",
                            )

                        # track new species in pool
                        if p_std not in pool:
                            pool.add(p_std)
                            any_new_species = True
                            new_species_this_step += 1

                        # target check (map-free canonical)
                        if target_norm is not None and target_node is None:
                            p_norm = self._canonical_nomap(p_std)
                            if p_norm == target_norm:
                                target_node = pid
                                target_step = step
                                break

                    if (
                        self.max_new_species_per_step is not None
                        and new_species_this_step >= self.max_new_species_per_step
                    ):
                        break
                    if target_node is not None:
                        break

                if (
                    self.max_new_species_per_step is not None
                    and new_species_this_step >= self.max_new_species_per_step
                ):
                    break
                if target_node is not None:
                    break

            # ---- optional beam pruning of the active pool ----
            if target_node is None and any_new_species and self.beam_width is not None:
                if target_norm is not None and len(pool) > self.beam_width:
                    scored = sorted(
                        pool,
                        key=lambda s: self._score(s, target_norm),
                    )
                    keep_set = set(scored[: self.beam_width])

                    # always keep seed species
                    seed_smiles = {
                        self.dag.graph.nodes[n]["smiles"] for n in seed_nodes
                    }
                    keep_set |= seed_smiles

                    removed = pool - keep_set
                    if removed:
                        logger.debug(
                            "TargetDAG beam pruning at step %d: "
                            "pool %d -> %d (removed %d)",
                            step,
                            len(pool),
                            len(keep_set),
                            len(removed),
                        )
                    pool = keep_set

            if target_node is not None:
                break
            if not any_new_species:
                break

        if target_node is None:
            logger.info(
                "TargetDAG: target not reached within %s repeats "
                "(last step explored = %s).",
                self.dag.repeats,
                last_step,
            )
            return self.dag.graph, None, last_step

        # ---- path search (BFS or A*) ----
        mode = self.search_mode.lower()
        if mode == "astar":
            path = self._astar_path(seed_nodes, target_node, target_norm)
            if path is None:
                path = self._bfs_path(seed_nodes, target_node)
        else:
            path = self._bfs_path(seed_nodes, target_node)

        return self.dag.graph, path, target_step
