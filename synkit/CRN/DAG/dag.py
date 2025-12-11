from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Dict, Set, Any, Tuple
import logging
from concurrent.futures import ProcessPoolExecutor
from collections import deque

import networkx as nx

from synkit.Synthesis.Reactor.syn_reactor import SynReactor

logger = logging.getLogger(__name__)

try:
    from rdkit import Chem
except ImportError:
    Chem = None


# --------------------------------------------------------------------------- #
# Worker for parallel rule application (used by DAG and DirectedConstructor)
# --------------------------------------------------------------------------- #


def _apply_rule_worker(
    args: Tuple[int, Any, str, bool, bool, Optional[str]],
) -> Tuple[int, List[str]]:
    """
    Worker function for parallel rule application.

    Parameters
    ----------
    args :
        Tuple (rule_index, rule, substrate, explicit_h, implicit_temp, strategy)

    Returns
    -------
    rule_index :
        Index of the rule in the original rules list.
    products_list :
        List of product mixture SMILES produced by SynReactor.
    """
    idx, rule, substrate, explicit_h, implicit_temp, strategy = args

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
    products_list = list(reactor.smiles_list)
    return idx, products_list


# --------------------------------------------------------------------------- #
# DAG: forward expansion of CRN
# --------------------------------------------------------------------------- #


@dataclass
class DAG:
    """
    Directed reaction-derivation graph builder on top of :class:`SynReactor`.
    """

    rules: List[Any]
    repeats: int = 2
    explicit_h: bool = False
    implicit_temp: bool = False
    strategy: Optional[str] = None
    keep_aam: bool = True

    graph: nx.DiGraph = field(init=False)
    _species_index: Dict[str, int] = field(init=False)  # standardized_smiles -> node_id
    _rule_index: Dict[int, int] = field(init=False)  # rule_idx -> node_id
    _next_node_id: int = field(init=False)
    _smiles_cache: Dict[str, Optional[str]] = field(init=False)  # raw -> standardized
    _next_map_num: int = field(init=False)  # global atom-map counter

    def __post_init__(self) -> None:
        self.graph = nx.DiGraph()
        self._species_index = {}
        self._rule_index = {}
        self._next_node_id = 1
        self._smiles_cache = {}
        self._next_map_num = 1

        # Pre-create rule nodes
        for i, rule in enumerate(self.rules):
            node_id = self._next_node_id
            self._next_node_id += 1
            self._rule_index[i] = node_id
            rule_name = getattr(rule, "name", f"r{i}")
            self.graph.add_node(
                node_id,
                kind="rule",
                rule_index=i,
                rule_name=rule_name,
                rule_repr=repr(rule),
            )

    # ---------------- internal helpers ---------------- #

    def _standardize_smiles(self, smiles: str) -> Optional[str]:
        """
        Standardize a SMILES string using RDKit and cache the result.
        """
        if smiles in self._smiles_cache:
            return self._smiles_cache[smiles]

        if Chem is None:
            logger.warning(
                "RDKit not available; cannot standardize SMILES. "
                "Returning original for %s",
                smiles,
            )
            self._smiles_cache[smiles] = smiles
            return smiles

        if not smiles:
            self._smiles_cache[smiles] = None
            return None

        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            logger.warning("RDKit failed to parse SMILES %s; skipping", smiles)
            self._smiles_cache[smiles] = None
            return None

        try:
            Chem.SanitizeMol(mol)
        except Exception:
            logger.warning("RDKit failed to sanitize SMILES %s; skipping", smiles)
            self._smiles_cache[smiles] = None
            return None

        if self.keep_aam:
            has_map = any(a.GetAtomMapNum() > 0 for a in mol.GetAtoms())
            if not has_map:
                for atom in mol.GetAtoms():
                    atom.SetAtomMapNum(self._next_map_num)
                    self._next_map_num += 1
            std = Chem.MolToSmiles(mol, canonical=True)
        else:
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)
            mol = Chem.RemoveAllHs(mol)
            std = Chem.MolToSmiles(mol, canonical=True)

        self._smiles_cache[smiles] = std
        return std

    def _add_species_node(self, smiles: str) -> Optional[int]:
        """
        Add (or retrieve) a species node corresponding to the given SMILES.
        """
        s = self._standardize_smiles(smiles)
        if s is None:
            return None

        if s in self._species_index:
            return self._species_index[s]

        node_id = self._next_node_id
        self._next_node_id += 1
        self._species_index[s] = node_id

        self.graph.add_node(
            node_id,
            kind="species",
            smiles=s,
        )
        return node_id

    def _get_rule_node(self, idx: int) -> int:
        """
        Return the node id corresponding to the rule with index ``idx``.
        """
        return self._rule_index[idx]

    # ---------------- main API ---------------- #

    def build(
        self,
        seeds: Iterable[str],
        parallel: bool = False,
        max_workers: Optional[int] = None,
    ) -> nx.DiGraph:
        """
        Expand the DAG starting from a pool of seed molecules.
        """
        pool: Set[str] = set()

        # Standardize + add initial species
        for s in seeds:
            sid = self._add_species_node(s)
            if sid is None:
                continue
            smi_std = self.graph.nodes[sid]["smiles"]
            pool.add(smi_std)

        for step in range(1, self.repeats + 1):
            current_pool = set(pool)
            if not current_pool:
                break

            substrate = ".".join(sorted(current_pool))
            logger.debug("Step %d substrate: %s", step, substrate)

            reactant_ids = [self._species_index[s] for s in current_pool]
            any_new_species = False

            # ---------- compute products per rule (optionally in parallel) ----------
            rule_results: Dict[int, List[str]] = {}

            if parallel and len(self.rules) > 1:
                tasks = [
                    (
                        idx,
                        rule,
                        substrate,
                        self.explicit_h,
                        self.implicit_temp,
                        self.strategy,
                    )
                    for idx, rule in enumerate(self.rules)
                ]
                with ProcessPoolExecutor(max_workers=max_workers) as ex:
                    for idx, products_list in ex.map(_apply_rule_worker, tasks):
                        rule_results[idx] = products_list
            else:
                # sequential fallback (original behaviour)
                for idx, rule in enumerate(self.rules):
                    kwargs = dict(
                        smiles=substrate,
                        template=rule,
                        invert=False,
                        explicit_h=self.explicit_h,
                        implicit_temp=self.implicit_temp,
                    )
                    if self.strategy is not None:
                        kwargs["strategy"] = self.strategy

                    reactor = SynReactor.from_smiles(**kwargs)
                    products_list = list(reactor.smiles_list)
                    rule_results[idx] = products_list

            # ---------- integrate results into the graph (single process) ----------
            for idx, products_list in rule_results.items():
                rnode = self._get_rule_node(idx)
                logger.debug(
                    "Step %d, rule %d: %d mappings", step, idx, len(products_list)
                )
                logger.debug("%r", products_list)

                if not products_list:
                    continue

                # reactants → rule (hyper-edge "in" pins)
                for rid in reactant_ids:
                    if not self.graph.has_edge(rid, rnode):
                        self.graph.add_edge(
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
                        pid = self._add_species_node(p_raw)
                        if pid is None:
                            continue

                        p_std = self.graph.nodes[pid]["smiles"]

                        if not self.graph.has_edge(rnode, pid):
                            self.graph.add_edge(
                                rnode,
                                pid,
                                step=step,
                                rule_index=idx,
                                role="product",
                            )
                        if p_std not in pool:
                            pool.add(p_std)
                            any_new_species = True

            if not any_new_species:
                break

        return self.graph

    @property
    def species_nodes(self) -> List[int]:
        """
        Return the list of node ids corresponding to species nodes.
        """
        return [n for n, d in self.graph.nodes(data=True) if d.get("kind") == "species"]

    @property
    def rule_nodes(self) -> List[int]:
        """
        Return the list of node ids corresponding to rule nodes.
        """
        return [n for n, d in self.graph.nodes(data=True) if d.get("kind") == "rule"]


def build_dag_from_smarts(
    rules: List[str],
    seeds: List[str],
    repeats: int = 2,
    explicit_h: bool = False,
    implicit_temp: bool = False,
    strategy: Optional[str] = None,
    keep_aam: bool = True,
    parallel: bool = False,
    max_workers: Optional[int] = None,
) -> nx.DiGraph:
    """
    Convenience wrapper to build a :class:`DAG` from reaction SMARTS strings.
    """
    dag = DAG(
        rules=rules,
        repeats=repeats,
        explicit_h=explicit_h,
        implicit_temp=implicit_temp,
        strategy=strategy,
        keep_aam=keep_aam,
    )
    return dag.build(seeds, parallel=parallel, max_workers=max_workers)


# --------------------------------------------------------------------------- #
# DirectedConstructor: find a pathway from seeds → target
# --------------------------------------------------------------------------- #


@dataclass
class DirectedConstructor:
    """
    Forward-directed pathway constructor on top of :class:`DAG`.

    Returns both a CRN DAG and, if reachable, a node-id path from a seed
    species to the target species, together with the expansion depth.
    """

    rules: List[Any]
    repeats: int = 50  # generous default upper bound
    explicit_h: bool = False
    implicit_temp: bool = False
    strategy: Optional[str] = None
    keep_aam: bool = True

    dag: DAG = field(init=False)
    _nomap_cache: Dict[str, Optional[str]] = field(init=False)

    def __post_init__(self) -> None:
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

        Used only for *matching* the target; independent of DAG.keep_aam.
        """
        if smiles in self._nomap_cache:
            return self._nomap_cache[smiles]

        if Chem is None:
            # fall back: string equality
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

    # --------------- internal: BFS path reconstruction -------------------- #

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

        Returns (graph, path, target_step), where target_step is the
        expansion round in which the target first appeared (or None if
        it was never reached).
        """
        target_norm = self._canonical_nomap(target_smiles)
        if target_norm is None:
            logger.warning(
                "DirectedConstructor: target %s could not be canonicalised.",
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
            logger.debug("[Directed] Step %d substrate: %s", step, substrate)

            reactant_ids = [self.dag._species_index[s] for s in current_pool]
            any_new_species = False

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
                        pid = self.dag._add_species_node(p_raw)
                        if pid is None:
                            continue

                        p_std = self.dag.graph.nodes[pid]["smiles"]

                        if not self.dag.graph.has_edge(rnode, pid):
                            self.dag.graph.add_edge(
                                rnode,
                                pid,
                                step=step,
                                rule_index=idx,
                                role="product",
                            )
                        if p_std not in pool:
                            pool.add(p_std)
                            any_new_species = True

                        if target_norm is not None and target_node is None:
                            p_norm = self._canonical_nomap(p_std)
                            if p_norm == target_norm:
                                target_node = pid
                                target_step = step
                                break
                    if target_node is not None:
                        break
                if target_node is not None:
                    break

            if target_node is not None:
                break
            if not any_new_species:
                break

        if target_node is None:
            logger.info(
                "DirectedConstructor: target not reached within %s repeats "
                "(last step explored = %s).",
                self.dag.repeats,
                last_step,
            )
            return self.dag.graph, None, last_step

        path = self._bfs_path(seed_nodes, target_node)
        return self.dag.graph, path, target_step


# --------------------------------------------------------------------------- #
# Convenience wrapper for directed construction
# --------------------------------------------------------------------------- #


def build_path_to_target(
    rules: List[str],
    seeds: List[str],
    target_smiles: str,
    repeats: Optional[int] = None,
    explicit_h: bool = False,
    implicit_temp: bool = False,
    strategy: Optional[str] = None,
    keep_aam: bool = True,
    parallel: bool = False,
    max_workers: Optional[int] = None,
) -> Tuple[nx.DiGraph, Optional[List[int]], Optional[int]]:
    """
    Convenience wrapper around :class:`DirectedConstructor`.

    :param repeats: Optional maximum number of global expansion rounds. If None,
        a default (50) is used; the actual depth at which the target is found
        is returned as ``target_step``.
    :return: (DAG graph, node-id path, target_step)
    """
    max_repeats = repeats if repeats is not None else 50

    dc = DirectedConstructor(
        rules=rules,
        repeats=max_repeats,
        explicit_h=explicit_h,
        implicit_temp=implicit_temp,
        strategy=strategy,
        keep_aam=keep_aam,
    )
    return dc.build_to_target(
        seeds=seeds,
        target_smiles=target_smiles,
        parallel=parallel,
        max_workers=max_workers,
    )
