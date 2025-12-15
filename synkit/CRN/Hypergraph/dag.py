from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Dict, Set, Any
import logging

import networkx as nx

from synkit.Synthesis.Reactor.syn_reactor import SynReactor

logger = logging.getLogger(__name__)

try:
    from rdkit import Chem
except ImportError:
    Chem = None


@dataclass
class DAG:
    """
    Directed reaction-derivation graph builder on top of :class:`SynReactor`.

    This class expands an initial pool of seed molecules by repeated application
    of reaction rules using :meth:`synkit.Synthesis.Reactor.syn_reactor.SynReactor.from_smiles`.
    The result is stored in :attr:`graph` as a directed bipartite graph with

    * species nodes (``kind='species'``) labelled by a standardized SMILES string, and
    * rule nodes (``kind='rule'``) for each entry in :attr:`rules`.

    Edges always follow the pattern ``species -> rule -> species`` and are annotated
    with the integer ``step`` at which they were created and a ``role`` indicating
    ``"reactant"`` or ``"product"``.

    :param rules: Reaction templates accepted by :meth:`SynReactor.from_smiles`
        (e.g. reaction SMARTS strings or pre-constructed template objects).
        One rule node is created per element in this list.
    :type rules: list[Any]
    :param repeats: Maximum number of global expansion rounds. In each round,
        all rules are applied to the *current* species pool; newly created species
        are only considered in subsequent rounds.
    :type repeats: int
    :param explicit_h: Whether to treat explicit hydrogens as present in
        :class:`SynReactor`. Passed through to :meth:`SynReactor.from_smiles`.
    :type explicit_h: bool
    :param implicit_temp: Whether templates are specified with implicit hydrogens.
        Passed through to :meth:`SynReactor.from_smiles`.
    :type implicit_temp: bool
    :param strategy: Optional SynReactor strategy (e.g. ``"comp"``, ``"bt"``, …).
        If ``None``, the default strategy of :class:`SynReactor` is used.
    :type strategy: str or None
    :param keep_aam: If ``True``, species SMILES are kept atom-mapped. Any molecule
        with no atom maps at all is assigned global map indices using a single
        counter shared across the whole DAG. If ``False``, all atom maps are stripped
        and canonical RDKit SMILES are stored.
    :type keep_aam: bool

    :ivar graph: Directed NetworkX graph storing species and rule nodes and
        species→rule→species edges.
    :vartype graph: :class:`networkx.DiGraph`

    **Examples**

    .. code-block:: python

       from synkit.CRN.dag import DAG

       rules = [
           "[CH2:1]=[O:2].[OH:3][CH:4]([CH:5]=[O:6])[H:7]>>"
           "[CH2:1]=[O:2].[OH:3][CH:4]=[CH:5][O:6][H:7]"
       ]
       seeds = ["C=O", "OCC=O"]

       dag = DAG(rules=rules, repeats=3, explicit_h=True, keep_aam=True)
       G = dag.build(seeds)

       print(dag.species_nodes)
       print(dag.rule_nodes)
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

        - If :attr:`keep_aam` is ``True``:
          - If the molecule has no atom maps at all, assign new map indices using
            the global counter :attr:`_next_map_num`.
          - Existing atom maps are preserved.
          - SMILES is canonicalized with RDKit while keeping map numbers.

        - If :attr:`keep_aam` is ``False``:
          - All atom maps are set to 0 and then SMILES is canonicalized.
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

    def build(self, seeds: Iterable[str]) -> nx.DiGraph:
        """
        Expand the DAG starting from a pool of seed molecules.

        The method performs at most :attr:`repeats` global rounds. In each round:

        * The current pool of species is concatenated into a single mixture
          SMILES (joined with ``"."``).
        * Each rule in :attr:`rules` is applied once to this mixture via
          :meth:`SynReactor.from_smiles`.
        * For each successful application (entry in ``reactor.smiles_list``),
          edges are added:

          - from all current species to the rule node (role ``"reactant"``),
          - from the rule node to each product species (role ``"product"``).

        Newly discovered species are added to the global pool and become part
        of the substrate mixture in subsequent rounds.

        :param seeds: Initial set of molecules to populate the pool. Each entry
            should be a SMILES string parseable by RDKit.
        :type seeds: iterable[str]
        :return: The underlying directed NetworkX graph with species and rule nodes.
        :rtype: :class:`networkx.DiGraph`
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

            for idx, rule in enumerate(self.rules):
                rnode = self._get_rule_node(idx)
                logger.debug("%r", rule)

                kwargs = dict(
                    smiles=substrate,
                    template=rule,  # rule can be SMARTS, GML, SynRule, etc.
                    invert=False,
                    explicit_h=self.explicit_h,
                    implicit_temp=self.implicit_temp,
                )
                if self.strategy is not None:
                    kwargs["strategy"] = self.strategy

                reactor = SynReactor.from_smiles(**kwargs)

                products_list = list(reactor.smiles_list)
                logger.debug("%d mapping(s) discovered", len(products_list))
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
) -> nx.DiGraph:
    """
    Convenience wrapper to build a :class:`DAG` from reaction SMARTS strings.

    :param rules: Reaction SMARTS (or other templates) accepted by
        :meth:`SynReactor.from_smiles`. One rule node is created per element.
    :type rules: list[str]
    :param seeds: Initial pool of molecules given as SMILES strings.
    :type seeds: list[str]
    :param repeats: Maximum number of global expansion rounds.
    :type repeats: int
    :param explicit_h: Passed through to :meth:`SynReactor.from_smiles`.
    :type explicit_h: bool
    :param implicit_temp: Passed through to :meth:`SynReactor.from_smiles`.
    :type implicit_temp: bool
    :param strategy: Optional SynReactor strategy (e.g. ``"comp"``, ``"bt"``).
        If ``None``, the default strategy is used.
    :type strategy: str or None
    :param keep_aam: If ``True``, species SMILES are kept atom-mapped (and unmapped
        molecules receive new global map indices). If ``False``, all maps are
        stripped and canonical SMILES are stored.
    :type keep_aam: bool
    :return: Directed NetworkX graph with species and rule nodes.
    :rtype: :class:`networkx.DiGraph`

    **Examples**

    .. code-block:: python

       from synkit.CRN.dag import build_dag_from_smarts

       rules = [
           "[CH2:1]=[O:2].[OH:3][CH:4]([CH:5]=[O:6])[H:7]>>"
           "[CH2:1]=[O:2].[OH:3][CH:4]=[CH:5][O:6][H:7]",
       ]
       seeds = ["C=O", "OCC=O"]

       G = build_dag_from_smarts(
           rules=rules,
           seeds=seeds,
           repeats=4,
           explicit_h=True,
           implicit_temp=False,
           strategy=None,
           keep_aam=True,
       )

       print(G.number_of_nodes(), G.number_of_edges())
    """
    dag = DAG(
        rules=rules,
        repeats=repeats,
        explicit_h=explicit_h,
        implicit_temp=implicit_temp,
        strategy=strategy,
        keep_aam=keep_aam,
    )
    return dag.build(seeds)
