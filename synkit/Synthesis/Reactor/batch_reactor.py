import logging
from typing import Any, Dict, Iterable, List, Optional, Union

import networkx as nx
from joblib import Parallel, delayed

from synkit.IO import smiles_to_graph, rsmi_to_its
from synkit.Synthesis.Reactor.mod_reactor import MODReactor
from synkit.Synthesis.Reactor.rule_filter import RuleFilter
from synkit.Synthesis.Reactor.syn_reactor import SynReactor

__all__ = ["BatchReactor"]

# =============================================================================
# Low-level rule application
# =============================================================================


def _apply_rule_raw(
    substrate: nx.Graph,
    rule: nx.Graph,
    invert: bool,
    engine: str,
    *,
    strategy: str,
    explicit_h: bool,
    implicit_temp: bool,
) -> List[str]:
    """
    Apply **one** rule graph to a substrate graph using the specified reactor engine.

    :param substrate: Graph representing the substrate molecule.
    :type substrate: networkx.Graph
    :param rule: Graph representing the reaction rule (ITS template).
    :type rule: networkx.Graph
    :param invert: Whether to apply the rule in reverse.
    :type invert: bool
    :param engine: Which reactor engine to use (`"syn"` or `"mod"`).
    :type engine: str
    :param strategy: Matching strategy passed to SynReactor.
    :type strategy: str
    :param explicit_h: Use explicit hydrogens in SynReactor.
    :type explicit_h: bool
    :param implicit_temp: Use implicit templates in SynReactor.
    :type implicit_temp: bool
    :returns: A list of product SMARTS strings or reaction SMILES.
    :rtype: list of str
    """
    try:
        if engine == "syn":
            reactor = SynReactor(
                substrate=substrate,
                template=rule,
                invert=invert,
                strategy=strategy,
                explicit_h=explicit_h,
                implicit_temp=implicit_temp,
            )
            return list(getattr(reactor, "smarts_list", getattr(reactor, "smarts", [])))

        reactor = MODReactor(
            substrate=substrate,
            rule=rule,
            invert=invert,
            strategy=strategy,
        )
        return [reactor.run().get_reaction_smiles()]

    except Exception as exc:  # pragma: no cover
        logging.getLogger(__name__).debug(
            "%s reactor failed (invert=%s): %s", engine, invert, exc
        )
        return []


# =============================================================================
# Picklable rule-applier with small FIFO cache
# =============================================================================


class _RuleApplier:
    """
    Callable wrapper around `_apply_rule_raw` with a FIFO cache.

    The cache is a per-process dict keyed by (substrate_id, rule_id, invert).
    """

    __slots__ = (
        "_engine",
        "_strategy",
        "_explicit_h",
        "_implicit_temp",
        "_cache",
        "_cache_max",
    )

    def __init__(
        self,
        engine: str,
        *,
        strategy: str,
        explicit_h: bool,
        implicit_temp: bool,
        cache_enabled: bool,
        cache_maxsize: int,
    ) -> None:
        """
        Initialize the rule applier.

        :param engine: Reactor engine, 'syn' or 'mod'.
        :param strategy: Matching strategy for SynReactor.
        :param explicit_h: Pass explicit hydrogens to SynReactor.
        :param implicit_temp: Pass implicit template flag to SynReactor.
        :param cache_enabled: Whether to enable in-process caching.
        :param cache_maxsize: Maximum cache entries before eviction.
        """
        self._engine = engine
        self._strategy = strategy
        self._explicit_h = explicit_h
        self._implicit_temp = implicit_temp
        self._cache = {} if cache_enabled else None  # type: ignore[var-annotated]
        self._cache_max = cache_maxsize

    def _execute(
        self,
        substrate: nx.Graph,
        rule: nx.Graph,
        inv: bool,
    ) -> List[str]:
        """Directly call `_apply_rule_raw` without caching."""
        return _apply_rule_raw(
            substrate,
            rule,
            inv,
            self._engine,
            strategy=self._strategy,
            explicit_h=self._explicit_h,
            implicit_temp=self._implicit_temp,
        )

    def __call__(
        self,
        substrate: nx.Graph,
        rule: nx.Graph,
        inv: bool,
    ) -> List[str]:
        """
        Apply a rule to a substrate, using the cache if enabled.

        :param substrate: Substrate graph.
        :param rule: Rule graph.
        :param inv: Inversion flag.
        :returns: List of reaction outputs.
        """
        if self._cache is None:
            return self._execute(substrate, rule, inv)

        key = (id(substrate), id(rule), inv)
        if key in self._cache:
            return self._cache[key]

        res = self._execute(substrate, rule, inv)
        if len(self._cache) >= self._cache_max:
            self._cache.pop(next(iter(self._cache)))  # FIFO eviction
        self._cache[key] = res
        return res


# =============================================================================
# Helper utilities
# =============================================================================


def _dedupe(items: Iterable[Any]) -> List[Any]:
    """Remove duplicates while preserving order."""
    seen: set[Any] = set()
    out: List[Any] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# =============================================================================
# Public API
# =============================================================================


class BatchReactor:
    """
    Parallel, cache-aware batch application of reaction rules to SMILES substrates.

    :param data: List of SMILES strings or dicts containing SMILES under `host_key`.
    :type data: list of str or dict
    :param host_key: Key to extract SMILES from dict entries (optional).
    :type host_key: str or None
    :param react_engine: Reactor engine: 'syn' or 'mod'.
    :type react_engine: str
    :param pre_filter_engine: Pre-filtering engine for rules (None to skip).
    :type pre_filter_engine: str or None
    :param explicit_h: Use explicit hydrogens in SynReactor.
    :type explicit_h: bool
    :param implicit_temp: Use implicit templates in SynReactor.
    :type implicit_temp: bool
    :param strategy: Matching strategy for SynReactor.
    :type strategy: str
    :param dedupe: Deduplicate results per-substrate.
    :type dedupe: bool
    :param entry_n_jobs: Number of parallel jobs for substrates.
    :type entry_n_jobs: int
    :param rule_n_jobs: Number of parallel jobs for rules per substrate.
    :type rule_n_jobs: int
    :param parallel_rules: Enable parallelism over rules.
    :type parallel_rules: bool
    :param allow_nested: Allow nested parallelism.
    :type allow_nested: bool
    :param cache_enabled: Enable in-process per-rule caching.
    :type cache_enabled: bool
    :param cache_maxsize: Max entries in per-process cache.
    :type cache_maxsize: int
    :param logger: Optional custom logger.
    :type logger: logging.Logger or None

    :raises ValueError: If react_engine is invalid or SMILES/rule conversion fails.
    """

    def __init__(
        self,
        data: List[Union[str, Dict[str, Any]]],
        host_key: Optional[str] = None,
        *,
        react_engine: str = "syn",
        pre_filter_engine: Optional[str] = None,
        explicit_h: bool = True,
        implicit_temp: bool = False,
        strategy: str = "bt",
        dedupe: bool = True,
        entry_n_jobs: int = 1,
        rule_n_jobs: int = 1,
        parallel_rules: bool = False,
        allow_nested: bool = False,
        cache_enabled: bool = True,
        cache_maxsize: int = 32768,
        logger: Optional[logging.Logger] = None,
        enable_logging: bool = True,
    ) -> None:
        """
        Initialize the BatchReactor.

        See class docstring for parameter details.
        """
        if react_engine.lower() not in {"syn", "mod"}:
            raise ValueError("react_engine must be 'syn' or 'mod'")

        self._data = list(data)
        self._host_key = host_key
        self._engine = react_engine.lower()
        self._pre_filter = pre_filter_engine.lower() if pre_filter_engine else None
        self._explicit_h = explicit_h
        self._implicit_temp = implicit_temp
        self._strategy = strategy
        self._dedupe = dedupe
        self._entry_jobs = max(1, entry_n_jobs)
        self._rule_jobs = max(1, rule_n_jobs)
        self._parallel_rules = parallel_rules
        self._allow_nested = allow_nested
        self._log = logger or logging.getLogger(__name__)
        if enable_logging is False:
            logging.disable(logging.CRITICAL)

        self._apply_rule = _RuleApplier(
            self._engine,
            strategy=strategy,
            explicit_h=explicit_h,
            implicit_temp=implicit_temp,
            cache_enabled=cache_enabled,
            cache_maxsize=cache_maxsize,
        )

    def help(self) -> str:
        """
        Return usage examples and API description.

        :returns: Multi-line help text.
        :rtype: str
        """
        return (
            "BatchReactor(data, host_key=None, react_engine='syn', pre_filter_engine=None, "
            "explicit_h=True, implicit_temp=False, strategy='bt', dedupe=True, "
            "entry_n_jobs=1, rule_n_jobs=1, parallel_rules=False, allow_nested=False, "
            "cache_enabled=True, cache_maxsize=32768, logger=None)" + "\n"
            "Use .fit(rules, invert=False) to apply reaction rules to your SMILES batch."
        )

    def describe(self) -> str:
        """
        Return a configuration summary.

        :returns: Human-readable settings overview.
        :rtype: str
        """
        lines = [
            "BatchReactor configuration:",
            f"  entries         : {len(self._data)}",
            f"  engine          : {self._engine}",
            f"  pre_filter      : {self._pre_filter}",
            f"  explicit_h      : {self._explicit_h}",
            f"  implicit_temp   : {self._implicit_temp}",
            f"  strategy        : {self._strategy}",
            f"  dedupe          : {self._dedupe}",
            f"  entry_n_jobs    : {self._entry_jobs}",
            f"  rule_n_jobs     : {self._rule_jobs}",
            f"  parallel_rules  : {self._parallel_rules}",
            f"  allow_nested    : {self._allow_nested}",
            f"  cache_enabled   : {self._apply_rule._cache is not None}",
            f"  cache_maxsize   : {self._apply_rule._cache_max}",
        ]
        return "\n".join(lines)

    def __len__(self) -> int:
        """Return the number of substrates in the batch."""
        return len(self._data)

    def __iter__(self):
        """Yield each entry from the batch data."""
        yield from self._data

    def __getitem__(self, idx: int) -> Union[str, Dict[str, Any]]:
        """Get the data entry at index `idx`."""
        return self._data[idx]

    def __repr__(self) -> str:
        """Concise summary of the BatchReactor instance."""
        return (
            f"<BatchReactor entries={len(self._data)} engine={self._engine} "
            f"pre_filter={self._pre_filter} entry_jobs={self._entry_jobs} "
            f"rule_jobs={self._rule_jobs} parallel_rules={self._parallel_rules}>"
        )

    def fit(
        self,
        rules: Iterable[Any],
        *,
        invert: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Apply reaction rules to each substrate in the batch.

        :param rules: Iterable of rule graphs or SMILES strings.
        :type rules: iterable
        :param invert: Whether to apply rules in reverse direction.
        :type invert: bool
        :returns: A list of dicts, each with:
            - "out": list of product SMARTS/reaction SMILES
            - "count": number of outputs
        :rtype: list of dict
        """
        rule_graphs = self._ensure_graph_rules(rules)
        direction = "bw" if invert else "fw"

        def worker(entry: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
            g = self._to_graph(entry)
            filtered = (
                RuleFilter(g, rule_graphs, invert=invert, engine=self._pre_filter).new_rules  # type: ignore[arg-type]
                if self._pre_filter and self._engine == "syn"
                else rule_graphs
            )
            out = self._apply_bulk(g, filtered, invert)
            return {f"{self._engine}_{direction}": out, "count": len(out)}

        if self._entry_jobs == 1:
            return [worker(e) for e in self._data]

        return Parallel(
            n_jobs=self._entry_jobs,
            backend="loky",
            prefer="processes",
        )(delayed(worker)(e) for e in self._data)

    # ------------------------------------------------------------------
    # Additional helpers
    # ------------------------------------------------------------------

    def _to_graph(self, entry: Union[str, Dict[str, Any]]) -> nx.Graph:
        """
        Convert a SMILES string or dict entry into a networkx Graph.

        :param entry: SMILES or dict holding SMILES under host_key.
        :type entry: str or dict
        :returns: Molecule graph.
        :rtype: networkx.Graph
        :raises KeyError: if host_key is missing for dict.
        :raises TypeError: if entry is not str or dict.
        :raises ValueError: if SMILES conversion fails.
        """
        if isinstance(entry, dict):
            if self._host_key is None:
                raise KeyError("host_key missing for dict entry")
            entry = entry[self._host_key]
        if not isinstance(entry, str):
            raise TypeError("Each entry must be a SMILES string or dict containing one")
        g = smiles_to_graph(entry, drop_non_aam=False, use_index_as_atom_map=False)
        if g is None:
            raise ValueError(f"Invalid SMILES: {entry}")
        return g

    @staticmethod
    def _ensure_graph_rules(rules: Iterable[Any]) -> List[nx.Graph]:
        """
        Convert an iterable of SMILES or graph rules into graph objects.

        :param rules: Iterable of rule SMILES or networkx.Graph.
        :type rules: iterable
        :returns: List of rule graphs.
        :rtype: list of networkx.Graph
        :raises ValueError: if a rule SMILES is invalid.
        :raises TypeError: for unsupported rule types.
        """
        out: List[nx.Graph] = []
        for r in rules:
            if isinstance(r, str):
                g = rsmi_to_its(r, core=True)
                if g is None:
                    raise ValueError(f"Invalid rule SMILES: {r}")
                out.append(g)
            elif isinstance(r, nx.Graph):
                out.append(r)
            else:
                raise TypeError("Rules must be str or nx.Graph")
        return out

    def _apply_bulk(
        self, g: nx.Graph, rules: List[nx.Graph], invert: bool
    ) -> List[str]:
        """
        Apply a list of rule graphs to a substrate graph.

        :param g: Substrate graph.
        :type g: networkx.Graph
        :param rules: List of rule graphs.
        :type rules: list of networkx.Graph
        :param invert: Direction flag.
        :type invert: bool
        :returns: Flattened list of reaction outputs.
        :rtype: list of str
        """
        jobs = 1
        if (
            self._parallel_rules
            and self._rule_jobs > 1
            and (self._allow_nested or self._entry_jobs == 1)
        ):
            jobs = self._rule_jobs
        if jobs == 1:
            nested = [self._apply_rule(g, r, invert) for r in rules]
        else:
            nested = Parallel(n_jobs=jobs, backend="loky", prefer="processes")(
                delayed(self._apply_rule)(g, r, invert) for r in rules
            )
        flat = [x for sub in nested for x in sub]
        return _dedupe(flat) if self._dedupe else flat
