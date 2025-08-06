import logging
from typing import Any, Dict, Iterable, List, Optional
from synkit.Synthesis.Reactor.batch_reactor import BatchReactor

# =============================================================================
# Benchmark subclass
# =============================================================================


class Benchmark(BatchReactor):  # pylint: disable=too-many-arguments
    """
    Extension of BatchReactor to benchmark forward/backward application on reaction-SMILES entries.

    :param data: List of dicts containing reaction SMILES under `reaction_key`.
    :type data: list of dict
    :param reaction_key: Key for reaction-SMILES strings (format 'reactants>>products').
    :type reaction_key: str
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
    :param dedupe: Deduplicate results per substrate.
    :type dedupe: bool
    :param entry_n_jobs: Parallel jobs for substrates.
    :type entry_n_jobs: int
    :param rule_n_jobs: Parallel jobs for rules per substrate.
    :type rule_n_jobs: int
    :param parallel_rules: Enable rule-level parallelism.
    :type parallel_rules: bool
    :param allow_nested: Allow nested parallelism.
    :type allow_nested: bool
    :param cache_enabled: Enable per-process caching.
    :type cache_enabled: bool
    :param cache_maxsize: Max cache entries before eviction.
    :type cache_maxsize: int
    :param logger: Optional custom logger.
    :type logger: logging.Logger or None
    :raises ValueError: If reaction_key entry malformed or SMILES invalid.
    """

    def __init__(
        self,
        data: List[Dict[str, Any]],
        reaction_key: str = "reactions",
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
        Initialize Benchmark with reaction entries.

        Splits each reaction-SMILES into reactant 'r' and product 'p'.
        All other parameters mirror BatchReactor (host_key set to 'r').
        """
        data_prepped = self._get_host(data, reaction_key)
        super().__init__(
            data_prepped,
            host_key="r",
            react_engine=react_engine,
            pre_filter_engine=pre_filter_engine,
            explicit_h=explicit_h,
            implicit_temp=implicit_temp,
            strategy=strategy,
            dedupe=dedupe,
            entry_n_jobs=entry_n_jobs,
            rule_n_jobs=rule_n_jobs,
            parallel_rules=parallel_rules,
            allow_nested=allow_nested,
            cache_enabled=cache_enabled,
            cache_maxsize=cache_maxsize,
            logger=logger,
            enable_logging=enable_logging,
        )

    @staticmethod
    def _get_host(
        data: List[Dict[str, Any]], reaction_key: str = "reactions"
    ) -> List[Dict[str, Any]]:
        """
        Populate 'r' and 'p' SMILES fields for each dict entry.

        :param data: List of dict entries.
        :type data: list of dict
        :param reaction_key: Key for reaction-SMILES string.
        :type reaction_key: str
        :returns: Same list with 'r' and 'p' keys added.
        :rtype: list of dict
        :raises ValueError: If any string lacks '>>'.
        """
        for entry in data:
            rxn = entry.get(reaction_key)
            if not isinstance(rxn, str) or ">>" not in rxn:
                raise ValueError(
                    f"Invalid reaction string for key '{reaction_key}': {rxn!r}"
                )
            r, p = rxn.split(">>", 1)
            entry["r"], entry["p"] = r, p
        return data

    def fit(
        self,
        rules: Iterable[Any],
    ) -> List[Dict[str, Any]]:
        """
        Perform forward (invert=False) on 'r' and backward (invert=True) on 'p'.

        :param rules: Iterable of rule graphs or SMILES.
        :type rules: iterable
        :param reaction_key: Key for reaction-SMILES (unused here).
        :type reaction_key: str
        :returns: List of dicts each with keys 'fw','bw','fw_count','bw_count'.
        :rtype: list of dict
        """
        fw_out = super().fit(rules, invert=False)
        self._host_key = "p"
        bw_out = super().fit(rules, invert=True)
        self._host_key = "r"

        for entry, fw, bw in zip(self._data, fw_out, bw_out):
            entry["fw"] = fw.get(f"{self._engine}_fw", fw.get("out", fw))
            entry["fw_count"] = fw.get("count", len(entry["fw"]))
            entry["bw"] = bw.get(f"{self._engine}_bw", bw.get("out", bw))
            entry["bw_count"] = bw.get("count", len(entry["bw"]))
        return self._data

    def describe(self) -> str:
        """
        Return detailed configuration for Benchmark, including reaction_key.

        :returns: Multi-line summary.
        :rtype: str
        """
        base = super().describe().splitlines()
        return "".join(base + ["Benchmark host_key: 'r' (product host: 'p')"])
