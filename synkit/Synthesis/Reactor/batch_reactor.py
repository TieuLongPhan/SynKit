import networkx as nx
from typing import List, Union, Dict, Any, Optional

from synkit.IO import smiles_to_graph
from synkit.Synthesis.Reactor.rule_filter import RuleFilter
from synkit.Synthesis.Reactor.syn_reactor import SynReactor
from synkit.Synthesis.Reactor.mod_reactor import MODReactor


class BatchReactor:
    """
    Apply a collection of pattern-graphs (rules) to a batch of substrates.

    Each data entry can be:
      - a dict (expects substrate under `host_key`)
      - a SMILES string
      - a networkx.Graph

    For 'syn' engine, the substrate must be a networkx.Graph (either directly,
    or extracted from dict via `host_key`).
    For 'mod' engine, the substrate must be a SMILES string (either directly,
    or extracted from dict via `host_key`).

    :param data:            List of substrates (dict, SMILES, or Graph).
    :type data:             list
    :param host_key:        If data entries are dicts, key under which to find the substrate.
    :type host_key:         str or None
    :param react_engine:    Engine for reaction application: 'syn' or 'mod'.
    :type react_engine:     str
    :param filter_engine:   Engine for rule filtering: 'turbo', 'sing', 'nx', 'mod', or None.
    :type filter_engine:    str or None
    :param invert:          Whether to invert the rule set before filtering.
    :type invert:           bool
    :param explicit_h:      Pass-through to SynReactor explicit hydrogens flag.
    :type explicit_h:       bool
    :param implicit_temp:   Pass-through to SynReactor implicit template flag.
    :type implicit_temp:    bool
    :param strategy:        Matching strategy for reactors.
    :type strategy:         str

    :returns:               An instance ready to apply `fit()`.
    :rtype:                 BatchReactor
    """

    def __init__(
        self,
        data: List[Union[Dict[str, Any], str, nx.Graph]],
        host_key: Optional[str] = None,
        react_engine: str = "syn",
        filter_engine: Optional[str] = "turbo",
        invert: bool = False,
        explicit_h: bool = True,
        implicit_temp: bool = False,
        strategy: str = "bt",
    ) -> None:
        """
        Initialize batch reactor configuration.

        :param data:            Batch of substrates to process.
        :type data:             list
        :param host_key:        Key to extract graph/SMILES from dict entries.
        :type host_key:         str or None
        :param react_engine:    Which reactor engine to use ('syn' or 'mod').
        :type react_engine:     str
        :param filter_engine:   RuleFilter engine (or None to skip filtering).
        :type filter_engine:    str or None
        :param invert:          Use inverted rule patterns if True.
        :type invert:           bool
        :param explicit_h:      Use explicit hydrogens in SynReactor.
        :type explicit_h:       bool
        :param implicit_temp:   Use implicit templates in SynReactor.
        :type implicit_temp:    bool
        :param strategy:        Matching strategy identifier.
        :type strategy:         str
        """
        self._data = data
        self._host_key = host_key
        self._react_engine = react_engine.lower()
        if self._react_engine not in ("syn", "mod"):
            raise ValueError(
                f"Unknown react_engine '{react_engine}', use 'syn' or 'mod'."
            )
        self._filter_engine = filter_engine.lower() if filter_engine else None
        self._invert = invert
        self._explicit_h = explicit_h
        self._implicit_temp = implicit_temp
        self._strategy = strategy

    def _get_substrate(
        self, entry: Union[Dict[str, Any], str, nx.Graph]
    ) -> Union[nx.Graph, str]:
        """
        Normalize and validate an entry based on react_engine.

        :param entry:   The substrate entry (dict, SMILES, or Graph).
        :type entry:    dict or str or nx.Graph
        :returns:       networkx.Graph (for 'syn') or SMILES string (for 'mod').
        :rtype:         nx.Graph or str
        """
        # extract from dict if needed
        if isinstance(entry, dict):
            if not self._host_key:
                raise ValueError(
                    "host_key must be set to extract graph/SMILES from dict entries."
                )
            entry = entry.get(self._host_key)

        # now entry is either Graph or str
        if self._react_engine == "syn":
            if not isinstance(entry, nx.Graph):
                raise TypeError("For 'syn' engine, substrate must be a networkx.Graph.")
            return entry
        # 'mod' engine expects SMILES
        if not isinstance(entry, str):
            raise TypeError("For 'mod' engine, substrate must be a SMILES string.")
        # validate SMILES
        graph = smiles_to_graph(entry)
        if graph is None:
            raise ValueError(f"Invalid SMILES string: {entry}")
        return entry

    def _filter_rules(
        self, substrate: Union[nx.Graph, str], rules_list: List[Any]
    ) -> List[Any]:
        """
        Apply rule filtering if configured.

        :param substrate:    Host graph or SMILES to filter against.
        :type substrate:     nx.Graph or str
        :param rules_list:   List of rule patterns.
        :type rules_list:    list
        :returns:            Filtered list of rules.
        :rtype:              list
        """
        if self._filter_engine and self._react_engine == "syn":
            rf = RuleFilter(
                substrate,
                rules_list,
                invert=self._invert,
                engine=self._filter_engine,
            )
            return rf.new_rules
        return rules_list

    def fit(self, rules_list: List[Any]) -> List[List[str]]:
        """
        Apply each rule to every substrate, returning product SMARTS or reaction SMILES.

        :param rules_list:  List of rules (pattern-graphs or objects).
        :type rules_list:   list
        :returns:           Nested list: outputs[i] for substrate i.
        :rtype:             List[List[str]]
        """
        results: List[List[str]] = []
        for entry in self._data:
            substrate = self._get_substrate(entry)
            rules_to_apply = self._filter_rules(substrate, rules_list)

            entry_out: List[str] = []
            for rule in rules_to_apply:
                if self._react_engine == "syn":
                    reactor = SynReactor(
                        substrate,
                        rule,
                        explicit_h=self._explicit_h,
                        implicit_temp=self._implicit_temp,
                        invert=self._invert,
                        strategy=self._strategy,
                    )
                    entry_out.append(reactor.smarts)
                else:
                    reactor = MODReactor(
                        substrate,
                        rule,
                        invert=self._invert,
                        strategy=self._strategy,
                    )
                    entry_out.append(reactor.run().get_reaction_smiles())
            results.append(entry_out)
        return results

    @property
    def data(self) -> List[Union[Dict[str, Any], str, nx.Graph]]:
        """
        Original batch input data.

        :returns: The list of data entries.
        :rtype: list
        """
        return list(self._data)

    @property
    def filter_engine(self) -> Optional[str]:
        """
        The engine used for rule filtering.

        :returns: Name of filter engine or None.
        :rtype: str or None
        """
        return self._filter_engine

    @property
    def react_engine(self) -> str:
        """
        The engine used for reaction application.

        :returns: Name of react engine ('syn' or 'mod').
        :rtype: str
        """
        return self._react_engine

    def __repr__(self) -> str:
        """
        Concise summary of BatchReactor configuration.

        :returns: Representation string.
        :rtype: str
        """
        return (
            f"<BatchReactor react={self._react_engine!r} "
            f"filter={self._filter_engine!r} entries={len(self._data)}>"
        )

    def __help__(self) -> str:
        """
        Return class documentation for interactive help.

        :returns: The class docstring.
        :rtype: str
        """
        return self.__doc__
