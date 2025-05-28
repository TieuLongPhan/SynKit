import networkx as nx
from typing import Union, List, Any
from synkit.Graph.Matcher.turbo_iso import TurboISO
from synkit.Graph.Matcher.sing import SING
from synkit.Graph.ITS import its_decompose
from synkit.Graph.Matcher.subgraph_matcher import SubgraphMatch
from synkit.Graph.Hyrogen._misc import h_to_explicit


class RuleFilter:
    """
    Filter a host graph by a list of transformation rules (patterns),
    keeping only those rules whose (decomposed) pattern appears as a
    subgraph in the host.

    :param host_graph: The host graph to search within (will be converted to explicit H).
    :type host_graph: nx.Graph
    :param rules_list: A list of rule objects to filter against.
    :type rules_list: list
    :param invert: If True, use the "modifier" component of each decomposition; otherwise use the normal part.
    :type invert: bool
    :param engine: Matching engine to use: "turbo", "sing", "nx", or "mod".
    :type engine: str
    :param node_label: Node attribute(s) for TurboISO to match on.
    :type node_label: str or list
    :param edge_label: Edge attribute(s) for TurboISO to match on.
    :type edge_label: str or list
    :param distance_threshold: Threshold to skip distance filtering in TurboISO.
    :type distance_threshold: int
    :param sing_max_path: Maximum path length for SING engine.
    :type sing_max_path: int

    :returns: An instance with only the rules that matched.
    :rtype: RuleFilter
    """

    def __init__(
        self,
        host_graph: nx.Graph,
        rules_list: List[Any],
        invert: bool = False,
        engine: str = "turbo",
        node_label: Union[str, List[str]] = ["element", "charge"],
        edge_label: Union[str, List[str]] = "order",
        distance_threshold: int = 5000,
        sing_max_path: int = 3,
    ) -> None:
        """
        Initialize the RuleFilter and perform the filtering pass.

        :param host_graph: The host graph to search within.
        :type host_graph: nx.Graph
        :param rules_list: A list of rule objects to filter against.
        :type rules_list: list
        :param invert: If True, use the "modifier" component of each decomposition.
        :type invert: bool
        :param engine: Matching engine to use.
        :type engine: str
        :param node_label: Node attribute(s) for TurboISO to match on.
        :type node_label: str or list
        :param edge_label: Edge attribute(s) for TurboISO to match on.
        :type edge_label: str or list
        :param distance_threshold: Threshold to skip distance filtering in TurboISO.
        :type distance_threshold: int
        :param sing_max_path: Maximum path length for SING engine.
        :type sing_max_path: int
        """
        # Convert host to explicit hydrogen version once
        self._host = h_to_explicit(host_graph)
        self._rules = list(rules_list)
        self._invert = invert
        self._engine = engine.lower()

        # Decompose patterns via ITS
        self._patterns = [
            its_decompose(r)[1] if self._invert else its_decompose(r)[0]
            for r in self._rules
        ]

        # Instantiate matcher based on engine
        if self._engine == "turbo":
            self._matcher = TurboISO(
                self._host,
                node_label=node_label,
                edge_label=edge_label,
                distance_threshold=distance_threshold,
            )
        elif self._engine == "sing":
            self._matcher = SING(self._host, max_path_length=sing_max_path)
        elif self._engine in ("nx", "mod"):
            self._matcher = SubgraphMatch()
        else:
            raise ValueError(f"Unknown matching engine '{engine}'")

        # Perform filtering and collect matched rules
        self._matches = [self._match(p) for p in self._patterns]
        self._new_rules = [r for r, m in zip(self._rules, self._matches) if m]

    def _match(self, pattern: nx.Graph) -> bool:
        """
        Test whether the given pattern occurs as a subgraph in the host.

        :param pattern: The query graph pattern to match.
        :type pattern: nx.Graph
        :returns: True if pattern is found, False otherwise.
        :rtype: bool
        """
        if self._engine == "turbo":
            return bool(self._matcher.search(pattern, prune=True))
        if self._engine == "sing":
            return bool(self._matcher.search(pattern, prune=True))
        if self._engine == "nx":
            return bool(
                self._matcher.subgraph_isomorphism(
                    pattern, self._host, check_type="mono"
                )
            )
        # "mod"
        return bool(self._matcher.rule_subgraph_morphism(pattern, self._host))

    @property
    def host(self) -> nx.Graph:
        """
        The explicit host graph.

        :returns: The host graph used for matching.
        :rtype: nx.Graph
        """
        return self._host

    @property
    def rules(self) -> List[Any]:
        """
        Original list of rules provided.

        :returns: The list of rules.
        :rtype: list
        """
        return list(self._rules)

    @property
    def patterns(self) -> List[nx.Graph]:
        """
        Decomposed subgraph queries used internally.

        :returns: List of ITS-decomposed query graphs.
        :rtype: list of nx.Graph
        """
        return list(self._patterns)

    @property
    def matches(self) -> List[bool]:
        """
        Boolean list indicating which patterns were found.

        :returns: List of booleans aligned with `patterns`.
        :rtype: list of bool
        """
        return list(self._matches)

    @property
    def new_rules(self) -> List[Any]:
        """
        Subset of rules for which `matches[i]` is True.

        :returns: Filtered list of matching rules.
        :rtype: list
        """
        return list(self._new_rules)

    @property
    def engine(self) -> str:
        """
        Matching engine in use.

        :returns: The name of the engine.
        :rtype: str
        """
        return self._engine

    def __repr__(self) -> str:
        """
        Concise representation of the filter.

        :returns: Representation string.
        :rtype: str
        """
        return (
            f"<RuleFilter engine={self._engine!r} "
            f"patterns={len(self._patterns)} "
            f"matches={sum(self._matches)}/{len(self._matches)}>"
        )

    def __help__(self) -> str:
        """
        Return the class docstring for interactive help.

        :returns: The class documentation.
        :rtype: str
        """
        return self.__doc__
