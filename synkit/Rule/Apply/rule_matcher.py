"""rule_matcher.py
=================
Immutable matcher for applying a reaction‑template rule to a reaction SMILES.

Key features
------------
* **Standardization** – canonicalize the input RSMI.
* **Balanced vs. partial matching** – uses stoichiometric balance checks.
* **SMARTS extraction** – extracts SMARTS that reproduce the RSMI.
* **Introspective API** – stores the match on init; exposes `get_result()`, `help()`,
  `__str__()`, and `__repr__()` for inspection.

Quick start
-----------
>>> from synkit.Graph.rule_matcher import RuleMatcher
>>> matcher = RuleMatcher('CCO>>CC=O', some_rule_graph)
>>> smarts, rule = matcher.get_result()
"""

from typing import List, Optional, Tuple, Union

import networkx as nx
from synkit.IO import rsmi_to_graph, rsmi_to_its
from synkit.Chem.Reaction.standardize import Standardize
from synkit.Chem.Reaction.balance_check import BalanceReactionCheck
from synkit.Synthesis.Reactor.syn_reactor import SynReactor

__all__ = ["RuleMatcher"]


class RuleMatcher:
    """Match a reaction SMILES against a transformation‑rule graph and extract
    the SMARTS pattern that reproduces the reaction.

    On initialization, the matcher standardizes the RSMI, builds reactant/product
    graphs, checks balance, and finds the matching SMARTS (stored in `self.result`).

    :param rsmi: Reaction SMILES in 'reactant>>product' format.
    :type rsmi: str
    :param rule: A NetworkX graph encoding the reaction template.
    :type rule: nx.Graph

    :ivar std: SMILES standardizer instance.
    :vartype std: Standardize
    :ivar rsmi: Standardized reaction SMILES.
    :vartype rsmi: str
    :ivar r_graph: Reactant graph extracted from `rsmi`.
    :vartype r_graph: nx.Graph
    :ivar p_graph: Product graph extracted from `rsmi`.
    :vartype p_graph: nx.Graph
    :ivar balanced: True if reaction passes stoichiometric balance check.
    :vartype balanced: bool
    :ivar result: The matching SMARTS and rule graph tuple.
    :vartype result: Tuple[str, nx.Graph]
    """

    def __init__(
        self, rsmi: str, rule: Union[str, nx.Graph], explicit_h: bool = True
    ) -> None:
        """Initialize the matcher by standardizing the RSMI, building graphs,
        checking balance, and computing the match.

        :param rsmi: Reaction SMILES in 'reactant>>product' format.
        :type rsmi: str
        :param rule: Transformation-rule graph.
        :type rule: nx.Graph
        :raises ValueError: If no SMARTS reproduces the RSMI under the
            given rule.
        """
        self.std = Standardize()
        self.rsmi = self.std.fit(rsmi)
        self.r_graph, self.p_graph = rsmi_to_graph(self.rsmi, drop_non_aam=False)
        if isinstance(rule, str):
            rule = rsmi_to_its(rule, core=True)
        self.rule = rule
        self.explicit_h = explicit_h
        self.balanced = BalanceReactionCheck(n_jobs=1).rsmi_balance_check(self.rsmi)

        # Compute and store the match result
        if self.balanced:
            match = self._match_valid()
        else:
            match = self._match_reverse()

        if match is None:
            raise ValueError(
                f"No matching SMARTS for RSMI '{self.rsmi}' with given rule"
            )
        self.result = match

    def get_result(self) -> Tuple[str, nx.Graph]:
        """Return the SMARTS and rule graph found during initialization.

        :returns: A tuple (smarts, rule_graph).
        :rtype: tuple[str, nx.Graph]
        """
        return self.result

    def _match_valid(self) -> Optional[Tuple[str, nx.Graph]]:
        """Attempt a direct (balanced) match of the rule.

        :returns: (smarts, rule) if direct match succeeds; otherwise
            None.
        :rtype: Optional[tuple[str, nx.Graph]]
        """
        reactor = SynReactor(substrate=self.r_graph, template=self.rule)
        for smarts in reactor.smarts_list:
            if self.std.fit(smarts) == self.rsmi:
                return smarts, self.rule
        return None

    def _match_reverse(self) -> Optional[Tuple[str, nx.Graph]]:
        """Attempt a reverse‑balance (partial) match for unbalanced reactions.

        First tries matching on product fragments, then on reactant
        fragments with the template inverted.

        :returns: (smarts, rule) if a partial match is found; otherwise
            None.
        :rtype: Optional[tuple[str, nx.Graph]]
        """
        # Product‑side fragments
        reactor = SynReactor(substrate=self.r_graph, template=self.rule)
        for smarts in reactor.smarts_list:
            std_r = self.std.fit(smarts)
            if self.all_in(
                self.rsmi.split(">>")[1].split("."), std_r.split(">>")[1].split(".")
            ):
                return smarts, self.rule

        # Reactant‑side with inverted template
        reactor = SynReactor(
            substrate=self.p_graph,
            template=self.rule,
            invert=True,
            explicit_h=self.explicit_h,
        )
        for smarts in reactor.smarts_list:
            std_r = self.std.fit(smarts)
            if self.all_in(
                self.rsmi.split(">>")[0].split("."), std_r.split(">>")[0].split(".")
            ):
                return smarts, self.rule

        return None

    @staticmethod
    def all_in(a: List[str], b: List[str]) -> bool:
        """Check if every element of list `a` appears in list `b`.

        :param a: List of elements to test for membership.
        :type a: list[str]
        :param b: List in which to test membership.
        :type b: list[str]
        :returns: True if set(a) is a subset of set(b); otherwise False.
        :rtype: bool
        """
        return set(a).issubset(b)

    def help(self) -> None:
        """Print internal state and candidate SMARTS patterns for debugging.

        :returns: None
        :rtype: NoneType
        """
        print(f"RuleMatcher for RSMI: {self.rsmi!r}")
        print(f"Balanced: {self.balanced}")
        print("Candidate SMARTS patterns:")
        reactor = SynReactor(substrate=self.r_graph, template=self.rule)
        for smarts in reactor.smarts_list:
            print("  ", smarts)

    def __str__(self) -> str:
        """Short string showing the RSMI and balance status.

        :returns: Human‑readable summary.
        :rtype: str
        """
        status = "balanced" if self.balanced else "unbalanced"
        return f"<RuleMatcher {self.rsmi!r} ({status})>"

    def __repr__(self) -> str:
        """Detailed representation including rule size and balance.

        :returns: repr string.
        :rtype: str
        """
        try:
            v, e = self.rule.number_of_nodes(), self.rule.number_of_edges()
        except Exception:
            v = e = 0
        return (
            f"RuleMatcher(rsmi={self.rsmi!r}, "
            f"rule=(|V|={v},|E|={e}), balanced={self.balanced})"
        )
