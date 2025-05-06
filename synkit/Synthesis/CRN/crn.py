from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Dict, List, Sequence, Union

from synkit.Chem.Reaction.cleanning import Cleanning
from synkit.Chem.utils import (
    count_carbons,
    get_max_fragment,
    process_smiles_list,
)
from synkit.Synthesis.reactor_utils import _remove_reagent
from synkit.Synthesis.Reactor.mod_reactor import MODReactor
from synkit.Synthesis.Reactor.strategy import Strategy

logger = logging.getLogger("CRN")


class CRN:
    """
    Expand an initial pool of molecules through several rounds of rule
    application using **MODReactor** under the hood.

    Public attributes
    -----------------
    initial_smiles : List[str]
        The starting set of molecules.
    n_repeats : int
        Number of expansion rounds requested.
    rounds : List[Tuple[str, List[str]]]
        `[("Round 1", [rxn₁, …]), …]` — kept for backwards compatibility.
    final_smiles : List[str]
        Unique molecule SMILES present after the last round.
    rule_count : int
        How many rules were supplied.

    Public helpers
    --------------
    run() -> CRN
        Rebuild the network from scratch (chainable).
    product_sets -> Dict[str, List[str]]
        Mapping of round‑tag → reaction‑SMILES list.
    get_reaction_smiles() -> Dict[str, List[str]]
        Same as `product_sets` (alias).
    help()
        Human‑readable summary.
    """

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        rule_list: List[Dict[str, Any]],
        smiles_list: Union[str, Sequence[str]],
        *,
        n_repeats: int = 3,
        prune: bool = True,
        strategy: Union[str, Strategy] = Strategy.BACKTRACK,
        verbosity: int = 0,
    ) -> None:
        if not rule_list:
            raise ValueError("rule_list must contain at least one rule dict")

        self.rule_list = rule_list
        self.initial_smiles: List[str] = (
            smiles_list.split(".")
            if isinstance(smiles_list, str)
            else list(smiles_list)
        )
        self.n_repeats = max(1, n_repeats)
        self._prune = prune
        self.strategy = Strategy.from_string(strategy)
        self.verbosity = verbosity

        # populated by _build_crn()
        self.rounds: List[Dict[str, List[str]]] = []
        self.final_smiles: List[str] = []

        self._build_crn()  # auto‑run on construction

    # ------------------------------------------------------------------ API
    def run(self) -> "CRN":
        """Re‑run the expansion pipeline and return *self* for chaining."""
        self._build_crn()
        return self

    # ---------- properties -------------------------------------------------
    @property
    def rule_count(self) -> int:
        return len(self.rule_list)

    @property
    def product_sets(self) -> Dict[str, List[str]]:
        """
        Dict view of the per‑round reaction SMILES.

        Handles both shapes:

        * self.rounds == [{"Round 1": [...]}, {"Round 2": [...]}, ...]
        * self.rounds == [("Round 1", [...]), ("Round 2", [...]), ...]
        """
        if not self.rounds:
            return {}

        # rounds as list[dict[str, list[str]]]
        if isinstance(self.rounds[0], dict):
            out: Dict[str, List[str]] = {}
            for d in self.rounds:  # type: ignore[arg-type]
                out.update(d)
            return out

        # fallback: list[tuple[str, list[str]]]
        return {tag: rxns for tag, rxns in self.rounds}  # type: ignore[misc]

    # Alias kept for tests / external callers
    def get_reaction_smiles(self) -> Dict[str, List[str]]:
        return self.product_sets

    # ---------------------------------------------------------------- help
    def help(self) -> None:
        print("CRN\n---")
        print(" Initial SMILES :", self.initial_smiles)
        print(" Rules          :", self.rule_count)
        print(" Rounds         :", self.n_repeats)
        print(" Final molecules:", len(self.final_smiles))
        print(" Final SMILES   :", self.final_smiles)

    # ---------------------------------------------------------------- repr
    def __repr__(self) -> str:
        return (
            f"<CRN rules={self.rule_count} start={len(self.initial_smiles)} "
            f"rounds={self.n_repeats} final={len(self.final_smiles)}>"
        )

    __str__ = __repr__

    # ============================================================ internals
    def _expand_once(self, smiles: List[str]) -> List[str]:
        """Apply every rule once to the molecule pool and return reaction RSMI."""
        rxn_results: List[str] = []
        smiles_for_mod = process_smiles_list(smiles)

        for rule in self.rule_list:
            reactor = MODReactor(
                smiles_for_mod,
                rule["gml"],
                invert=False,
                strategy=self.strategy,
                verbosity=self.verbosity,
            )
            reactor.run()
            rsmi = reactor.get_reaction_smiles()
            rsmi = Cleanning().clean_smiles(rsmi)
            rsmi = [_remove_reagent(r) for r in rsmi]
            rxn_results.extend(rsmi)

        return rxn_results

    def _update_smiles_pool(
        self,
        current: List[str],
        reactions: List[str],
        *,
        starting: str,
        target: str,
    ) -> List[str]:
        """Merge products from *reactions* into *current* with optional pruning."""
        new: List[str] = []

        for rsmi in reactions:
            products = rsmi.split(">>")[1].split(".")
            if self._prune:
                products = get_max_fragment(products)
                if count_carbons(products) <= count_carbons(target) and count_carbons(
                    products
                ) >= count_carbons(starting):
                    new.append(products)
            else:
                new.extend(products)

        return list(set(current).union(new))

    def _build_crn(self) -> None:
        """Populate `rounds` and `final_smiles`."""
        self.rounds.clear()
        smiles_pool = deepcopy(self.initial_smiles)

        starting = min(smiles_pool, key=count_carbons)
        target = max(smiles_pool, key=count_carbons)

        last_rxns: List[str] = []
        for idx in range(1, self.n_repeats + 1):
            if idx > 1:
                smiles_pool = self._update_smiles_pool(
                    smiles_pool,
                    last_rxns,
                    starting=starting,
                    target=target,
                )

            last_rxns = self._expand_once(smiles_pool)
            self.rounds.append({f"Round {idx}": last_rxns})

        # Final molecules
        self.final_smiles = self._update_smiles_pool(
            smiles_pool, last_rxns, starting=starting, target=target
        )
