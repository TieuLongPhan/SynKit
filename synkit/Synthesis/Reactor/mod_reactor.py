from __future__ import annotations

"""modreactor.py
=========================
A **hardened** and **typed** re-write of the original ``MODReactor`` wrapper
around the MØD toolkit. The public API remains 100 % compatible but the
internals are now:

* **Safer**  – avoids mutating inputs, validates arguments, logs diagnostics.
* **Faster** – lazy-builds the derivation graph and reaction SMILES only when first accessed.
* **Cleaner** – exhaustive doc-strings, typing everywhere, and single-purpose
  helpers. All heavy lifting lives in private methods prefixed `_`.

External behavior is unchanged:
```python
r = MODReactor("CC.O", "rule.gml", strategy="bt").run()
smiles = r.get_reaction_smiles()
"""
import importlib.util
from pathlib import Path
from collections import Counter
from typing import Any, List, Optional, Union


from synkit.IO.debug import setup_logging
from synkit.IO.data_io import load_gml_as_text
from synkit.IO.chem_converter import smart_to_gml
from synkit.Chem.Molecule.standardize import sanitize_and_canonicalize_smiles

from synkit.Synthesis.Reactor.strategy import Strategy
from synkit.Synthesis.reactor_utils import _deduplicateGraphs


# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
log = setup_logging(task_type="MODReactor")
if importlib.util.find_spec("mod"):
    from mod import smiles, ruleGMLString, DG, config
else:
    ruleGMLString = None
    smiles = None
    DG = None
    config = None
    log.warning("Optional 'mod' package not found")


# ──────────────────────────────────────────────────────────────────────────────
# MODReactor
# ──────────────────────────────────────────────────────────────────────────────
class MODReactor:
    """
    Lazy, ergonomic wrapper around the MØD toolkit’s derivation pipeline.

    Workflow
    --------
    1. Instantiate: give substrate SMILES and a rule GML (path or string).
    2. Call `.run()` to execute the reaction strategy.
    3. Inspect results via `.get_reaction_smiles()`, `.product_sets`, `.get_dg()`, etc.

    Attributes
    ----------
    initial_smiles : List[str]
        List of SMILES strings for reactants (or products, if inverted).
    rule_file : Path
        Filesystem path or raw GML string or raw smart with AAM for the reaction rule.
    invert : bool
        If True, apply the rule in reverse (products → reactants).
    strategy : Strategy
        One of ALL, COMPONENT, or BACKTRACK.
    verbosity : int
        Verbosity level for the MØD DG.apply() call.
    print_results : bool
        If True, prints the derivation graph to stdout.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        substrate: Union[str, List[str]],
        rule_file: Union[str, Path],
        *,
        invert: bool = False,
        strategy: Union[str, Strategy] = Strategy.BACKTRACK,
        verbosity: int = 0,
        print_results: bool = False,
    ) -> None:

        self.initial_smiles: List[str] = (
            substrate if isinstance(substrate, list) else substrate.split(".")
        )
        self.rule_file = rule_file
        self.invert = bool(invert)
        self.strategy = Strategy.from_string(strategy)
        self.verbosity = verbosity
        self.print_results = print_results

        # Prepared artefacts (lazy)
        self._initial_molecules: List[Any] = self._prepare_initial_molecules()
        self._reaction_rule: Any = self._parse_reaction_rule()
        self._dg: Optional[DG] = None
        self._temp_results: Optional[List[List[str]]] = None
        self._reaction_smiles: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Public high‑level API
    # ------------------------------------------------------------------
    def run(self) -> "MODReactor":
        """
        Execute the chosen strategy **once** and return *self* so you can chain:

        ```python
        r = MODReactor(...).run()
        smiles = r.get_reaction_smiles()
        ```
        """
        if self._temp_results is None:
            self._temp_results = self._predict()  # ← may build DG
        return self

    # helpers for outside world ------------------------------------------------
    def get_reaction_smiles(self) -> List[str]:
        """
        Retrieve the reaction SMILES strings (lazy).

        Returns
        -------
        List[str]
            List of reaction SMILES, in “A>>B” format.
        """
        return self.reaction_smiles

    def get_dg(self) -> DG:
        """
        Access the underlying derivation graph.

        Returns
        -------
        DG
            The MØD derivation graph constructed during `.run()`.

        Raises
        ------
        RuntimeError
            If `.run()` has not yet been called.
        """
        if self._dg is None:
            raise RuntimeError("Call `.run()` before accessing the derivation graph.")
        return self._dg

    # ------------------------------------------------------------------
    # Introspection / niceties
    # ------------------------------------------------------------------
    def __str__(self) -> str:
        return (
            f"<MODReactor n_substrate={len(self.initial_smiles)} "
            f"invert={self.invert} strategy={self.strategy.value} "
            f"predictions={self.prediction_count}>"
        )

    __repr__ = __str__

    def help(self) -> None:
        """
        Print a one-page summary of reactor configuration and results.
        """
        print("MODReactor".ljust(60, "─"))
        print(f"Rule file     : {self.rule_file}")
        print(f"Substrate     : {'.'.join(self.initial_smiles)}")
        print(f"Invert rule   : {self.invert}")
        print(f"Strategy      : {self.strategy.value}")
        print(f"Verbosity     : {self.verbosity}")
        print(f"Predictions   : {self.prediction_count}")
        if self._reaction_smiles:
            print(f"First result  : {self._reaction_smiles[0]}")
        print("─" * 60)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------
    @property
    def dg(self) -> Optional[DG]:
        """
        DG or None – cached derivation graph.

        See also
        --------
        get_dg
        """
        return self._dg

    @property
    def product_sets(self) -> List[List[str]]:
        """
        Raw product sets (lists of SMILES) before joining into full reactions.
        """
        return self.temp_results

    @property
    def product_smiles(self) -> List[str]:
        """
        Flattened list of all product SMILES (may contain duplicates).
        """
        return [s for batch in self.temp_results for s in batch]

    @property
    def prediction_count(self) -> int:
        """
        Number of distinct prediction batches generated.
        """
        return len(self._temp_results or [])

    # ------------------------------------------------------------------
    # Internals – lazy properties
    # ------------------------------------------------------------------
    @property
    def temp_results(self) -> List[List[str]]:
        """
        Lazy-loaded raw product lists.

        Returns
        -------
        List[List[str]]
        """
        if self._temp_results is None:
            self._temp_results = self._predict()
        return self._temp_results

    @property
    def reaction_smiles(self) -> List[str]:
        """
        Lazy-loaded reaction SMILES strings of form “A>>B”.

        Returns
        -------
        List[str]
        """
        if self._reaction_smiles is None:
            base = ".".join(self.initial_smiles)
            self._reaction_smiles = self.generate_reaction_smiles(
                self.temp_results, base, invert=self.invert
            )
        return self._reaction_smiles

    # ------------------------------------------------------------------
    # Internals – setup
    # ------------------------------------------------------------------
    def _prepare_initial_molecules(self) -> List[Any]:
        """
        Convert SMILES → MØD molecule objects, dedupe, and sort.

        Returns
        -------
        List[Any]
        """
        mols = [smiles(s, add=False) for s in self.initial_smiles]
        mols = _deduplicateGraphs(mols)
        mols.sort(key=lambda m: getattr(m, "numVertices", 0))
        log.debug("Prepared %d initial molecules", len(mols))
        return mols

    def _parse_reaction_rule(self) -> Any:
        """
        Load or parse the reaction rule from raw GML or file.

        Returns
        -------
        Any
            Rule object from ruleGMLString().
        """
        # First try raw text parse
        try:
            raw = str(self.rule_file)
            rule = ruleGMLString(raw, invert=self.invert, add=False)
            log.debug("Parsed rule from raw text")
            return rule
        except Exception:
            log.debug("Raw parse failed; trying file load", exc_info=True)
        # Second assume this is smart
        try:
            raw = smart_to_gml(self.rule_file)
            rule = ruleGMLString(raw, invert=self.invert, add=False)
            log.debug("Parsed smart from raw text")
            return rule
        except Exception:
            log.debug("Smart parse failed; trying file load", exc_info=True)
        # Then try file
        try:
            gml = load_gml_as_text(self.rule_file)
            rule = ruleGMLString(gml, invert=self.invert, add=False)
            log.debug("Loaded rule from file %s", self.rule_file)
            return rule
        except Exception:
            log.error(
                "Failed to load rule from text or file %s",
                self.rule_file,
                exc_info=True,
            )
            raise

    # ------------------------------------------------------------------
    # Internals – strategy dispatch
    # ------------------------------------------------------------------
    def _predict(self) -> List[List[str]]:
        """
        Dispatch to the appropriate application strategy.

        Returns
        -------
        List[List[str]]
            Raw product batches.
        """
        dispatch = {
            Strategy.ALL: self._apply_all,
            Strategy.COMPONENT: self._apply_components,
            Strategy.BACKTRACK: self._apply_backtrack,
        }
        func = dispatch[self.strategy]
        log.info("Running strategy %s", self.strategy.value)
        results = func()
        if not results:
            log.warning("No predictions generated")
        return results

    # ------------------------------------------------------------------
    # Internals – concrete strategy routines
    # ------------------------------------------------------------------
    def _apply_components(self) -> List[List[str]]:
        """
        Component-aware application: no cross-CC backtracking.

        Returns
        -------
        List[List[str]]
            Product batches.
        """
        self._dg = DG(graphDatabase=self._initial_molecules)
        config.dg.doRuleIsomorphismDuringBinding = False
        self._dg.build().apply(
            self._initial_molecules, self._reaction_rule, verbosity=self.verbosity
        )
        if self.print_results:
            self._dg.print()
        products = []
        for e in self._dg.edges:
            productSmiles = [v.graph.smiles for v in e.targets]
            products.append(productSmiles)
        return products

    def _apply_all(self) -> List[List[str]]:
        """
        Classic “ALL” strategy: VF2 with reagents included.

        Returns
        -------
        List[List[str]]
            Product batches (including unused reagents).
        """
        self._dg = DG(graphDatabase=self._initial_molecules)
        config.dg.doRuleIsomorphismDuringBinding = False
        self._dg.build().apply(
            self._initial_molecules,
            self._reaction_rule,
            verbosity=self.verbosity,
            onlyProper=False,
        )
        if self.print_results:
            self._dg.print()

        products, educts = [], []
        for e in self._dg.edges:
            products.append([v.graph.smiles for v in e.targets])
            educts.append([v.graph.smiles for v in e.sources])

        # re‑attach unused reagents
        base = Counter(sanitize_and_canonicalize_smiles(s) for s in self.initial_smiles)
        for batch, used in zip(products, educts):
            missing = list(
                (
                    base - Counter(sanitize_and_canonicalize_smiles(s) for s in used)
                ).elements()
            )
            batch.extend(missing)
        return products

    def _apply_backtrack(self) -> List[List[str]]:
        """
        BACKTRACK strategy: try COMPONENT, sanitize, else fall back to ALL.

        Returns
        -------
        List[List[str]]
            Sanitized product batches.
        """
        prod = self._apply_components()
        prod = [[sanitize_and_canonicalize_smiles(s) for s in batch] for batch in prod]
        if prod:
            return prod
        log.info("Component strategy returned 0 → falling back to ALL")
        return self._apply_all()

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------
    @staticmethod
    def generate_reaction_smiles(
        temp_results: List[List[str]],
        base_smiles: str,
        *,
        invert: bool = False,
        arrow: str = ">>",
        separator: str = ".",
    ) -> List[str]:
        """
        Build reaction SMILES of the form “A>>B”, where A and B swap
        roles if invert=True.

        Parameters
        ----------
        temp_results : List[List[str]]
            Batches of product (or reactant) SMILES.
        base_smiles : str
            The “other side” of the reaction: the reactant side when
            invert=False, or the product side when invert=True.
        invert : bool
            If False, generates “base_smiles>>joined_batch”;
            if True, generates “joined_batch>>base_smiles”.
        arrow : str
            The reaction arrow to use (default ">>").
        separator : str
            How to join multiple SMILES in a batch (default ".").

        Returns
        -------
        List[str]
            Reaction SMILES strings, one per batch.
        """
        reactions: List[str] = []
        for batch in temp_results:
            if all(x is not None for x in batch):
                joined = separator.join(batch) if batch else ""
                left, right = (joined, base_smiles) if invert else (base_smiles, joined)
                reactions.append(f"{left}{arrow}{right}")
        return reactions
