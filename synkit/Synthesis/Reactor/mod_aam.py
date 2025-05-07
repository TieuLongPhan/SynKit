from __future__ import annotations

"""modaam.py
=========================
A **hardened**, **typed**, and **lazy** wrapper around the MØD toolkit’s
derivation pipeline, with built-in AAM (atom-atom mapping) post-processing
and SMARTS/ITS expansion.

This class exposes the same public API as MODReactor but automatically
runs:

  1. MØD derivation (DG)
  2. AAM normalization
  3. Reagent re-addition (with optional inversion)
  4. ITS-based AAM expansion
  5. SMILES sanitization & deduplication

External API remains compatible with the original MODReactor:

```python
aam = MODAAM("CC.O", "rule.gml", strategy="bt")
smiles = aam.get_reaction_smiles()
"""
from pathlib import Path
from typing import Any, List, Optional, Union

from synkit.IO.dg_to_gml import DGToGML
from synkit.IO.debug import setup_logging
from synkit.Graph.ITS.its_expand import ITSExpand
from synkit.Graph.ITS.normalize_aam import NormalizeAAM
from synkit.Chem.Reaction.standardize import Standardize
from synkit.Chem.Reaction.rsmi_utils import reverse_reaction
from synkit.Synthesis.reactor_utils import _get_unique_aam, _get_reagent, _add_reagent

from synkit.Synthesis.Reactor.strategy import Strategy
from synkit.Synthesis.Reactor.mod_reactor import MODReactor

logger = setup_logging(task_type="MODAAM")


class MODAAM:
    """
    Runs MØD (via MODReactor) then a full AAM/ITS post-processing pipeline.

    Parameters
    ----------
    substrate : Union[str, List[str]]
        Dot-delimited SMILES or list of SMILES for reactants.
    rule_file : Union[str, Path]
        GML rule file path or raw GML/SMARTS string.
    invert : bool, optional
        If True, apply the rule in reverse (default False).
    strategy : Union[str, Strategy], optional
        Matching strategy: ALL, COMPONENT, or BACKTRACK (default BACKTRACK).
    verbosity : int, optional
        Verbosity for MODReactor (default 0).
    print_results : bool, optional
        If True, print the derivation graph (default False).
    check_isomorphic : bool, optional
        If True, deduplicate results by isomorphism (default True).
    """

    def __init__(
        self,
        substrate: Union[str, List[str]],
        rule_file: Union[str, Path],
        *,
        invert: bool = False,
        strategy: Union[str, Strategy] = Strategy.BACKTRACK,
        verbosity: int = 0,
        print_results: bool = False,
        check_isomorphic: bool = True,
    ) -> None:
        # Normalize substrate to list
        self.initial_smiles: List[str] = (
            substrate if isinstance(substrate, list) else substrate.split(".")
        )
        self.rule_file = rule_file
        self.invert = invert
        self.strategy = Strategy.from_string(strategy)
        self.verbosity = verbosity
        self.print_results = print_results
        self.check_isomorphic = check_isomorphic

        # Prepare internal MODReactor
        self._mod_reactor = MODReactor(
            self.initial_smiles,
            rule_file,
            invert=self.invert,
            strategy=self.strategy,
            verbosity=self.verbosity,
            print_results=self.print_results,
        )

        # Placeholders (populated immediately)
        self._dg: Any
        self._aam_smiles: List[str]

        # Run pipeline now
        self._run_pipeline()

    def _run_pipeline(self) -> None:
        """Execute MØD derivation and AAM post-processing."""
        self._mod_reactor.run()
        self._dg = self._mod_reactor.dg
        self._aam_smiles = self._process_aam(self._dg)

    def run(self) -> List[str]:
        """
        Re-run the entire pipeline (MØD + AAM) and return fresh results.
        """
        self._run_pipeline()
        return self._aam_smiles

    @property
    def dg(self) -> Any:
        """The MØD derivation graph (DG)."""
        return self._dg

    @property
    def reaction_smiles(self) -> List[str]:
        """The post-processed reaction SMILES."""
        return self._aam_smiles

    def get_reaction_smiles(self) -> List[str]:
        """Alias for accessing the processed reaction SMILES."""
        return self._aam_smiles

    def get_smarts(self) -> List[str]:
        """Synonym for `.get_reaction_smiles()`."""
        return self._aam_smiles

    @property
    def product_count(self) -> int:
        """Number of product SMILES generated."""
        return len(self._aam_smiles)

    def help(self) -> None:
        """Print a summary of inputs and outputs."""
        print("MODAAM\n-----")
        print(f" Substrate    : {self.initial_smiles}")
        print(f" Rule file    : {self.rule_file}")
        print(f" Inverted     : {self.invert}")
        print(f" Strategy     : {self.strategy}")
        print(f" Products     : {self.product_count}")
        print(" SMILES list  :")
        for smi in self._aam_smiles:
            print("   ", smi)

    def __repr__(self) -> str:
        return (
            f"<MODAAM substrates={self.initial_smiles} "
            f"rule={self.rule_file!r} invert={self.invert} "
            f"strategy={self.strategy} products={self.product_count}>"
        )

    __str__ = __repr__

    # ——— Internal AAM steps ————————————

    def _process_aam(self, dg: Any) -> List[str]:
        raw = self._extract_raw_smiles(dg)
        if not raw:
            return []
        normed = self._normalize_aam(raw)
        if not normed:
            return []
        expanded = self._expand_with_reagents_and_its(normed)
        if not expanded:
            return []
        curated = self._standardize(expanded)
        filtered = self._filter_failures(expanded, curated)
        return self._deduplicate(filtered)

    def _extract_raw_smiles(self, dg: Any) -> List[str]:
        try:
            rxn_map = DGToGML.getReactionSmiles(dg)
            return [vals[0] for vals in rxn_map.values()]
        except Exception as e:
            logger.error("Failed to extract reaction SMILES: %s", e)
            return []

    def _normalize_aam(self, raw: List[str]) -> List[str]:
        norm = NormalizeAAM()
        out: List[str] = []
        for smi in raw:
            try:
                out.append(norm.fit(smi))
            except Exception:
                logger.warning("AAM normalization failed for %s; skipping", smi)
        return out

    def _expand_with_reagents_and_its(self, normalized: List[str]) -> List[str]:
        expander = ITSExpand()
        out: List[str] = []
        for smi in normalized:
            try:
                reagents = _get_reagent(self.initial_smiles, smi)
                rsmi = _add_reagent(smi, reagents)
                if self.invert:
                    rsmi = reverse_reaction(rsmi)
                out.append(expander.expand_aam_with_its(rsmi))
            except Exception:
                logger.warning("ITS expansion failed for %s; skipping", smi)
        return out

    def _standardize(self, expanded: List[str]) -> List[Optional[str]]:
        std = Standardize()
        out: List[Optional[str]] = []
        for smi in expanded:
            try:
                out.append(std.fit(smi))
            except Exception:
                logger.warning("Standardization failed for %s; marking None", smi)
                out.append(None)
        return out

    def _filter_failures(
        self, expanded: List[str], curated: List[Optional[str]]
    ) -> List[str]:
        return [exp for exp, ok in zip(expanded, curated) if ok is not None]

    def _deduplicate(self, smiles: List[str]) -> List[str]:
        if not (self.check_isomorphic and smiles):
            return smiles
        try:
            return _get_unique_aam(smiles)
        except Exception:
            logger.error("Isomorphic deduplication failed; returning unfiltered")
            return smiles


def expand_aam(rsmi: str, rule: str) -> List[str]:
    """
    Expand Atom–Atom Mapping (AAM) for a given reaction SMARTS/SMILES (rsmi)
    using a pre‐sanitized GML rule string.

    Parameters
    ----------
    rsmi : str
        Reaction SMILES/SMARTS in 'reactants>>products' form.
    rule : str
        A GML rule string (already sanitized upstream).

    Returns
    -------
    List[str]
        All reaction SMILES from MODAAM whose standardized form matches `rsmi`.
    """
    std = Standardize()

    # Extract reactant side
    try:
        substrate, _ = rsmi.split(">>", 1)
    except ValueError:
        logger.error("Invalid rsmi format (missing '>>'): %r", rsmi)
        return []

    # Run the AAM reactor
    try:
        reactor = MODAAM(substrate=substrate, rule_file=rule, check_isomorphic=True)
        candidates = reactor.get_reaction_smiles()
    except Exception as e:
        logger.error("MODAAM failed: %s", e)
        return []

    # Standardize once and filter + dedupe
    target = std.fit(rsmi)
    seen = set()
    out: List[str] = []

    for sm in candidates:
        try:
            std_sm = std.fit(sm)
        except Exception as e:
            logger.debug("Skipping unparsable candidate %r: %s", sm, e)
            continue

        if std_sm == target and sm not in seen:
            seen.add(sm)
            out.append(sm)

    return out
