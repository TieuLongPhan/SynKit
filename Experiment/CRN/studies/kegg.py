#!/usr/bin/env python
"""Analyse cached KEGG metabolic modules end to end."""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import argparse

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from synkit.CRN.Query.cache import load_kegg_cache as read_kegg_cache, build_cached_kegg_crn  # noqa: E402
from synkit.CRN.Petrinet.persistence import siphon_persistence_details  # noqa: E402
from synkit.CRN.Petrinet.siphon_search import SiphonSearchLimit  # noqa: E402
from synkit.CRN.Petrinet.structure import find_siphons  # noqa: E402
from synkit.CRN.Props.deficiency import crnt_summary  # noqa: E402
from synkit.CRN.Props.helper import _species_and_rule_order  # noqa: E402
from synkit.CRN.Props.stoich import conserved_moieties  # noqa: E402
from synkit.CRN.Query.to_syncrn import CURRENCY_COMPOUNDS  # noqa: E402
from synkit.CRN.Structure.syncrn import SynCRN  # noqa: E402
from Experiment.CRN.tools.common import environment, timed, write_report, native_experiment  # noqa: E402

__all__ = [
    "CASE_STUDY_MODULES",
    "KeggModuleAnalysis",
    "analyze_kegg_module",
    "build_kegg_crn",
    "kegg_case_study_report",
    "load_kegg_cache",
    "module_ids",
]

#: Cached KEGG modules used by the case study, in pathway order.
CASE_STUDY_MODULES: Tuple[str, ...] = ("M00001", "M00307", "M00009", "M00004")

_CACHE_FILE = "kegg_modules.json"
_cache: Optional[Dict[str, Any]] = None


def load_kegg_cache() -> Dict[str, Any]:
    """Load the cached KEGG modules owned by this experiment.

    :return:
        Cache payload with ``source``, ``retrieved`` and ``modules``.
    :rtype: Dict[str, Any]

    .. rubric:: Example

    .. code-block:: python

        cache = load_kegg_cache()
        print(sorted(cache["modules"]))
    """
    global _cache
    if _cache is None:
        _cache = read_kegg_cache(Path(__file__).resolve().parents[1] / "data" / _CACHE_FILE)
    return _cache


def module_ids() -> List[str]:
    """Return the module ids available in the cache.

    :return:
        Sorted KEGG module identifiers.
    :rtype: List[str]
    """
    return sorted(load_kegg_cache()["modules"])


def build_kegg_crn(
    module_id: str,
    *,
    drop_currency: bool = True,
    expand_reversible: bool = True,
) -> SynCRN:
    """Build the network for one cached KEGG module.

    :param module_id:
        KEGG module identifier such as ``"M00001"``.
    :type module_id: str

    :param drop_currency:
        Whether currency metabolites are removed; see the module docstring.
    :type drop_currency: bool

    :param expand_reversible:
        Whether reversible KEGG equations become two directed reactions.
    :type expand_reversible: bool

    :return:
        Network for the module, with compound names as species labels.
    :rtype: SynCRN

    :raises KeyError:
        If the module is not in the cache.

    .. rubric:: Example

    .. code-block:: python

        crn = build_kegg_crn("M00001")
        print(crn.n_species, crn.n_reactions)
    """
    return build_cached_kegg_crn(
        load_kegg_cache(),
        module_id,
        expand_reversible=expand_reversible,
        drop_compounds=CURRENCY_COMPOUNDS if drop_currency else (),
    )


@dataclass
class KeggModuleAnalysis:
    """Structural report for one KEGG module.

    :param module_id:
        KEGG module identifier.
    :type module_id: str

    :param module_name:
        Human-readable pathway name.
    :type module_name: str

    :param n_species:
        Number of metabolites after currency removal.
    :type n_species: int

    :param n_reactions:
        Number of directed reactions.
    :type n_reactions: int

    :param n_complexes:
        Number of distinct complexes.
    :type n_complexes: int

    :param n_linkage_classes:
        Number of linkage classes.
    :type n_linkage_classes: int

    :param rank:
        Rank of the stoichiometric matrix.
    :type rank: int

    :param deficiency:
        Network deficiency.
    :type deficiency: int

    :param weakly_reversible:
        Whether the network is weakly reversible.
    :type weakly_reversible: bool

    :param moieties:
        Conserved moieties as lists of metabolite names.
    :type moieties: List[List[str]]

    :param siphons:
        Minimal siphons as lists of metabolite names.
    :type siphons: List[List[str]]

    :param persistent:
        Structural persistence verdict.
    :type persistent: Optional[bool]

    :param uncovered_siphons:
        Minimal siphons containing no conserved moiety; these are the ones that
        break persistence, and the biologically informative part of the verdict.
    :type uncovered_siphons: List[List[str]]
    """

    module_id: str
    module_name: str
    n_species: int
    n_reactions: int
    n_complexes: int
    n_linkage_classes: int
    rank: int
    deficiency: int
    weakly_reversible: bool
    moieties: List[List[str]] = field(default_factory=list)
    siphons: List[List[str]] = field(default_factory=list)
    persistent: Optional[bool] = None
    uncovered_siphons: List[List[str]] = field(default_factory=list)
    siphons_complete: bool = True
    siphon_termination_reason: str = "complete"

    def to_dict(self) -> Dict[str, Any]:
        """Return the report as a serializable mapping.

        :return:
            Report contents.
        :rtype: Dict[str, Any]
        """
        return {
            "module_id": self.module_id,
            "module_name": self.module_name,
            "n_species": self.n_species,
            "n_reactions": self.n_reactions,
            "n_complexes": self.n_complexes,
            "n_linkage_classes": self.n_linkage_classes,
            "rank": self.rank,
            "deficiency": self.deficiency,
            "weakly_reversible": self.weakly_reversible,
            "moieties": [list(m) for m in self.moieties],
            "siphons": [list(s) for s in self.siphons],
            "persistent": self.persistent,
            "uncovered_siphons": [list(s) for s in self.uncovered_siphons],
            "siphons_complete": self.siphons_complete,
            "siphon_termination_reason": self.siphon_termination_reason,
        }

    def __str__(self) -> str:
        """Return a readable multi-line report.

        :return:
            Human-readable report.
        :rtype: str
        """
        lines = [
            f"{self.module_id} — {self.module_name}",
            f"  {self.n_species} metabolites, {self.n_reactions} reactions, "
            f"{self.n_complexes} complexes, {self.n_linkage_classes} linkage classes",
            f"  rank s={self.rank}, deficiency delta={self.deficiency}, "
            f"weakly reversible: {self.weakly_reversible}",
            f"  conserved moieties ({len(self.moieties)}):",
        ]
        for moiety in self.moieties:
            lines.append(f"    {' + '.join(moiety)}")
        lines.append(f"  minimal siphons ({len(self.siphons)}):")
        if not self.siphons_complete:
            lines.append(f"    search incomplete: {self.siphon_termination_reason}")
        for siphon in self.siphons[:5]:
            lines.append(f"    {{{', '.join(siphon)}}}")
        if len(self.siphons) > 5:
            lines.append(f"    ... and {len(self.siphons) - 5} more")
        lines.append(f"  structurally persistent: {self.persistent}")
        if self.uncovered_siphons:
            lines.append(
                f"  siphons with no conserved moiety "
                f"({len(self.uncovered_siphons)}): these metabolite sets cannot "
                f"be regenerated by the module alone"
            )
        return "\n".join(lines)


def _label_lookup(crn: SynCRN) -> Dict[str, str]:
    """Return a mapping from any species handle to its display label.

    Siphon searches report species by whichever handle the Petri-net view uses,
    so both the internal id and the source node id are mapped.

    :param crn: Network to inspect.
    :type crn: SynCRN
    :return: Mapping from handle to label.
    :rtype: Dict[str, str]
    """
    lookup: Dict[str, str] = {}
    for sid, record in crn.species.items():
        label = record.label or sid
        lookup[str(sid)] = label
        lookup[str(record.source_node_id)] = label
        lookup[label] = label
    return lookup


def _species_labels(crn: SynCRN) -> List[str]:
    """Return species labels in stoichiometric-matrix row order.

    Row order comes from the reconstructed digraph, whose node ids are the
    *source* ids (KEGG compound ids here), not the internal ``s_1`` ids, so the
    labels are resolved through :func:`_label_lookup` rather than by indexing
    ``crn.species``.

    :param crn: Network to inspect.
    :type crn: SynCRN
    :return: Labels aligned with the rows of ``S``.
    :rtype: List[str]
    """
    order, _, _, _ = _species_and_rule_order(crn)
    lookup = _label_lookup(crn)
    return [lookup.get(str(node), str(node)) for node in order]


def analyze_kegg_module(
    module_id: str,
    *,
    drop_currency: bool = True,
    expand_reversible: bool = True,
    max_siphon_size: Optional[int] = None,
) -> KeggModuleAnalysis:
    """Run the full structural analysis on one cached KEGG module.

    :param module_id:
        KEGG module identifier such as ``"M00001"``.
    :type module_id: str

    :param drop_currency:
        Whether currency metabolites are removed before analysis.
    :type drop_currency: bool

    :param expand_reversible:
        Whether reversible KEGG equations become two directed reactions.
    :type expand_reversible: bool

    :param max_siphon_size:
        Optional ceiling on siphon size.
    :type max_siphon_size: Optional[int]

    :return:
        Structural report for the module.
    :rtype: KeggModuleAnalysis

    .. rubric:: Example

    .. code-block:: python

        result = analyze_kegg_module("M00001")
        print(result)
    """
    entry = load_kegg_cache()["modules"][module_id]
    crn = build_kegg_crn(
        module_id,
        drop_currency=drop_currency,
        expand_reversible=expand_reversible,
    )

    report = crnt_summary(crn)
    labels = _species_labels(crn)
    lookup = _label_lookup(crn)

    raw_moieties = conserved_moieties(crn)
    moieties = [sorted(label for label, coeff in zip(labels, vector) if coeff) for vector in raw_moieties]
    moiety_sets = [frozenset(m) for m in moieties]

    siphons_complete, siphon_reason = True, "complete"
    try:
        all_siphons = find_siphons(crn, names="id", strict_limit=True)
    except SiphonSearchLimit as exc:
        # Keep the report usable, but never infer persistence from a truncated
        # enumeration. The list API cannot promise globally minimal partials.
        all_siphons = []
        siphons_complete = False
        result = getattr(exc, "result", None)
        siphon_reason = result.termination_reason if result is not None else "node_limit"
    siphons_raw = [s for s in all_siphons if max_siphon_size is None or len(s) <= max_siphon_size]
    siphons = [sorted(lookup.get(str(item), str(item)) for item in siphon) for siphon in siphons_raw]

    uncovered = [siphon for siphon in siphons if not any(moiety <= frozenset(siphon) for moiety in moiety_sets)]

    try:
        import numpy as np

        persistent: Optional[bool] = (
            siphon_persistence_details(
                crn,
                max_siphon_size=max_siphon_size,
                p_semiflows=np.asarray(raw_moieties).T,
                _complete_siphons=all_siphons,
            ).persistence_ok
            if siphons_complete
            else None
        )
    except Exception:  # pragma: no cover - defensive
        persistent = None

    return KeggModuleAnalysis(
        module_id=module_id,
        module_name=entry.get("name", module_id),
        n_species=report.n_species,
        n_reactions=report.n_reactions,
        n_complexes=report.n_complexes,
        n_linkage_classes=report.n_linkage_classes,
        rank=report.rank,
        deficiency=report.deficiency,
        weakly_reversible=report.is_weakly_reversible,
        moieties=moieties,
        siphons=siphons,
        persistent=persistent,
        uncovered_siphons=uncovered,
        siphons_complete=siphons_complete,
        siphon_termination_reason=siphon_reason,
    )


#: The canonical glycolytic route through M00001, as KEGG reaction ids.
#:
#: One firing of each: hexokinase, phosphoglucose isomerase, phosphofructokinase,
#: aldolase, triose-phosphate isomerase, GAPDH, phosphoglycerate kinase,
#: phosphoglycerate mutase, enolase, pyruvate kinase. The aldolase step produces
#: both trioses, so the lower half runs twice per glucose; the flux below fires
#: it twice, which is what makes it stoichiometrically coherent.
GLYCOLYSIS_FLUX: Dict[str, int] = {
    "R01786": 1,  # glucose + ATP -> G6P
    "R13199": 1,  # G6P -> F6P
    "R00756": 1,  # F6P + ATP -> FBP
    "R01068": 1,  # FBP -> DHAP + GAP
    "R01015": 1,  # DHAP -> GAP
    "R01061": 2,  # GAP -> 1,3-BPG
    "R01512": 2,  # 1,3-BPG -> 3PG
    "R01518": 2,  # 3PG -> 2PG
    "R00658": 2,  # 2PG -> PEP
    "R00200": 2,  # PEP -> pyruvate
}


def check_glycolysis_flux(
    flux: Optional[Dict[str, int]] = None,
    *,
    initial_glucose: int = 1,
    max_states: int = 100_000,
) -> Dict[str, Any]:
    """Check whether the canonical glycolytic flux is firable from glucose.

    Stoichiometric feasibility is a linear-algebra question; *realizability* is
    stronger — it asks whether the reactions can be fired one at a time, in some
    order, without any metabolite count ever going negative. This runs the
    Petri-net realizability check on the glycolysis module and returns the
    firing sequence as a certificate when it succeeds.

    :param flux:
        Reaction firing counts keyed by KEGG reaction id. Defaults to
        :data:`GLYCOLYSIS_FLUX`.
    :type flux: Optional[Dict[str, int]]

    :param initial_glucose:
        Tokens of alpha-D-glucose in the initial marking.
    :type initial_glucose: int

    :param max_states:
        Ceiling on explored Petri-net markings.
    :type max_states: int

    :return:
        Mapping with ``realizable``, ``certificate`` (the firing sequence, or
        ``None``), ``flux`` and ``initial_marking``.
    :rtype: Dict[str, Any]

    .. rubric:: Example

    .. code-block:: python

        result = check_glycolysis_flux()
        print(result["realizable"], len(result["certificate"] or []))
    """
    from synkit.CRN.Pathway.realizability import PathwayRealizability

    requested = dict(GLYCOLYSIS_FLUX if flux is None else flux)
    crn = build_kegg_crn("M00001")

    checker = PathwayRealizability().load_syncrn_and_flow(
        crn,
        flow=requested,
        initial_marking={"alpha-D-Glucose": initial_glucose},
        species="label",
        reaction="label",
    )
    checker.build_petri_net_from_flow()
    outcome = checker.realizability_result(max_states=max_states)

    return {
        **outcome.to_dict(),
        "flux": requested,
        "initial_marking": {"alpha-D-Glucose": initial_glucose},
    }


def kegg_case_study_report(
    modules: Sequence[str] = CASE_STUDY_MODULES,
    *,
    drop_currency: bool = True,
    fmt: str = "text",
) -> str:
    """Analyse every case-study module and render the result.

    :param modules:
        Module identifiers to analyse.
    :type modules: Sequence[str]

    :param drop_currency:
        Whether currency metabolites are removed before analysis.
    :type drop_currency: bool

    :param fmt:
        ``"text"`` for the per-module narrative report, or ``"markdown"`` for a
        one-row-per-module summary table.
    :type fmt: str

    :return:
        Rendered report.
    :rtype: str

    :raises ValueError:
        If ``fmt`` is not supported.

    .. rubric:: Example

    .. code-block:: python

        print(kegg_case_study_report(fmt="markdown"))
    """
    if fmt not in {"text", "markdown"}:
        raise ValueError(f"fmt must be 'text' or 'markdown', got {fmt!r}")

    results = [analyze_kegg_module(module_id, drop_currency=drop_currency) for module_id in modules]

    if fmt == "text":
        return "\n\n".join(str(result) for result in results)

    headers = [
        "module",
        "pathway",
        "metabolites",
        "reactions",
        "n",
        "l",
        "s",
        "delta",
        "WR",
        "moieties",
        "siphons",
        "persistent",
    ]
    rows = [
        [
            result.module_id,
            result.module_name.split(",")[0],
            str(result.n_species),
            str(result.n_reactions),
            str(result.n_complexes),
            str(result.n_linkage_classes),
            str(result.rank),
            str(result.deficiency),
            "yes" if result.weakly_reversible else "no",
            str(len(result.moieties)),
            str(len(result.siphons)),
            ("-" if result.persistent is None else ("yes" if result.persistent else "no")),
        ]
        for result in results
    ]

    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]

    def line(values: Sequence[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(values)) + " |"

    return "\n".join([line(headers), line(["-" * w for w in widths])] + [line(row) for row in rows])


SCHEMA = "synkit.crn-kegg/1"

#: Findings the study asserts, so that a silent change in KEGG or in the
#: analysis stack fails the run rather than quietly changing the expected results.
EXPECTED_GLYCOLYSIS_MOIETY = {
    "alpha-D-Glucose",
    "alpha-D-Glucose 6-phosphate",
    "D-Fructose 6-phosphate",
    "D-Fructose 1,6-bisphosphate",
    "Glycerone phosphate",
    "D-Glyceraldehyde 3-phosphate",
    "3-Phospho-D-glyceroyl phosphate",
    "3-Phospho-D-glycerate",
    "2-Phospho-D-glycerate",
    "Phosphoenolpyruvate",
    "Pyruvate",
}


def kegg_report(
    modules: Sequence[str],
    *,
    drop_currency: bool,
) -> Dict[str, Any]:
    """Analyse each module and assemble the evidence report.

    :param modules: KEGG module identifiers to analyse.
    :type modules: Sequence[str]
    :param drop_currency: Whether currency metabolites are removed.
    :type drop_currency: bool
    :return: Report payload.
    :rtype: Dict[str, Any]
    """
    cache = load_kegg_cache()

    analyses = []
    for module_id in modules:
        seconds, result, error = timed(lambda mid=module_id: analyze_kegg_module(mid, drop_currency=drop_currency))
        if result is None:
            analyses.append({"module_id": module_id, "error": error})
            continue
        payload = result.to_dict()
        payload["seconds"] = seconds
        analyses.append(payload)

    flux_seconds, flux, flux_error = timed(check_glycolysis_flux)
    flux_payload: Dict[str, Any] = {"error": flux_error} if flux is None else dict(flux)
    flux_payload["seconds"] = flux_seconds

    glycolysis = next((a for a in analyses if a.get("module_id") == "M00001"), {})
    moieties = [set(m) for m in glycolysis.get("moieties", [])]
    siphons = [set(s) for s in glycolysis.get("siphons", [])]

    fired: Dict[str, int] = {}
    for reaction in flux_payload.get("certificate") or []:
        fired[reaction] = fired.get(reaction, 0) + 1

    checks = {
        "every_module_analysed": all("error" not in a for a in analyses),
        "all_enumerations_complete": all(
            a.get("siphons_complete") is True and a.get("persistent") is not None for a in analyses
        ),
        "deficiency_identity_holds": all(
            a["deficiency"] == a["n_complexes"] - a["n_linkage_classes"] - a["rank"]
            for a in analyses
            if "error" not in a
        ),
        "glycolysis_carbon_backbone_recovered": EXPECTED_GLYCOLYSIS_MOIETY in moieties,
        "glucose_is_a_minimal_siphon": {"alpha-D-Glucose"} in siphons,
        "ferredoxin_couple_recovered": {
            "Oxidized ferredoxin",
            "Reduced ferredoxin",
        }
        in moieties,
        "glycolytic_flux_is_realizable": bool(flux_payload.get("realizable")),
        "certificate_matches_requested_flux": fired == dict(GLYCOLYSIS_FLUX),
    }

    return {
        "schema": SCHEMA,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "claim_boundary": (
            "Conserved moieties, siphons and persistence are structural "
            "properties of the stoichiometry as curated by KEGG. They describe "
            "the module as written, not the in vivo pathway, and say nothing "
            "about flux magnitude, regulation or kinetics."
        ),
        "environment": environment(),
        "source": {
            "database": cache.get("source"),
            "retrieved": cache.get("retrieved"),
            "drop_currency": drop_currency,
        },
        "checks": checks,
        "modules": analyses,
        "glycolysis_flux": flux_payload,
    }


def main() -> int:
    """Parse arguments, run the study, and write the evidence file.

    :return: Process exit status.
    :rtype: int
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--modules", nargs="+", default=list(CASE_STUDY_MODULES))
    parser.add_argument(
        "--keep-currency",
        action="store_true",
        help="retain ATP/NAD(H)/water and the other currency metabolites",
    )
    parser.add_argument("--output", type=Path, help="write the evidence file here")
    arguments = parser.parse_args()

    report = kegg_report(
        arguments.modules,
        drop_currency=not arguments.keep_currency,
    )
    return write_report(report, arguments.output)


main = native_experiment(main)

if __name__ == "__main__":
    raise SystemExit(main())
