"""End-to-end metabolic case study over cached KEGG modules.

This is the worked biological example: a KEGG module goes in, and a structural
report comes out, exercising the whole chain — retrieval and parsing
(:mod:`synkit.CRN.Query`), canonical representation
(:mod:`synkit.CRN.Structure`), stoichiometry and CRNT
(:mod:`synkit.CRN.Props`), the Petri-net layer (:mod:`synkit.CRN.Petrinet`) and
flux realizability (:mod:`synkit.CRN.Pathway`).

Four modules ship cached with the package, so the study is reproducible offline
and does not depend on KEGG being reachable or unchanged:

======= ==========================================================
Module  Pathway
======= ==========================================================
M00001  Glycolysis (Embden-Meyerhof), glucose to pyruvate
M00307  Pyruvate oxidation, pyruvate to acetyl-CoA
M00009  Citrate cycle (TCA cycle)
M00004  Pentose phosphate pathway
======= ==========================================================

What the analysis reports, and how to read it biologically:

**Conserved moieties**
    Non-negative minimal conservation laws. In a metabolic module these are the
    cofactor couples that the module cycles rather than consumes — the
    ATP/ADP/AMP adenylate pool, the NAD+/NADH pool, oxidized/reduced ferredoxin.
    Finding them confirms the module is balanced on those pools; a missing pool
    points at a reaction written with an unbalanced cofactor.

**Siphons**
    Sets of metabolites that, once all absent, can never be produced again. A
    siphon that contains no conserved moiety marks a metabolite set the module
    cannot regenerate on its own — in practice, a dependency on an input the
    module does not itself produce.

**Structural persistence**
    The Angeli-De Leenheer-Sontag condition: every minimal siphon contains the
    support of a conservation law. When it holds, no metabolite can be driven to
    extinction for any kinetics. A linear catabolic module is *not* expected to
    be persistent — it consumes its substrate by design — so a ``False`` here is
    information about the module's openness, not a defect.

**Flux realizability**
    Whether a proposed reaction flux can actually be fired from a given starting
    marking under integer token semantics, with a firing sequence as
    certificate. This is the check that distinguishes a flux the stoichiometry
    permits from one the network can really carry out step by step.

Currency metabolites are dropped by default
(:data:`~synkit.CRN.Query.to_syncrn.CURRENCY_COMPOUNDS`), because otherwise
every siphon and most semiflows describe cofactor recycling rather than the
pathway's carbon skeleton. Pass ``drop_currency=False`` to see the full picture.

.. rubric:: Example

.. code-block:: python

    from synkit.CRN.Benchmark import analyze_kegg_module, kegg_case_study_report

    result = analyze_kegg_module("M00001")
    print(result.n_species, result.n_reactions, result.deficiency)
    print(kegg_case_study_report())
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..Petrinet.persistence import siphon_persistence_details
from ..Petrinet.structure import find_siphons
from ..Props.deficiency import crnt_summary
from ..Props.helper import _species_and_rule_order
from ..Props.stoich import conserved_moieties
from ..Query.to_syncrn import CURRENCY_COMPOUNDS, syncrn_from_kegg_equations
from ..Structure.syncrn import SynCRN

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
    """Load the cached KEGG module data shipped with the package.

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
        data_dir = resources.files(__package__).joinpath("data")
        with data_dir.joinpath(_CACHE_FILE).open("r", encoding="utf-8") as handle:
            _cache = json.load(handle)
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
    modules = load_kegg_cache()["modules"]
    if module_id not in modules:
        raise KeyError(
            f"Module {module_id!r} is not cached. Available: "
            f"{', '.join(sorted(modules))}"
        )

    entry = modules[module_id]
    return syncrn_from_kegg_equations(
        entry["equations"],
        names=entry.get("compound_names"),
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
    ``crn.species`` directly.

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

    moieties = [
        sorted(label for label, coeff in zip(labels, vector) if coeff)
        for vector in conserved_moieties(crn)
    ]
    moiety_sets = [frozenset(m) for m in moieties]

    siphons_raw = find_siphons(crn, max_size=max_siphon_size)
    siphons = [
        sorted(lookup.get(str(item), str(item)) for item in siphon)
        for siphon in siphons_raw
    ]

    uncovered = [
        siphon
        for siphon in siphons
        if not any(moiety <= frozenset(siphon) for moiety in moiety_sets)
    ]

    try:
        persistent: Optional[bool] = bool(
            siphon_persistence_details(crn).persistence_ok
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
    from ..Pathway.realizability import PathwayRealizability

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
    realizable, certificate = checker.is_realizable(max_states=max_states)

    return {
        "realizable": bool(realizable),
        "certificate": certificate,
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

    results = [
        analyze_kegg_module(module_id, drop_currency=drop_currency)
        for module_id in modules
    ]

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
            "-" if result.persistent is None else ("yes" if result.persistent else "no"),
        ]
        for result in results
    ]

    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]

    def line(values: Sequence[str]) -> str:
        return "| " + " | ".join(
            value.ljust(widths[i]) for i, value in enumerate(values)
        ) + " |"

    return "\n".join(
        [line(headers), line(["-" * w for w in widths])] + [line(row) for row in rows]
    )
