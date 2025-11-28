from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .core import CRNNetwork
from .Props.structure import CRNStructuralProperties, compute_structural_properties
from .crn_theory import (
    # Feinberg
    is_deficiency_zero_applicable,
    compute_linkage_class_deficiencies,
    is_deficiency_one_theorem_applicable,
    is_regular_network,
    DeficiencyOneAlgorithmResult,
    run_deficiency_one_algorithm,
    # SR graph / autocatalysis / SSD
    is_autocatalytic,
    build_species_reaction_graph,
    find_sr_graph_cycles,
    check_species_reaction_graph_conditions,
    is_SSD,
    # Petri-net layer
    compute_P_semiflows,
    compute_T_semiflows,
    find_siphons,
    find_traps,
    check_persistence_sufficient,
    # Concordance / Endotactic
    is_concordant,
    is_accordant,
    is_endotactic,
    is_strongly_endotactic,
)

# optional thermo layer (if you already created thermo.py)
try:
    from .thermo import (
        CRNThermoProperties,
        compute_thermo_properties,
    )

    _THERMO_AVAILABLE = True
except Exception:  # pragma: no cover
    CRNThermoProperties = Any  # type: ignore
    compute_thermo_properties = None  # type: ignore
    _THERMO_AVAILABLE = False


@dataclass
class CRNAnalysisResult:
    """
    Aggregated structural + CRN-theoretic analysis.

    Includes:

    - Structural invariants (:class:`CRNStructuralProperties`).
    - Deficiency Zero / One info, linkage-class deficiencies.
    - Simplified regularity + Deficiency One Algorithm stub.
    - SR graph, cycles, autocatalysis, SSD flag.
    - Petri-net layer: P/T-semiflows, candidate siphons & traps,
      persistence flag.
    - Concordance / accordance placeholders.
    - Endotactic / strongly endotactic flags (simplified).
    - Optional thermo properties (if available).
    """

    structural: CRNStructuralProperties

    # Feinberg deficiency decomposition
    linkage_class_deficiencies: List[int] = field(default_factory=list)
    deficiency_zero_applicable: bool = False
    deficiency_one_applicable: bool = False
    regular_network: bool = False
    deficiency_one_result: Optional[DeficiencyOneAlgorithmResult] = None

    # SR graph / autocatalysis / SSD
    autocatalytic: bool = False
    sr_cycles: List[List[str]] = field(default_factory=list)
    sr_conditions_ok: Optional[bool] = None
    ssd: Optional[bool] = None

    # Petri-net layer
    P_semiflows: List[Any] = field(default_factory=list)
    T_semiflows: List[Any] = field(default_factory=list)
    siphons: List[Any] = field(default_factory=list)
    traps: List[Any] = field(default_factory=list)
    persistence_sufficient: Optional[bool] = None

    # Concordance / endotactic
    concordant: Optional[bool] = None
    accordant: Optional[bool] = None
    endotactic: Optional[bool] = None
    strongly_endotactic: Optional[bool] = None

    # Thermo layer
    thermo: Optional[CRNThermoProperties] = None

    # Free-form details and tagged conclusions
    details: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    conclusions: List[str] = field(default_factory=list)

    def add_conclusion(self, tag: str, msg: str, **extra: Any) -> None:
        self.tags.append(tag)
        self.conclusions.append(msg)
        if extra:
            self.details.setdefault("conclusions_extra", []).append(extra)

    @property
    def summary(self) -> str:
        props = self.structural
        lines: List[str] = []

        # Structural summary
        lines.append(
            f"CRN structural summary: "
            f"{len(props.N)} species, {props.N.shape[1]} reactions."
        )
        lines.append(
            f"  Stoichiometric rank: rank(N) = {props.rank_N}, "
            f"deficiency δ = {props.deficiency}."
        )
        lines.append(
            f"  Complexes: {len(props.complexes)}; "
            f"linkage classes: {len(props.linkage_classes)}; "
            f"weakly reversible: {props.weakly_reversible}."
        )
        if self.linkage_class_deficiencies:
            s = sum(self.linkage_class_deficiencies)
            lines.append(
                "  Linkage-class deficiencies δ_ℓ (Feinberg decomposition): "
                f"{self.linkage_class_deficiencies} (sum = {s})."
            )

        # Feinberg theorems
        if self.deficiency_zero_applicable:
            lines.append(
                "Deficiency Zero Theorem (Feinberg, 1977) applies: "
                "weakly reversible network with δ = 0 is complex-balanced "
                "and monostationary under mass-action kinetics."
            )
        else:
            lines.append(
                "Deficiency Zero Theorem: structural hypotheses not satisfied."
            )

        if self.deficiency_one_applicable:
            reg_str = "regular" if self.regular_network else "non-regular"
            lines.append(
                "Deficiency One Theorem (Feinberg, 1987): structural "
                "prerequisites met (δ = 1, δ_ℓ ≤ 1, Σ δ_ℓ = 1). "
                f"Network is {reg_str} under simplified regularity check."
            )
            if self.deficiency_one_result is not None:
                if self.deficiency_one_result.multiple_equilibria:
                    lines.append(
                        "  Deficiency One Algorithm (Feinberg, 1988) indicates "
                        "existence of multiple positive equilibria "
                        "(details not yet implemented)."
                    )
                else:
                    lines.append(
                        "  Deficiency One Algorithm backend not implemented — "
                        "no explicit multiple equilibria constructed."
                    )
        else:
            lines.append(
                "Deficiency One Theorem: not in the δ = 1 regime or δ_ℓ "
                "decomposition incompatible; theorem does not apply."
            )

        # SR graph + autocatalysis
        lines.append(
            f"Autocatalytic reactions present (stoichiometric heuristic): "
            f"{self.autocatalytic}."
        )
        lines.append(
            f"Species–Reaction (Craciun–Feinberg) graph: "
            f"{len(self.sr_cycles)} directed cycle(s) detected."
        )
        if self.sr_conditions_ok is not None:
            lines.append(
                "  Simple SR-graph injectivity condition "
                "(acyclic SR graph ⇒ no CF cycles): "
                f"{self.sr_conditions_ok}."
            )

        if self.ssd is not None:
            lines.append(
                "Stoichiometric matrix SSD (Banaji–Donnell–Baigent, 2007): "
                f"{self.ssd} (placeholder implementation)."
            )

        # Petri-net layer
        lines.append(
            f"Petri-net invariants: {len(self.P_semiflows)} P-semiflow(s), "
            f"{len(self.T_semiflows)} T-semiflow(s)."
        )
        lines.append(
            f"Candidate siphons: {len(self.siphons)}, candidate traps: "
            f"{len(self.traps)}."
        )
        if self.persistence_sufficient is not None:
            lines.append(
                "  Angeli–De Leenheer–Sontag-style persistence condition "
                f"(each siphon contains support of a P-semiflow): "
                f"{self.persistence_sufficient}."
            )

        # Concordance / endotactic
        if self.concordant is not None:
            lines.append(
                f"Concordance (Shinar–Feinberg injectivity for weakly "
                f"monotonic kinetics): {self.concordant}."
            )
        if self.accordant is not None:
            lines.append(
                f"Accordance (related CFSTR injectivity notion): " f"{self.accordant}."
            )

        if self.endotactic is not None:
            lines.append(f"Endotactic (Gopalkrishnan–Miller–Shiu): {self.endotactic}.")
        if self.strongly_endotactic is not None:
            lines.append(
                f"Strongly endotactic (strong permanence / GAC cases): "
                f"{self.strongly_endotactic}."
            )

        # Thermo layer
        if self.thermo is not None:
            th = self.thermo
            lines.append("Thermodynamic / conservation summary:")
            lines.append(
                f"  Conservative in Feinberg sense (∃ m ≫ 0 with mᵀN = 0): "
                f"{th.conservative}."
            )
            if th.positive_conservation_law is not None:
                m = th.positive_conservation_law
                min_pos = float(m[m > 0].min())
                m_scaled = (m / min_pos).round().astype(int)
                lines.append(
                    f"    Example positive conservation law (scaled): "
                    f"{m_scaled.tolist()}."
                )
            lines.append(
                "  Irreversible futile cycles (nonnegative v with Nv=0 using "
                f"some irreversible reaction): {th.has_irreversible_futile_cycle}."
            )
            lines.append(
                "  Strict thermodynamic soundness of irreversible part "
                f"(no irreversible futile cycles): "
                f"{th.thermodynamically_sound_irreversible}."
            )

        # Key conclusions
        if self.conclusions:
            lines.append("")
            lines.append("Key theorem-based conclusions:")
            for msg in self.conclusions:
                lines.append(f"- {msg}")

        return "\n".join(lines)


@dataclass
class CRNAnalyzer:
    """
    High-level orchestrator for structural & CRN-theoretic analysis.

    Called by :meth:`ReactionNetwork.run_full_crnt_analysis`.
    """

    def analyze_model(
        self,
        network: CRNNetwork,
        *,
        kinetics: str = "mass_action",
    ) -> CRNAnalysisResult:
        props = compute_structural_properties(network)
        result = CRNAnalysisResult(structural=props)

        # Feinberg deficiency decomposition
        lc_def = compute_linkage_class_deficiencies(props)
        result.linkage_class_deficiencies = lc_def

        # Deficiency Zero
        if kinetics == "mass_action" and is_deficiency_zero_applicable(props):
            result.deficiency_zero_applicable = True
            result.add_conclusion(
                "deficiency_zero",
                "Deficiency Zero Theorem applies: weakly reversible network "
                "with δ = 0 is complex-balanced and monostationary under "
                "mass-action kinetics.",
            )

        # Deficiency One
        if kinetics == "mass_action" and is_deficiency_one_theorem_applicable(
            props, lc_def
        ):
            result.deficiency_one_applicable = True
            result.regular_network = is_regular_network(props)
            d1 = run_deficiency_one_algorithm(network, props)
            result.deficiency_one_result = d1

        # SR graph & autocatalysis
        sr = build_species_reaction_graph(network)
        result.sr_cycles = find_sr_graph_cycles(sr)
        result.sr_conditions_ok = check_species_reaction_graph_conditions(sr)
        result.autocatalytic = is_autocatalytic(network)
        result.ssd = is_SSD(props.N)

        # Petri-net layer
        try:
            result.P_semiflows = compute_P_semiflows(props.N)
            result.T_semiflows = compute_T_semiflows(props.N)
        except RuntimeError:
            result.P_semiflows = []
            result.T_semiflows = []

        result.siphons = find_siphons(network)
        result.traps = find_traps(network)

        if result.P_semiflows and result.siphons:
            result.persistence_sufficient = check_persistence_sufficient(
                props.N, result.siphons, result.P_semiflows
            )
        else:
            result.persistence_sufficient = None

        # Concordance / endotactic (placeholders / simplified)
        result.concordant = is_concordant(network)
        result.accordant = is_accordant(network)
        result.endotactic = is_endotactic(props)
        result.strongly_endotactic = is_strongly_endotactic(props)

        # Thermo layer if available
        if _THERMO_AVAILABLE and compute_thermo_properties is not None:
            try:
                result.thermo = compute_thermo_properties(network)
            except Exception:
                result.thermo = None

        # If no theorem gave a strong conclusion, add a generic note
        if not result.conclusions:
            result.add_conclusion(
                "inconclusive",
                "Structural criteria did not yield a definitive global verdict "
                "on multistationarity; further numerical or symbolic analysis "
                "may be required.",
            )

        return result
