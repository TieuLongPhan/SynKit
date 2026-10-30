"""Scenario-based biochemical audits and independently replayed pathways.

Expensive enumeration is deliberately separate from the routine audit. Instances
own a detached snapshot; construct a new instance after editing a network.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from itertools import product
from math import prod
from time import monotonic
from typing import Any, Mapping

import numpy as np
from scipy.optimize import linprog

from .Structure import SynCRN
from .Props.stoich import _exact_null_space
from .Props.deficiency import crnt_summary
from .Pathway import PathwayRealizability


@dataclass(frozen=True)
class AnalysisResult:
    value: Any
    status: str = "complete"
    complete: bool = True
    method: str = "structural"
    assumptions: dict = field(default_factory=dict)
    certificate: Any = None
    termination_reason: str = "complete"

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class Scenario:
    """Species selectors accept unique ids, source ids or labels.

    Chemostats have unlimited supply in discrete execution. Currency deletion
    removes incidences and changes the model. Neither happens implicitly.
    """

    chemostats: tuple[str, ...] = ()
    drop_species: tuple[str, ...] = ()
    initial_marking: Mapping[str, int] = field(default_factory=dict)

    @classmethod
    def from_model(cls, crn, *, initial_marking=None):
        """Use declared SBML boundary species; never infer token counts from concentrations."""
        return cls(
            chemostats=tuple(
                sid
                for sid, sp in crn.species.items()
                if str(sp.source_attrs.get("boundaryCondition", "")).lower() == "true"
            ),
            initial_marking={} if initial_marking is None else dict(initial_marking),
        )


class CRNAnalysis:
    """Prepared, detached model and explicit biochemical scenario."""

    def __init__(self, crn: Any, scenario: Scenario | None = None):
        self.scenario = deepcopy(scenario or Scenario())
        self._original = deepcopy(
            crn if isinstance(crn, SynCRN) else SynCRN.from_digraph(crn)
        )
        self._crn = deepcopy(self._original)
        dropped = {self._resolve(x) for x in self.scenario.drop_species}
        self.chemostats = {self._resolve(x) for x in self.scenario.chemostats}
        if dropped & self.chemostats:
            raise ValueError("A species cannot be both removed and chemostatted")
        for reaction in self._crn.reactions.values():
            for side in (reaction.lhs, reaction.rhs):
                for sid in dropped:
                    side.counts.pop(sid, None)
        for sid in dropped:
            del self._crn.species[sid]
        self._matrices = self._crn.to_stoichiometric_matrices()
        self.species_order = tuple(self._matrices["species_order"])
        self.reaction_order = tuple(self._matrices["reaction_order"])
        self._structure = None

    @property
    def model(self):
        return deepcopy(self._crn)

    @property
    def matrices(self):
        return deepcopy(self._matrices)

    def _resolve(self, token, *, reaction=False):
        records = self._crn.reactions if reaction else self._crn.species
        if token in records:
            return token
        matches = [
            key
            for key, record in records.items()
            if str(token) in {str(record.label), str(record.source_node_id)}
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Unknown or ambiguous {'reaction' if reaction else 'species'}: {token!r}"
            )
        return matches[0]

    @staticmethod
    def _counts(values):
        if any(
            isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 0
            for v in values.values()
        ):
            raise ValueError(
                "Discrete inventories and flows require nonnegative integers"
            )
        return {k: int(v) for k, v in values.items() if v}

    def _species_compositions(self):
        from rdkit import Chem

        compositions = {}
        for sid, species in self._crn.species.items():
            formula = species.source_attrs.get("chemicalFormula")
            charge = species.source_attrs.get("charge")
            if formula and charge is not None:
                import re

                parts = re.findall(r"([A-Z][a-z]?)([0-9]*)", formula)
                if "".join(e + n for e, n in parts) == formula:
                    counts = {}
                    for element, number in parts:
                        counts[element] = counts.get(element, 0) + int(number or 1)
                    counts["charge"] = float(charge)
                    compositions[sid] = counts
            mol = Chem.MolFromSmiles(species.smiles) if species.smiles else None
            if mol is not None:
                counts = {}
                for atom in Chem.AddHs(mol).GetAtoms():
                    counts[atom.GetSymbol()] = counts.get(atom.GetSymbol(), 0) + 1
                counts["charge"] = sum(a.GetFormalCharge() for a in mol.GetAtoms())
                compositions[sid] = counts
        return compositions

    def audit(self):
        """Cheap model diagnostics; chemical balance is unknown without structures."""
        signatures, duplicates, empty, unbalanced, unknown = {}, [], [], [], []
        compositions = self._species_compositions()
        involved = set()
        for rid, reaction in self._crn.reactions.items():
            sig = (
                tuple(sorted(reaction.lhs.items())),
                tuple(sorted(reaction.rhs.items())),
            )
            if sig in signatures:
                duplicates.append((signatures[sig], rid))
            signatures[sig] = rid
            if not reaction.lhs or not reaction.rhs:
                empty.append(rid)
            participants = set(dict(reaction.lhs.items())) | set(
                dict(reaction.rhs.items())
            )
            involved |= participants
            if not participants <= compositions.keys():
                unknown.append(rid)
                continue
            balance = {}
            for sign, side in ((-1, reaction.lhs), (1, reaction.rhs)):
                for sid, coeff in side.items():
                    for element, count in compositions[sid].items():
                        balance[element] = (
                            balance.get(element, 0) + sign * coeff * count
                        )
            if any(balance.values()):
                unbalanced.append(
                    {
                        "reaction": rid,
                        "imbalance": {k: v for k, v in balance.items() if v},
                    }
                )
        return AnalysisResult(
            dict(
                n_species=len(self.species_order),
                n_reactions=len(self.reaction_order),
                duplicate_reactions=duplicates,
                open_reactions=empty,
                isolated_species=sorted(set(self.species_order) - involved),
                unbalanced_reactions=unbalanced,
                balance_unknown=unknown,
            ),
            assumptions=self._assumptions(),
        )

    def _assumptions(self):
        return {
            "chemostats": sorted(self.chemostats),
            "drop_species": list(self.scenario.drop_species),
            "inventory_semantics": "integer tokens; concentrations are not converted",
        }

    def conservation(self):
        s = self._matrices["S"]
        order = tuple(sid for sid in self.species_order if sid not in self.chemostats)
        s = s[
            [
                i
                for i, sid in enumerate(self.species_order)
                if sid not in self.chemostats
            ],
            :,
        ]
        laws = _exact_null_space(s.T.tolist(), len(order))
        return AnalysisResult(
            {
                "species_order": order,
                "basis": laws,
                "dimension": len(laws),
                "rank": len(order) - len(laws),
            },
            method="rational elimination",
            assumptions=self._assumptions(),
        )

    def structure(self):
        if self._structure is None:
            self._structure = crnt_summary(self._crn).to_dict()
        value = deepcopy(self._structure)
        value["scope"] = "reaction equations including external species"
        if self.chemostats:
            value["deficiency_zero"] = {
                "applicable": None,
                "reason": "Chemostatted dynamics require a separately reduced model",
            }
            value["deficiency_one"] = {
                "applicable": None,
                "reason": "Chemostatted dynamics require a separately reduced model",
            }
        return AnalysisResult(value, assumptions=self._assumptions())

    def accessibility(self, seeds=()):
        available = {self._resolve(s) for s in seeds} | self.chemostats
        fired, witnesses = set(), {}
        changed = True
        while changed:
            changed = False
            for rid, reaction in self._crn.reactions.items():
                if rid not in fired and set(dict(reaction.lhs.items())) <= available:
                    fired.add(rid)
                    for sid, _ in reaction.rhs.items():
                        if sid not in available:
                            witnesses[sid] = rid
                            available.add(sid)
                            changed = True
        return AnalysisResult(
            {
                "reachable_species": sorted(available),
                "enabled_reactions": sorted(fired),
                "unreachable_species": sorted(set(self.species_order) - available),
            },
            method="qualitative closure",
            certificate=witnesses,
            assumptions=self._assumptions(),
        )

    def flux_feasibility(self, *, bounds=None, required=None, time_limit=30.0):
        """Steady-state feasibility on internal species, not discrete execution.

        Bounds default to nonnegative directed fluxes. Require a reaction to
        exclude the zero solution. External supply must be declared explicitly.
        """
        if time_limit <= 0:
            raise ValueError("time_limit must be positive")
        bounds = (
            {}
            if bounds is None
            else {self._resolve(k, reaction=True): v for k, v in bounds.items()}
        )
        required = (
            {}
            if required is None
            else {self._resolve(k, reaction=True): v for k, v in required.items()}
        )
        limits = []
        for rid in self.reaction_order:
            lo, hi = bounds.get(rid, (0, None))
            lo = max(0 if lo is None else lo, required.get(rid, 0))
            if hi is not None and hi < lo:
                return AnalysisResult(
                    False,
                    certificate={"conflicting_bounds": rid},
                    assumptions=self._assumptions(),
                )
            limits.append((lo, hi))
        if not limits:
            return AnalysisResult(True, certificate={}, assumptions=self._assumptions())
        internal = [
            i for i, sid in enumerate(self.species_order) if sid not in self.chemostats
        ]
        s = self._matrices["S"][internal, :]
        answer = linprog(
            np.zeros(len(limits)),
            A_eq=s,
            b_eq=np.zeros(len(internal)),
            bounds=limits,
            method="highs",
            options={"time_limit": time_limit},
        )
        if answer.status == 2:
            return AnalysisResult(
                False, method="HiGHS numerical LP", assumptions=self._assumptions()
            )
        valid = answer.success and np.max(np.abs(s @ answer.x), initial=0) <= 1e-7
        if valid:
            valid = all(
                x >= (lo * (1 - 1e-7) if lo > 0 else lo - 1e-7)
                and (hi is None or x <= hi + 1e-7)
                for x, (lo, hi) in zip(answer.x, limits)
            )
        return AnalysisResult(
            True if valid else None,
            "complete" if valid else "inconclusive",
            bool(valid),
            "HiGHS numerical LP",
            self._assumptions(),
            dict(zip(self.reaction_order, map(float, answer.x))) if valid else None,
            "complete" if valid else str(answer.message),
        )

    def blocked_reactions(self, *, bounds=None, time_limit=30.0, threshold=1e-6):
        """Classify ability to carry at least threshold forward steady-state flux."""
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        deadline = monotonic() + time_limit
        blocked, active, unknown = [], [], []
        for rid in self.reaction_order:
            remaining = deadline - monotonic()
            result = (
                self.flux_feasibility(
                    bounds=bounds, required={rid: threshold}, time_limit=remaining
                )
                if remaining > 0
                else AnalysisResult(None)
            )
            (
                blocked
                if result.value is False
                else active if result.value is True else unknown
            ).append(rid)
        return AnalysisResult(
            {"blocked": blocked, "active": active, "unknown": unknown},
            "inconclusive" if unknown else "complete",
            not unknown,
            "threshold LP feasibility",
            {**self._assumptions(), "threshold": threshold},
        )

    def verify_pathway(
        self, flow, initial_marking=None, *, max_states=100000, max_depth=10000
    ):
        flow = {
            self._resolve(k, reaction=True): v for k, v in self._counts(flow).items()
        }
        marking = (
            self.scenario.initial_marking
            if initial_marking is None
            else initial_marking
        )
        marking = {self._resolve(k): v for k, v in self._counts(marking).items()}
        model = deepcopy(self._crn)
        for reaction in model.reactions.values():
            for side in (reaction.lhs, reaction.rhs):
                for sid in self.chemostats:
                    side.counts.pop(sid, None)
        checker = PathwayRealizability().load_syncrn_and_flow(
            model, flow=flow, initial_marking=marking, species="id", reaction="id"
        )
        checker.build_petri_net_from_flow()
        outcome = checker.realizability_result(
            max_states=max_states, max_depth=max_depth
        ).to_dict()
        inventory = dict(marking)
        trajectory = [dict(inventory)]
        counts = {}
        for rid in outcome.get("certificate") or []:
            reaction = model.reactions[rid]
            if any(inventory.get(s, 0) < n for s, n in reaction.lhs.items()):
                raise RuntimeError("Returned certificate is not enabled")
            for sign, side in ((-1, reaction.lhs), (1, reaction.rhs)):
                for sid, n in side.items():
                    inventory[sid] = inventory.get(sid, 0) + sign * n
            counts[rid] = counts.get(rid, 0) + 1
            trajectory.append(dict(inventory))
        if outcome["status"] == "realizable" and counts != flow:
            raise RuntimeError("Certificate does not match the requested flow")
        outcome.update(
            trajectory=trajectory,
            replay_verified=outcome["status"] == "realizable",
            assumptions=self._assumptions(),
        )
        return outcome

    def pathways(
        self,
        seeds,
        targets,
        *,
        initial_marking=None,
        max_depth=20,
        max_paths=10,
        max_states=100000,
    ):
        """Suggest qualitative routes and verify each against finite inventory.

        Candidate search is bounded and qualitative; absent candidates never
        constitute a proof of discrete unreachability or route completeness.
        """
        from .Pathway import PathwayFinder

        source = {self._resolve(s) for s in seeds} | self.chemostats
        target = {self._resolve(s) for s in targets}
        finder = PathwayFinder().load_syncrn(self._crn, species="id", reaction="id")
        candidates = finder.find_paths_set(
            source_species=source,
            target_species=target,
            max_depth=max_depth,
            max_paths=max_paths,
            stop_after_first=False,
        )
        rows = []
        for candidate in candidates:
            outcome = self.verify_pathway(
                candidate.flow,
                initial_marking,
                max_states=max_states,
                max_depth=max_depth,
            )
            rows.append(
                {
                    "flow": candidate.flow,
                    "qualitative_sequence": candidate.reactions,
                    "execution": outcome,
                }
            )
        return AnalysisResult(
            rows,
            "inconclusive",
            False,
            "bounded qualitative candidates plus discrete replay",
            self._assumptions(),
            termination_reason="candidate_search_not_exhaustive",
        )

    def to_cobra(self, *, bounds, exchange_bounds=None):
        """Optional COBRApy bridge with caller-specified flux bounds.

        Every reaction and chemostat exchange must have explicit bounds; no
        physiological capacities or exchange directions are inferred.
        """
        try:
            import cobra
        except ImportError as exc:
            raise ImportError(
                "Install cobra separately to use the optional COBRApy bridge"
            ) from exc
        bounds = {self._resolve(k, reaction=True): value for k, value in bounds.items()}
        if set(bounds) != set(self.reaction_order):
            raise ValueError("Explicit bounds are required for every directed reaction")
        exchanges = (
            {}
            if exchange_bounds is None
            else {self._resolve(k): value for k, value in exchange_bounds.items()}
        )
        if set(exchanges) != self.chemostats:
            raise ValueError(
                "Explicit exchange bounds are required for every chemostat"
            )
        model = cobra.Model("synkit_crn")
        metabolites = {}
        for sid, sp in self._crn.species.items():
            metabolites[sid] = cobra.Metabolite(
                sid,
                name=sp.label,
                compartment=sp.source_attrs.get("compartment", "cell"),
            )
        model.add_metabolites(list(metabolites.values()))
        for rid, record in self._crn.reactions.items():
            reaction = cobra.Reaction(rid, name=record.label or rid)
            reaction.bounds = bounds[rid]
            counts = {}
            for sign, side in ((-1, record.lhs), (1, record.rhs)):
                for sid, coefficient in side.items():
                    counts[metabolites[sid]] = (
                        counts.get(metabolites[sid], 0) + sign * coefficient
                    )
            reaction.add_metabolites(counts)
            model.add_reactions([reaction])
        for sid, limits in exchanges.items():
            identifier = "EX_" + sid
            if identifier in model.reactions:
                raise ValueError("Generated exchange id conflicts with a reaction")
            reaction = cobra.Reaction(identifier)
            reaction.bounds = limits
            reaction.add_metabolites({metabolites[sid]: -1})
            model.add_reactions([reaction])
        return model

    def minimum_supplements(
        self,
        flow,
        *,
        allowed,
        initial_marking=None,
        costs=None,
        max_added=4,
        max_candidates=10000,
        max_states=100000,
        max_depth=10000,
    ):
        """Minimum weighted added tokens within a bounded, explicit search domain.

        `allowed` lists species; each may receive 0..max_added tokens. A result
        is optimal only within that box, and only if every cheaper candidate
        was conclusively rejected. No global minimality is implied.
        """
        if max_added < 0 or max_candidates < 1:
            raise ValueError("Invalid supplementation search limits")
        ids = sorted({self._resolve(s) for s in allowed})
        costs = {} if costs is None else {self._resolve(s): c for s, c in costs.items()}
        if any(not np.isfinite(c) or c <= 0 for c in costs.values()):
            raise ValueError("Supplement costs must be finite and positive")
        domain = prod([max_added + 1] * len(ids))
        if domain > max_candidates:
            return AnalysisResult(
                None,
                "inconclusive",
                False,
                "bounded exhaustive supplementation",
                termination_reason="candidate_limit",
            )
        base = (
            self.scenario.initial_marking
            if initial_marking is None
            else initial_marking
        )
        base = {self._resolve(s): n for s, n in self._counts(base).items()}
        candidates = sorted(
            product(range(max_added + 1), repeat=len(ids)),
            key=lambda values: (
                sum(costs.get(s, 1) * n for s, n in zip(ids, values)),
                values,
            ),
        )
        uncertain = []
        for values in candidates:
            score = sum(costs.get(s, 1) * n for s, n in zip(ids, values))
            added = {s: n for s, n in zip(ids, values) if n}
            marking = dict(base)
            for s, n in added.items():
                marking[s] = marking.get(s, 0) + n
            outcome = self.verify_pathway(
                flow, marking, max_states=max_states, max_depth=max_depth
            )
            if outcome["status"] == "realizable":
                optimal = not any(c < score for c in uncertain)
                return AnalysisResult(
                    {
                        "supplements": added,
                        "cost": score,
                        "optimal_within_bounds": optimal,
                        "max_added_per_species": max_added,
                    },
                    "complete" if optimal else "inconclusive",
                    optimal,
                    "bounded exhaustive supplementation",
                    self._assumptions(),
                    outcome,
                    "optimum" if optimal else "unresolved_cheaper_candidates",
                )
            if outcome["status"] == "inconclusive":
                uncertain.append(score)
        return AnalysisResult(
            None,
            "inconclusive" if uncertain else "complete",
            not uncertain,
            "bounded exhaustive supplementation",
            self._assumptions(),
            termination_reason=(
                "search_limit" if uncertain else "no_solution_within_bounds"
            ),
        )

    def compare(self, other):
        """Compare scenarios sharing the same stable species/reaction identities."""
        if not isinstance(other, CRNAnalysis):
            other = CRNAnalysis(other)
        species = sorted(set(self.species_order) | set(other.species_order))
        reactions = sorted(set(self.reaction_order) | set(other.reaction_order))

        def aligned(analysis, name):
            a = np.zeros((len(species), len(reactions)))
            si, ri = {s: i for i, s in enumerate(species)}, {
                r: i for i, r in enumerate(reactions)
            }
            for i, s in enumerate(analysis.species_order):
                for j, r in enumerate(analysis.reaction_order):
                    a[si[s], ri[r]] = analysis._matrices[name][i, j]
            return a

        return AnalysisResult(
            {
                "same_net_matrix": bool(
                    np.array_equal(aligned(self, "S"), aligned(other, "S"))
                ),
                "same_reactants": bool(
                    np.array_equal(aligned(self, "S_minus"), aligned(other, "S_minus"))
                ),
                "same_products": bool(
                    np.array_equal(aligned(self, "S_plus"), aligned(other, "S_plus"))
                ),
                "before": self.structure().value,
                "after": other.structure().value,
                "conservation_before": self.conservation().value,
                "conservation_after": other.conservation().value,
            },
            assumptions={"before": self._assumptions(), "after": other._assumptions()},
        )
