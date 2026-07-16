"""Audits for radical state propagation and matching policies."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any, Literal, Mapping

import networkx as nx

from .model import VerificationIssue
from synkit.Graph.Mech.electron_accounting import recompute_charge

RadicalPolicy = Literal["strict", "lower_bound", "ignore"]


def radical_counts_by_atom_map(graph: nx.Graph) -> dict[int, int]:
    """Return scalar radical counts keyed by non-zero atom map."""
    return {
        int(attrs["atom_map"]): int(attrs.get("radical", 0))
        for _, attrs in graph.nodes(data=True)
        if int(attrs.get("atom_map", 0) or 0) > 0
    }


def radical_match(
    pattern: Mapping[int, int],
    host: Mapping[int, int],
    *,
    policy: RadicalPolicy = "strict",
) -> bool:
    """Compare mapped radical resources under an explicit policy."""
    if policy == "ignore":
        return True
    if policy == "strict":
        return dict(pattern) == dict(host)
    if policy == "lower_bound":
        return all(
            host.get(atom_map, -1) >= value for atom_map, value in pattern.items()
        )
    raise ValueError(f"Unsupported radical policy: {policy!r}")


@dataclass(frozen=True)
class RadicalStateAudit:
    """Result of a mapped radical round-trip or graph-pair audit."""

    before: Mapping[int, int]
    after: Mapping[int, int]
    policy: RadicalPolicy = "strict"

    @property
    def matches(self) -> bool:
        return radical_match(self.before, self.after, policy=self.policy)

    @property
    def issues(self) -> tuple[VerificationIssue, ...]:
        if self.matches:
            return ()
        atom_maps = tuple(sorted(set(self.before) | set(self.after)))
        return (
            VerificationIssue(
                code="RADICAL_STATE_MISMATCH",
                message="Mapped radical counts changed across the audited boundary.",
                atom_maps=atom_maps,
                expected=dict(self.before),
                observed=dict(self.after),
            ),
        )

    @classmethod
    def between_graphs(
        cls, before: nx.Graph, after: nx.Graph, *, policy: RadicalPolicy = "strict"
    ) -> "RadicalStateAudit":
        return cls(
            radical_counts_by_atom_map(before),
            radical_counts_by_atom_map(after),
            policy,
        )

    @classmethod
    def molecule_round_trip(cls, mol: Any) -> "RadicalStateAudit":
        """Audit RDKit → SynKit graph → RDKit molecule radical preservation."""
        from synkit.IO.graph_to_mol import GraphToMol
        from synkit.IO.mol_to_graph import MolToGraph

        before = MolToGraph().transform(mol)
        rebuilt = GraphToMol().graph_to_mol(before, sanitize=False)
        # Diagnostic radical states may intentionally be unsanitized.  RDKit
        # still needs its property cache refreshed before hcount inspection.
        rebuilt.UpdatePropertyCache(strict=False)
        after = MolToGraph().transform(rebuilt)
        return cls.between_graphs(before, after)


@dataclass(frozen=True)
class LocalElectronStateAudit:
    """Structured local Lewis-state identity audit."""

    issues: tuple[VerificationIssue, ...]
    repaired_atom_maps: tuple[int, ...] = ()

    @property
    def valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)


def audit_local_electron_state(  # noqa: C901
    graph: nx.Graph, *, repair: bool = False, tolerance: float = 1e-9
) -> LocalElectronStateAudit:
    """Check scalar Lewis-resource domains and formal-charge identities.

    With ``repair=True`` the represented charge is updated explicitly and the
    repair is retained in the returned audit. No other Lewis-state field is
    guessed or modified.
    """
    issues: list[VerificationIssue] = []
    repaired: list[int] = []
    resource_fields = ("hcount", "lone_pairs", "radical")
    maps: dict[int, Any] = {}
    global_values_are_scalar = True

    for node, attrs in graph.nodes(data=True):
        raw_atom_map = attrs.get("atom_map", node)
        try:
            atom_map = int(raw_atom_map)
        except (TypeError, ValueError):
            atom_map = 0
        if atom_map > 0:
            if atom_map in maps:
                issues.append(
                    VerificationIssue(
                        "DUPLICATE_ATOM_MAP",
                        "Lewis-state graphs require unique positive atom maps.",
                        atom_maps=(atom_map,),
                        observed=(maps[atom_map], node),
                    )
                )
            else:
                maps[atom_map] = node

        node_resources_are_valid = True
        for field_name in resource_fields:
            value = attrs.get(field_name, 0)
            valid = (
                isinstance(value, Real)
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and float(value) >= 0
                and abs(float(value) - round(float(value))) <= tolerance
            )
            if valid:
                continue
            node_resources_are_valid = False
            global_values_are_scalar = False
            issues.append(
                VerificationIssue(
                    "INVALID_ELECTRON_RESOURCE",
                    f"{field_name} must be a non-negative integer resource.",
                    atom_maps=(atom_map,) if atom_map > 0 else (),
                    expected="non-negative integer",
                    observed=value,
                )
            )

        if "valence_electrons" not in attrs:
            global_values_are_scalar = False
            continue
        valence = attrs["valence_electrons"]
        charge = attrs.get("charge", 0)
        if not all(
            isinstance(value, Real)
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in (valence, charge)
        ):
            global_values_are_scalar = False
            issues.append(
                VerificationIssue(
                    "INVALID_ELECTRON_SCALAR",
                    "Valence-electron and charge fields must be finite scalars.",
                    atom_maps=(atom_map,) if atom_map > 0 else (),
                    observed={"valence_electrons": valence, "charge": charge},
                )
            )
            continue
        if (
            float(valence) < 0
            or abs(float(valence) - round(float(valence))) > tolerance
        ):
            node_resources_are_valid = False
            global_values_are_scalar = False
            issues.append(
                VerificationIssue(
                    "INVALID_VALENCE_ELECTRON_RESOURCE",
                    "valence_electrons must be a non-negative integer.",
                    atom_maps=(atom_map,) if atom_map > 0 else (),
                    expected="non-negative integer",
                    observed=valence,
                )
            )
        if not node_resources_are_valid:
            continue
        expected = recompute_charge(graph, node)
        observed = charge
        if abs(float(expected) - float(observed)) <= tolerance:
            continue
        if repair:
            attrs["charge"] = expected
            repaired.append(atom_map)
            issues.append(
                VerificationIssue(
                    "LOCAL_ELECTRON_MISMATCH_REPAIRED",
                    "Formal charge was explicitly repaired from Lewis-state resources.",
                    severity="info",
                    atom_maps=(atom_map,),
                    expected=expected,
                    observed=observed,
                )
            )
        else:
            issues.append(
                VerificationIssue(
                    "LOCAL_ELECTRON_MISMATCH",
                    "Formal charge disagrees with stored Lewis-state resources.",
                    atom_maps=(atom_map,),
                    expected=expected,
                    observed=observed,
                )
            )
    for left, right, attrs in graph.edges(data=True):
        sigma = attrs.get("sigma_order", 0.0)
        pi = attrs.get("pi_order", 0.0)
        endpoints = tuple(
            sorted(
                atom_map
                for endpoint in (left, right)
                for atom_map in [int(graph.nodes[endpoint].get("atom_map", 0) or 0)]
                if atom_map > 0
            )
        )
        if not all(
            isinstance(value, Real)
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in (sigma, pi)
        ):
            global_values_are_scalar = False
            issues.append(
                VerificationIssue(
                    "INVALID_BOND_RESOURCE",
                    "Sigma and pi orders must be finite scalar resources.",
                    atom_maps=endpoints,
                    observed={"sigma_order": sigma, "pi_order": pi},
                )
            )
            continue
        sigma_value = float(sigma)
        pi_value = float(pi)
        if (
            sigma_value < 0
            or pi_value < 0
            or abs(sigma_value - round(sigma_value)) > tolerance
            or abs(pi_value - round(pi_value)) > tolerance
        ):
            global_values_are_scalar = False
            issues.append(
                VerificationIssue(
                    "INVALID_BOND_RESOURCE",
                    "Sigma and pi orders must be non-negative whole electron-pair resources.",
                    atom_maps=endpoints,
                    expected="non-negative integers",
                    observed={"sigma_order": sigma, "pi_order": pi},
                )
            )
        if sigma_value > 1 + tolerance:
            issues.append(
                VerificationIssue(
                    "INVALID_SIGMA_ORDER",
                    "A two-center Lewis bond can contain at most one sigma pair.",
                    atom_maps=endpoints,
                    expected="0 or 1",
                    observed=sigma,
                )
            )
        if pi_value > tolerance and sigma_value <= tolerance:
            issues.append(
                VerificationIssue(
                    "PI_WITHOUT_SIGMA",
                    "A pi bond requires a supporting sigma bond.",
                    atom_maps=endpoints,
                    observed={"sigma_order": sigma, "pi_order": pi},
                )
            )

    if global_values_are_scalar:
        expected_total = sum(
            float(attrs["valence_electrons"])
            + float(attrs.get("hcount", 0))
            - float(attrs.get("charge", 0))
            for _, attrs in graph.nodes(data=True)
        )
        represented_total = sum(
            2 * float(attrs.get("lone_pairs", 0))
            + float(attrs.get("radical", 0))
            + 2 * float(attrs.get("hcount", 0))
            for _, attrs in graph.nodes(data=True)
        ) + 2 * sum(
            float(attrs.get("sigma_order", 0.0)) + float(attrs.get("pi_order", 0.0))
            for _, _, attrs in graph.edges(data=True)
        )
        if abs(expected_total - represented_total) > tolerance:
            issues.append(
                VerificationIssue(
                    "GLOBAL_ELECTRON_MISMATCH",
                    "The full Lewis-state electron inventory is not conserved.",
                    expected=expected_total,
                    observed=represented_total,
                )
            )

    return LocalElectronStateAudit(tuple(issues), tuple(sorted(repaired)))
