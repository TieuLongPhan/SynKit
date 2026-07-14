"""Audits for radical state propagation and matching policies."""

from __future__ import annotations

from dataclasses import dataclass
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


def audit_local_electron_state(
    graph: nx.Graph, *, repair: bool = False, tolerance: float = 1e-9
) -> LocalElectronStateAudit:
    """Check charge against valence, nonbonding, bond, and H resources.

    With ``repair=True`` the represented charge is updated explicitly and the
    repair is retained in the returned audit. No other Lewis-state field is
    guessed or modified.
    """
    issues: list[VerificationIssue] = []
    repaired: list[int] = []
    for node, attrs in graph.nodes(data=True):
        if "valence_electrons" not in attrs:
            continue
        atom_map = int(attrs.get("atom_map", node) or node)
        expected = recompute_charge(graph, node)
        observed = attrs.get("charge", 0)
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
    return LocalElectronStateAudit(tuple(issues), tuple(sorted(repaired)))
