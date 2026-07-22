"""Proof-bearing stereo refinements of immutable graph morphisms."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Hashable, Mapping

import networkx as nx

from synkit.Graph.Stereo.identity import StereoIdentityError

from .morphism import GraphMorphism
from .stereo_morphism_helpers import (
    _build_certificates,
    _frame_profile,
    _transport_configuration,
    _validate_stereo_port_bindings,
    _validate_stored_port_bindings,
)
from .stereo_morphism_types import (
    LocalStereoCertificate,
    StereoCertificateStatus,
    StereoInformationPolicy,
    StereoMorphismError,
    StereoMorphismIssue,
    StereoMorphismIssueCode,
    StereoPresenceMode,
)


@dataclass(frozen=True)
class StereoMorphism:
    """A graph morphism refined by complete endpoint-local stereo evidence."""

    graph_morphism: GraphMorphism
    presence_mode: StereoPresenceMode
    information_policy: StereoInformationPolicy
    certificates: tuple[LocalStereoCertificate, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.presence_mode, StereoPresenceMode):
            object.__setattr__(
                self,
                "presence_mode",
                StereoPresenceMode(self.presence_mode),
            )
        if not isinstance(self.information_policy, StereoInformationPolicy):
            object.__setattr__(
                self,
                "information_policy",
                StereoInformationPolicy(self.information_policy),
            )
        certificates = tuple(
            sorted(
                self.certificates,
                key=lambda item: (
                    item.layer,
                    repr(item.source_configuration.canonical_form()),
                ),
            )
        )
        object.__setattr__(self, "certificates", certificates)
        _validate_stored_port_bindings(self.graph_morphism, certificates)
        for certificate in certificates:
            if certificate.status is not StereoCertificateStatus.MATCHED:
                continue
            transported = _transport_configuration(
                certificate.source_configuration,
                self.graph_morphism.mapping,
                self.graph_morphism.substitutions,
            )
            target = certificate.target_configuration
            witness = certificate.witness
            relation = certificate.relation
            assert target is not None and witness is not None and relation is not None
            if relation.witness != witness:
                raise StereoMorphismError(
                    StereoMorphismIssue(
                        StereoMorphismIssueCode.WITNESS_MISMATCH,
                        "The relation and local certificate store different witnesses.",
                    )
                )
            if witness.apply(transported.frame) != target.frame:
                raise StereoMorphismError(
                    StereoMorphismIssue(
                        StereoMorphismIssueCode.WITNESS_MISMATCH,
                        "A local witness does not replay the transported frame.",
                        {"layer": certificate.layer, "shape": transported.shape},
                    )
                )
            direct = transported.relation_to(target)
            if direct.kind is not relation.kind or direct.class_id != relation.class_id:
                raise StereoMorphismError(
                    StereoMorphismIssue(
                        StereoMorphismIssueCode.INVALID_CERTIFICATE,
                        "A stored stereo relation disagrees with endpoint classification.",
                    )
                )

    @classmethod
    def from_graphs(
        cls,
        graph_morphism: GraphMorphism,
        source_graph: nx.Graph,
        target_graph: nx.Graph,
        *,
        presence_mode: StereoPresenceMode | str = StereoPresenceMode.REQUIRE,
        information_policy: StereoInformationPolicy | str = (
            StereoInformationPolicy.EXACT
        ),
        information_policies: (
            Mapping[
                str,
                StereoInformationPolicy | str,
            ]
            | None
        ) = None,
    ) -> "StereoMorphism":
        """Construct and validate stereo evidence without retaining either graph."""
        mode = StereoPresenceMode(presence_mode)
        policy = StereoInformationPolicy(information_policy)
        if graph_morphism.source_nodes != frozenset(source_graph.nodes) or (
            graph_morphism.target_nodes != frozenset(target_graph.nodes)
        ):
            raise StereoMorphismError(
                StereoMorphismIssue(
                    StereoMorphismIssueCode.GRAPH_NODE_MISMATCH,
                    "GraphMorphism endpoint node sets must equal the supplied graphs.",
                )
            )
        try:
            _validate_stereo_port_bindings(
                graph_morphism,
                source_graph,
                target_graph,
            )
            certificates = _build_certificates(
                graph_morphism,
                source_graph,
                target_graph,
                mode,
                policy,
                information_policies or {},
            )
        except StereoIdentityError as exc:
            raise StereoMorphismError(
                StereoMorphismIssue(
                    StereoMorphismIssueCode.INVALID_REFERENCE,
                    str(exc),
                )
            ) from exc
        return cls(graph_morphism, mode, policy, certificates)

    @classmethod
    def identity(
        cls,
        object_id: Hashable,
        graph: nx.Graph,
        *,
        presence_mode: StereoPresenceMode | str = StereoPresenceMode.REQUIRE,
        information_policy: StereoInformationPolicy | str = (
            StereoInformationPolicy.EXACT
        ),
        information_policies: (
            Mapping[
                str,
                StereoInformationPolicy | str,
            ]
            | None
        ) = None,
    ) -> "StereoMorphism":
        graph_morphism = GraphMorphism.identity(object_id, frozenset(graph.nodes))
        return cls.from_graphs(
            graph_morphism,
            graph,
            graph,
            presence_mode=presence_mode,
            information_policy=information_policy,
            information_policies=information_policies,
        )

    def then(self, after: "StereoMorphism") -> "StereoMorphism":
        """Return ``after ∘ self`` and reclassify every endpoint relation."""
        if self.presence_mode is not after.presence_mode or (
            self.information_policy is not after.information_policy
        ):
            raise StereoMorphismError(
                StereoMorphismIssue(
                    StereoMorphismIssueCode.POLICY_MISMATCH,
                    "Composable stereo morphisms must use identical policies.",
                )
            )
        graph_morphism = self.graph_morphism.then(after.graph_morphism)
        remaining = list(after.certificates)
        composed_certificates = []
        for first in self.certificates:
            if first.status is not StereoCertificateStatus.MATCHED:
                composed_certificates.append(
                    replace(
                        first, target_configuration=None, relation=None, witness=None
                    )
                )
                continue
            intermediate = first.target_configuration
            assert intermediate is not None
            match_index = next(
                (
                    index
                    for index, candidate in enumerate(remaining)
                    if candidate.layer == first.layer
                    and candidate.source_configuration.same_configuration(intermediate)
                ),
                None,
            )
            if match_index is None:
                raise StereoMorphismError(
                    StereoMorphismIssue(
                        StereoMorphismIssueCode.INTERMEDIATE_MISMATCH,
                        "A matched intermediate descriptor has no continuation.",
                        {"layer": first.layer, "shape": intermediate.shape},
                    )
                )
            second = remaining.pop(match_index)
            if second.status is not StereoCertificateStatus.MATCHED:
                composed_certificates.append(
                    LocalStereoCertificate(
                        first.layer,
                        first.source_configuration,
                        None,
                        None,
                        None,
                        second.status,
                        first.information_policy,
                    )
                )
                continue
            target = second.target_configuration
            assert target is not None and first.witness is not None
            assert second.witness is not None
            transported = _transport_configuration(
                first.source_configuration,
                graph_morphism.mapping,
                graph_morphism.substitutions,
            )
            direct = transported.relation_to(target)
            witness = first.witness.then(second.witness)
            if witness.apply(transported.frame) != target.frame:
                raise StereoMorphismError(
                    StereoMorphismIssue(
                        StereoMorphismIssueCode.WITNESS_MISMATCH,
                        "Composed local witnesses do not replay the endpoint frame.",
                    )
                )
            relation = replace(direct, witness=witness)
            composed_certificates.append(
                LocalStereoCertificate(
                    first.layer,
                    first.source_configuration,
                    target,
                    relation,
                    witness,
                    StereoCertificateStatus.MATCHED,
                    first.information_policy,
                )
            )
        return StereoMorphism(
            graph_morphism,
            self.presence_mode,
            self.information_policy,
            tuple(composed_certificates),
        )

    def compose(self, after: "StereoMorphism") -> "StereoMorphism":
        return self.then(after)

    def relabel(
        self,
        source_labels: Mapping[Hashable, Hashable],
        target_labels: Mapping[Hashable, Hashable],
        *,
        source: Hashable | None = None,
        target: Hashable | None = None,
    ) -> "StereoMorphism":
        graph_morphism = self.graph_morphism.relabel(
            source_labels,
            target_labels,
            source=source,
            target=target,
        )
        certificates = []
        for certificate in self.certificates:
            source_configuration = _transport_configuration(
                certificate.source_configuration,
                source_labels,
            )
            target_configuration = (
                None
                if certificate.target_configuration is None
                else _transport_configuration(
                    certificate.target_configuration,
                    target_labels,
                )
            )
            relation = certificate.relation
            if target_configuration is not None:
                transported = _transport_configuration(
                    source_configuration,
                    graph_morphism.mapping,
                    graph_morphism.substitutions,
                )
                direct = transported.relation_to(target_configuration)
                relation = replace(direct, witness=certificate.witness)
            certificates.append(
                replace(
                    certificate,
                    source_configuration=source_configuration,
                    target_configuration=target_configuration,
                    relation=relation,
                )
            )
        return StereoMorphism(
            graph_morphism,
            self.presence_mode,
            self.information_policy,
            tuple(certificates),
        )

    def canonical_signature(self) -> tuple[Any, ...]:
        """Return a numbering-independent structural/stereo proof signature."""
        wildcard_nodes = frozenset(self.graph_morphism.substitutions)
        records = (
            (
                certificate.layer,
                certificate.status.value,
                certificate.information_policy.value,
                certificate.source_configuration.shape,
                certificate.source_configuration.specification.value,
                (
                    None
                    if certificate.target_configuration is None
                    else certificate.target_configuration.specification.value
                ),
                (
                    None
                    if certificate.relation is None
                    else certificate.relation.kind.value
                ),
                (
                    None
                    if certificate.relation is None
                    else certificate.relation.class_id
                ),
                (
                    None
                    if certificate.witness is None
                    else certificate.witness.permutation.image
                ),
                _frame_profile(
                    certificate.source_configuration,
                    wildcard_nodes,
                ),
            )
            for certificate in self.certificates
        )
        evidence = tuple(sorted(records, key=repr))
        return (
            self.graph_morphism.canonical_signature(),
            self.presence_mode.value,
            self.information_policy.value,
            evidence,
        )


__all__ = [
    "LocalStereoCertificate",
    "StereoCertificateStatus",
    "StereoInformationPolicy",
    "StereoMorphism",
    "StereoMorphismError",
    "StereoMorphismIssue",
    "StereoMorphismIssueCode",
    "StereoPresenceMode",
]
