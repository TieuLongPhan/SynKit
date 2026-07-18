"""Proof-bearing generic stereo rule extraction and replay."""

from __future__ import annotations

import pytest

from synkit.Rule import (
    EXTRACTION_SCHEMA,
    GenericStereoExtractionError,
    GenericStereoExtractionIssueCode,
    GenericStereoRuleExtractor,
    GenericStereoRulePolicy,
)
from synkit.Synthesis.Reactor.syn_reactor import SynReactor
from synkit.Synthesis.Reactor import StereoWildcardAssignmentLimitError

SN2 = "[CH3:1][C@H:2]([F:3])[Cl:4].[OH-:5]>>" "[CH3:1][C@@H:2]([F:3])[OH:5].[Cl-:4]"


def _reactor(substrate, rule):
    return SynReactor(
        substrate,
        rule,
        template_format="tuple",
        explicit_h=False,
        stereo_mode="strict",
    )


def test_observed_extraction_emits_ports_and_exact_replay_certificate() -> None:
    result = GenericStereoRuleExtractor().extract(SN2)

    assert result.certificate.schema == EXTRACTION_SCHEMA
    assert result.certificate.source_replay_exact
    assert result.certificate.reverse_status == "exact"
    assert result.certificate.source_mapping_count == 1
    assert len(result.certificate.digest) == 64
    assert {port.reference for port in result.certificate.ports} == {1, 3}
    assert all(
        result.rule.left.raw.nodes[reference]["element"] == "*" for reference in (1, 3)
    )
    assert {
        result.rule.left.raw.nodes[reference]["stereo_slot"] for reference in (1, 3)
    } == {0, 1}


def test_class_domains_apply_extracted_rule_to_a_new_tetrahedral_substrate() -> None:
    policy = GenericStereoRulePolicy(
        domain_source="class",
        explicit_domains={
            1: {"elements": {"C"}},
            3: {"elements": {"F", "Br"}},
        },
    )
    result = GenericStereoRuleExtractor(policy).extract(SN2)
    reactor = _reactor("CC[C@H](Br)Cl.[OH-]", result.rule)

    assert reactor.mapping_count == 1
    assert len(reactor.its_list) == 1
    assert "[C@@H:3]" in reactor.smarts[0].split(">>", 1)[1]


def test_observed_domain_rejects_unobserved_ligand() -> None:
    rule = GenericStereoRuleExtractor().extract(SN2).rule

    assert _reactor("CC[C@@H](Br)Cl.[OH-]", rule).mapping_count == 0


def test_exact_policy_retains_concrete_rule() -> None:
    result = GenericStereoRuleExtractor(
        GenericStereoRulePolicy(domain_source="exact")
    ).extract(SN2)

    assert result.certificate.ports == ()
    assert all(
        attrs["element"] != "*" for _, attrs in result.rule.left.raw.nodes(data=True)
    )


def test_selected_reaction_locus_reference_fails_closed() -> None:
    extractor = GenericStereoRuleExtractor(
        GenericStereoRulePolicy(selected_references={4})
    )

    with pytest.raises(GenericStereoExtractionError) as error:
        extractor.extract(SN2)

    assert error.value.issues[0].code is (
        GenericStereoExtractionIssueCode.REFERENCE_NOT_ELIGIBLE
    )


def test_class_generalization_requires_domains_for_every_selected_port() -> None:
    extractor = GenericStereoRuleExtractor(
        GenericStereoRulePolicy(
            domain_source="class",
            selected_references={1, 3},
            explicit_domains={1: {"elements": {"C"}}},
        )
    )

    with pytest.raises(GenericStereoExtractionError) as error:
        extractor.extract(SN2)

    assert error.value.issues[0].code is (
        GenericStereoExtractionIssueCode.DOMAIN_REQUIRED
    )


def test_non_stereo_source_is_rejected() -> None:
    with pytest.raises(GenericStereoExtractionError) as error:
        GenericStereoRuleExtractor().extract("[CH3:1][Cl:2]>>[CH3:1].[Cl-:2]")

    assert error.value.issues[0].code is GenericStereoExtractionIssueCode.NO_STEREO


def test_certificate_digest_is_map_and_fragment_order_invariant() -> None:
    relabeled = (
        "[OH-:9].[CH3:11][C@H:20]([F:7])[Cl:44]>>"
        "[Cl-:44].[CH3:11][C@@H:20]([F:7])[OH:9]"
    )

    first = GenericStereoRuleExtractor().extract(SN2).certificate
    second = GenericStereoRuleExtractor().extract(relabeled).certificate

    assert first.source_rule_digest == second.source_rule_digest
    assert first.generic_rule_digest == second.generic_rule_digest
    assert first.digest == second.digest


def test_unmapped_source_is_rejected_before_index_identity_can_be_invented() -> None:
    source = "C[C@H](F)Cl.[OH-]>>C[C@@H](F)O.[Cl-]"

    with pytest.raises(GenericStereoExtractionError) as error:
        GenericStereoRuleExtractor().extract(source)

    assert error.value.issues[0].code is (
        GenericStereoExtractionIssueCode.INPUT_NOT_MAPPED
    )


def test_assignment_cap_raises_instead_of_truncating_port_permutations() -> None:
    policy = GenericStereoRulePolicy(
        domain_source="class",
        explicit_domains={
            1: {"elements": {"C", "F"}},
            3: {"elements": {"C", "F"}},
        },
    )
    rule = GenericStereoRuleExtractor(policy).extract(SN2).rule
    reactor = SynReactor(
        "C[C@H](F)Cl.[OH-]",
        rule,
        template_format="tuple",
        explicit_h=False,
        stereo_mode="strict",
        automorphism=False,
        stereo_assignment_limit=1,
    )

    with pytest.raises(StereoWildcardAssignmentLimitError) as error:
        _ = reactor.mappings

    assert error.value.limit == 1
    assert error.value.discovered == 2


def test_exact_product_quotient_retains_homotopic_application_multiplicity() -> None:
    rule = GenericStereoRuleExtractor().extract(SN2).rule
    reactor = SynReactor(
        "C[C@H](F)Cl.[OH-].[OH-]",
        rule,
        template_format="tuple",
        explicit_h=False,
        stereo_mode="strict",
        automorphism=True,
    )

    assert reactor.mapping_count == 2
    assert len(reactor.its_list) == 1
    orbit = reactor.its_list[0].graph["application_orbit"]
    assert orbit["multiplicity"] == 2
    assert [application["mapping_index"] for application in orbit["applications"]] == [
        0,
        1,
    ]
    assert all(
        application["port_assignment"]
        and application["stereo_morphism"]["certificates"]
        for application in orbit["applications"]
    )


def test_observed_policy_cannot_be_silently_broadened() -> None:
    with pytest.raises(ValueError, match="class.*corpus"):
        GenericStereoRulePolicy(explicit_domains={3: {"elements": {"F", "Br"}}})


def test_corpus_domains_require_and_record_aligned_evidence() -> None:
    missing = GenericStereoRuleExtractor(
        GenericStereoRulePolicy(
            domain_source="corpus",
            explicit_domains={
                1: {"elements": {"C"}},
                3: {"elements": {"F", "Br"}},
            },
        )
    )

    with pytest.raises(GenericStereoExtractionError) as error:
        missing.extract(SN2)
    assert (
        error.value.issues[0].code is GenericStereoExtractionIssueCode.DOMAIN_REQUIRED
    )

    result = GenericStereoRuleExtractor(
        GenericStereoRulePolicy(
            domain_source="corpus",
            explicit_domains={
                1: {"elements": {"C"}},
                3: {"elements": {"F", "Br"}},
            },
            domain_evidence={
                1: ("record-001", "record-002"),
                3: ("record-001", "record-002"),
            },
        )
    ).extract(SN2)

    assert all(port.domain_evidence for port in result.certificate.ports)
