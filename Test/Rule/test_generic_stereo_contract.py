from copy import deepcopy

import pytest

from synkit.Graph.Stereo import StereoChange, TetrahedralStereo
from synkit.IO.chem_converter import rsmi_to_its
from synkit.Rule import (
    GenericStereoExtractionError,
    GenericStereoExtractionIssueCode,
    GenericStereoRuleExtractor,
    GenericStereoRulePolicy,
    RuleExtractionCertificate,
)

SN2 = "[CH3:1][C@H:2]([F:3])[Cl:4].[OH-:5]>>" "[CH3:1][C@@H:2]([F:3])[OH:5].[Cl-:4]"


def test_extraction_certificate_round_trip_verifies_its_digest():
    certificate = GenericStereoRuleExtractor().extract(SN2).certificate
    payload = certificate.to_dict()
    restored = RuleExtractionCertificate.from_dict(payload)

    assert len(payload["digest"]) == 64
    assert restored == certificate
    assert restored.digest == payload["digest"]


def test_extraction_certificate_tamper_is_rejected():
    payload = GenericStereoRuleExtractor().extract(SN2).certificate.to_dict()
    tampered = deepcopy(payload)
    tampered["source_replay"]["mapping_count"] += 1

    with pytest.raises(GenericStereoExtractionError) as error:
        RuleExtractionCertificate.from_dict(tampered)

    assert (
        error.value.issues[0].code
        is GenericStereoExtractionIssueCode.CERTIFICATE_INVALID
    )
    assert "digest mismatch" in error.value.issues[0].message


def test_extraction_rejects_dangling_stereo_support_before_rule_creation():
    its = rsmi_to_its(
        SN2,
        format="tuple",
        drop_non_aam=False,
        use_index_as_atom_map=True,
    )
    before = TetrahedralStereo((2, 1, 3, 99, "@H:2"), 1)
    after = before.invert()
    its.graph["stereo_descriptors"] = {
        "reactant": {"atom:2": before},
        "product": {"atom:2": after},
    }
    its.graph["stereo_changes"] = {"atom:2": StereoChange.from_endpoints(before, after)}

    extractor = GenericStereoRuleExtractor(
        GenericStereoRulePolicy(domain_source="exact")
    )
    with pytest.raises(GenericStereoExtractionError) as error:
        extractor.extract(its)

    assert (
        error.value.issues[0].code
        is GenericStereoExtractionIssueCode.INVALID_STEREO_SUPPORT
    )
    refusal = error.value.issues[0].context["refusals"][0]
    assert refusal["code"] == "INVALID_REFERENCE"
    assert "ligand atom map 99 is absent" in refusal["detail"]


def test_certificate_rejects_duplicate_port_bindings_even_with_new_digest():
    payload = GenericStereoRuleExtractor().extract(SN2).certificate.to_dict()
    payload["ports"].append(deepcopy(payload["ports"][0]))
    payload.pop("digest")

    with pytest.raises(GenericStereoExtractionError) as error:
        RuleExtractionCertificate.from_dict(payload)

    assert (
        error.value.issues[0].code
        is GenericStereoExtractionIssueCode.CERTIFICATE_INVALID
    )
    assert "duplicate port" in error.value.issues[0].message
