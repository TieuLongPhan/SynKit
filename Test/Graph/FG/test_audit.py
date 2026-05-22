from synkit.Graph.FG.audit import audit_reaction_smiles


def test_audit_reaction_smiles_summarizes_small_corpus():
    report = audit_reaction_smiles(
        [
            "CCO>>CC=O",
            "c1ncnnc1>>c1ncnnc1",
        ]
    )

    assert report.reactions == 2
    assert report.molecules == 4
    assert report.parse_failures == 0
    assert report.label_counts["primary_alcohol"] == 1
    assert report.label_counts["aldehyde"] == 1
    assert report.label_counts["heteroaromatic_ring"] == 2
    assert report.label_counts["triazine"] == 2
    assert report.heteroaromatic_systems == 2
    assert report.named_heteroaromatic_systems == 2
    assert report.unnamed_heteroaromatic_count == 0
