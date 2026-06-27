from synkit.Chem.Reaction.Mapper.chem.its import dedup_mapped_rxns


def test_its_dedup_keeps_unhashable_raw_mappings():
    results = [{"smiles": "not-a-reaction"}, {"smiles": "not-a-reaction"}]

    deduped = dedup_mapped_rxns(results)

    assert len(deduped) == 1
    assert deduped[0]["its_hash"] is None
