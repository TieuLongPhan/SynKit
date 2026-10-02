from Experiment.Synister.select_development import endpoint_key, group, stratum
from synkit.Chem.Mapper.prediction_adapter import unmapped_input


def test_endpoint_key_removes_map_numbers_order_and_direction_only():
    key = lambda x: endpoint_key(unmapped_input(x))
    assert key("[CH3:10][OH:2].Cl>>CO.Cl") == key("Cl.OC>>Cl.CO")
    assert key("CC.O>>CO.C") == key("C.CO>>O.CC")
    assert key("CO.Cl>>CO.Cl") != key("CO.Cl.Cl>>CO.Cl.Cl")


def test_source_groups_and_input_bins():
    assert group("123:p1") == group("123:2") == "123"
    assert [stratum(n) for n in (20, 21, 40, 41, 60, 61, 80, 81)] == [
        "le20", "21to40", "21to40", "41to60", "41to60", "61to80", "61to80", "gt80"]
