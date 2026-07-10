import importlib


def test_slap_package_exports_matcher_and_lap_helpers():
    module = importlib.import_module("synkit.Chem.Reaction.Mapper.slap")

    assert module.GraphMatcher is not None
    assert callable(module.solve_lap)
    assert callable(module.chemical_distance)
