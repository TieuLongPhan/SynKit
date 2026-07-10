import importlib


def test_chem_package_exports_core_helpers():
    module = importlib.import_module("synkit.Chem.Reaction.Mapper.chem")

    assert module.AAMapper is not None
    assert callable(module.smiles2lgp)
    assert callable(module.dedup_mapped_rxns)
