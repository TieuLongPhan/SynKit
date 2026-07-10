import importlib


def test_mapper_package_imports_public_api():
    module = importlib.import_module("synkit.Chem.Reaction.Mapper")

    assert module.AAMapper is not None
    assert module.GraphMatcher is not None
    assert module.LabeledGraph is not None
    assert module.RESEARCH_BASIS_DOI == "10.26434/chemrxiv-2025-hthwn"
