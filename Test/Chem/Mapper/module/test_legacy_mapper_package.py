import importlib


def test_legacy_mapper_namespace_aliases_canonical_modules():
    canonical = importlib.import_module("synkit.Chem.Mapper")
    legacy = importlib.import_module("synkit.Chem.Reaction.Mapper")
    canonical_kernel = importlib.import_module("synkit.Chem.Mapper.exact.kernel")
    legacy_kernel = importlib.import_module(
        "synkit.Chem.Reaction.Mapper.exact.kernel"
    )

    assert legacy.AAMapper is canonical.AAMapper
    assert legacy_kernel is canonical_kernel
    assert legacy_kernel.Kernel is canonical_kernel.Kernel
