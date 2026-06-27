import importlib


def test_io_package_exports_idxmap_helpers():
    module = importlib.import_module("synkit.Chem.Reaction.Mapper.io")

    assert callable(module.parse_index_string)
    assert callable(module.parse_index_mapping_string)
    assert callable(module.fmt_idxs)
