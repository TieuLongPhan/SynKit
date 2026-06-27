import importlib


def test_exact_package_exports_solvers():
    module = importlib.import_module("synkit.Chem.Reaction.Mapper.exact")

    assert module.Certificate is not None
    assert callable(module.extract_kernel)
    assert callable(module.enumerate_kernel_optima)
