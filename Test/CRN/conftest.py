"""Shared native build used by CRN regression tests."""

import pytest

from synkit.CRN import build_native, bundled_native_library
from synkit.CRN.native import _boost_include


@pytest.fixture(scope="session")
def exact_library(native_build_cache):
    try:
        return bundled_native_library()
    except FileNotFoundError:
        pass
    if _boost_include() is None:
        pytest.skip("Install Boost headers for exact native enumeration tests")
    return build_native(native_build_cache / "crn", with_semiflows=True)
