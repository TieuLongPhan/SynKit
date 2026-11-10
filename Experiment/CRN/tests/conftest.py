"""Native build shared by the CRN experiment tests."""

import os
from pathlib import Path

import pytest

from synkit.CRN import build_native, bundled_native_library
from synkit.CRN.native import _boost_include


@pytest.fixture(scope="session")
def exact_library(tmp_path_factory):
    try:
        return bundled_native_library()
    except FileNotFoundError:
        pass
    if _boost_include() is None:
        pytest.skip("Install Boost headers for exact native enumeration tests")
    configured = os.environ.get("SYNKIT_TEST_NATIVE_CACHE")
    output = Path(configured).expanduser().resolve() / "crn" if configured else tmp_path_factory.mktemp("crn-exact")
    return build_native(output, with_semiflows=True)
