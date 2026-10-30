"""Use checkout imports and share the optional native compiler fixture."""

import os
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]

if (ROOT / "synkit").is_dir() and str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

loaded = sys.modules.get("synkit")
if (
    (ROOT / "synkit").is_dir()
    and loaded is not None
    and not str(getattr(loaded, "__file__", "")).startswith(str(ROOT))
):
    for name in list(sys.modules):
        if name == "synkit" or name.startswith("synkit."):
            del sys.modules[name]


@pytest.fixture(scope="session")
def native_library(tmp_path_factory):
    """Compile the optional kernel once, or use an explicitly supplied artifact."""
    verified = os.environ.get("SYNKIT_TEST_NATIVE_LIBRARY")
    if verified:
        path = Path(verified).resolve()
        assert path.is_file(), path
        return path
    if shutil.which("g++") is None:
        pytest.skip("optional native kernel requires a C++17 compiler")
    from synkit.Chem.Mapper.exact.native_build import build_native

    return build_native(tmp_path_factory.mktemp("synkit-native"))


@pytest.fixture(scope="session")
def mapper_core_library(tmp_path_factory):
    """Use an explicit shared core or build the current shipped sources once."""
    configured = os.environ.get("SYNKIT_CORE_LIBRARY") or os.environ.get(
        "SYNKIT_MAPPER_OPTIONAL_LIBRARY"
    )
    if configured:
        path = Path(configured).resolve()
        assert path.is_file(), path
        return path
    if shutil.which("g++") is None:
        pytest.skip("optional mapper core requires a C++17 compiler")
    from synkit.Chem.Mapper._native.build import build_mapper_native

    return build_mapper_native(
        tmp_path_factory.mktemp("synkit-core"),
        ortools_dir=os.environ.get("SYNKIT_ORTOOLS_DIR"),
    )


@pytest.fixture(scope="session")
def mapper_core_test_library(tmp_path_factory):
    """Build oracle-only exports explicitly, apart from production artifacts."""
    configured = os.environ.get("SYNKIT_CORE_TEST_LIBRARY")
    if configured:
        path = Path(configured).resolve()
        assert path.is_file(), path
        return path
    if shutil.which("g++") is None:
        pytest.skip("optional mapper test core requires a C++17 compiler")
    from synkit.Chem.Mapper._native.build import build_mapper_native

    return build_mapper_native(
        tmp_path_factory.mktemp("synkit-core-oracles"),
        ortools_dir=os.environ.get("SYNKIT_ORTOOLS_DIR"),
        test_hooks=True,
    )
