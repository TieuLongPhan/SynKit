"""Distribution and SDK identity checks without compiling or loading a solver."""

import hashlib
import re
from pathlib import Path

import pytest

from scripts import check_package_artifacts
from synkit.Chem.Mapper._native.build import _ortools_libraries, build_mapper_native


@pytest.mark.parametrize(
    "relative",
    [
        "lib/libortools.so.9",
        "lib64/libortools.dylib",
        "lib/ortools.lib",
        "bin/ortools.dll",
    ],
)
def test_optional_sdk_identity_covers_host_library_names(tmp_path, relative):
    library = tmp_path / relative
    library.parent.mkdir(parents=True)
    library.write_bytes(b"SDK library identity")
    assert _ortools_libraries(tmp_path) == {
        str(library.resolve()): hashlib.sha256(library.read_bytes()).hexdigest()
    }
    library.write_bytes(b"different SDK library")
    assert (
        _ortools_libraries(tmp_path)[str(library.resolve())]
        == hashlib.sha256(library.read_bytes()).hexdigest()
    )


def test_missing_optional_sdk_has_an_explicit_error(tmp_path):
    with pytest.raises(RuntimeError, match="OR-Tools C\\+\\+ libraries"):
        _ortools_libraries(tmp_path)


@pytest.mark.parametrize("value", [0, 1, None, "false"])
def test_oracle_exports_require_an_explicit_boolean(tmp_path, value):
    with pytest.raises(TypeError, match="test_hooks"):
        build_mapper_native(tmp_path, test_hooks=value)


@pytest.mark.parametrize(
    "omitted",
    ["provider.inc", "vendor.c", "COPYRIGHT", "LICENSE-2.0.txt"],
)
def test_distribution_cannot_omit_native_support_files(tmp_path, monkeypatch, omitted):
    project = tmp_path / "project"
    monkeypatch.setattr(
        check_package_artifacts,
        "__file__",
        str(project / "scripts/check_package_artifacts.py"),
    )
    native = project / "synkit/Chem/Mapper/solvers/native/cpp/src/vendor"
    native.mkdir(parents=True)
    for name in ("provider.inc", "vendor.c", "COPYRIGHT", "LICENSE-2.0.txt"):
        (native / name).write_text("required source or license\n")
    members = {
        "synkit/__init__.py",
        "synkit/Chem/Mapper/mapper.py",
        "synkit/Chem/Mapper/solvers/exact/native_build.py",
        "synkit/Chem/Mapper/solvers/exact/native/build.py",
        "synkit/Chem/Mapper/solvers/exact/native/native_distance.cpp",
        "synkit/Chem/Mapper/solvers/native/build.py",
        "synkit/Chem/Mapper/solvers/native/implicit.py",
        "synkit/Chem/Mapper/solvers/native/cpp/CMakeLists.txt",
        "synkit/Chem/Mapper/solvers/native/cpp/include/synkit_mapper/mapper.h",
        *(f"synkit/Data/{name}" for name in check_package_artifacts.DATA_FILES),
        *(
            f"synkit/{package}/{name}"
            for package, name in check_package_artifacts.PACKAGE_DATA
        ),
        *(path.relative_to(project).as_posix() for path in native.iterdir()),
    }
    artifact = Path("test.whl")
    check_package_artifacts._assert_member_suffixes(members, artifact)
    member = (native / omitted).relative_to(project).as_posix()
    members.remove(member)
    with pytest.raises(RuntimeError, match=re.escape(omitted)):
        check_package_artifacts._assert_member_suffixes(members, artifact)


@pytest.mark.parametrize("omitted", check_package_artifacts.GUIDE_DOCS)
def test_source_distribution_cannot_omit_mapper_documentation(omitted):
    members = {"synkit-1.6.3/" + name for name in check_package_artifacts.GUIDE_DOCS}
    artifact = Path("test.tar.gz")
    check_package_artifacts._assert_source_documentation(members, artifact)
    members.remove("synkit-1.6.3/" + omitted)
    with pytest.raises(RuntimeError, match=re.escape(omitted)):
        check_package_artifacts._assert_source_documentation(members, artifact)
