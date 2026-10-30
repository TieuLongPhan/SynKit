"""Bundled discovery must use a real compiled core and preserve explicit overrides."""

import shutil
import sys

import pytest

from synkit.Chem.Mapper import Mapper, map_reaction_implicit
from synkit.Chem.Mapper._native import runtime


def test_missing_bundle_reports_build_instructions_without_compiling(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "__file__", str(tmp_path / "runtime.py"))
    with pytest.raises(FileNotFoundError, match="library_path explicitly"):
        Mapper("native").map("CO>>CO")


def test_bundled_core_supports_public_and_implicit_apis(
    tmp_path, monkeypatch, mapper_core_library
):
    monkeypatch.setattr(runtime, "__file__", str(tmp_path / "runtime.py"))
    directory = tmp_path / "lib"
    directory.mkdir()
    name = (
        "synkit_mapper.dll" if sys.platform == "win32"
        else "libsynkit_mapper.dylib" if sys.platform == "darwin"
        else "libsynkit_mapper.so"
    )
    shutil.copyfile(mapper_core_library, directory / name)
    assert runtime.resolve_library_path() == directory / name
    available = Mapper.native_methods(available_only=True)
    assert {"slap", "pabs", "cpabs", "astar", "dfs", "sepdp"} <= set(available)
    assert set(available) <= set(Mapper.native_methods())
    result = Mapper("native").map("CO>>CO")
    assert result.complete and result.optimality_proven and result.minimum_cost == 0
    minimum = map_reaction_implicit("CCO>>COC")
    assert minimum.complete and minimum.minimum_cost > 0
    exact = map_reaction_implicit("CCO>>COC", CD=minimum.minimum_cost)
    assert exact.complete
    assert exact.mapped_reactions == minimum.mapped_reactions


def test_explicit_core_overrides_a_missing_bundle(
    tmp_path, monkeypatch, mapper_core_library
):
    monkeypatch.setattr(runtime, "__file__", str(tmp_path / "runtime.py"))
    result = Mapper("native", library_path=mapper_core_library).map("CO>>CO")
    assert result.complete and result.minimum_cost == 0
    assert "pabs" in Mapper.native_methods(
        available_only=True, library_path=mapper_core_library
    )
