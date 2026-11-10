"""Rule fusion ownership and compatibility across public import paths."""

import importlib
import pickle
import subprocess
import sys

import pytest


@pytest.mark.parametrize("module", ["", ".candidate", ".constructor", ".interface", ".search"])
def test_legacy_import_is_the_same_module(module):
    old = importlib.import_module("synkit.Graph.Fusion" + module)
    current = importlib.import_module("synkit.Rule.Fusion" + module)
    if module:
        assert old is current
    else:
        assert old.__all__ == current.__all__
        assert all(getattr(old, name) is getattr(current, name) for name in current.__all__)


def test_graph_identity_is_shared_and_class_imports_still_unpickle():
    old = importlib.import_module("synkit.Graph.Fusion.identity")
    current = importlib.import_module("synkit.Graph.identity")
    assert old is current
    from synkit.Rule.Fusion import FusionScore

    assert pickle.loads(pickle.dumps(FusionScore)) is FusionScore
    # Protocol zero names the historical module explicitly, as old pickles do.
    legacy = b"csynkit.Graph.Fusion.candidate\nFusionScore\n."
    assert pickle.loads(legacy) is FusionScore


@pytest.mark.parametrize("first", [
    "synkit.Graph.Fusion.constructor", "synkit.Rule.Fusion",
    "synkit.Graph.identity", "synkit.Synthesis.RBL.engine",
])
def test_fresh_import_orders(first):
    code = (
        "import importlib; importlib.import_module(" + repr(first) + "); "
        "from synkit.Graph.Fusion import FusionInterface as old; "
        "from synkit.Rule.Fusion import FusionInterface as current; "
        "assert old is current"
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)
