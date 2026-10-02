import sys
from types import SimpleNamespace

from Experiment.Synister import development


def test_vendored_distribution_cannot_override_selected_version(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "rxnmapper", SimpleNamespace(__file__=str(tmp_path / "__init__.py")))
    distributions = [SimpleNamespace(metadata={"Name": "packaging"}, version="26.2"),
                     SimpleNamespace(metadata={"Name": "packaging"}, version="24.2")]
    monkeypatch.setattr(development.importlib.metadata, "distributions", lambda: distributions)
    monkeypatch.setattr(development.importlib.metadata, "version", lambda name: "26.2")
    assert development.environment()["packages"] == {"packaging": "26.2"}
