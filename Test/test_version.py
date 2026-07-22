"""Release-version consistency tests."""

from pathlib import Path
import tomllib

import synkit


def test_source_version_matches_project_metadata() -> None:
    project = Path(__file__).resolve().parents[1] / "pyproject.toml"
    version = tomllib.loads(project.read_text(encoding="utf-8"))["project"]["version"]

    assert synkit.__version__ == version
    assert version == "1.6.0"
