"""Set thread defaults and collect experiment suites when available."""

import base64
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(variable, "1")


_EXPERIMENT_TESTS = {
    "MTG": [
        "Benchmark/test_mtg_validation.py",
        "Benchmark/test_mtg_paper_assets.py",
    ],
    "RBL": ["Benchmark/RBL", "Synthesis/RBL/test_verified_search.py"],
    "Stereo": ["Benchmark/Stereo"],
    "StereoReaction": ["Benchmark/test_stereo_rxn_validation.py"],
}
_EXPERIMENT_ROOT = Path(__file__).resolve().parents[1] / "Experiment"
collect_ignore = []
for experiment, tests in _EXPERIMENT_TESTS.items():
    if not (_EXPERIMENT_ROOT / experiment).is_dir():
        collect_ignore.extend(tests)


@pytest.fixture
def native_wheel_record(monkeypatch):
    """Register real library bytes and their exact synthetic installation path."""
    from synkit import _native

    def register(library, package):
        library, package = Path(library).resolve(), Path(package).resolve()
        digest = (
            base64.urlsafe_b64encode(hashlib.sha256(library.read_bytes()).digest())
            .decode()
            .rstrip("=")
        )

        class RecordPath(str):
            hash = SimpleNamespace(mode="sha256", value=digest)

        entry = RecordPath("synkit/" + library.relative_to(package).as_posix())
        installed = SimpleNamespace(
            files=[entry], locate_file=lambda name: package.parent / str(name)
        )
        monkeypatch.setattr(_native, "__file__", str(package / "_native.py"))
        monkeypatch.setattr(_native, "distribution", lambda name: installed)
        _native._conda_record_digest.cache_clear()
        _native._wheel_record_digest.cache_clear()
        return entry, installed

    yield register
    _native._conda_record_digest.cache_clear()
    _native._wheel_record_digest.cache_clear()
