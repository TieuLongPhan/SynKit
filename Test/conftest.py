"""Set thread defaults and collect experiment suites when available."""

import os
from pathlib import Path

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
