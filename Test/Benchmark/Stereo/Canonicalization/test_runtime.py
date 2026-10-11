"""Platform regressions for experiment imports, native memory and deadlines."""

import json
import os
from pathlib import Path
import subprocess
import sys
from textwrap import dedent
from types import SimpleNamespace

import pytest

from Experiment.Stereo.Canonicalization import runtime


@pytest.mark.parametrize("platform,reported_peak", [("linux", 128 * 1024), ("darwin", 128 * 1024 * 1024)])
def test_posix_peak_memory_uses_the_platform_units(monkeypatch, platform, reported_peak):
    monkeypatch.setattr(runtime, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setattr(runtime, "resource", SimpleNamespace(
        RUSAGE_SELF=0, getrusage=lambda _: SimpleNamespace(ru_maxrss=reported_peak),
    ))
    assert runtime.peak_rss_mib() == 128


def test_windows_measurements_include_native_peak_memory_and_cpu(monkeypatch):
    process = SimpleNamespace(
        memory_info=lambda: SimpleNamespace(rss=32 * runtime.MIB, peak_wset=128 * runtime.MIB),
        cpu_times=lambda: SimpleNamespace(user=1.0, system=0.5),
    )
    monkeypatch.setattr(runtime, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(runtime, "resource", None)
    monkeypatch.setattr(runtime.psutil, "Process", lambda _: process)
    assert runtime.process_rss_mib(123) == runtime.peak_rss_mib() == 128
    assert runtime.process_cpu_seconds(123) == 1.5


def test_missing_load_average_is_explicit(monkeypatch):
    monkeypatch.setattr(runtime, "os", SimpleNamespace())
    assert runtime.load_average() == (None, None, None)


def test_without_posix_alarms_a_late_result_is_rejected(monkeypatch):
    clock = iter((10.0, 12.0))
    monkeypatch.setattr(runtime, "signal", SimpleNamespace())
    monkeypatch.setattr(runtime.time, "monotonic", lambda: next(clock))
    with pytest.raises(TimeoutError, match="expired"):
        with runtime.wall_time_limit(1.0, TimeoutError("expired")):
            pass


def test_experiments_work_without_unix_resource_timers_or_load_average():
    script = dedent("""
        import importlib
        import json
        import os
        import signal
        import sys

        # Linux psutil itself uses resource; load its platform backend first.
        import psutil
        sys.modules['resource'] = None
        for name in ('SIGALRM', 'ITIMER_REAL', 'setitimer'):
            if hasattr(signal, name):
                delattr(signal, name)
        for name in ('getloadavg', 'sched_getaffinity'):
            if hasattr(os, name):
                delattr(os, name)

        for name in (
            'Canonicalization.independent_orbits', 'Canonicalization.independent_runner',
            'Canonicalization.performance', 'Canonicalization.registration',
            'ChemicalValidation.evaluate', 'ChemicalValidation.rota_evaluate',
        ):
            importlib.import_module('Experiment.Stereo.' + name)
        from Experiment.Stereo.Canonicalization.independent_fixtures import fixtures
        from Experiment.Stereo.Canonicalization.independent_model import formal_states
        from Experiment.Stereo.Canonicalization.independent_orbits import reference_partition
        from Experiment.Stereo.Canonicalization.independent_runner import _code
        from Experiment.Stereo.Canonicalization.performance import hardware
        from Experiment.Stereo.Canonicalization.runtime import peak_rss_mib
        from Experiment.Stereo.ChemicalValidation.common import wall_limit
        from synkit.Graph.Stereo import TetrahedralStereo

        fixture = next(value for value in fixtures() if value.name == 'tetrahedral:distinct')
        choices = tuple(formal_states(seed) for seed in fixture.seeds)
        with wall_limit(10):
            complete = reference_partition(fixture.graph, choices, atom_color='color', bond_color='color')
            refused = reference_partition(fixture.graph, choices, max_rss_mb=0)
            code = _code(fixture.graph, (TetrahedralStereo((0, 1, 2, 3, 4), 1),),
                         {'atom_color': 'color', 'bond_color': 'color'}, 10)
        assert complete['status'] == 'complete'
        assert len(complete['partition']) == 2
        assert refused['status'] == 'oracle_incomplete'
        assert refused['reason'] == 'oracle_process_peak_rss_limit'
        assert code and peak_rss_mib() > 0
        host = hardware()
        assert host['load_average_at_report'] == (None, None, None)
        assert host['cpu_affinity'] is None
        print(json.dumps({'status': 'passed'}))
    """)
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[4],
        env=dict(os.environ), capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "passed"
