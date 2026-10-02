"""Portable imports and optional Unix limits for native mapper workers."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from synkit.Chem.Mapper.exact import native_parallel


def test_mapper_and_campaign_imports_without_unix_resource():
    root = Path(__file__).resolve().parents[4]
    script = """
import sys
sys.modules['resource'] = None
from synkit.Chem.Mapper.exact import native_frontier, native_parallel
from synkit.Chem.Mapper import native_analysis
from scripts import run_synister_global_shells, run_synister_alternative_its
assert native_frontier.resource is native_parallel.resource is None
for configure in (
    run_synister_global_shells._configure_memory_limit,
    run_synister_alternative_its._configure_memory_limit,
):
    try:
        configure(1)
    except RuntimeError as error:
        assert 'Unix address-space resource limits' in str(error)
    else:
        raise AssertionError('campaign claimed an unavailable memory limit')
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=root, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_native_worker_initializes_without_unix_limits(monkeypatch):
    monkeypatch.setattr(native_parallel, "resource", None)
    monkeypatch.setattr(native_parallel, "_STATE", None)
    monkeypatch.setattr(native_parallel, "_READY", True)
    native_parallel._init_worker("counter", "deadline", "barrier", 12)
    assert native_parallel._STATE == ("counter", "deadline", "barrier", 12)
    assert native_parallel._READY is False


def test_native_worker_retains_supported_address_space_limit(monkeypatch):
    calls = []
    resource = SimpleNamespace(
        RLIMIT_AS=9, setrlimit=lambda kind, limits: calls.append((kind, limits))
    )
    monkeypatch.setattr(native_parallel, "resource", resource)
    monkeypatch.setattr(native_parallel, "_STATE", None)
    monkeypatch.setattr(native_parallel, "_READY", False)
    native_parallel._init_worker(None, None, None, None)
    assert calls == [(9, (4 * 1024**3, 4 * 1024**3))]
