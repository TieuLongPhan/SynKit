"""Portable imports and optional Unix limits for native mapper workers."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from synkit.Chem.Mapper.exact import native_parallel
from synkit.Chem.Mapper.exact.native import runtime


def test_mapper_imports_without_unix_resource():
    root = Path(__file__).resolve().parents[4]
    script = """
import sys
sys.modules['resource'] = None
from synkit.Chem.Mapper.exact import native_frontier, native_parallel
from synkit.Chem.Mapper import native_analysis
assert native_frontier.resource is native_parallel.resource is None
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
        RLIMIT_AS=9, RLIM_INFINITY=-1, getrlimit=lambda _kind: (-1, -1),
        setrlimit=lambda kind, limits: calls.append((kind, limits)),
    )
    monkeypatch.setattr(native_parallel, "resource", resource)
    monkeypatch.setattr(native_parallel, "_STATE", None)
    monkeypatch.setattr(native_parallel, "_READY", False)
    native_parallel._init_worker(None, None, None, None)
    assert calls == [(9, (4 * 1024**3, -1))]


@pytest.mark.parametrize("soft,hard,expected", [(-1, -1, 4), (-1, 2, 2), (1, 2, 1), (5, 6, 4)])
def test_worker_memory_limit_respects_inherited_ceilings(soft, hard, expected):
    gib = 1024**3
    values = tuple(value * gib if value >= 0 else value for value in (soft, hard))
    calls = []
    resource = SimpleNamespace(
        RLIMIT_AS=9, RLIM_INFINITY=-1, getrlimit=lambda _kind: values,
        setrlimit=lambda kind, limits: calls.append((kind, limits)),
    )
    assert runtime.limit_worker_address_space(resource) == expected * gib
    assert calls == [(9, (expected * gib, values[1]))]


@pytest.mark.parametrize("error", [ValueError("current limit exceeds maximum limit"), OSError("unsupported")])
def test_worker_starts_when_optional_memory_limit_is_rejected(monkeypatch, error):
    def reject(_kind, _limits):
        raise error

    resource = SimpleNamespace(RLIMIT_AS=9, RLIM_INFINITY=-1, getrlimit=lambda _kind: (-1, -1), setrlimit=reject)
    assert runtime.limit_worker_address_space(resource) is None
    monkeypatch.setattr(native_parallel, "resource", resource)
    monkeypatch.setattr(native_parallel, "_STATE", None)
    monkeypatch.setattr(native_parallel, "_READY", True)
    native_parallel._init_worker("counter", "deadline", "barrier", 12)
    assert native_parallel._STATE == ("counter", "deadline", "barrier", 12)
    assert native_parallel._READY is False


def test_worker_cpu_pinning_is_optional(monkeypatch):
    monkeypatch.delattr(runtime.os, "sched_getaffinity", raising=False)
    monkeypatch.delattr(runtime.os, "sched_setaffinity", raising=False)
    assert runtime.worker_cpus(2) == []
    runtime.pin_worker_cpu(None)


def test_worker_cpu_pinning_failure_does_not_abort_search(monkeypatch):
    def denied(*_args):
        raise OSError("permission denied")

    monkeypatch.setattr(runtime.os, "sched_getaffinity", denied, raising=False)
    monkeypatch.setattr(runtime.os, "sched_setaffinity", denied, raising=False)
    assert runtime.worker_cpus(2) == []
    runtime.pin_worker_cpu(0)
