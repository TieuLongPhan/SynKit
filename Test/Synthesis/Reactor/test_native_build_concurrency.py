"""Concurrent explicit builds must not replace a library already in use."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import ctypes
import errno
import hashlib
import json
import threading
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from synkit.Synthesis.Reactor._native import build


def test_two_builders_publish_one_complete_library(tmp_path, monkeypatch):
    original_run = build.subprocess.run
    compilations = []
    ready = threading.Barrier(2)

    def recorded_run(command, *args, **kwargs):
        compilations.append(tuple(command))
        return original_run(command, *args, **kwargs)

    def worker():
        ready.wait()
        path = build.build_native_reactor(tmp_path)
        return path, ctypes.CDLL(str(path))

    monkeypatch.setattr(build.subprocess, "run", recorded_run)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: worker(), range(2)))
    assert results[0][0] == results[1][0]
    assert len([command for command in compilations if "--version" not in command]) == 2
    assert all(library.skp_abi() == 1 for _, library in results)
    path = results[0][0]
    manifest = json.loads(path.with_suffix(".json").read_text())
    assert manifest["library_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("failure_errno", [errno.EACCES, errno.EAGAIN])
def test_windows_build_lock_retries_contention_without_accessing_lock_bytes(tmp_path, monkeypatch, failure_errno):
    calls = []
    sleeps = []
    windows = SimpleNamespace(LK_NBLCK=1, LK_UNLCK=2)

    def locking(fd, mode, size):
        calls.append((mode, size))
        if mode == windows.LK_NBLCK and len(calls) <= 12:
            raise OSError(failure_errno, "lock held by another builder")

    windows.locking = locking
    monkeypatch.setitem(sys.modules, "msvcrt", windows)
    monkeypatch.setattr(build, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(build, "time", SimpleNamespace(sleep=sleeps.append))
    stream = Mock()
    stream.read.side_effect = PermissionError(errno.EACCES, "another handle owns the lock byte")
    original_open = type(tmp_path).open

    @contextmanager
    def open_lock_file(path, *args, **kwargs):
        with original_open(path, *args, **kwargs) as opened:
            stream.seek.side_effect = opened.seek
            stream.fileno.side_effect = opened.fileno
            yield stream

    monkeypatch.setattr(type(tmp_path), "open", open_lock_file)
    with build._build_lock(tmp_path, "test-lock"):
        assert calls[-1] == (windows.LK_NBLCK, 1)
        assert len(sleeps) == 12
    assert calls[-1] == (windows.LK_UNLCK, 1)
    stream.read.assert_not_called()
    stream.write.assert_not_called()


@pytest.mark.parametrize("failure_errno", [errno.EBADF, errno.EINVAL])
def test_windows_build_lock_propagates_unrelated_errors(tmp_path, monkeypatch, failure_errno):
    windows = SimpleNamespace(LK_NBLCK=1, LK_UNLCK=2)
    windows.locking = Mock(side_effect=OSError(failure_errno, "invalid lock request"))
    monkeypatch.setitem(sys.modules, "msvcrt", windows)
    monkeypatch.setattr(build, "os", SimpleNamespace(name="nt"))
    with pytest.raises(OSError) as caught:
        with build._build_lock(tmp_path, "test-lock"):
            pytest.fail("failed locking must not enter the build")
    assert caught.value.errno == failure_errno
    windows.locking.assert_called_once()
