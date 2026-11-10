"""Concurrent explicit builds must not replace a library already in use."""

from concurrent.futures import ThreadPoolExecutor
import ctypes
import hashlib
import json
import threading

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
