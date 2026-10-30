#!/usr/bin/env python
"""Shared helpers for the SynKit CRN experiments.

Keeps the environment block, evidence-file writing, and timing behaviour
identical across the studies so their outputs can be compared and merged.
"""

from __future__ import annotations

import json
import hashlib
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

#: Directory holding cached inputs that the studies must not re-download.
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def source_hashes() -> Dict[str, str]:
    """Fingerprint analysis sources and the reviewed fixture, excluding outputs."""
    files = []
    for directory in (
        REPOSITORY_ROOT / "synkit/CRN",
        REPOSITORY_ROOT / "Experiment/CRN",
    ):
        for path in directory.rglob("*"):
            if not path.is_file() or path.suffix not in {".py", ".cpp", ".h", ".hpp"}:
                continue
            if "results" in path.relative_to(directory).parts:
                continue
            files.append(path)
    files.extend(sorted(DATA_DIR.glob("*.json")))
    return {
        str(path.relative_to(REPOSITORY_ROOT)): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(files)
    }


def environment() -> Dict[str, Any]:
    """Return the environment block recorded in every evidence file.

    Timings are only interpretable next to the machine that produced them, so
    every study records this.

    :return: Interpreter, library and platform description.
    :rtype: Dict[str, Any]
    """
    import networkx as nx
    import numpy as np
    import scipy
    from rdkit import rdBase

    import synkit
    from synkit.CRN.backend import backend_metadata

    def _git(*arguments: str) -> Optional[str]:
        """Return stripped Git output, or ``None`` when unavailable.

        :param arguments: Git command arguments.
        :type arguments: str
        :return: Command output or ``None``.
        :rtype: Optional[str]
        """
        try:
            result = subprocess.run(
                ["git", *arguments],
                cwd=REPOSITORY_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")

    return {
        "crn_backend": backend_metadata(),
        "synkit": getattr(synkit, "__version__", "unknown"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "networkx": nx.__version__,
        "rdkit": rdBase.rdkitVersion,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "git_commit": commit,
        "git_dirty": None if status is None else bool(status),
    }


def timed(
    function: Callable[[], Any],
    *,
    budget: Optional[float] = None,
) -> Tuple[Optional[float], Any, Optional[str]]:
    """Time a call, optionally abandoning it after a wall-clock budget.

    The budget is enforced with :mod:`signal` alarms, so it applies only on
    platforms with ``SIGALRM`` and only to the main thread. Where unavailable
    the call runs to completion.

    :param function: Zero-argument callable to time.
    :type function: Callable[[], Any]
    :param budget: Optional wall-clock budget in seconds.
    :type budget: Optional[float]
    :return:
        Triple ``(seconds, result, error)``. On timeout ``seconds`` and
        ``result`` are ``None`` and ``error`` is ``"timeout"``.
    :rtype: Tuple[Optional[float], Any, Optional[str]]
    """
    if budget is None or not hasattr(__import__("signal"), "SIGALRM"):
        started = time.perf_counter()
        try:
            value = function()
        except Exception as exc:  # pragma: no cover - reported, not raised
            return None, None, f"{type(exc).__name__}: {exc}"
        return time.perf_counter() - started, value, None

    import signal

    class _Timeout(Exception):
        pass

    def _raise(_signum: int, _frame: Any) -> None:
        raise _Timeout()

    previous = signal.signal(signal.SIGALRM, _raise)
    signal.setitimer(signal.ITIMER_REAL, budget)
    started = time.perf_counter()
    try:
        value = function()
        return time.perf_counter() - started, value, None
    except _Timeout:
        return None, None, "timeout"
    except Exception as exc:  # pragma: no cover - reported, not raised
        return None, None, f"{type(exc).__name__}: {exc}"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def write_report(report: Dict[str, Any], output: Optional[Path]) -> int:
    """Serialize an evidence report and return the process exit status.

    :param report: Report payload; must carry ``schema`` and ``status``.
    :type report: Dict[str, Any]
    :param output: Destination path, or ``None`` to print to standard output.
    :type output: Optional[pathlib.Path]
    :return: ``0`` when the report status is ``PASS``, otherwise ``1``.
    :rtype: int
    """
    payload = json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if output is None:
        print(payload, end="")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    return 0 if report.get("status") == "PASS" else 1


def load_report(path: Path, *, expected_schema: str) -> Dict[str, Any]:
    """Load an evidence file and reject an unexpected schema or a failure.

    :param path: Evidence file to load.
    :type path: pathlib.Path
    :param expected_schema: Schema the file must declare.
    :type expected_schema: str
    :return: Parsed report.
    :rtype: Dict[str, Any]
    :raises ValueError: If the schema does not match or the status is not ``PASS``.
    """
    report = json.loads(path.read_text(encoding="utf-8"))
    schema = report.get("schema")
    if schema != expected_schema:
        raise ValueError(
            f"{path}: expected schema {expected_schema!r}, found {schema!r}"
        )
    if report.get("status") != "PASS":
        raise ValueError(f"{path}: status is {report.get('status')!r}, not 'PASS'")
    return report


def native_experiment(function):
    """CLI production analyses default to a verified C++ backend.

    Reference calculations retain their independent implementations. Compilation
    happens only when an experiment is invoked, never during imports.
    """
    from functools import wraps
    import inspect

    @wraps(function)
    def wrapped(*args, **kwargs):
        import argparse
        from synkit.CRN import build_native, native_backend
        from synkit.CRN.backend import resolve_backend, python_backend

        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--backend", choices=["cpp", "python"], default="cpp")
        parser.add_argument("--native-library", type=Path)
        parser.add_argument(
            "--native-build-dir",
            type=Path,
            default=Path("/tmp/synkit-crn-native"),
        )
        parser.add_argument("--boost-include-dir", type=Path)
        supplied = args[0] if args else kwargs.get("argv")
        options, remaining = parser.parse_known_args(
            sys.argv[1:] if supplied is None else supplied
        )

        def invoke():
            if "argv" in inspect.signature(function).parameters:
                return function(remaining)
            previous = sys.argv
            try:
                sys.argv = [previous[0], *remaining]
                return function()
            finally:
                sys.argv = previous

        if "--help" in remaining or "-h" in remaining:
            print(parser.format_help())
            return invoke()
        if options.backend == "python":
            with python_backend():
                return invoke()
        _, active = resolve_backend()
        library = (
            options.native_library
            or active
            or build_native(
                options.native_build_dir,
                boost_include_dir=options.boost_include_dir,
                with_semiflows=True,
            )
        )
        manifest = json.loads(Path(library).with_suffix(".json").read_text())
        required = {
            "exact_semiflows",
            "exact_algebra",
            "crnt",
            "pathway",
            "symmetry",
            "structural_sets",
            "exhaustive_siphons",
        }
        if not required <= set(manifest.get("capabilities", [])):
            raise RuntimeError(
                "CRN experiments require a full native library; rebuild with build_native(..., with_semiflows=True)"
            )
        with native_backend(library):
            return invoke()

    return wrapped
