"""Native rule-based CRN expansion against an RDKit C++ SDK.

The expansion controller runs in C++ (``cpp/expand.cpp``): it keeps the
universe and frontier of canonical molecules, binds each molecule to rule
reactant templates once, forms only mixtures that admit a complete binding
(MOD-style incremental binding), applies the rules with RDKit's compiled
reaction engine, and deduplicates reaction events. Python only builds the
request and writes the returned events into the reaction graph.

The executable runs out of process, so the RDKit C++ SDK it links never shares
an address space with the RDKit Python package. Builds are explicit and
immutable; importing this module never invokes a compiler.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROTOCOL = 1
EXECUTABLE = "synkit_crn_expand"
_SOURCE = Path(__file__).resolve().parent / "cpp"
_EXPANDER = ContextVar("synkit_crn_native_expander", default=None)
_SERVER = ContextVar("synkit_crn_native_expander_server", default=None)


class _Server:
    """One verified ``--serve`` process answering newline-delimited requests."""

    def __init__(self, executable: Path):
        import threading

        self.executable = executable
        self._lock = threading.Lock()
        self._process = None

    def _start(self):
        self._process = subprocess.Popen(
            [str(self.executable), "--serve"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

    def request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                self._start()
            self._process.stdin.write(json.dumps(request) + "\n")
            self._process.stdin.flush()
            line = self._process.stdout.readline()
        if not line:
            self.close()
            raise RuntimeError("Native CRN expander server exited without a result")
        return _check(json.loads(line), "")

    def close(self):
        process, self._process = self._process, None
        if process is not None:
            try:
                process.stdin.close()
                process.wait(timeout=10)
            except Exception:
                process.kill()


def _digest(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _run(command: Sequence[str]) -> str:
    result = subprocess.run(list(command), capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError("Native CRN expander build failed:\n" + (result.stdout + result.stderr)[-6000:])
    return result.stdout


def _sdk_inputs(sdk: Path) -> Dict[str, str]:
    records = {}
    for directory in (sdk / "include/rdkit", sdk / "lib/cmake/rdkit"):
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                records[str(path.relative_to(sdk))] = _digest(path)
    for path in sorted((sdk / "lib").glob("libRDKit*.so*")):
        if path.is_file() and not path.is_symlink():
            records[str(path.relative_to(sdk))] = _digest(path)
    return records


def build_native_expander(
    output_dir,
    *,
    rdkit_dir,
    compiler: Optional[str] = None,
) -> Path:
    """Build and verify the native expansion executable.

    :param output_dir: Directory receiving immutable, content-addressed builds.
    :param rdkit_dir: Prefix of an RDKit C++ SDK (headers, libraries and the
        ``rdkit`` CMake package, plus Boost headers).
    :param compiler: Optional C++20 compiler.
    :return: Path of the verified executable. Its manifest sits beside it.
    """
    sdk = Path(rdkit_dir).expanduser().resolve()
    if not (sdk / "include/rdkit/GraphMol/ChemReactions/Reaction.h").is_file():
        raise ValueError("rdkit_dir lacks RDKit C++ reaction headers")
    cmake = shutil.which("cmake")
    if cmake is None:
        raise RuntimeError("CMake is required for a native CRN expander build")
    compiler_path = None
    if compiler is not None:
        compiler_path = shutil.which(compiler)
        if compiler_path is None:
            raise RuntimeError(f"C++20 compiler unavailable: {compiler}")
    sources = {
        path.name: _digest(path)
        for path in sorted(_SOURCE.iterdir())
        if path.is_file() and path.suffix in {".cpp", ".h", ".txt"}
    }
    configuration = {
        "protocol": PROTOCOL,
        "builder_sha256": _digest(__file__),
        "sources": sources,
        "rdkit_dir": str(sdk),
        "sdk_inputs_sha256": hashlib.sha256(json.dumps(_sdk_inputs(sdk), sort_keys=True).encode()).hexdigest(),
        "compiler": compiler_path,
        "platform": sys.platform,
        "machine": platform.machine(),
        "environment": {name: os.environ.get(name) for name in ("CXX", "CXXFLAGS", "LDFLAGS")},
    }
    key = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    suffix = ".exe" if os.name == "nt" else ""
    executable = output / f"{EXECUTABLE}_{key[:16]}{suffix}"
    manifest = executable.with_name(executable.name + ".json")
    if executable.exists():
        return _verify(executable, manifest, configuration)
    with tempfile.TemporaryDirectory(prefix="crn-expand-build-", dir=output) as folder:
        work = Path(folder)
        snapshot = work / "sources"
        snapshot.mkdir()
        for name in sources:
            shutil.copy2(_SOURCE / name, snapshot / name)
        build = work / "build"
        command = [
            cmake,
            "-S",
            str(snapshot),
            "-B",
            str(build),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_PREFIX_PATH={sdk}",
            f"-DCMAKE_BUILD_RPATH={sdk / 'lib'}",
        ]
        if compiler_path is not None:
            command.append(f"-DCMAKE_CXX_COMPILER={compiler_path}")
        _run(command)
        _run([cmake, "--build", str(build), "--config", "Release", "--parallel", "2"])
        candidates = [p for p in build.rglob(EXECUTABLE + suffix) if p.is_file()]
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one native expander, found {candidates}")
        built = candidates[0]
        probe = _probe(built)
        cache = {}
        for line in (build / "CMakeCache.txt").read_text().splitlines():
            name, separator, value = line.partition("=")
            if separator and name.startswith(("CMAKE_CXX_COMPILER", "CMAKE_CXX_FLAGS", "CMAKE_BUILD_TYPE")):
                cache[name] = value
        shutil.copy2(built, executable)
        record = {
            "configuration": configuration,
            "build_sha256": key,
            "executable_sha256": _digest(executable),
            "rdkit_version": probe["rdkit_version"],
            "cmake_cache": cache,
        }
        manifest.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return executable


def _probe(executable: Path) -> Dict[str, Any]:
    result = _invoke(executable, {"protocol": PROTOCOL, "rules": [], "seeds": ["C"]})
    if result.get("status") != "complete" or result.get("species") != ["C"]:
        raise RuntimeError(f"Native CRN expander failed its probe: {result}")
    return result


def _verify(executable: Path, manifest: Path, configuration=None) -> Path:
    try:
        record = json.loads(manifest.read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError("Native CRN expander has no valid manifest") from exc
    if configuration is not None and record.get("configuration") != configuration:
        raise RuntimeError("Native CRN expander configuration mismatch")
    if record.get("executable_sha256") != _digest(executable):
        raise RuntimeError("Native CRN expander checksum mismatch")
    return executable


def expander_manifest(executable) -> Dict[str, Any]:
    """Return the verified build manifest of a native expander."""
    executable = Path(executable).resolve()
    manifest = executable.with_name(executable.name + ".json")
    _verify(executable, manifest)
    return json.loads(manifest.read_text())


def _invoke(executable: Path, request: Dict[str, Any]) -> Dict[str, Any]:
    result = subprocess.run(
        [str(executable)],
        input=json.dumps(request),
        capture_output=True,
        text=True,
    )
    try:
        payload = json.loads(result.stdout)
    except ValueError:
        raise RuntimeError("Native CRN expander produced no result:\n" + result.stderr[-3000:]) from None
    return _check(payload, result.stderr)


def _check(payload: Dict[str, Any], stderr: str) -> Dict[str, Any]:
    if payload.get("status") == "error":
        raise RuntimeError(f"Native CRN expansion failed: {payload.get('error')}")
    if payload.get("protocol") != PROTOCOL:
        raise RuntimeError("Unsupported native CRN expander protocol")
    return payload


def resolve_expander(executable=None) -> Optional[Path]:
    """Return an explicit or task-local native expander, if any."""
    path = executable if executable is not None else _EXPANDER.get()
    return None if path is None else Path(path).resolve()


@contextmanager
def native_expander(executable):
    """Use a verified native expander for ``CRNExpand(rule_backend="native")``.

    The executable is verified once and kept running for the whole context,
    so repeated expansions do not pay process start-up and RDKit loading.
    """
    path = Path(executable).resolve()
    expander_manifest(path)
    server = _Server(path)
    token = _EXPANDER.set(path)
    server_token = _SERVER.set(server)
    try:
        yield path
    finally:
        _SERVER.reset(server_token)
        _EXPANDER.reset(token)
        server.close()


def run_native_expansion(
    executable,
    *,
    rules: List[str],
    arities: List[int],
    seeds: List[str],
    options: Dict[str, Any],
) -> Dict[str, Any]:
    """Run one expansion and return the native event stream."""
    path = Path(executable).resolve()
    server = _SERVER.get()
    if server is None or server.executable != path:
        server = None
        expander_manifest(path)
    request = {
        "protocol": PROTOCOL,
        "rules": [{"smarts": r, "arity": int(a)} for r, a in zip(rules, arities)],
        "seeds": list(seeds),
        "options": options,
    }
    from ..backend import record_operation

    record_operation("native_rule_expansion")
    return server.request(request) if server is not None else _invoke(path, request)
