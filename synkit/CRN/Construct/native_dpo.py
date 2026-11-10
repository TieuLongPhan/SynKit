"""Build and run the C++ CRN controller with the typed DPO reactor kernel."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import threading

PROTOCOL = 1
EXECUTABLE = "synkit_dpo_crn"
_SOURCE = Path(__file__).with_name("dpo_cpp")
_ACTIVE = ContextVar("synkit_native_dpo_expander", default=None)
_SERVER = ContextVar("synkit_native_dpo_server", default=None)


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _check(payload):
    if payload.get("protocol") != PROTOCOL:
        raise RuntimeError("Unsupported native DPO expansion protocol")
    if payload.get("status") in {"error", "unsupported"}:
        raise RuntimeError("Native DPO expansion " + payload["status"] + ": " + payload.get("error", "unknown error"))
    return payload


class _Server:
    def __init__(self, path):
        self.path = path
        self.process = None
        self.lock = threading.Lock()

    def request(self, payload):
        with self.lock:
            if self.process is None or self.process.poll() is not None:
                self.process = subprocess.Popen(
                    [str(self.path), "--serve"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    bufsize=1,
                )
            self.process.stdin.write(json.dumps(payload) + "\n")
            self.process.stdin.flush()
            line = self.process.stdout.readline()
        if not line:
            self.close()
            raise RuntimeError("Native DPO controller exited without a complete response")
        return _check(json.loads(line))

    def close(self):
        process, self.process = self.process, None
        if process is not None:
            try:
                process.stdin.close()
                process.wait(timeout=10)
            except Exception:
                process.kill()
                process.wait()


def _invoke(path, payload):
    result = subprocess.run([str(path)], input=json.dumps(payload), capture_output=True, text=True)
    try:
        response = json.loads(result.stdout)
    except ValueError:
        raise RuntimeError("Native DPO controller returned no JSON: " + result.stderr[-3000:]) from None
    return _check(response)


def build_native_dpo_expander(output_dir, *, rdkit_dir, reactor_library=None, compiler=None):
    """Build the DPO controller, using RDKit for molecular I/O and identity."""
    from .native_expand import _run, _sdk_inputs
    from synkit.Synthesis.Reactor import build_native_reactor
    from synkit.Synthesis.Reactor._native import runtime

    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    sdk = Path(rdkit_dir).expanduser().resolve()
    if not (sdk / "include/rdkit/GraphMol/GraphMol.h").is_file():
        raise ValueError("rdkit_dir lacks RDKit C++ molecular graph headers")
    if reactor_library is None:
        reactor_library = build_native_reactor(out / "kernel", compiler=compiler)
    _, kernel = runtime.load(reactor_library)
    kernel = kernel.resolve()
    kernel_manifest = json.loads(kernel.with_suffix(".json").read_text())
    # The kernel's SONAME is libsynkit_partial.so, while explicit builds publish
    # content-addressed filenames. Retain a private immutable SONAME copy.
    kernel_digest = _digest(kernel)
    runtime_dir = out / ("linked-kernel-" + kernel_digest[:20])
    runtime_dir.mkdir(exist_ok=True)
    runtime_name = (
        "synkit_partial.dll"
        if os.name == "nt"
        else "libsynkit_partial.dylib" if sys.platform == "darwin" else "libsynkit_partial.so"
    )
    runtime_kernel = runtime_dir / runtime_name
    if runtime_kernel.exists():
        if _digest(runtime_kernel) != kernel_digest:
            raise RuntimeError("Private native DPO kernel integrity check failed")
    else:
        shutil.copy2(kernel, runtime_kernel)
    kernel_include = Path(runtime.__file__).resolve().parents[1] / "cpp/include"
    header = kernel_include / "synkit_partial/partial.h"
    sources = {p.name: _digest(p) for p in sorted(_SOURCE.iterdir()) if p.is_file()}
    cxx = None if compiler is None else shutil.which(compiler)
    if compiler is not None and cxx is None:
        raise RuntimeError("Requested C++ compiler is unavailable")
    configuration = {
        "protocol": PROTOCOL,
        "builder_sha256": _digest(__file__),
        "sources": sources,
        "rdkit_dir": str(sdk),
        "sdk_inputs_sha256": hashlib.sha256(json.dumps(_sdk_inputs(sdk), sort_keys=True).encode()).hexdigest(),
        "kernel_library": str(kernel),
        "kernel_library_sha256": kernel_digest,
        "kernel_runtime_library": str(runtime_kernel),
        "kernel_header_sha256": _digest(header),
        "rule_compiler_sha256": _digest(runtime.__file__),
        "compiler": cxx,
        "platform": sys.platform,
        "machine": platform.machine(),
        "environment": {k: os.environ.get(k) for k in ("CXX", "CXXFLAGS", "LDFLAGS")},
    }
    key = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    executable = out / (EXECUTABLE + "_" + key[:16] + (".exe" if os.name == "nt" else ""))
    manifest = executable.with_name(executable.name + ".json")
    if executable.exists():
        record = dpo_expander_manifest(executable)
        if record["configuration"] != configuration:
            raise RuntimeError("Native DPO build configuration mismatch")
        return executable
    cmake = shutil.which("cmake")
    if cmake is None:
        raise RuntimeError("CMake is required for native DPO expansion")
    with tempfile.TemporaryDirectory(prefix="dpo-crn-build-", dir=out) as folder:
        work = Path(folder)
        snapshot = work / "source"
        shutil.copytree(_SOURCE, snapshot)
        build = work / "build"
        cmd = [
            cmake,
            "-S",
            str(snapshot),
            "-B",
            str(build),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_PREFIX_PATH=" + str(sdk),
            "-DDPO_KERNEL_LIBRARY=" + str(runtime_kernel),
            "-DDPO_KERNEL_INCLUDE=" + str(kernel_include),
            "-DCMAKE_BUILD_RPATH=" + str(sdk / "lib") + ";" + str(runtime_dir),
        ]
        if cxx is not None:
            cmd.append("-DCMAKE_CXX_COMPILER=" + cxx)
        _run(cmd)
        _run([cmake, "--build", str(build), "--parallel", "2"])
        candidates = [p for p in build.rglob(EXECUTABLE + (".exe" if os.name == "nt" else "")) if p.is_file()]
        if len(candidates) != 1:
            raise RuntimeError("Expected one native DPO controller executable")
        probe = _invoke(candidates[0], {"protocol": PROTOCOL, "probe": True})
        if probe.get("kernel_abi") != 1 or probe.get("controller") != "cpp":
            raise RuntimeError("Native DPO controller failed its ABI probe")
        shutil.copy2(candidates[0], executable)
        record = {
            "configuration": configuration,
            "build_sha256": key,
            "executable_sha256": _digest(executable),
            "rdkit_version": probe["rdkit_version"],
            "kernel_library": str(kernel),
            "kernel_library_sha256": kernel_digest,
            "kernel_runtime_library": str(runtime_kernel),
            "kernel_abi": 1,
            "kernel_manifest": kernel_manifest,
            "engine": "synkit_dpo_cpp",
            "controller": "cpp",
            "kernel": "synkit_dpo_cpp",
        }
        manifest.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return executable


def dpo_expander_manifest(executable):
    """Verify executable and linked DPO library fingerprints."""
    path = Path(executable).expanduser().resolve()
    try:
        manifest = json.loads(path.with_name(path.name + ".json").read_text())
    except (ValueError, OSError):
        raise RuntimeError("Native DPO controller has no valid build manifest") from None
    if manifest.get("executable_sha256") != _digest(path):
        raise RuntimeError("Native DPO controller integrity check failed")
    if manifest.get("kernel_library_sha256") != _digest(manifest["kernel_library"]):
        raise RuntimeError("Linked native DPO kernel integrity check failed")
    if manifest.get("kernel_library_sha256") != _digest(manifest["kernel_runtime_library"]):
        raise RuntimeError("Private native DPO kernel integrity check failed")
    return manifest


def resolve_native_dpo_expander(executable=None):
    value = executable if executable is not None else _ACTIVE.get()
    return None if value is None else Path(value).expanduser().resolve()


@contextmanager
def native_dpo_expander(executable):
    """Retain one verified native DPO controller process across builds."""
    path = Path(executable).expanduser().resolve()
    dpo_expander_manifest(path)
    server = _Server(path)
    active = _ACTIVE.set(path)
    token = _SERVER.set(server)
    try:
        yield path
    finally:
        _SERVER.reset(token)
        _ACTIVE.reset(active)
        server.close()


def compile_dpo_request(*, rules, seeds, options, radical_policy="strict"):
    """Serialize tuple rules and seed graphs for the typed DPO ABI."""
    from synkit.Rule import SynRule
    from synkit.Synthesis.Reactor import WildcardEngine
    from synkit.Synthesis.Reactor._native.runtime import serialize_native_request

    compiled = []
    for rule in rules:
        if isinstance(rule, str):
            rule = SynRule.from_smart(rule, format="tuple", implicit_h=True)
        if not isinstance(rule, SynRule) or rule._format != "tuple":
            raise ValueError("Native DPO expansion requires tuple-format SynRule objects or reaction strings")
        compiled.append(rule)
    source = ".".join(seeds) if seeds else "C"

    def encode(host, rule):
        return serialize_native_request(
            WildcardEngine(
                host, rule, partial=False, explicit_h=False, implicit_temp=True, radical_policy=radical_policy
            )
        )

    encoded_rules = [encode(source, rule) for rule in compiled]
    # The identity template below is used only to invoke the shared host wire
    # compiler for an empty rule set; it is never part of the generated network.
    host_rule = compiled[0] if compiled else SynRule.from_smart("[CH4:1]>>[CH4:1]", format="tuple", implicit_h=True)
    encoded_seeds = []
    for seed in seeds:
        encoded = encode(seed, host_rule)
        encoded_seeds.append({k: encoded[k] for k in ("symbols", "atoms", "edges")})
    return {
        "protocol": PROTOCOL,
        "rules": encoded_rules,
        "seeds": encoded_seeds,
        "options": dict(options),
        "rule_compiler": "synkit_typed_dpo_abi1",
    }


def run_native_dpo_expansion(executable, *, rules, seeds, options, radical_policy="strict"):
    """Expand one complete network or return an explicit native limit result."""
    path = Path(executable).expanduser().resolve()
    payload = compile_dpo_request(rules=rules, seeds=seeds, options=options, radical_policy=radical_policy)
    server = _SERVER.get()
    if server is not None and server.path == path:
        result = server.request(payload)
    else:
        dpo_expander_manifest(path)
        result = _invoke(path, payload)
    from ..backend import record_operation

    record_operation("native_dpo_rule_expansion")
    return result
