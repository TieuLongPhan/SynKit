"""Independent compiled-C checks, with explicit archival-text provenance."""
from __future__ import annotations

import ctypes
import hashlib
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from scipy.integrate import solve_ivp

from models import ParentParameters, parent, reference_initial


REFERENCE = Path(__file__).resolve().parent / "reference"
ARCHIVE_MD5 = {"generalb.c": "27c3f0e8afe043fda1e7dbdae3b3f8d4",
               "generalf.c": "43a135f995309b2d6ffbe7dbb187db92"}


def source_manifest():
    records = []
    for filename, md5 in ARCHIVE_MD5.items():
        data = (REFERENCE/filename).read_bytes()
        records.append({"file": filename, "source_url":
                        f"https://zenodo.org/records/15299074/files/{filename}",
                        "archive_md5": md5, "local_md5": hashlib.md5(data).hexdigest(),
                        "local_sha256": hashlib.sha256(data).hexdigest(),
                        "bytes": len(data),
                        "byte_identical_to_archive": hashlib.md5(data).hexdigest() == md5})
    return {"archive_doi": "10.5281/zenodo.15299074", "archive_version": 1,
            "access_date": "2026-10-07", "files": records,
            "retrieval": "Reconstructed from browser-rendered numbered source lines; not a raw-byte download.",
            "validation_scope": "Compile preserved deriv() equations; compare RHS and selected finite-time RK4 trajectories. The complete archived main() sweep was not run.",
            "positive_kdim_caveat": "Archived C returns LD hydrolysis to 2L and DL hydrolysis to 2D in monomer equations. The explicit-reaction compiler returns each to L+D. They coincide at kdim=0, or when LD=DL; off this subspace at positive kdim they differ."}


class CReference:
    def __init__(self, mechanism, directory):
        filename = "generalf.c" if mechanism == "constructive" else "generalb.c"
        target = Path(directory)/(mechanism+".so")
        command = ["/usr/bin/gcc", "-O2", "-shared", "-fPIC", "-std=c99",
                   f'-DREFERENCE_SOURCE="{REFERENCE/filename}"',
                   f'-DCONSTRUCTIVE={int(mechanism == "constructive")}',
                   str(REFERENCE/"bridge.c"), "-lm", "-o", str(target)]
        compiled = subprocess.run(command, check=True, text=True, capture_output=True)
        self.compiler_stderr = compiled.stderr
        self.lib = ctypes.CDLL(str(target))
        pointer = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
        self.lib.configure_reference.argtypes = [pointer]
        self.lib.configure_reference.restype = None
        self.lib.evaluate_reference.argtypes = [pointer, pointer]
        self.lib.evaluate_reference.restype = None
        self.lib.integrate_reference.argtypes = [pointer, ctypes.c_double, ctypes.c_int]
        self.lib.integrate_reference.restype = None

    def configure(self, p):
        self.lib.configure_reference(np.array([p.u, p.k1, p.k2, p.a1, p.a2,
                                              p.kmon, p.kdim, p.phi, p.total_feed]))

    def rhs(self, x):
        x = np.ascontiguousarray(x, dtype=float)
        out = np.empty(7)
        self.lib.evaluate_reference(x, out)
        return out

    def integrate(self, x, step, steps):
        out = np.array(x, dtype=float)
        self.lib.integrate_reference(out, step, steps)
        return out


def check_reference(seed=1729):
    rng = np.random.default_rng(seed)
    reports = []
    with tempfile.TemporaryDirectory(prefix="synkit-autocatalytic-c-") as temporary:
        for mechanism, ratios in (("constructive", (1., 4., 8.)),
                                  ("destructive", (2., 8., 12.))):
            ref = CReference(mechanism, temporary)
            for ratio in ratios:
                p = (ParentParameters(k1=.75*ratio, k2=.75) if mechanism == "constructive"
                     else ParentParameters(mechanism=mechanism, k1=10., k2=10.*ratio))
                ref.configure(p)
                model = parent(p)
                errors = [float(np.max(np.abs(ref.rhs(x)-model.rhs(0, x))))
                          for x in rng.uniform(1e-3, .2, (64, 7))]
                initial = reference_initial()
                c_endpoint = ref.integrate(initial, .01, 10000)
                solution = solve_ivp(model.rhs, (0, 100), initial, method="DOP853",
                                     rtol=1e-11, atol=1e-13)
                if not solution.success:
                    raise RuntimeError(solution.message)
                trajectory_error = float(np.max(np.abs(solution.y[:, -1]-c_endpoint)))
                if max(errors) > 1e-12 or trajectory_error > 2e-7:
                    raise RuntimeError("Independent C reproduction failed")
                reports.append({"mechanism": mechanism, "ratio": ratio,
                                "parameters": model.metadata["parameters"],
                                "rhs_samples": 64, "rhs_max_abs_error": max(errors),
                                "finite_time": 100, "reference_rk4_step": .01,
                                "reference_steps": 10000,
                                "finite_time_max_abs_error": trajectory_error,
                                "c_endpoint": c_endpoint.tolist(),
                                "python_endpoint": solution.y[:, -1].tolist()})
            # Explicitly exercise the known positive-hydrolysis convention.
            p = ParentParameters(mechanism=mechanism, kdim=.01)
            ref.configure(p)
            model = parent(p)
            x = np.array([.2, .03, .08, .04, .02, .06, .01])
            difference = ref.rhs(x)-model.rhs(0, x)
            expected = np.zeros(7)
            expected[1] = p.a2*p.kdim*(x[4]-x[5])
            expected[2] = -expected[1]
            if np.max(np.abs(difference-expected)) > 1e-12:
                raise RuntimeError("Hydrolysis convention discrepancy changed")
            equal = x.copy()
            equal[5] = equal[4]
            reports.append({"mechanism": mechanism, "hydrolysis_discrepancy_test": True,
                            "kdim": p.kdim, "off_subspace_difference": difference.tolist(),
                            "predicted_difference": expected.tolist(),
                            "equal_LD_DL_max_abs_error": float(np.max(np.abs(
                                ref.rhs(equal)-model.rhs(0, equal))))})
    return {"seed": seed, "checks": reports, "passed": True,
            "scope": source_manifest()["validation_scope"]}
