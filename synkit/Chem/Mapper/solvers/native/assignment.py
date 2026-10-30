"""Native assignment B&B and explicitly restricted uncertainty kernels."""

import ctypes as ct
from numbers import Integral

import numpy as np

from .core import enumerate_core_mappings
from .runtime import INT, load_core


def kernel_domains(mappings, types_r, types_p, *, library_path, close_block_size=4):
    """Build restricted domains in C++ from compatible supplied candidates.

    Agreement among heuristic candidates is not a global-optimality proof.
    """
    left, right = list(types_r), list(types_p)
    n = len(left)
    rows = list(mappings)
    if not 1 <= n <= 1024 or len(right) != n or not rows:
        raise ValueError(
            "Kernel requires nonempty equal endpoint inventories and candidates"
        )
    if any(len(row) != n for row in rows) or any(
        isinstance(j, bool) or not isinstance(j, Integral) or not 0 <= j < n
        for row in rows
        for j in row
    ):
        raise ValueError("Kernel candidates require valid integer image indices")
    if (
        isinstance(close_block_size, bool)
        or not isinstance(close_block_size, Integral)
        or not 0 <= close_block_size < 2**31
    ):
        raise ValueError("close_block_size must be a nonnegative C ABI integer")
    registry = {}
    codes = [
        np.asarray(
            [registry.setdefault(v, len(registry)) for v in side], dtype=np.int32
        )
        for side in (left, right)
    ]
    permutations = np.ascontiguousarray(rows, dtype=np.int32)
    domains = np.zeros((n, n), dtype=np.uint8)
    lib = load_core(str(library_path))
    if not hasattr(lib, "synkit_mapper_kernel_domains"):
        raise RuntimeError("Library lacks native uncertainty-kernel construction")
    lib.synkit_mapper_assignment_error.restype = ct.c_char_p
    lib.synkit_mapper_kernel_domains.argtypes = [
        ct.c_int32,
        ct.c_int64,
        INT,
        INT,
        INT,
        ct.c_int32,
        ct.POINTER(ct.c_uint8),
    ]
    result = lib.synkit_mapper_kernel_domains(
        n,
        len(rows),
        permutations.ctypes.data_as(INT),
        *[v.ctypes.data_as(INT) for v in codes],
        close_block_size,
        domains.ctypes.data_as(ct.POINTER(ct.c_uint8)),
    )
    if result:
        raise ValueError(lib.synkit_mapper_assignment_error().decode())
    return domains


def enumerate_assignment_mappings(lgp, *, library_path, **options):
    return enumerate_core_mappings(
        lgp, library_path=library_path, backend="abb", **options
    )


def enumerate_kernel_mappings(kernel, *, library_path, backend="kbb", **options):
    """Solve an existing Kernel in its fixed/uncertain domains, entirely natively."""
    if backend not in {"kbb", "kmilp"}:
        raise ValueError("Kernel backend must be kbb or kmilp")
    n = len(kernel.lgp[0].labels)
    allowed = np.zeros((n, n), dtype=np.uint8)
    for i, j in kernel.fixed_mapping.items():
        allowed[i, j] = 1
    for position, i in enumerate(kernel.r_idx):
        candidates = (
            kernel.p_idx
            if kernel.candidate_images is None
            else kernel.candidate_images[position]
        )
        allowed[i, candidates] = 1
    return enumerate_core_mappings(
        kernel.lgp,
        library_path=library_path,
        backend=backend,
        binary=kernel.binary,
        allowed_domains=allowed,
        **options,
    )
