"""Python interfaces to native exact algebra, CRNT and Petri-net searches."""

from __future__ import annotations
import ctypes as ct
import json
from math import gcd
from functools import reduce
import numpy as np
from .native import _load_library
from .analysis import AnalysisResult
from .Petrinet.minimal_semiflows import _integer_stoichiometry

PTR = ct.POINTER(ct.c_int64)


class NativeSearchLimit(RuntimeError):
    def __init__(self, result):
        self.result = result
        super().__init__("Native CRN analysis incomplete: " + result.termination_reason)


def _call(library_path, name, types, values):
    library = _load_library(library_path)
    if not hasattr(library, name):
        raise RuntimeError(f"Rebuild the CRN library: missing native operation {name}")
    from .backend import record_operation

    record_operation(name.removeprefix("synkit_crn_"))
    function = getattr(library, name)
    function.argtypes = [*types, ct.POINTER(ct.c_void_p)]
    function.restype = ct.c_int
    library.synkit_crn_free_text.argtypes = [ct.c_void_p]
    library.synkit_crn_free_text.restype = None
    output = ct.c_void_p()
    try:
        code = function(*values, ct.byref(output))
        if code or not output.value:
            raise RuntimeError(f"Native CRN operation failed: {name}")
        result = json.loads(ct.string_at(output).decode("ascii"))
    finally:
        if output.value:
            library.synkit_crn_free_text(output)
    if result.get("termination_reason") not in {
        "complete",
        "certificate",
        "exhausted",
        "depth_limit",
        "state_limit",
        "frontier_limit",
        "time_limit",
        "count_limit",
        "allocation_failure",
    }:
        raise RuntimeError(f"Invalid native result: {result}")
    return result


def _array(matrix):
    return np.ascontiguousarray(_integer_stoichiometry(matrix), dtype=np.int64)


def _seconds(seconds):
    if not np.isfinite(seconds) or not 0 < seconds <= 1e9:
        raise ValueError("time_limit must be finite, positive and at most 1e9 seconds")
    return seconds


def matrix_analysis(matrix, *, library_path, kernels="left", time_limit=30.0):
    s = _array(matrix)
    flags = {"none": 0, "left": 1, "right": 2, "both": 3}
    if kernels not in flags:
        raise ValueError("Invalid kernel selection")
    result = _call(
        library_path,
        "synkit_crn_matrix",
        [ct.c_size_t, ct.c_size_t, PTR, ct.c_int, ct.c_double],
        [*s.shape, s.ctypes.data_as(PTR), flags[kernels], _seconds(time_limit)],
    )
    if result["termination_reason"] != "complete":
        raise NativeSearchLimit(
            AnalysisResult(
                None,
                "inconclusive",
                False,
                "native rational algebra",
                termination_reason=result["termination_reason"],
            )
        )
    for name, matrix in [("left", s.T), ("right", s)]:
        if name not in kernels and kernels != "both":
            continue
        basis = result[name]
        if len(basis) != matrix.shape[1] - result["rank"]:
            raise RuntimeError("Invalid native nullity")
        for row in basis:
            if len(row) != matrix.shape[1] or reduce(gcd, row, 0) != 1:
                raise RuntimeError("Invalid native primitive kernel vector")
            if any(sum(int(v) * c for v, c in zip(r, row)) for r in matrix):
                raise RuntimeError("Native kernel witness failed exact verification")
    return result


def crnt_summary_native(crn, *, library_path, time_limit=30.0):
    from .Props.stoich import build_S_minus_plus
    from .Props.deficiency import (
        CRNTSummary,
        Complex,
        _species_labels,
        _deficiency_zero_values,
        _deficiency_one_values,
    )

    _, _, minus, plus = build_S_minus_plus(crn)
    pre, post = _array(minus), _array(plus)
    result = _call(
        library_path,
        "synkit_crn_crnt",
        [ct.c_size_t, ct.c_size_t, PTR, PTR, ct.c_double],
        [
            *pre.shape,
            pre.ctypes.data_as(PTR),
            post.ctypes.data_as(PTR),
            _seconds(time_limit),
        ],
    )
    if result["termination_reason"] != "complete":
        raise NativeSearchLimit(
            AnalysisResult(
                None,
                "inconclusive",
                False,
                "native exact CRNT",
                termination_reason=result["termination_reason"],
            )
        )
    result.pop("termination_reason")
    labels = tuple(_species_labels(crn))
    result["complexes"] = [
        str(Complex(coefficients=tuple(c), species_labels=labels))
        for c in result.pop("complex_coefficients")
    ]
    terminals = result.pop("terminal_counts")
    result["deficiency_zero"] = _deficiency_zero_values(
        result["deficiency"], result["is_weakly_reversible"]
    )
    result["deficiency_one"] = _deficiency_one_values(
        result["deficiency"],
        result["linkage_class_deficiencies"],
        result["is_weakly_reversible"],
        terminals,
    )
    return CRNTSummary(**result)


def petri_reachability(
    net,
    initial,
    goal_exact,
    goal_atleast,
    *,
    library_path,
    max_states=100000,
    max_depth=10000,
    time_limit=30.0,
    qualitative=False,
):
    if (
        not isinstance(max_states, int)
        or not 0 < max_states < 2**59
        or not isinstance(max_depth, int)
        or not 0 <= max_depth < 2**59
    ):
        raise ValueError("Invalid state/depth budget")
    places, transitions = net.place_order, net.transition_order
    pre = np.zeros((len(places), len(transitions)), dtype=np.int64)
    post = pre.copy()
    for j, t in enumerate(transitions):
        for i, s in enumerate(places):
            pre[i, j] = net.transitions[t].pre.get(s, 0)
            post[i, j] = net.transitions[t].post.get(s, 0)
    m = _array([[initial.get(s, 0) for s in places]]).ravel()
    exact = _array([[goal_exact.get(s, -1) for s in places]]).ravel()
    atleast = _array([[goal_atleast.get(s, 0) for s in places]]).ravel()
    result = _call(
        library_path,
        "synkit_crn_pathway",
        [
            ct.c_size_t,
            ct.c_size_t,
            PTR,
            PTR,
            PTR,
            PTR,
            PTR,
            ct.c_size_t,
            ct.c_size_t,
            ct.c_double,
            ct.c_int,
        ],
        [
            len(places),
            len(transitions),
            pre.ctypes.data_as(PTR),
            post.ctypes.data_as(PTR),
            m.ctypes.data_as(PTR),
            exact.ctypes.data_as(PTR),
            atleast.ctypes.data_as(PTR),
            max_states,
            max_depth,
            _seconds(time_limit),
            int(qualitative),
        ],
    )
    if result["certificate"] is not None:
        certificate = [transitions[i] for i in result["certificate"]]
        marking = dict(initial)
        for t in certificate:
            if qualitative:
                if not set(net.transitions[t].pre) <= {
                    s for s, n in marking.items() if n > 0
                }:
                    raise RuntimeError("Invalid native qualitative certificate")
                marking.update({s: 1 for s in net.transitions[t].post})
            else:
                if not net.enabled(marking, t):
                    raise RuntimeError("Invalid native firing certificate")
                marking = net.fire(marking, t)
        if any(marking.get(s, 0) != n for s, n in goal_exact.items()) or any(
            marking.get(s, 0) < n for s, n in goal_atleast.items()
        ):
            raise RuntimeError("Native certificate misses its goal")
        result["certificate"] = certificate
    return result


def graph_symmetry(
    crn, *, library_path, time_limit=30.0, max_states=1000000, max_count=1000000
):
    from .Petrinet.net import extract_syncrn_incidence
    import hashlib

    incidence = extract_syncrn_incidence(crn)
    nodes = list(incidence.species_order) + list(incidence.reaction_order)
    n = len(nodes)
    index = {s: i for i, s in enumerate(nodes)}
    adjacency = np.zeros((n, n), dtype=np.int64)
    for rid in incidence.reaction_order:
        for sid, value in incidence.pre[rid].items():
            adjacency[index[sid], index[rid]] = value
        for sid, value in incidence.post[rid].items():
            adjacency[index[rid], index[sid]] = value
    kinds = np.array(
        [0] * len(incidence.species_order) + [1] * len(incidence.reaction_order),
        dtype=np.int64,
    )
    result = _call(
        library_path,
        "synkit_crn_symmetry",
        [ct.c_size_t, PTR, PTR, ct.c_double, ct.c_size_t, ct.c_size_t],
        [
            n,
            kinds.ctypes.data_as(PTR),
            adjacency.ctypes.data_as(PTR),
            _seconds(time_limit),
            max_states,
            max_count,
        ],
    )
    complete = result["termination_reason"] == "complete"
    if complete:
        order = result["order"]
        # Sparse canonical certificate avoids storing an n*n matrix in evidence.
        inverse = {v: i for i, v in enumerate(order)}
        edges = sorted(
            (inverse[int(i)], inverse[int(j)], int(adjacency[i, j]))
            for i, j in zip(*adjacency.nonzero())
        )
        certificate = ([int(kinds[i]) for i in order], edges)
        result["canonical_key"] = hashlib.sha256(
            json.dumps(certificate, separators=(",", ":")).encode()
        ).hexdigest()
        result["canonical_order"] = [nodes[i] for i in order]
        result["orbits"] = [[nodes[i] for i in group] for group in result["orbits"]]
    else:
        result["canonical_key"] = None
    return AnalysisResult(
        result,
        "complete" if complete else "inconclusive",
        complete,
        "native exact individualization/refinement",
        termination_reason=result["termination_reason"],
    )


def exhaustive_siphons(crn, *, library_path, max_species=22, time_limit=30.0):
    from .Petrinet.net import PetriNet

    net = crn if isinstance(crn, PetriNet) else PetriNet.from_syncrn(crn)
    places, transitions = net.place_order, net.transition_order
    if len(places) > min(max_species, 24):
        raise ValueError("Exhaustive enumeration exceeds species limit")
    pre = np.zeros(len(transitions), dtype=np.uint64)
    post = pre.copy()
    index = {s: i for i, s in enumerate(places)}
    for j, t in enumerate(transitions):
        for s in net.transitions[t].pre:
            pre[j] |= np.uint64(1 << index[s])
        for s in net.transitions[t].post:
            post[j] |= np.uint64(1 << index[s])
    pointer = ct.POINTER(ct.c_uint64)
    result = _call(
        library_path,
        "synkit_crn_exhaustive_siphons",
        [ct.c_size_t, ct.c_size_t, pointer, pointer, ct.c_double],
        [
            len(places),
            len(transitions),
            pre.ctypes.data_as(pointer),
            post.ctypes.data_as(pointer),
            _seconds(time_limit),
        ],
    )
    if result["termination_reason"] != "complete":
        raise NativeSearchLimit(
            AnalysisResult(
                result,
                "inconclusive",
                False,
                "native exhaustive subsets",
                termination_reason=result["termination_reason"],
            )
        )
    return [{places[i] for i in group} for group in result["sets"]]


def qualitative_path(
    crn,
    seeds,
    targets,
    *,
    library_path,
    max_states=100000,
    max_depth=100,
    time_limit=30.0,
):
    from .Petrinet.net import PetriNet

    net = PetriNet.from_syncrn(crn)
    result = petri_reachability(
        net,
        {s: 1 for s in seeds},
        {},
        {s: 1 for s in targets},
        library_path=library_path,
        max_states=max_states,
        max_depth=max_depth,
        time_limit=time_limit,
        qualitative=True,
    )
    return result


def positive_kernel_certificate(crn, *, kind="p", include_problem=False):
    """Decide strict positive-kernel feasibility only with an exact witness.

    A negative answer carries a separating vector y with A.T @ y >= 0 and
    at least one strictly positive component (Gordan's alternative). Set
    include_problem to record the integer matrix for standalone verification.
    """
    from scipy.optimize import linprog
    from fractions import Fraction
    from .Props.stoich import stoichiometric_matrix, _primitive_integer_vector
    from .backend import record_operation

    if kind not in {"p", "t"}:
        raise ValueError("kind must be p or t")
    s = _array(stoichiometric_matrix(crn))
    a = s.T if kind == "p" else s
    record_operation("highs_positive_kernel")
    rows, columns = a.shape
    problem = {"kind": kind, "shape": [rows, columns]}
    if include_problem:
        problem["problem_matrix"] = a.tolist()
    unknown = {**problem, "holds": None, "exact": False}
    if columns == 0 or rows == 0:
        return {
            **problem,
            "holds": True,
            "exact": True,
            "primal_witness": [1] * columns,
        }
    result = linprog(
        np.zeros(columns),
        A_eq=a,
        b_eq=np.zeros(rows),
        bounds=(1, None),
        method="highs",
    )
    if result.success:
        try:
            witness = _primitive_integer_vector(
                [Fraction(float(v)).limit_denominator(10**6) for v in result.x]
            )
        except (ValueError, OverflowError):
            return unknown
        if all(v > 0 for v in witness) and all(
            sum(int(x) * v for x, v in zip(row, witness)) == 0 for row in a
        ):
            return {
                **problem,
                "holds": True,
                "exact": True,
                "primal_witness": witness,
            }
        return unknown
    if result.status != 2:
        return unknown
    # A floating infeasibility status alone is insufficient. Find and verify a
    # rational separator instead, normalized by sum(A.T @ y) = 1.
    column_sums = np.array([sum(int(v) for v in row) for row in a], dtype=float)
    dual = linprog(
        np.zeros(rows),
        A_ub=-a.astype(float).T,
        b_ub=np.zeros(columns),
        A_eq=column_sums.reshape(1, -1),
        b_eq=np.ones(1),
        bounds=[(None, None)] * rows,
        method="highs",
    )
    if not dual.success:
        return unknown
    try:
        witness = [Fraction(float(v)).limit_denominator(10**6) for v in dual.x]
    except (ValueError, OverflowError):
        return unknown
    image = [
        sum(int(a[i, j]) * witness[i] for i in range(rows)) for j in range(columns)
    ]
    if all(v >= 0 for v in image) and any(v > 0 for v in image):
        return {
            **problem,
            "holds": False,
            "exact": True,
            "dual_witness": [str(v) for v in witness],
            "dual_image": [str(v) for v in image],
        }
    return unknown


def positive_kernel(crn, *, kind="p"):
    """Compatibility interface backed by exactly verified certificates."""
    from fractions import Fraction

    certificate = positive_kernel_certificate(crn, kind=kind)
    witness = certificate.get("primal_witness")
    if certificate["holds"] is True and witness:
        total = sum(witness)
        return True, np.array([float(Fraction(v, total)) for v in witness])
    return certificate["holds"], None
