"""
Optimality certificates and gap reporting.

A :class:`Certificate` pairs the cost of a concrete mapping (an *upper bound* on
the optimal chemical distance) with an admissible *lower bound*, and reports the
gap between them. When the gap is zero the mapping is *proven optimal*:

* ``method="dual-lap"`` -- the lower bound is the Gilmore-Lawler / dual-LAP
  relaxation (:func:`mapper.slap.lap.dual_lap_lower_bound`). A zero gap means the
  cheap relaxation already certifies optimality; a positive gap is honest about
  the relaxation not being tight, *not* about the mapping being suboptimal.
* ``method="milp-exact"`` -- the lower bound equals the optimum returned by the
  exact MILP/QAP solver (:mod:`mapper.exact.milp`), which CBC proves optimal, so
  the gap is zero by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..slap.lap import dual_lap_lower_bound, chemical_distance, recover_mapping

# Costs are integer/half-integer in cd units; this tolerance absorbs float noise.
_TOL = 1e-6


@dataclass
class Certificate:
    """Optimality certificate for a single mapping.

    Attributes
    ----------
    upper_bound : float
        Cost of the concrete mapping (chemical distance, cd units).
    lower_bound : float
        Admissible lower bound on the optimal chemical distance.
    method : str
        How ``lower_bound`` was obtained (``"dual-lap"`` or ``"milp-exact"``).
    """

    upper_bound: float
    lower_bound: float
    method: str = "dual-lap"

    @property
    def gap(self) -> float:
        """Optimality gap ``upper_bound - lower_bound`` (>= 0)."""
        return self.upper_bound - self.lower_bound

    @property
    def proven_optimal(self) -> bool:
        """True iff the lower bound certifies the mapping is optimal."""
        import math

        return math.isfinite(self.lower_bound) and self.gap <= _TOL

    def __str__(self) -> str:
        status = "PROVEN OPTIMAL" if self.proven_optimal else "not certified"
        return (
            f"Certificate(cost={self.upper_bound:g}, "
            f"lower_bound={self.lower_bound:g}, gap={self.gap:g}, "
            f"method={self.method!r}) -> {status}"
        )


def certify_result(result, binary, method="dual-lap"):
    """Build and attach a :class:`Certificate` to a single result dict.

    Uses the result's ``"cd"`` as the incumbent upper bound (falling back to a
    recomputed chemical distance) and the dual-LAP lower bound. The certificate
    is stored under ``result["certificate"]`` and returned.

    Parameters
    ----------
    result : dict
        A result entry with ``"lgp"`` and (optionally) ``"cd"``.
    binary : bool
        Whether bond orders are binarised (matches the matcher's ``binary``).
    method : str, optional
        Lower-bound provenance label.

    Returns
    -------
    Certificate
    """
    lgp = result["lgp"]
    if "cd" in result:
        ub = float(result["cd"])
    else:
        ub = chemical_distance(lgp, recover_mapping(lgp), binary)
    lb = dual_lap_lower_bound(lgp, binary)
    cert = Certificate(upper_bound=ub, lower_bound=lb, method=method)
    result["certificate"] = cert
    return cert


def certify_results_exact(results, binary):
    """Attach exact (MILP/QAP) certificates to a set of optimal results.

    The uncertainty-region kernel is extracted from the results and solved to
    proven optimality with the MILP/QAP solver. The proven kernel optimum becomes
    the lower bound for every result, so results whose cost equals it are
    certified optimal over the uncertainty region (``method="milp-exact"``).

    Falls back to the dual-LAP certificate when PuLP is unavailable or the kernel
    is degenerate.

    Parameters
    ----------
    results : list[dict]
        Optimal results sharing a common reaction (each with ``"lgp"``/``"cd"``).
    binary : bool
        Whether bond orders are binarised.
    """
    if not results:
        return

    from .kernel import extract_kernel
    from .milp import solve_kernel_milp, HAS_PULP

    if not HAS_PULP:
        for r in results:
            certify_result(r, binary)
        return

    lgp = results[0]["lgp"]
    kernel = extract_kernel(results, lgp, binary=binary)
    sol = solve_kernel_milp(kernel)
    lb = sol.cost
    method = "milp-exact" if sol.proven_optimal else "milp"
    for r in results:
        if "cd" in r:
            ub = float(r["cd"])
        else:
            ub = chemical_distance(r["lgp"], recover_mapping(r["lgp"]), binary)
        r["certificate"] = Certificate(upper_bound=ub, lower_bound=lb, method=method)
