"""Declared finite chemical-lattice domain for the identifiability study.

This guard is not a repair of the generic floating-point solver API. The study
uses undirected loop-free endpoints with doubled orders in {2,3,4,6}, at most
512 atoms, and exact integer independent witness rescoring. The current Python
assignment route verifies bounded lattice LAP answers using integer primal--dual
certificates; its operation-level argument is in the Synister supplement. This
input guard alone does not certify another execution route or an archived source
version, and does not broaden the generic floating-point API guarantee.
"""

MAX_STUDY_ATOMS = 512


def validate_study_domain(reactant, product):
    from collections import Counter

    n = len(reactant.atomic_numbers)
    if not 1 <= n <= MAX_STUDY_ATOMS or len(product.atomic_numbers) != n:
        raise ValueError("Study arithmetic domain requires 1–512 balanced heavy atoms")
    if Counter(reactant.atomic_numbers) != Counter(product.atomic_numbers):
        raise ValueError("Study endpoints must be element balanced")
    for endpoint in (reactant, product):
        endpoint.__post_init__()
    # 4*value must fit exactly in binary64. This deliberately loose polynomial
    # envelope is far above the O(n^2) matrix/DFS/profile quantities actually
    # used; it also leaves headroom for LAP potential/shortest-path updates.
    scaled_working_envelope = 4 * 64 * n**4
    if scaled_working_envelope >= 2**53:
        raise ValueError("Study arithmetic envelope exceeds binary64 integer range")
    return {"domain": "bounded-half-integer-chemical-v1", "heavy_atoms": n,
            "max_heavy_atoms": MAX_STUDY_ATOMS,
            "doubled_orders": [2, 3, 4, 6],
            "scaled_working_envelope": scaled_working_envelope,
            "binary64_exact_integer_limit": 2**53,
            "generic_float_api_certified": False}
