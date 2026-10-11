"""
Reduced uncertainty subproblems derived from supplied SLAP mapping results.

The supplied heuristic mappings determine which atom assignments agree and
which vary. ``extract_kernel`` holds agreeing assignments fixed and builds
candidate domains for the remaining atoms, with fallback label-class domains
and closure of small same-element uncertain blocks where applicable. Agreement
among supplied SLAP results is not a proof that an assignment is forced across
all global optima.

``Kernel`` is this reduced problem, not a mapping algorithm. Branch-and-bound
or optional MILP solves its remaining assignments exactly while respecting the
fixed assignments and candidate domains. ``apply_kernel_solution`` combines
the subsolution with the fixed part. An exact kernel solution is optimal within
the supplied restrictions; a full-space global optimum requires a full-space
search or an independent global proof. WL colour agreement alone is not used
as a certainty test because a reaction can change an atom's environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from ...slap.lap import recover_mapping

_MAX_CLOSED_ELEMENT_BLOCK = 4


@dataclass
class Kernel:
    """Reduced atom-assignment problem with fixed assignments and candidate domains.

    :ivar r_idx, p_idx: Original reactant/product atom indices that remain uncertain.
    :vartype r_idx, p_idx: list[int]
    :ivar r_colors, p_colors: Atom type (atomic number) of each kernel atom; reactant atom
                              ``r_idx[a]`` may map to product atom ``p_idx[b]`` only if
                              ``r_colors[a] == p_colors[b]``.
    :vartype r_colors, p_colors: list[int]
    :ivar fixed_mapping: Assignments held fixed: original reactant index -> product index.
    :vartype fixed_mapping: dict[int, int]
    :ivar candidate_images: Allowed product atom indices for each uncertain atom in ``r_idx``.
                            Extraction derives these from supplied SLAP results, with fallback
                            label-class domains and closure of small uncertain element blocks.
                            The exact kernel solver stays inside the supplied candidate sets.
    :vartype candidate_images: list[list[int]]
    :ivar lgp: The original reactant/product graph pair (for adjacency look-ups).
    :vartype lgp: list
    :ivar binary: Whether bond orders are binarised.
    :vartype binary: bool
    """

    r_idx: List[int]
    p_idx: List[int]
    r_colors: List[int]
    p_colors: List[int]
    fixed_mapping: Dict[int, int]
    lgp: list
    binary: bool = False
    candidate_images: Optional[List[List[int]]] = None

    @property
    def size(self) -> int:
        """Number of uncertain reactant atoms (== uncertain product atoms)."""
        return len(self.r_idx)

    @property
    def is_trivial(self) -> bool:
        """True when the supplied kernel has no uncertain atoms to optimize."""
        return self.size == 0


def _atomic_numbers(lg):
    nums = lg.props.get("atomic numbers")
    if nums is None:
        nums = list(lg.labels)
    return list(nums)


def extract_kernel(results, lgp, binary=False):
    """Extract a restricted uncertainty problem from supplied SLAP results.

    Agreement among the supplied heuristic mappings defines fixed assignments;
    variation defines uncertain domains. These restrictions are local to the
    supplied results and do not establish certainty over every global optimum.

    :param results: Best observed results from :meth:`GraphMatcher.get_maps`; each must carry a
                    fully resolved ``"lgp"`` graph pair.
    :type results: list[dict]
    :param lgp: The original reactant/product graph pair (equal atom counts).
    :type lgp: list[LabeledGraph]
    :param binary: Whether bond orders are binarised (recorded on the kernel).
    :type binary: bool, optional

    :rtype: Kernel
    """
    n = len(lgp[0].labels)
    mappings = [recover_mapping(r["lgp"]) for r in results]
    if not mappings:
        mappings = [list(range(n))]

    images = [set(m[i] for m in mappings) for i in range(n)]
    elements_r = _atomic_numbers(lgp[0])
    elements_p = _atomic_numbers(lgp[1])
    fixed_mapping, r_idx, p_idx, candidate_images = _compact_kernel_parts(images)

    if not _kernel_parts_are_valid(fixed_mapping, r_idx, p_idx, candidate_images):
        images = _label_class_candidate_images(results, lgp, elements_r, elements_p)
        fixed_mapping, r_idx, p_idx, candidate_images = _propagated_kernel_parts(images)

    candidate_images = _close_small_element_blocks(
        r_idx,
        p_idx,
        candidate_images,
        elements_r,
        elements_p,
    )

    return Kernel(
        r_idx=r_idx,
        p_idx=p_idx,
        r_colors=[elements_r[i] for i in r_idx],
        p_colors=[elements_p[p] for p in p_idx],
        fixed_mapping=fixed_mapping,
        lgp=lgp,
        binary=binary,
        candidate_images=candidate_images,
    )


def _compact_kernel_parts(images):
    fixed_mapping = {}
    r_idx = []
    for i, candidates in enumerate(images):
        if len(candidates) == 1:
            fixed_mapping[i] = next(iter(candidates))
        else:
            r_idx.append(i)

    p_set = set()
    for i in r_idx:
        p_set.update(images[i])
    p_idx = sorted(p_set)
    candidate_images = [sorted(images[i]) for i in r_idx]
    return fixed_mapping, r_idx, p_idx, candidate_images


def _kernel_parts_are_valid(fixed_mapping, r_idx, p_idx, candidate_images):
    if len(p_idx) != len(r_idx):
        return False
    p_set = set(p_idx)
    return all(
        candidates and set(candidates) <= p_set for candidates in candidate_images
    )


def _label_class_candidate_images(results, lgp, elements_r, elements_p):
    n = len(lgp[0].labels)
    images: List[Set[int]] = [set() for _ in range(n)]
    for result in results:
        left, right = result["lgp"]
        fallback = recover_mapping(result["lgp"])
        for i, label in enumerate(left.labels):
            for p in right.label2idxs.get(label, []):
                if 0 <= p < n and elements_r[i] == elements_p[p]:
                    images[i].add(p)
            p = fallback[i] if i < len(fallback) else -1
            if 0 <= p < n and elements_r[i] == elements_p[p]:
                images[i].add(p)
    if not results:
        images = [{i} for i in range(n)]
    return images


def _propagated_kernel_parts(images):
    images = [set(candidates) for candidates in images]
    fixed_mapping = {}
    unresolved = set(range(len(images)))
    used_products = set()
    changed = True
    while changed:
        changed = False
        for i in list(unresolved):
            images[i].difference_update(used_products)
            if len(images[i]) != 1:
                continue
            p = next(iter(images[i]))
            fixed_mapping[i] = p
            used_products.add(p)
            unresolved.remove(i)
            for j in unresolved:
                images[j].discard(p)
            changed = True
            break

    r_idx = sorted(unresolved)
    p_set = set()
    for i in r_idx:
        p_set.update(images[i])
    p_idx = sorted(p_set)
    candidate_images = [sorted(images[i]) for i in r_idx]
    return fixed_mapping, r_idx, p_idx, candidate_images


def _close_small_element_blocks(r_idx, p_idx, candidate_images, elements_r, elements_p):
    """Let exact mode fully permute small same-element uncertain blocks."""
    if not candidate_images:
        return candidate_images

    out = [sorted(set(candidates)) for candidates in candidate_images]
    positions_by_element = {}
    products_by_element = {}
    for a, i in enumerate(r_idx):
        positions_by_element.setdefault(elements_r[i], []).append(a)
    for p in p_idx:
        products_by_element.setdefault(elements_p[p], []).append(p)

    for element, positions in positions_by_element.items():
        pool = sorted(products_by_element.get(element, []))
        if (
            not pool
            or len(pool) != len(positions)
            or len(pool) > _MAX_CLOSED_ELEMENT_BLOCK
        ):
            continue
        pool_set = set(pool)
        if any(not set(out[a]) <= pool_set for a in positions):
            continue
        for a in positions:
            out[a] = pool
    return out


def apply_kernel_solution(kernel, sub_mapping):
    """Combine a kernel sub-mapping with its fixed assignments into a full mapping.

    :param kernel:
    :type kernel: Kernel
    :param sub_mapping: Mapping over kernel positions: either a dict ``a -> b`` (positions into
                        ``kernel.r_idx`` / ``kernel.p_idx``) or a sequence ``sub_mapping[a] = b``.
    :type sub_mapping: dict[int, int] or sequence[int]

    :return: ``mapping[i] = p`` over original atom indices.
    :rtype: list[int]
    """
    n = len(kernel.lgp[0].labels)
    mapping = [-1] * n
    for i, p in kernel.fixed_mapping.items():
        mapping[i] = p
    if isinstance(sub_mapping, dict):
        items = sub_mapping.items()
    else:
        items = enumerate(sub_mapping)
    for a, b in items:
        mapping[kernel.r_idx[a]] = kernel.p_idx[b]
    return mapping
