"""
Kernelization of the atom-to-atom mapping problem from SLAP uncertainty regions.

SLAP returns one or more equal-cost optimal mappings. Most atoms are *certain*:
they map to the same product atom in **every** optimal mapping, so their image is
effectively forced. The atoms whose image varies across the optimal mappings form
the *uncertainty region*; together with the symmetry that produced them they are
the only thing an exact solver still has to decide. The certain atoms contribute a
constant to the chemical distance.

Note that colour-refinement compatibility (matching Weisfeiler-Lehman colours
across reactant and product) is *not* a valid certainty test here: a reaction
changes bonds, so an atom's environment genuinely differs between reactant and
product. We therefore read the uncertainty region directly off the variation in
SLAP's optimal mappings, which is reaction-correct.

:func:`extract_kernel` partitions atoms into certain/uncertain and returns a small
:class:`Kernel` over the uncertain atoms (candidate images restricted to the same
element, within the uncertain region). The exact solvers
(:mod:`mapper.exact.milp`, :mod:`mapper.exact.branching`) optimise only the
kernel -- which keeps the factorial/exponential exact methods tractable -- and
:func:`apply_kernel_solution` stitches the kernel mapping back with the certain
part into a full atom mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from ..slap.lap import recover_mapping

_MAX_CLOSED_ELEMENT_BLOCK = 4


@dataclass
class Kernel:
    """The uncertain sub-problem extracted from a reaction.

    Attributes
    ----------
    r_idx, p_idx : list[int]
        Original reactant/product atom indices that remain uncertain.
    r_colors, p_colors : list[int]
        Atom type (atomic number) of each kernel atom; reactant atom
        ``r_idx[a]`` may map to product atom ``p_idx[b]`` only if
        ``r_colors[a] == p_colors[b]``.
    fixed_mapping : dict[int, int]
        Certain atoms: original reactant index -> original product index.
    candidate_images : list[list[int]]
        For each uncertain reactant atom in ``r_idx``, product atom indices
        observed for that atom across the SLAP optima. The exact kernel solver
        must stay inside these candidate sets; otherwise an omitted byproduct
        padded with disconnected dummy atoms can create a huge artificial
        same-element search space.
    lgp : list
        The original reactant/product graph pair (for adjacency look-ups).
    binary : bool
        Whether bond orders are binarised.
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
        """True when SLAP already determined the whole mapping unambiguously."""
        return self.size == 0


def _atomic_numbers(lg):
    nums = lg.props.get("atomic numbers")
    if nums is None:
        nums = list(lg.labels)
    return list(nums)


def extract_kernel(results, lgp, binary=False):
    """Extract the uncertainty-region kernel from SLAP's optimal mappings.

    Parameters
    ----------
    results : list[dict]
        Optimal results from :meth:`GraphMatcher.get_maps`; each must carry a
        fully resolved ``"lgp"`` graph pair.
    lgp : list[LabeledGraph]
        The original reactant/product graph pair (equal atom counts).
    binary : bool, optional
        Whether bond orders are binarised (recorded on the kernel).

    Returns
    -------
    Kernel
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
    """Combine a kernel sub-mapping with the certain mapping into a full mapping.

    Parameters
    ----------
    kernel : Kernel
    sub_mapping : dict[int, int] or sequence[int]
        Mapping over kernel positions: either a dict ``a -> b`` (positions into
        ``kernel.r_idx`` / ``kernel.p_idx``) or a sequence ``sub_mapping[a] = b``.

    Returns
    -------
    list[int]
        ``mapping[i] = p`` over original atom indices.
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
