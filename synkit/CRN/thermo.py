# thermo.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List, Type

import numpy as np

from .core import CRNNetwork, CRNReaction, CRNSpecies  # existing core types


# -----------------------------------------------------------------------------
# Optional SciPy backend
# -----------------------------------------------------------------------------
try:
    from scipy.linalg import null_space
    from scipy.optimize import linprog

    _SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover - env dependent
    null_space = None
    linprog = None
    _SCIPY_AVAILABLE = False


class ThermoBackendError(RuntimeError):
    """Raised when SciPy (required for thermo LP/LA) is not installed."""

    pass


def _require_scipy() -> None:
    if not _SCIPY_AVAILABLE:
        raise ThermoBackendError(
            "Thermodynamic analysis requires SciPy. "
            "Install it via `pip install scipy`."
        )


# -----------------------------------------------------------------------------
# Dataclass for thermodynamic / mass-conservation properties
# -----------------------------------------------------------------------------
@dataclass
class CRNThermoProperties:
    """
    Thermodynamic-style properties of a CRN derived purely from stoichiometry.

    Attributes
    ----------
    N : np.ndarray
        Stoichiometric matrix (species x reactions).
    reversible_mask : np.ndarray[bool]
        True for reversible reactions, False for irreversible.
    irreversible_mask : np.ndarray[bool]
        Logical negation of `reversible_mask`.
    left_nullspace : np.ndarray
        Basis of ker(N^T) (columns).
    right_nullspace : np.ndarray
        Basis of ker(N) (columns).
    positive_conservation_law : Optional[np.ndarray]
        Strictly positive vector m with m^T N = 0, if it exists.
    conservative : bool
        True iff a positive conservation law exists.
    nonnegative_left_kernel_vector : Optional[np.ndarray]
        A nonnegative left-kernel vector y >= 0 with y^T N = 0, if found.
    has_irreversible_futile_cycle : bool
        True iff there exists v >= 0 with N v = 0 that uses some
        irreversible reaction.
    thermodynamically_sound_irreversible : bool
        True iff there is no irreversible futile cycle.
    """

    N: np.ndarray
    reversible_mask: np.ndarray
    irreversible_mask: np.ndarray
    left_nullspace: np.ndarray
    right_nullspace: np.ndarray
    positive_conservation_law: Optional[np.ndarray]
    conservative: bool
    nonnegative_left_kernel_vector: Optional[np.ndarray]
    has_irreversible_futile_cycle: bool
    thermodynamically_sound_irreversible: bool


# -----------------------------------------------------------------------------
# Core linear-algebra helpers
# -----------------------------------------------------------------------------
def _compute_nullspaces(
    N: np.ndarray, tol: float = 1e-10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute bases of ker(N^T) (left) and ker(N) (right).
    """
    _require_scipy()
    left = null_space(N.T, rcond=tol)
    right = null_space(N, rcond=tol)
    return left, right


def find_positive_conservation_law(
    N: np.ndarray,
    *,
    min_mass: float = 1e-6,
    tol_eq: float = 1e-9,
    max_iter: int = 1,
) -> Optional[np.ndarray]:
    """
    Find m > 0 with m^T N = 0 (conservativity).

    LP:
        minimize 1^T m
        s.t. N^T m = 0
             m_i >= min_mass
    """
    _require_scipy()

    n_species, n_rxn = N.shape
    c = np.ones(n_species)
    A_eq = N.T
    b_eq = np.zeros(n_rxn)
    bounds = [(min_mass, None) for _ in range(n_species)]

    for _ in range(max_iter):
        res = linprog(
            c=c,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )
        if res.success:
            m = res.x
            if np.all(m >= min_mass - 1e-12) and np.all(np.abs(N.T @ m) <= tol_eq):
                return m
        min_mass *= 0.1  # relax lower bound and retry

    return None


def find_nonnegative_left_kernel_vector(
    N: np.ndarray,
    *,
    min_value: float = 0.0,
    enforce_nonzero: bool = True,
    tol_eq: float = 1e-9,
) -> Optional[np.ndarray]:
    """
    Find y >= min_value with y^T N = 0 (approximate moiety-like law).

    Returns a real-valued y; you can scale/round externally if you need
    integer moiety laws.
    """
    _require_scipy()

    n_species, n_rxn = N.shape
    c = np.ones(n_species)
    A_eq = N.T
    b_eq = np.zeros(n_rxn)
    bounds = [(min_value, None) for _ in range(n_species)]

    A_ub = None
    b_ub = None
    if enforce_nonzero:
        # -sum(y) <= -1  <=>  sum(y) >= 1
        A_ub = np.array([-np.ones(n_species)])
        b_ub = np.array([-1.0])

    res = linprog(
        c=c,
        A_eq=A_eq,
        b_eq=b_eq,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs",
    )
    if not res.success:
        return None

    y = res.x
    if enforce_nonzero and np.sum(y) < 1e-6:
        return None
    if np.any(y < min_value - 1e-12):
        return None
    if np.any(np.abs(N.T @ y) > tol_eq):
        return None

    return y


def has_irreversible_futile_cycle(
    N: np.ndarray,
    irreversible_mask: np.ndarray,
    *,
    min_flux: float = 1e-6,
    tol_eq: float = 1e-9,
) -> bool:
    """
    Check whether there exists an irreversible futile cycle:

        v >= 0, N v = 0, sum_{r in irr} v_r >= min_flux
    """
    _require_scipy()

    n_species, n_rxn = N.shape
    if not np.any(irreversible_mask):
        return False

    c = np.ones(n_rxn)
    A_eq = N
    b_eq = np.zeros(n_species)

    # -sum(v_irr) <= -min_flux
    A_ub = np.zeros((1, n_rxn))
    A_ub[0, irreversible_mask] = -1.0
    b_ub = np.array([-min_flux])

    bounds = [(0.0, None) for _ in range(n_rxn)]

    res = linprog(
        c=c,
        A_eq=A_eq,
        b_eq=b_eq,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs",
    )
    if not res.success:
        return False

    v = res.x
    if np.any(np.abs(N @ v) > tol_eq):
        return False
    if np.sum(v[irreversible_mask]) < 0.5 * min_flux:
        return False

    return True


def in_image_of_N_transpose(
    N: np.ndarray,
    g: np.ndarray,
    tol: float = 1e-8,
) -> bool:
    """
    Check if g lies in im(N^T), i.e. ∃ G with N^T G = g.
    """
    g = np.asarray(g, dtype=float).reshape(-1)
    _, n_rxn = N.shape
    if g.shape[0] != n_rxn:
        raise ValueError(f"g must have length {n_rxn}, got {g.shape[0]}")

    N_T = N.T
    G, *_ = np.linalg.lstsq(N_T, g, rcond=None)
    residual = np.linalg.norm(N_T @ G - g, ord=2)
    return residual <= tol


# -----------------------------------------------------------------------------
# High-level thermo property computation
# -----------------------------------------------------------------------------
def compute_thermo_properties(
    network: CRNNetwork,
    *,
    min_mass: float = 1e-6,
    min_flux: float = 1e-6,
) -> CRNThermoProperties:
    """
    Compute thermodynamic-style properties of a CRN using its stoichiometry.

    This is independent of any particular kinetic parameterization.
    """
    N = network.stoichiometric_matrix()
    reversible_mask = np.array(
        [rxn.reversible for rxn in network.reactions],
        dtype=bool,
    )
    irreversible_mask = ~reversible_mask

    left_ns, right_ns = _compute_nullspaces(N)
    m_pos = find_positive_conservation_law(
        N, min_mass=min_mass, tol_eq=1e-9, max_iter=2
    )
    conservative = m_pos is not None
    y_nonneg = find_nonnegative_left_kernel_vector(
        N, min_value=0.0, enforce_nonzero=True
    )
    has_cycle = has_irreversible_futile_cycle(
        N,
        irreversible_mask,
        min_flux=min_flux,
    )

    thermo_sound = not has_cycle

    return CRNThermoProperties(
        N=N,
        reversible_mask=reversible_mask,
        irreversible_mask=irreversible_mask,
        left_nullspace=left_ns,
        right_nullspace=right_ns,
        positive_conservation_law=m_pos,
        conservative=conservative,
        nonnegative_left_kernel_vector=y_nonneg,
        has_irreversible_futile_cycle=has_cycle,
        thermodynamically_sound_irreversible=thermo_sound,
    )


# -----------------------------------------------------------------------------
# Reversible completion
# -----------------------------------------------------------------------------
def reversible_completion(
    network: CRNNetwork,
    g: Optional[np.ndarray] = None,
) -> Tuple[CRNNetwork, Optional[np.ndarray]]:
    """
    Build the reversible completion of a CRN.

    For each irreversible reaction r:
        - keep r but mark it reversible,
        - add a reverse reaction r_rev with reactants/products swapped,
          also marked reversible.
    If a reaction-energy vector g is provided, g_rev = -g_r for each
    newly added reverse reaction.
    """
    species_new: List[CRNSpecies] = [
        CRNSpecies(name=s.name, metadata=dict(s.metadata)) for s in network.species
    ]

    # Start with copies of all original reactions, but mark them reversible.
    reactions_new: List[CRNReaction] = []
    irr_indices: List[int] = []

    for j, rxn in enumerate(network.reactions):
        reactions_new.append(
            CRNReaction(
                reactants=dict(rxn.reactants),
                products=dict(rxn.products),
                reversible=True,
                metadata=dict(rxn.metadata),
            )
        )
        if not rxn.reversible:
            irr_indices.append(j)

    # Add reverse reactions for irreversibles
    for j in irr_indices:
        rxn = network.reactions[j]
        reactions_new.append(
            CRNReaction(
                reactants=dict(rxn.products),
                products=dict(rxn.reactants),
                reversible=True,
                metadata={**rxn.metadata, "reverse_of": j},
            )
        )

    # Preserve subclass type (CRNNetwork vs ReactionNetwork)
    NetworkCls: Type[CRNNetwork] = type(network)
    net_star = NetworkCls(species=species_new, reactions=reactions_new)

    g_star: Optional[np.ndarray] = None
    if g is not None:
        g = np.asarray(g, dtype=float).reshape(-1)
        if g.shape[0] != len(network.reactions):
            raise ValueError(
                f"g must have length {len(network.reactions)}, got {g.shape[0]}"
            )
        g_extra = -g[irr_indices]
        g_star = np.concatenate([g, g_extra])

    return net_star, g_star
