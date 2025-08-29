# synkit/Chem/Molecule/descriptors.py
from __future__ import annotations

from dataclasses import dataclass
from typing import (
    List,
    Tuple,
    Optional,
    Protocol,
    runtime_checkable,
    Dict,
    Any,
    Sequence,
)

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit import RDLogger

# Keep RDKit quiet in helper scope (best effort)
try:
    RDLogger.DisableLog("rdApp.*")
except Exception:
    pass


@runtime_checkable
class _MolLike(Protocol):
    """
    Minimal protocol for RDKit-like molecules used by this module.
    """

    def GetNumAtoms(self) -> int: ...
    def GetAtoms(self): ...
    def GetAtomWithIdx(self, idx: int): ...


# -----------------------------
# Internal small helpers
# -----------------------------
def compute_gasteiger_inplace(mol: Chem.Mol | Any) -> None:
    """
    Compatibility helper: compute Gasteiger charges in-place (best-effort).
    Reintroduced for backward compatibility with code that imports this from
    synkit.Chem.Molecule.descriptors.

    :param mol: RDKit Mol (or mol-like object) to annotate. Mutates in place.
    :returns: None
    """
    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception:
        # best-effort: swallow RDKit quirks and continue
        return


def _make_copy_and_sanitize(mol: Chem.Mol | _MolLike, sanitize: bool) -> Chem.Mol:
    """
    Make a defensive copy of ``mol`` and optionally sanitize it.

    :param mol: input molecule-like object
    :param sanitize: whether to call ``Chem.SanitizeMol`` on the copy
    :returns: copied (and possibly sanitized) RDKit Mol
    """
    m = Chem.Mol(mol)
    if sanitize:
        try:
            Chem.SanitizeMol(m)
        except Exception:
            # best-effort: keep working with the copy even if sanitization fails
            pass
    return m


def _safe_get_gasteiger_value(atom: Any) -> float:
    """
    Extract a Gasteiger charge value from an RDKit atom, tolerant to API variants.

    :param atom: RDKit atom-like object.
    :returns: float value (0.0 on failure).
    """
    try:
        if atom.HasProp("_GasteigerCharge"):
            try:
                return float(atom.GetProp("_GasteigerCharge"))
            except Exception:
                try:
                    return float(atom.GetDoubleProp("_GasteigerCharge"))
                except Exception:
                    return 0.0
    except Exception:
        return 0.0
    return 0.0


def _compute_gasteiger_list(m: Chem.Mol, n_atoms: int) -> List[float]:
    """
    Compute per-atom Gasteiger charges for molecule ``m``.

    Best-effort: returns a list of length ``n_atoms`` filled with floats.
    """
    gasteiger: List[float] = [0.0] * n_atoms
    try:
        AllChem.ComputeGasteigerCharges(m)
        for a in m.GetAtoms():
            try:
                idx = a.GetIdx()
            except Exception:
                continue
            gasteiger[idx] = _safe_get_gasteiger_value(a)
    except Exception:
        # On any failure leave zeros
        pass
    return gasteiger


def _compute_estate_list(m: Chem.Mol, n_atoms: int) -> List[float]:
    """
    Compute per-atom EState indices (best-effort). Returns zeros if unavailable.
    """
    try:
        estate_vals = list(
            __import__("rdkit.Chem.EState", fromlist=["EState"]).EState.EStateIndices(m)
        )
        if len(estate_vals) != n_atoms:
            estate_vals = [0.0] * n_atoms
    except Exception:
        estate_vals = [0.0] * n_atoms
    return estate_vals


def _calc_crippen_contribs(m: Chem.Mol) -> Optional[List[Tuple[float, float]]]:
    """
    Version-tolerant wrapper for RDKit's Crippen contribution routine.

    Kept small and used by _compute_crippen_lists.
    """
    fn = getattr(rdMolDescriptors, "CalcCrippenContribs", None)
    if fn is None:
        fn = getattr(rdMolDescriptors, "_CalcCrippenContribs", None)
    if fn is None:
        return None
    try:
        return list(fn(m))
    except Exception:
        return None


def _compute_crippen_lists(
    m: Chem.Mol, n_atoms: int
) -> Tuple[List[float], List[float]]:
    """
    Compute per-atom Crippen (logP, MR) lists. Returns two lists of length n_atoms.
    """
    cr_logp: List[float] = [0.0] * n_atoms
    cr_mr: List[float] = [0.0] * n_atoms
    cr = _calc_crippen_contribs(m)
    if cr is not None and len(cr) == n_atoms:
        try:
            cr_logp = [float(x[0]) for x in cr]
            cr_mr = [float(x[1]) for x in cr]
        except Exception:
            cr_logp = [0.0] * n_atoms
            cr_mr = [0.0] * n_atoms
    return cr_logp, cr_mr


def _normalize_vector(vec: Sequence[float], method: Optional[str]) -> List[float]:
    """
    Normalize a numeric vector in-place style (returns a new list).

    Supported methods: None, "zscore", "minmax".
    If method is None or vector length <= 1, returns a float copy of vec.
    """
    n = len(vec)
    if n == 0 or method is None:
        return [float(x) for x in vec]

    vals = [float(x) for x in vec]
    if method == "zscore":
        mean = sum(vals) / n
        var = sum((x - mean) ** 2 for x in vals) / n
        std = var**0.5
        if std == 0.0:
            return [0.0 for _ in vals]
        return [(x - mean) / std for x in vals]
    elif method == "minmax":
        lo = min(vals)
        hi = max(vals)
        if hi == lo:
            return [0.0 for _ in vals]
        return [(x - lo) / (hi - lo) for x in vals]
    else:
        raise ValueError(f"Unsupported normalization method: {method!r}")


def _apply_normalization_to_all(
    vectors: Dict[str, List[float]], method: Optional[str]
) -> Dict[str, List[float]]:
    """
    Apply normalization ``method`` to all vectors in the dict and return a new dict.
    """
    if method is None:
        # ensure float conversion
        return {k: [float(x) for x in v] for k, v in vectors.items()}
    normalized: Dict[str, List[float]] = {}
    for k, v in vectors.items():
        normalized[k] = _normalize_vector(v, method)
    return normalized


# -----------------------------
# Public dataclass + builder
# -----------------------------


@dataclass(frozen=True)
class PerMolDescriptors:
    """
    Immutable container for per-atom descriptor lists.

    :param gasteiger: per-atom Gasteiger charges
    :param estate: per-atom EState indices
    :param crippen_logp: per-atom Crippen logP contributions
    :param crippen_mr: per-atom Crippen MR contributions
    """

    gasteiger: List[float]
    estate: List[float]
    crippen_logp: List[float]
    crippen_mr: List[float]

    # ----------------- Convenience constructors ---------------------------
    @classmethod
    def compute(
        cls,
        mol: Chem.Mol | _MolLike,
        sanitize: bool = True,
        normalize: Optional[str] = None,
    ) -> "PerMolDescriptors":
        """
        Best-effort compute per-atom descriptors for the given molecule.

        This function delegates work to small helpers for clarity and easier
        testing. Behavior is identical to the previous implementation.

        :param mol: RDKit molecule (or duck-typed equivalent).
        :param sanitize: try to sanitize the copied molecule (default True).
        :param normalize: normalization method: None (default), "zscore", or "minmax".
        :returns: PerMolDescriptors instance.
        """
        # Defensive copy and optional sanitization
        m = _make_copy_and_sanitize(mol, sanitize)
        n_atoms = m.GetNumAtoms() if hasattr(m, "GetNumAtoms") else 0

        # Compute components (small focused helpers)
        gasteiger = _compute_gasteiger_list(m, n_atoms)
        estate_vals = _compute_estate_list(m, n_atoms)
        cr_logp, cr_mr = _compute_crippen_lists(m, n_atoms)

        # Apply normalization if requested
        vectors = {
            "gasteiger": gasteiger,
            "estate": estate_vals,
            "crippen_logp": cr_logp,
            "crippen_mr": cr_mr,
        }
        vectors = _apply_normalization_to_all(vectors, normalize)

        return cls(
            gasteiger=vectors["gasteiger"],
            estate=vectors["estate"],
            crippen_logp=vectors["crippen_logp"],
            crippen_mr=vectors["crippen_mr"],
        )

    @classmethod
    def from_smiles(
        cls, smiles: str, sanitize: bool = True, normalize: Optional[str] = None
    ) -> "PerMolDescriptors":
        """
        Parse SMILES and compute descriptors.

        :param smiles: SMILES string to parse.
        :param sanitize: try to sanitize the parsed molecule (default True).
        :param normalize: optional normalization ("zscore" | "minmax" | None).
        :returns: PerMolDescriptors instance.
        :raises ValueError: if SMILES fails to parse.
        """
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            raise ValueError(f"Failed to parse SMILES: {smiles!r}")
        return cls.compute(m, sanitize=sanitize, normalize=normalize)

    # ----------------- Small helpers / serialization ---------------------
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_atoms={self.num_atoms}, "
            f"has_gasteiger={any(v != 0.0 for v in self.gasteiger)})"
        )

    @property
    def num_atoms(self) -> int:
        """
        Infer atom count from stored lists (prefers gasteiger length).

        :returns: inferred atom count.
        """
        return len(self.gasteiger)

    def to_dict(self) -> Dict[str, List[float]]:
        """
        Convert to a plain dictionary.

        :returns: dict with the per-atom lists.
        """
        return {
            "gasteiger": list(self.gasteiger),
            "estate": list(self.estate),
            "crippen_logp": list(self.crippen_logp),
            "crippen_mr": list(self.crippen_mr),
        }


class PerMolDescriptorsBuilder:
    """
    Fluent builder for PerMolDescriptors.

    Usage example:
        desc = (
            PerMolDescriptorsBuilder(mol)
            .compute_gasteiger()
            .compute_estate()
            .compute_crippen()
            .normalize("zscore")
            .build()
            .descriptor
        )

    The builder's chainable methods return ``self``; call ``.build()`` then
    access the final PerMolDescriptors via the ``.descriptor`` property.
    """

    def __init__(self, mol: Chem.Mol | _MolLike, sanitize: bool = True):
        """
        Create a builder for the given molecule.

        :param mol: RDKit Mol or equivalent.
        :param sanitize: try to sanitize the internal copy (default True).
        """
        self._mol = Chem.Mol(mol)
        self._sanitize = sanitize
        if sanitize:
            try:
                Chem.SanitizeMol(self._mol)
            except Exception:
                pass

        n_atoms = self._mol.GetNumAtoms() if hasattr(self._mol, "GetNumAtoms") else 0
        self._n_atoms = n_atoms

        # Internal working vectors (None until computed)
        self._gasteiger: Optional[List[float]] = None
        self._estate: Optional[List[float]] = None
        self._crippen_logp: Optional[List[float]] = None
        self._crippen_mr: Optional[List[float]] = None
        self._normalized_method: Optional[str] = None
        self._built: Optional[PerMolDescriptors] = None

    # ---------- Chainable compute methods --------------------------------
    def compute_gasteiger(self) -> "PerMolDescriptorsBuilder":
        """
        Compute Gasteiger charges (best-effort) and store internally.

        :returns: self (chainable).
        """
        self._gasteiger = _compute_gasteiger_list(self._mol, self._n_atoms)
        self._built = None
        return self

    def compute_estate(self) -> "PerMolDescriptorsBuilder":
        """
        Compute EState indices (best-effort).

        :returns: self (chainable).
        """
        self._estate = _compute_estate_list(self._mol, self._n_atoms)
        self._built = None
        return self

    def compute_crippen(self) -> "PerMolDescriptorsBuilder":
        """
        Compute Crippen per-atom contributions (best-effort).

        :returns: self (chainable).
        """
        cr_logp, cr_mr = _compute_crippen_lists(self._mol, self._n_atoms)
        self._crippen_logp = cr_logp
        self._crippen_mr = cr_mr
        self._built = None
        return self

    # ---------- Normalization --------------------------------------------
    def normalize(self, method: Optional[str]) -> "PerMolDescriptorsBuilder":
        """
        Normalize any computed vectors using ``method`` ("zscore" | "minmax" | None).

        :param method: normalization method or None to skip.
        :returns: self (chainable).
        """
        if method is None:
            self._normalized_method = None
            return self

        if method not in ("zscore", "minmax"):
            raise ValueError(f"Unsupported normalization method: {method!r}")

        if self._gasteiger is not None:
            self._gasteiger = _normalize_vector(self._gasteiger, method)
        if self._estate is not None:
            self._estate = _normalize_vector(self._estate, method)
        if self._crippen_logp is not None:
            self._crippen_logp = _normalize_vector(self._crippen_logp, method)
        if self._crippen_mr is not None:
            self._crippen_mr = _normalize_vector(self._crippen_mr, method)

        self._normalized_method = method
        self._built = None
        return self

    # ---------- Finalize / retrieve -------------------------------------
    def build(self) -> "PerMolDescriptorsBuilder":
        """
        Finalize internal state and prepare the immutable PerMolDescriptors.

        The method stores the result internally and returns ``self``. Use the
        ``.descriptor`` property to access the final object.

        :returns: self
        """
        # If a vector wasn't computed, fill with zeros of appropriate length
        n = self._n_atoms
        g = self._gasteiger if self._gasteiger is not None else [0.0] * n
        e = self._estate if self._estate is not None else [0.0] * n
        crp = self._crippen_logp if self._crippen_logp is not None else [0.0] * n
        crm = self._crippen_mr if self._crippen_mr is not None else [0.0] * n

        self._built = PerMolDescriptors(
            gasteiger=g, estate=e, crippen_logp=crp, crippen_mr=crm
        )
        return self

    @property
    def descriptor(self) -> PerMolDescriptors:
        """
        Retrieve the built PerMolDescriptors. If not built yet, ``build()`` is
        called implicitly.

        :returns: PerMolDescriptors
        """
        if self._built is None:
            self.build()
        assert self._built is not None
        return self._built
