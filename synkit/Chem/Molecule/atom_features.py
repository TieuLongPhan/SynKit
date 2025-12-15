from __future__ import annotations

from collections import deque
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable

from rdkit import Chem

from .valence import ValenceResolver
from .descriptors import PerMolDescriptors


@runtime_checkable
class _AtomLike(Protocol):
    """
    Minimal protocol for RDKit Atom-like objects used by AtomFeatureExtractor.
    """

    def GetIdx(self) -> int: ...
    def GetSymbol(self) -> str: ...
    def GetIsAromatic(self) -> bool: ...
    def GetTotalNumHs(self) -> int: ...
    def GetFormalCharge(self) -> int: ...
    def GetNumRadicalElectrons(self) -> int: ...
    def GetHybridization(self) -> Any: ...
    def IsInRing(self) -> bool: ...
    def GetNumImplicitHs(self) -> int: ...
    def GetNeighbors(self): ...
    def GetAtomMapNum(self) -> int: ...
    def GetChiralTag(self) -> Any: ...
    def HasProp(self, name: str) -> bool: ...
    def GetProp(self, name: str) -> str: ...
    def GetDoubleProp(self, name: str) -> float: ...
    def GetDegree(self) -> int: ...
    def GetIsotope(self) -> int: ...


class AtomFeatureExtractor:
    """
    Build per-atom feature dictionaries for an RDKit molecule.

    The extractor supports two profiles:
      - ``"minimal"`` : a compact set of attributes (backwards compatible with
                        the original `_gather_atom_properties`).
      - ``"full"``    : includes valence, ring sizes, neighbor counts,
                        shortest distances to functional groups, and optional
                        descriptors from :class:`PerMolDescriptors`.

    The class exposes a fluent API: ``.build(atom)`` returns ``self`` and stores
    the result in ``.feature`` (dict). For batch processing use
    ``.build_all()`` and read ``.all_features`` afterwards. For one-off usage,
    the compatibility helper ``.build_dict(atom)`` returns the feature dict
    directly.

    :param mol: RDKit molecule to extract features from.
    :param per: Optional precomputed per-atom descriptors (EState, Crippen, etc.).
    :param profile: Feature profile to compute (``"minimal"`` or ``"full"``).
    """

    SUPPORTED_PROFILES = ("minimal", "full")

    def __init__(
        self,
        mol: Chem.Mol,
        per: Optional[PerMolDescriptors] = None,
        profile: str = "minimal",
    ):
        if profile not in self.SUPPORTED_PROFILES:
            raise ValueError(
                f"Unsupported profile: {profile!r}. Supported: {self.SUPPORTED_PROFILES}"
            )
        self.mol: Chem.Mol = mol
        self.per: Optional[PerMolDescriptors] = per
        self.profile: str = profile

        # results (filled by build/build_all)
        self._last_feature: Optional[Dict[str, Any]] = None
        self._all_features: Optional[List[Dict[str, Any]]] = None

    # ---------------- fluent / compatibility API -------------------------

    def build(self, atom: Chem.Atom | _AtomLike) -> "AtomFeatureExtractor":
        """
        Compute features for *one* atom and store them internally.

        Returns self to enable chaining. The result dictionary can be accessed
        via the ``feature`` property.

        :param atom: RDKit Atom instance (or Atom-like object).
        :returns: self
        """
        if self.profile == "full":
            self._last_feature = self._build_full(atom)
        else:
            self._last_feature = self._build_minimal(atom)
        return self

    def build_dict(self, atom: Chem.Atom | _AtomLike) -> Dict[str, Any]:
        """
        Backwards-compatible helper that returns the computed feature dict
        directly (does not alter ``.feature`` or ``.all_features``).

        :param atom: RDKit Atom instance (or Atom-like object).
        :returns: feature dictionary
        """
        if self.profile == "full":
            return self._build_full(atom)
        return self._build_minimal(atom)

    def build_all(self) -> "AtomFeatureExtractor":
        """
        Compute features for *all* atoms in the molecule and store them in
        ``.all_features``. Returns self for chaining.

        :returns: self
        """
        features: List[Dict[str, Any]] = []
        try:
            for atom in self.mol.GetAtoms():
                features.append(self.build_dict(atom))
        except Exception:
            # Defensive: if iteration fails return empty list
            features = []
        self._all_features = features
        return self

    # ---------------- properties to retrieve results ---------------------

    @property
    def feature(self) -> Dict[str, Any]:
        """
        The last computed feature dictionary (via ``build``).

        :raises RuntimeError: if ``build`` has not been called yet.
        """
        if self._last_feature is None:
            raise RuntimeError("No features computed yet — call `build(atom)` first.")
        return dict(self._last_feature)

    @property
    def all_features(self) -> List[Dict[str, Any]]:
        """
        List of feature dicts for every atom (populated by ``build_all``).

        If ``build_all`` was not called, this property will call it lazily.
        """
        if self._all_features is None:
            self.build_all()
        assert self._all_features is not None
        return list(self._all_features)

    # ---------------- minimal (backwards-compatible) --------------------

    def _build_minimal(self, atom: Chem.Atom | _AtomLike) -> Dict[str, Any]:
        """
        Minimal feature set, intended to match the original helper.

        :param atom: RDKit Atom instance (or Atom-like object).
        :returns: dict of features.
        """
        # Gasteiger (tolerant access)
        gcharge = 0.0
        try:
            if atom.HasProp("_GasteigerCharge"):
                try:
                    gcharge = float(atom.GetProp("_GasteigerCharge"))
                except Exception:
                    try:
                        gcharge = float(atom.GetDoubleProp("_GasteigerCharge"))
                    except Exception:
                        gcharge = 0.0
        except Exception:
            gcharge = 0.0

        # neighbors list (safe)
        try:
            neighbor_symbols = sorted(nb.GetSymbol() for nb in atom.GetNeighbors())
        except Exception:
            neighbor_symbols = []

        try:
            atom_map = int(atom.GetAtomMapNum())
        except Exception:
            atom_map = 0

        return {
            "element": atom.GetSymbol(),
            "aromatic": bool(atom.GetIsAromatic()),
            "hcount": int(atom.GetTotalNumHs()),
            "charge": int(atom.GetFormalCharge()),
            "radical": int(atom.GetNumRadicalElectrons()),
            "isomer": self._stereo_atom(atom),
            "partial_charge": round(float(gcharge), 3),
            "hybridization": str(atom.GetHybridization()),
            "in_ring": bool(atom.IsInRing()),
            "implicit_hcount": int(atom.GetNumImplicitHs()),
            "neighbors": neighbor_symbols,
            "atom_map": atom_map,
        }

    # ---------------- full profile (extra descriptors) ------------------

    def _build_full(self, atom: Chem.Atom | _AtomLike) -> Dict[str, Any]:
        """
        Full feature set, extends minimal with additional computed properties.

        :param atom: RDKit Atom instance (or Atom-like object).
        :returns: dict of features
        """
        d = self._build_minimal(atom)

        # valence (safe)
        try:
            ev = ValenceResolver.explicit(atom)
            iv = ValenceResolver.implicit(atom)
        except Exception:
            ev = 0
            iv = 0

        # core additions
        d.update(
            {
                "explicit_valence": int(ev),
                "implicit_valence": int(iv),
                "valence": int(ev + iv),
                "total_num_hs": (
                    int(atom.GetTotalNumHs()) if hasattr(atom, "GetTotalNumHs") else 0
                ),
                "chiral_tag": str(atom.GetChiralTag()),
                "is_chiral_center": (
                    bool(atom.HasProp("_ChiralityPossible") and atom.GetDegree() > 0)
                    if hasattr(atom, "HasProp")
                    else False
                ),
                "ring_sizes": self._ring_sizes(atom),
                "nbr_elements_counts_r1": self._neighbor_counts(atom),
                "dist_to_carbonyl": self._dist_to(
                    lambda a: self._is_carbonyl_atom(a), atom.GetIdx()
                ),
                "dist_to_hetero": self._dist_to(
                    lambda a: a.GetSymbol() not in {"C", "H"}, atom.GetIdx()
                ),
                "dist_to_halogen": self._dist_to(
                    lambda a: a.GetSymbol() in {"F", "Cl", "Br", "I"}, atom.GetIdx()
                ),
                "dist_to_aromatic": self._dist_to(
                    lambda a: a.GetIsAromatic(), atom.GetIdx()
                ),
                "alpha_to_carbonyl": any(
                    self._is_carbonyl_atom(nb) for nb in atom.GetNeighbors()
                ),
            }
        )

        # Optional: estates/crippen if provided in PerMolDescriptors
        if self.per is not None:
            idx = atom.GetIdx()
            if idx < len(self.per.estate):
                d["estate"] = float(self.per.estate[idx])
            if idx < len(self.per.crippen_logp):
                d["crippen_logp"] = float(self.per.crippen_logp[idx])
            if idx < len(self.per.crippen_mr):
                d["crippen_mr"] = float(self.per.crippen_mr[idx])

        return d

    # ---------------- helpers (small, well-typed) ------------------------

    @staticmethod
    def _stereo_atom(atom: Chem.Atom | _AtomLike) -> str:
        """
        Map RDKit chiral tags to simple stereodescriptors.

        Returns "S", "R" or "N" (none/unknown).
        """
        try:
            ch = atom.GetChiralTag()
            if ch == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
                return "S"
            if ch == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
                return "R"
        except Exception:
            pass
        return "N"

    def _ring_sizes(self, atom: Chem.Atom | _AtomLike) -> List[int]:
        """
        Return list of ring sizes the atom belongs to (empty if none).
        """
        sizes: List[int] = []
        try:
            ri = self.mol.GetRingInfo()
            for ring in ri.AtomRings():
                if atom.GetIdx() in ring:
                    sizes.append(len(ring))
        except Exception:
            pass
        return sizes

    @staticmethod
    def _neighbor_counts(atom: Chem.Atom | _AtomLike) -> Dict[str, int]:
        """
        Count neighbor element occurrences (e.g., {"H": 3, "C": 1}).
        """
        counts: Dict[str, int] = {}
        try:
            for nb in atom.GetNeighbors():
                s = nb.GetSymbol()
                counts[s] = counts.get(s, 0) + 1
        except Exception:
            pass
        return counts

    def _is_carbonyl_atom(self, a: Chem.Atom | _AtomLike) -> bool:
        """
        Heuristic: carbon atom double-bonded to oxygen (C=O).
        """
        try:
            if a.GetSymbol() != "C":
                return False
            for nb in a.GetNeighbors():
                if nb.GetSymbol() == "O":
                    b = self.mol.GetBondBetweenAtoms(a.GetIdx(), nb.GetIdx())
                    if b is not None and b.GetBondTypeAsDouble() >= 2.0:
                        return True
        except Exception:
            pass
        return False

    def _dist_to(
        self, predicate: Callable[[Chem.Atom], bool], start_idx: int, maxd: int = 99
    ) -> int:
        """
        Shortest-path distance (in bonds) from atom ``start_idx`` to the first
        atom that satisfies ``predicate``. Returns ``maxd`` if none found
        within the search limit.

        :param predicate: callable that accepts an RDKit atom and returns bool.
        :param start_idx: starting atom index.
        :param maxd: maximum distance to search (defaults to 99).
        :returns: integer distance (0 means start atom satisfies predicate).
        """
        try:
            seen = {start_idx}
            dq = deque([(start_idx, 0)])
            while dq:
                idx, d = dq.popleft()
                a = self.mol.GetAtomWithIdx(idx)
                try:
                    if predicate(a):
                        return d
                except Exception:
                    # ignore predicate failures for robustness
                    pass
                if d >= maxd:
                    continue
                for nb in a.GetNeighbors():
                    ni = nb.GetIdx()
                    if ni not in seen:
                        seen.add(ni)
                        dq.append((ni, d + 1))
        except Exception:
            pass
        return maxd

    # ---------------- utilities / metadata --------------------------------

    def __repr__(self) -> str:
        try:
            n_atoms = self.mol.GetNumAtoms()
        except Exception:
            n_atoms = -1
        return f"{self.__class__.__name__}(profile={self.profile!r}, n_atoms={n_atoms})"

    @classmethod
    def help(cls) -> str:
        """
        Short machine-readable help describing supported profiles.
        """
        return (
            "AtomFeatureExtractor.help() -> str\n\n"
            "Supported profiles:\n"
            "  - 'minimal': compact, standard atom properties\n"
            "  - 'full'   : includes valence, ring sizes, neighbor counts, "
            "distances to functional groups, and optional PerMolDescriptors fields\n"
        )
