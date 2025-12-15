# standardize.py
"""
Molecule standardization helpers and a chainable MolStandardizer class.

Provides:
 - lightweight helpers: sanitize_and_canonicalize_smiles, fix_radical_rsmi, remove_isotopes, ...
 - MolStandardizer: fluent, chainable standardization API with convenience constructors.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem.SaltRemover import SaltRemover

# rdMolStandardize may not be available in all RDKit builds; import defensively.
try:
    from rdkit.Chem.MolStandardize import rdMolStandardize  # type: ignore
except Exception:  # pragma: no cover - allow imports to fail gracefully
    rdMolStandardize = None  # type: ignore

# Prefer project's logger if available, otherwise fallback to stdlib logging
try:
    # If running in the synkit environment this will pick up project logger
    from synkit.IO.debug import setup_logging  # type: ignore

    logger = setup_logging()
except Exception:
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logging.basicConfig(level=logging.WARNING)


# -------------------------
# Simple functional helpers
# -------------------------
def sanitize_and_canonicalize_smiles(smiles: str) -> Optional[str]:
    """
    Sanitize and canonicalize a SMILES string.

    :param smiles: Input SMILES string.
    :type smiles: str
    :returns: Canonical SMILES if valid, otherwise ``None``.
    :rtype: Optional[str]

    Notes
    -----
    The function attempts to parse and sanitize the SMILES with RDKit. On any
    parsing/sanitization failure it returns ``None`` (best-effort policy).
    """
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        logger.debug(
            "sanitize_and_canonicalize_smiles failed for %r", smiles, exc_info=True
        )
        return None


def normalize_molecule(mol: Chem.Mol) -> Chem.Mol:
    """
    Normalize a molecule using rdMolStandardize.Normalizer when available.

    :param mol: RDKit Mol object to normalize.
    :type mol: Chem.Mol
    :returns: Normalized RDKit Mol object (or original if normalizer missing).
    :rtype: Chem.Mol
    """
    if rdMolStandardize is None:
        logger.debug("rdMolStandardize not available; normalize skipped.")
        return mol
    try:
        normalizer_cls = getattr(rdMolStandardize, "Normalizer", None)
        if normalizer_cls is None:
            logger.debug("Normalizer not found in rdMolStandardize; normalize skipped.")
            return mol
        return normalizer_cls().normalize(mol)
    except Exception as exc:
        logger.debug("normalize_molecule failed: %s", exc, exc_info=True)
        return mol


def canonicalize_tautomer(mol: Chem.Mol) -> Chem.Mol:
    """
    Canonicalize tautomeric form using rdMolStandardize.TautomerEnumerator if available.

    :param mol: RDKit Mol object to canonicalize.
    :type mol: Chem.Mol
    :returns: Canonicalized tautomer Mol (or original if unavailable).
    :rtype: Chem.Mol
    """
    if rdMolStandardize is None:
        logger.debug("rdMolStandardize not available; canonicalize_tautomer skipped.")
        return mol
    try:
        taut_enum_cls = getattr(rdMolStandardize, "TautomerEnumerator", None)
        if taut_enum_cls is None:
            logger.debug(
                "TautomerEnumerator not present; canonicalize_tautomer skipped."
            )
            return mol
        return taut_enum_cls().Canonicalize(mol)
    except Exception as exc:
        logger.debug("canonicalize_tautomer failed: %s", exc, exc_info=True)
        return mol


def salts_remover(mol: Chem.Mol, remover: Optional[SaltRemover] = None) -> Chem.Mol:
    """
    Remove salts from a molecule using RDKit's SaltRemover.

    :param mol: RDKit Mol object to process.
    :type mol: Chem.Mol
    :param remover: Optional SaltRemover instance to use.
    :type remover: Optional[SaltRemover]
    :returns: Mol object with salts removed (best-effort).
    :rtype: Chem.Mol
    """
    try:
        _rem = remover if remover is not None else SaltRemover()
        return _rem.StripMol(mol)
    except Exception as exc:
        logger.debug("salts_remover failed: %s", exc, exc_info=True)
        return mol


def uncharge_molecule(mol: Chem.Mol) -> Chem.Mol:
    """
    Neutralize/uncharge a molecule using rdMolStandardize.Uncharger if available.

    :param mol: RDKit Mol object to neutralize.
    :type mol: Chem.Mol
    :returns: Neutralized Mol object (or original if uncharger missing).
    :rtype: Chem.Mol
    """
    if rdMolStandardize is None:
        logger.debug("rdMolStandardize not available; uncharge skipped.")
        return mol
    try:
        uncharger_cls = getattr(rdMolStandardize, "Uncharger", None)
        if uncharger_cls is None:
            logger.debug("Uncharger not found; uncharge skipped.")
            return mol
        return uncharger_cls().uncharge(mol)
    except Exception as exc:
        logger.debug("uncharge_molecule failed: %s", exc, exc_info=True)
        return mol


def fragments_remover(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """
    Keep only the largest fragment by atom count.

    :param mol: RDKit Mol object to fragment.
    :type mol: Chem.Mol
    :returns: Mol of the largest fragment, or None if input is empty.
    :rtype: Optional[Chem.Mol]
    """
    try:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        return max(frags, default=None, key=lambda m: m.GetNumAtoms())
    except Exception:
        logger.debug("fragments_remover failed.", exc_info=True)
        return None


def remove_explicit_hydrogens(mol: Chem.Mol) -> Chem.Mol:
    """
    Remove explicit hydrogens from the molecule (Chem.RemoveHs wrapper).

    :param mol: RDKit Mol object to process.
    :type mol: Chem.Mol
    :returns: Mol object without explicit hydrogens.
    :rtype: Chem.Mol
    """
    try:
        return Chem.RemoveHs(mol)
    except Exception as exc:
        logger.debug("remove_explicit_hydrogens failed: %s", exc, exc_info=True)
        return mol


# -------------------------
# Radical-handling helpers (further split to reduce complexity)
# -------------------------
def _zero_out_radicals_on_atoms(mol: Chem.Mol) -> None:
    """
    Iterate atoms and zero radical electron counts; also increment explicit H count
    by the number of radical electrons for each atom (best-effort).

    This mutates the input molecule in-place.

    :param mol: RDKit Mol to operate on (mutated in place).
    :type mol: Chem.Mol
    :returns: None
    :rtype: None
    """
    for atom in mol.GetAtoms():
        try:
            rad = int(atom.GetNumRadicalElectrons())
        except Exception:
            rad = 0
        if rad > 0:
            try:
                atom.SetNumExplicitHs(atom.GetNumExplicitHs() + rad)
                atom.SetNumRadicalElectrons(0)
            except Exception:
                # ignore per-atom failures
                continue


def _add_explicit_hydrogens(mol: Chem.Mol) -> Chem.Mol:
    """
    Wrapper around rdmolops.AddHs with defensive fallback to return original mol
    if AddHs fails.

    :param mol: RDKit Mol to process.
    :type mol: Chem.Mol
    :returns: Molecule with explicit hydrogens added, or original on failure.
    :rtype: Chem.Mol
    """
    try:
        return rdmolops.AddHs(mol)
    except Exception:
        return mol


def _maybe_remove_explicit_hydrogens(mol: Chem.Mol, removeH: bool) -> Chem.Mol:
    """
    Remove explicit hydrogens if removeH is True, otherwise return the molecule unchanged.

    :param mol: RDKit Mol to process.
    :type mol: Chem.Mol
    :param removeH: whether to remove explicit hydrogens after adding them.
    :type removeH: bool
    :returns: Processed molecule.
    :rtype: Chem.Mol
    """
    if not removeH:
        return mol
    try:
        return Chem.RemoveHs(mol)
    except Exception:
        return mol


def _replace_radicals_with_hs_in_mol(
    mol: Chem.Mol, removeH: bool = True
) -> Optional[Chem.Mol]:
    """
    High-level helper that replaces radicals with hydrogens.

    Steps:
      1. zero out radicals (and add explicit H counters)
      2. call AddHs to create explicit H atoms
      3. optionally remove explicit H atoms again

    :param mol: RDKit Mol to process.
    :type mol: Chem.Mol
    :param removeH: whether to remove explicit hydrogens after addition.
    :type removeH: bool
    :returns: Processed RDKit Mol or None on extreme failure.
    :rtype: Optional[Chem.Mol]
    """
    if mol is None:
        return None
    try:
        _zero_out_radicals_on_atoms(mol)
        mol_with_h = _add_explicit_hydrogens(mol)
        return _maybe_remove_explicit_hydrogens(mol_with_h, removeH)
    except Exception as exc:
        logger.debug("_replace_radicals_with_hs_in_mol failed: %s", exc, exc_info=True)
        return None


def remove_radicals_and_add_hydrogens(
    mol: Chem.Mol, removeH: bool = True
) -> Optional[Chem.Mol]:
    """
    Replace radical electrons by adding hydrogens and optionally remove explicit H.

    :param mol: RDKit Mol with possible radical atoms.
    :type mol: Chem.Mol
    :param removeH: If True, remove explicit hydrogens after addition.
    :type removeH: bool
    :returns: Mol with radicals neutralized (or None on failure).
    :rtype: Optional[Chem.Mol]
    """
    return _replace_radicals_with_hs_in_mol(mol, removeH)


# -------------------------
# Reaction SMILES helpers
# -------------------------
def _parse_reaction_smiles(rsmi: str) -> Optional[Tuple[str, str]]:
    """
    Parse a reaction SMILES of the form 'reactant>>product' and return tuple,
    or None if not parseable.

    :param rsmi: reaction SMILES string.
    :type rsmi: str
    :returns: tuple (reactant_smiles, product_smiles) or None.
    :rtype: Optional[Tuple[str, str]]
    """
    if ">>" not in rsmi:
        return None
    parts = rsmi.split(">>", 1)
    if len(parts) != 2:
        return None
    return parts[0], parts[1]


def _fix_single_side_of_reaction(smiles: str, removeH: bool) -> Optional[Chem.Mol]:
    """
    Create a Mol from SMILES (without sanitization), sanitize best-effort,
    then replace radicals and return processed Mol.

    :param smiles: SMILES for one side of a reaction.
    :type smiles: str
    :param removeH: whether to remove explicit hydrogens after addition.
    :type removeH: bool
    :returns: Processed RDKit Mol or None.
    :rtype: Optional[Chem.Mol]
    """
    m = Chem.MolFromSmiles(smiles, sanitize=False)
    if m is None:
        return None
    try:
        Chem.SanitizeMol(m)
    except Exception:
        pass
    return _replace_radicals_with_hs_in_mol(m, removeH)


def fix_radical_rsmi(rsmi: str, removeH: bool = True) -> str:
    """
    Fix radicals in a reaction SMILES by converting them to hydrogens.

    :param rsmi: Reaction SMILES string (format 'reactant>>product').
    :type rsmi: str
    :param removeH: If True, remove explicit hydrogens after addition.
    :type removeH: bool
    :returns: Corrected reaction SMILES with radicals replaced (or original on failure).
    :rtype: str
    """
    try:
        parsed = _parse_reaction_smiles(rsmi)
        if parsed is None:
            return rsmi
        react_smiles, prod_smiles = parsed
        r_fixed = _fix_single_side_of_reaction(react_smiles, removeH)
        p_fixed = _fix_single_side_of_reaction(prod_smiles, removeH)
        r_out = Chem.MolToSmiles(r_fixed) if r_fixed is not None else react_smiles
        p_out = Chem.MolToSmiles(p_fixed) if p_fixed is not None else prod_smiles
        return f"{r_out}>>{p_out}"
    except Exception:
        logger.debug("fix_radical_rsmi failed for %r", rsmi, exc_info=True)
        return rsmi


def remove_isotopes(mol: Chem.Mol) -> Chem.Mol:
    """
    Clear isotope labels on every atom in the molecule.

    :param mol: RDKit Mol object to process.
    :type mol: Chem.Mol
    :returns: The same RDKit Mol instance with isotopic labels cleared.
    :rtype: Chem.Mol
    """
    try:
        for atom in mol.GetAtoms():
            try:
                atom.SetIsotope(0)
            except Exception:
                continue
    except Exception as exc:
        logger.debug("remove_isotopes encountered error: %s", exc, exc_info=True)
    return mol


def clear_stereochemistry(mol: Chem.Mol) -> Chem.Mol:
    """
    Remove stereochemical annotations from a molecule.

    :param mol: RDKit Mol object to process.
    :type mol: Chem.Mol
    :returns: Mol object with stereochemistry removed.
    :rtype: Chem.Mol
    """
    try:
        Chem.RemoveStereochemistry(mol)
    except Exception as exc:
        logger.debug("clear_stereochemistry failed: %s", exc, exc_info=True)
    return mol


# -------------------------
# MolStandardizer class
# -------------------------
class MolStandardizer:
    """
    Chainable molecule standardizer wrapper around RDKit utilities.

    Use the fluent API to apply a sequence of standardizations and then
    retrieve the resulting molecule via the ``.mol`` property or ``.to_smiles()``.

    Example
    -------
    >>> std = MolStandardizer.from_smiles("CC(=O)[O-]").remove_salts().uncharge().mol
    """

    def __init__(self, mol: Chem.Mol, sanitize: bool = True) -> None:
        """
        Create a MolStandardizer working on a defensive copy of ``mol``.

        :param mol: RDKit Mol to operate on (a copy is created).
        :type mol: Chem.Mol
        :param sanitize: attempt initial sanitization (default True).
        :type sanitize: bool
        """
        self._mol: Optional[Chem.Mol] = Chem.Mol(mol) if mol is not None else None
        self._last_error: Optional[Exception] = None

        if sanitize and self._mol is not None:
            try:
                Chem.SanitizeMol(self._mol)
            except Exception as exc:
                logger.debug("Initial SanitizeMol failed: %s", exc, exc_info=True)
                self._last_error = exc

    # ----- constructors / convenience -----
    @classmethod
    def from_smiles(cls, smiles: str, sanitize: bool = True) -> "MolStandardizer":
        """
        Parse SMILES and return a configured standardizer.

        :param smiles: SMILES string to parse.
        :type smiles: str
        :param sanitize: attempt sanitization on parse (default True).
        :type sanitize: bool
        :returns: MolStandardizer.
        :rtype: MolStandardizer
        :raises ValueError: if SMILES fails to parse.
        """
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            raise ValueError(f"Failed to parse SMILES: {smiles!r}")
        return cls(m, sanitize=sanitize)

    @classmethod
    def _maybe_normalize(cls, inst: "MolStandardizer") -> "MolStandardizer":
        """
        Normalize if rdMolStandardize available (small wrapper).

        :param inst: MolStandardizer instance to operate on.
        :type inst: MolStandardizer
        :returns: same instance (chainable).
        :rtype: MolStandardizer
        """
        if rdMolStandardize is None:
            return inst
        try:
            inst.normalize()
        except Exception:
            pass
        return inst

    @classmethod
    def _maybe_keep_largest_fragment(
        cls, inst: "MolStandardizer", keep: bool
    ) -> "MolStandardizer":
        """
        Keep largest fragment if requested (small wrapper).

        :param inst: MolStandardizer instance to operate on.
        :type inst: MolStandardizer
        :param keep: whether to keep the largest fragment.
        :type keep: bool
        :returns: same instance (chainable).
        :rtype: MolStandardizer
        """
        if not keep:
            return inst
        try:
            inst.keep_largest_fragment()
        except Exception:
            pass
        return inst

    @classmethod
    def _maybe_remove_salts(cls, inst: "MolStandardizer") -> "MolStandardizer":
        """
        Remove salts (small wrapper).

        :param inst: MolStandardizer instance to operate on.
        :type inst: MolStandardizer
        :returns: same instance (chainable).
        :rtype: MolStandardizer
        """
        try:
            inst.remove_salts()
        except Exception:
            pass
        return inst

    @classmethod
    def _maybe_uncharge(cls, inst: "MolStandardizer") -> "MolStandardizer":
        """
        Uncharge if rdMolStandardize available (small wrapper).

        :param inst: MolStandardizer instance to operate on.
        :type inst: MolStandardizer
        :returns: same instance (chainable).
        :rtype: MolStandardizer
        """
        if rdMolStandardize is None:
            return inst
        try:
            inst.uncharge()
        except Exception:
            pass
        return inst

    @classmethod
    def _maybe_canonicalize_tautomer(cls, inst: "MolStandardizer") -> "MolStandardizer":
        """
        Canonicalize tautomer if rdMolStandardize available (small wrapper).

        :param inst: MolStandardizer instance to operate on.
        :type inst: MolStandardizer
        :returns: same instance (chainable).
        :rtype: MolStandardizer
        """
        if rdMolStandardize is None:
            return inst
        try:
            inst.canonicalize_tautomer()
        except Exception:
            pass
        return inst

    @classmethod
    def _apply_default_pipeline(
        cls, inst: "MolStandardizer", keep_largest_fragment: bool
    ) -> "MolStandardizer":
        """
        Apply the default standardization pipeline to an instance.

        :param inst: MolStandardizer instance to operate on.
        :type inst: MolStandardizer
        :param keep_largest_fragment: whether to keep the largest fragment.
        :type keep_largest_fragment: bool
        :returns: same instance after applying pipeline.
        :rtype: MolStandardizer
        """
        cls._maybe_normalize(inst)
        cls._maybe_keep_largest_fragment(inst, keep_largest_fragment)
        cls._maybe_remove_salts(inst)
        cls._maybe_uncharge(inst)
        cls._maybe_canonicalize_tautomer(inst)
        return inst

    @classmethod
    def standardize_smiles(
        cls, smiles: str, *, keep_largest_fragment: bool = True
    ) -> Optional[str]:
        """
        Quick convenience: parse SMILES, apply a sensible default standardization,
        and return canonical SMILES or None on failure.

        Default pipeline:
          sanitize -> normalize (if available) -> keep largest fragment ->
          remove salts -> uncharge -> canonicalize tautomer (if available)

        :param smiles: Input SMILES string.
        :type smiles: str
        :param keep_largest_fragment: keep only the largest fragment (default True).
        :type keep_largest_fragment: bool
        :returns: Canonical SMILES or None.
        :rtype: Optional[str]
        """
        try:
            inst = cls.from_smiles(smiles, sanitize=True)
            cls._apply_default_pipeline(
                inst, keep_largest_fragment=keep_largest_fragment
            )
            return inst.to_smiles(canonical=True)
        except Exception:
            logger.debug("standardize_smiles failed for %r", smiles, exc_info=True)
            return None

    # ----- mutating chainable operations -----
    def normalize(self) -> "MolStandardizer":
        """
        Normalize the internal molecule using rdMolStandardize.Normalizer.

        :returns: self (chainable).
        :rtype: MolStandardizer
        """
        if self._mol is None:
            return self
        if rdMolStandardize is None:
            logger.debug("rdMolStandardize not available; normalize skipped.")
            return self
        try:
            normalizer_cls = getattr(rdMolStandardize, "Normalizer", None)
            if normalizer_cls is None:
                logger.debug("Normalizer not found in rdMolStandardize.")
                return self
            self._mol = normalizer_cls().normalize(self._mol)
        except Exception as exc:
            logger.debug("normalize failed: %s", exc, exc_info=True)
            self._last_error = exc
        return self

    def canonicalize_tautomer(self) -> "MolStandardizer":
        """
        Canonicalize tautomer using rdMolStandardize.TautomerEnumerator.

        :returns: self (chainable).
        :rtype: MolStandardizer
        """
        if self._mol is None:
            return self
        if rdMolStandardize is None:
            logger.debug(
                "rdMolStandardize not available; canonicalize_tautomer skipped."
            )
            return self
        try:
            taut_enum_cls = getattr(rdMolStandardize, "TautomerEnumerator", None)
            if taut_enum_cls is None:
                logger.debug("TautomerEnumerator not found in rdMolStandardize.")
                return self
            self._mol = taut_enum_cls().Canonicalize(self._mol)
        except Exception as exc:
            logger.debug("canonicalize_tautomer failed: %s", exc, exc_info=True)
            self._last_error = exc
        return self

    def remove_salts(
        self, salt_remover: Optional[SaltRemover] = None
    ) -> "MolStandardizer":
        """
        Remove salts using RDKit's SaltRemover.

        :param salt_remover: Optional SaltRemover instance to use; if None a new one is created.
        :type salt_remover: Optional[SaltRemover]
        :returns: self (chainable).
        :rtype: MolStandardizer
        """
        if self._mol is None:
            return self
        try:
            remover = salt_remover if salt_remover is not None else SaltRemover()
            self._mol = remover.StripMol(self._mol)
        except Exception as exc:
            logger.debug("remove_salts failed: %s", exc, exc_info=True)
            self._last_error = exc
        return self

    def uncharge(self) -> "MolStandardizer":
        """
        Neutralize charges using rdMolStandardize.Uncharger.

        :returns: self (chainable).
        :rtype: MolStandardizer
        """
        if self._mol is None:
            return self
        if rdMolStandardize is None:
            logger.debug("rdMolStandardize not available; uncharge skipped.")
            return self
        try:
            uncharger_cls = getattr(rdMolStandardize, "Uncharger", None)
            if uncharger_cls is None:
                logger.debug("Uncharger not found in rdMolStandardize.")
                return self
            self._mol = uncharger_cls().uncharge(self._mol)
        except Exception as exc:
            logger.debug("uncharge failed: %s", exc, exc_info=True)
            self._last_error = exc
        return self

    def keep_largest_fragment(self) -> "MolStandardizer":
        """
        Keep only the largest fragment by atom count.

        :returns: self (chainable).
        :rtype: MolStandardizer
        """
        if self._mol is None:
            return self
        try:
            frags = Chem.GetMolFrags(self._mol, asMols=True, sanitizeFrags=True)
            if frags:
                self._mol = max(frags, key=lambda m: m.GetNumAtoms())
        except Exception as exc:
            logger.debug("keep_largest_fragment failed: %s", exc, exc_info=True)
            self._last_error = exc
        return self

    def remove_explicit_hs(self) -> "MolStandardizer":
        """
        Remove explicit hydrogens (Chem.RemoveHs).

        :returns: self (chainable).
        :rtype: MolStandardizer
        """
        if self._mol is None:
            return self
        try:
            self._mol = Chem.RemoveHs(self._mol)
        except Exception as exc:
            logger.debug("remove_explicit_hs failed: %s", exc, exc_info=True)
            self._last_error = exc
        return self

    def add_hs_and_clear_radicals(self, removeH: bool = True) -> "MolStandardizer":
        """
        Replace radical electrons with explicit hydrogens and optionally remove them.

        :param removeH: if True remove explicit hydrogens after addition.
        :type removeH: bool
        :returns: self (chainable).
        :rtype: MolStandardizer
        """
        if self._mol is None:
            return self
        try:
            processed = _replace_radicals_with_hs_in_mol(self._mol, removeH=removeH)
            if processed is not None:
                self._mol = processed
        except Exception as exc:
            logger.debug("add_hs_and_clear_radicals failed: %s", exc, exc_info=True)
            self._last_error = exc
        return self

    def remove_isotopes(self) -> "MolStandardizer":
        """
        Clear isotope labels on all atoms.

        :returns: self (chainable).
        :rtype: MolStandardizer
        """
        if self._mol is None:
            return self
        try:
            for atom in self._mol.GetAtoms():
                try:
                    atom.SetIsotope(0)
                except Exception:
                    pass
        except Exception as exc:
            logger.debug("remove_isotopes failed: %s", exc, exc_info=True)
            self._last_error = exc
        return self

    def clear_stereochemistry(self) -> "MolStandardizer":
        """
        Remove stereochemical annotations (Chem.RemoveStereochemistry).

        :returns: self (chainable).
        :rtype: MolStandardizer
        """
        if self._mol is None:
            return self
        try:
            Chem.RemoveStereochemistry(self._mol)
        except Exception as exc:
            logger.debug("clear_stereochemistry failed: %s", exc, exc_info=True)
            self._last_error = exc
        return self

    # ----- retrieval / helpers -----
    @property
    def mol(self) -> Optional[Chem.Mol]:
        """
        Return the internal RDKit Mol (or None if absent).

        :returns: the internal RDKit Mol or None.
        :rtype: Optional[Chem.Mol]
        """
        return self._mol

    def to_smiles(self, canonical: bool = True) -> Optional[str]:
        """
        Return a SMILES string for the internal molecule.

        :param canonical: whether to return a canonical SMILES (default True).
        :type canonical: bool
        :returns: SMILES string or None.
        :rtype: Optional[str]
        """
        if self._mol is None:
            return None
        try:
            return Chem.MolToSmiles(self._mol, canonical=canonical)
        except Exception:
            try:
                Chem.SanitizeMol(self._mol)
                return Chem.MolToSmiles(self._mol, canonical=canonical)
            except Exception:
                logger.debug("to_smiles serialization failed.", exc_info=True)
                return None

    def summarize_last_error(self) -> Optional[str]:
        """
        Return a short string describing the last internal exception, if any.

        :returns: descriptive string for last error or None.
        :rtype: Optional[str]
        """
        if self._last_error is None:
            return None
        return f"{type(self._last_error).__name__}: {str(self._last_error)}"

    def __repr__(self) -> str:
        """
        Debug representation showing the number of atoms in the internal Mol.

        :returns: repr string.
        :rtype: str
        """
        n = -1
        try:
            n = self._mol.GetNumAtoms() if self._mol is not None else -1
        except Exception:
            n = -1
        return f"{self.__class__.__name__}(n_atoms={n})"

    @classmethod
    def help(cls) -> str:
        """
        Short machine-readable help describing capabilities.

        :returns: help string.
        :rtype: str
        """
        return (
            "MolStandardizer.help() -> str\n\n"
            "Fluent methods:\n"
            "  .normalize(), .canonicalize_tautomer(), .remove_salts(), .uncharge(),\n"
            "  .keep_largest_fragment(), .remove_explicit_hs(), .add_hs_and_clear_radicals(),\n"
            "  '  .remove_isotopes(), .clear_stereochemistry()\n\n"
            "Constructors:\n"
            "  MolStandardizer.from_smiles(smiles), MolStandardizer(mol)\n\n"
            "Convenience:\n"
            "  MolStandardizer.standardize_smiles(smiles) -> Optional[canonical_smiles]\n"
        )
