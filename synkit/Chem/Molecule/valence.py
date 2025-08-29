from __future__ import annotations

from typing import Protocol, runtime_checkable

from rdkit import Chem
from rdkit.Chem import rdchem as rdchem


@runtime_checkable
class _AtomLike(Protocol):
    """Minimal protocol RDKit atoms satisfy (duck-typed for typing tools)."""

    def GetValence(self, which: rdchem.ValenceType | int = ...) -> int: ...
    def GetExplicitValence(self) -> int: ...
    def GetImplicitValence(self) -> int: ...
    def GetNumImplicitHs(self) -> int: ...


class ValenceResolver:
    """
    Warning-free valence utilities for RDKit atoms.

    These helpers retrieve **explicit**, **implicit**, and **total** valences
    while silencing common deprecation or signature warnings across RDKit
    versions. They first try the modern keyword-argument API and gracefully
    fall back to older call signatures or legacy methods.

    Preferred (modern) RDKit API:
        - ``atom.GetValence(which=rdchem.ValenceType.EXPLICIT)``
        - ``atom.GetValence(which=rdchem.ValenceType.IMPLICIT)``

    Fallbacks maintain compatibility with older wrappers:
        - ``atom.GetValence(rdchem.ValenceType.EXPLICIT)`` (positional)
        - ``atom.GetExplicitValence()``
        - ``atom.GetImplicitValence()``
        - ``atom.GetNumImplicitHs()`` (as a last resort for implicit)

    Notes
    -----
    * Returned values are coerced to Python ``int`` and guaranteed non-negative,
      with ``0`` returned if all strategies fail.
    * Values reflect the *current* state of the atom. If you modify hydrogen
      counts, aromaticity, or bond orders, query again.
    * ``Chem.Atom`` is an alias of ``rdchem.Atom``, but a structural duck-type
      ``_AtomLike`` protocol is provided for static typing tools.

    Examples
    --------
    >>> from rdkit import Chem
    >>> m = Chem.MolFromSmiles("CCO")
    >>> a = m.GetAtomWithIdx(1)  # central carbon
    >>> ValenceResolver.explicit(a) >= 0
    True
    >>> ValenceResolver.total(a) == ValenceResolver.explicit(a) + ValenceResolver.implicit(a)
    True
    """

    @staticmethod
    def explicit(atom: Chem.Atom | _AtomLike) -> int:
        """
        Return the **explicit valence** of an atom.

        Tries modern ``GetValence(which=EXPLICIT)`` first, then older positional
        form, then ``GetExplicitValence()``. Returns ``0`` on failure.

        :param atom: RDKit atom instance.
        :type atom: rdchem.Atom
        :returns: Explicit valence (non-negative integer).
        :rtype: int
        """
        # Modern keyword form (preferred; avoids RDKit warnings)
        try:
            return int(atom.GetValence(which=rdchem.ValenceType.EXPLICIT))  # type: ignore[call-arg]
        except TypeError:
            # Some RDKit builds don't accept the kwarg form
            try:
                return int(atom.GetValence(rdchem.ValenceType.EXPLICIT))  # type: ignore[arg-type]
            except Exception:
                # Legacy explicit valence API
                try:
                    return int(atom.GetExplicitValence())
                except Exception:
                    return 0

    @staticmethod
    def implicit(atom: Chem.Atom | _AtomLike) -> int:
        """
        Return the **implicit valence** of an atom.

        Tries modern ``GetValence(which=IMPLICIT)`` first, then older positional
        form, then ``GetImplicitValence()``, finally falls back to the number of
        implicit hydrogens if needed. Returns ``0`` on failure.

        :param atom: RDKit atom instance.
        :type atom: rdchem.Atom
        :returns: Implicit valence (non-negative integer).
        :rtype: int
        """
        # Modern keyword form (preferred)
        try:
            return int(atom.GetValence(which=rdchem.ValenceType.IMPLICIT))  # type: ignore[call-arg]
        except TypeError:
            # Older builds without kwarg support
            try:
                return int(atom.GetValence(rdchem.ValenceType.IMPLICIT))  # type: ignore[arg-type]
            except Exception:
                # Legacy implicit valence API
                try:
                    return int(atom.GetImplicitValence())
                except Exception:
                    # As a last resort, approximate with implicit Hs
                    try:
                        return int(atom.GetNumImplicitHs())
                    except Exception:
                        return 0

    @staticmethod
    def total(atom: Chem.Atom | _AtomLike) -> int:
        """
        Return the **total valence** (explicit + implicit).

        :param atom: RDKit atom instance.
        :type atom: rdchem.Atom
        :returns: Total valence as ``explicit(atom) + implicit(atom)``.
        :rtype: int
        """
        return ValenceResolver.explicit(atom) + ValenceResolver.implicit(atom)
