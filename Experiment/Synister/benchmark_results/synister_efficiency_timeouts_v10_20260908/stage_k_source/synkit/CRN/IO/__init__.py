"""Interchange formats for reaction networks.

Currently one format is supported: **SBML**, the interchange format of the
existing CRN tooling (``crnpy``, CRNT4SBML, CoNtRol, COPASI, BioModels). The
adapter is self-contained — no ``libsbml`` installation is needed.

.. rubric:: Example

.. code-block:: python

    from synkit.CRN import SynCRN
    from synkit.CRN.IO import crn_to_sbml, crn_from_sbml

    crn = SynCRN.from_reaction_strings(["A+B>>C", "C>>A+B"])
    assert crn_from_sbml(crn_to_sbml(crn)).n_reactions == 2
"""

from __future__ import annotations

from .sbml import (
    SBML_NS,
    SYNKIT_NS,
    crn_from_sbml,
    crn_to_sbml,
    read_sbml,
    write_sbml,
)

__all__ = [
    "SBML_NS",
    "SYNKIT_NS",
    "crn_from_sbml",
    "crn_to_sbml",
    "read_sbml",
    "write_sbml",
]
