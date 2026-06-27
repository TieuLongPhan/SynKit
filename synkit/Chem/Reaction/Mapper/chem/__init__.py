"""
mapper.chem — chemistry front-end (RDKit SMILES, ITS analysis).

Submodules
----------
aam
    :class:`AAMapper`: high-level reaction-SMILES atom-to-atom mapper.
smiles
    SMILES ↔ LabeledGraph conversion and reaction-SMILES annotation utilities.
its
    ITS graph hashing, electron-balance checks, and symmetry-distinct
    deduplication via synkit.
"""

from .aam import AAMapper
from .its import (
    its_canonical_hash,
    dedup_mapped_rxns,
    mapped_rxn_is_electron_balanced,
    is_electron_balanced,
    electron_balance_status,
)
from .smiles import (
    HAS_RDKIT,
    smiles2lgp,
    get_numbered_rxn_smiles,
    canonicalize_rxn_smiles,
    expand_reaction_center_hydrogens,
)

__all__ = [
    # aam
    "AAMapper",
    # its
    "its_canonical_hash",
    "dedup_mapped_rxns",
    "mapped_rxn_is_electron_balanced",
    "is_electron_balanced",
    "electron_balance_status",
    # smiles
    "HAS_RDKIT",
    "smiles2lgp",
    "get_numbered_rxn_smiles",
    "canonicalize_rxn_smiles",
    "expand_reaction_center_hydrogens",
]
