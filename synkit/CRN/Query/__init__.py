"""KEGG retrieval, parsing and curation utilities for CRN construction.

This subpackage is the entry point for reaction networks derived from a curated
biological pathway rather than from rule-based expansion.

.. rubric:: Example

.. code-block:: python

    from synkit.CRN.Query import KEGGExtractor

    data = KEGGExtractor().build_module_json("M00001")
    print(sorted(data)[:5])
"""

from __future__ import annotations

from .kegg_api import KEGGClient
from .kegg_extract import KEGGExtractor
from .kegg_impute import KEGGImputer
from .kegg_parse import (
    KEGGEquation,
    equation_to_text,
    expand_stoichiometry,
    get_compound_ids_from_equations,
    get_compound_ids_from_text,
    molblock_to_smiles,
    normalize_module_id,
    orient_equation_to_module,
    parse_equation,
    parse_kegg_field_blocks,
    parse_module_reaction_directions,
    parse_side,
    reaction_smiles_from_equation,
)
from .to_syncrn import (
    CURRENCY_COMPOUNDS,
    syncrn_from_kegg_equations,
    syncrn_from_kegg_module,
)

__all__ = [
    "KEGGClient",
    "KEGGExtractor",
    "KEGGImputer",
    "KEGGEquation",
    "equation_to_text",
    "expand_stoichiometry",
    "get_compound_ids_from_equations",
    "get_compound_ids_from_text",
    "molblock_to_smiles",
    "normalize_module_id",
    "orient_equation_to_module",
    "parse_equation",
    "parse_kegg_field_blocks",
    "parse_module_reaction_directions",
    "parse_side",
    "reaction_smiles_from_equation",
    "CURRENCY_COMPOUNDS",
    "syncrn_from_kegg_equations",
    "syncrn_from_kegg_module",
]
