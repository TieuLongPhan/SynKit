from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # adjust level if needed
sys.path.insert(0, str(ROOT))
from pathlib import Path
from typing import Any, Dict, List

from synkit.CRN.Query.kegg_extract import KEGGExtractor
from synkit.CRN.Query.kegg_impute import KEGGImputer
from synkit.CRN.Construct.abstract import AbstractReactionExtractor

CASE_DIR = Path("./Data/Study/CRN/case_glycolysis")
RAW_JSON = CASE_DIR / "hsa00010_raw.json"
IMPUTED_JSON = CASE_DIR / "hsa00010_imputed.json"
ABSTRACT_JSON = CASE_DIR / "hsa00010_abstract.json"


def get_hsa00010_fixes() -> List[Dict[str, Any]]:
    """
    Return curated manual fixes for hsa00010 extraction/imputation.

    These include:
    - reaction text corrections for missing/ambiguous KEGG entries
    - compound SMILES for species absent from the default KEGG compound pool

    :returns:
        List of fix dictionaries accepted by ``KEGGImputer.impute_pathway``.
    :rtype: List[Dict[str, Any]]
    """
    return [
        {
            "id": "R02189",
            "reaction": "C00404 + C00267 => C99999 + C00668",
        },
        {
            "id": "C99999",
            "name": "Polyphosphate fragment",
            "smiles": "O=P(O)(O)OP(=O)(O)O",
        },
        {
            "id": "C00138",
            "name": "Reduced ferredoxin",
            "smiles": "S1[Fe]S[Fe+]1",
        },
        {
            "id": "C00139",
            "name": "Oxidized ferredoxin",
            "smiles": "S1[Fe+]S[Fe+]1",
        },
        {
            "id": "C02745",
            "smiles": (
                "CC2(C=C1(NC3(C(=O)NC(=O)NC("
                "N(CC(O)C(O)C(O)COP([O-])(=O)[O-])C1=CC(C)=2)=3)))"
            ),
        },
        {
            "id": "C02869",
            "smiles": (
                "CC2(C=C1(N=C3(C(=O)NC(=O)N=C("
                "N(CC(O)C(O)C(O)COP([O-])(=O)[O-])C1=CC(C)=2)3)))"
            ),
        },
        {
            "id": "C15972",
            "name": "Enzyme N6-(lipoyl)lysine",
            "smiles": "NC(=O)CCCC[C@@H]1CCSS1",
        },
        {
            "id": "C15973",
            "name": "Enzyme N6-(dihydrolipoyl)lysine",
            "smiles": "NC(=O)CCCC[C@@H](S)CCS",
        },
        {
            "id": "C16255",
            "name": "[Dihydrolipoyllysine-residue acetyltransferase] S-acetyldihydrolipoyllysine",
            "smiles": "NC(=O)CCCC[C@@H](S)CCSC(C)=O",
        },
    ]


def ensure_output_dir() -> None:
    """
    Ensure the case-study output directory exists.
    """
    CASE_DIR.mkdir(parents=True, exist_ok=True)


def extract_pathway() -> Dict[str, Any]:
    """
    Extract KEGG pathway data for hsa00010.

    :returns:
        Raw pathway JSON-like dictionary.
    :rtype: Dict[str, Any]
    """
    extractor = KEGGExtractor()
    pathway_data = extractor.build_pathway_json(
        "hsa00010",
        with_compounds=True,
        with_atom_maps=True,
        save_as=str(RAW_JSON),
    )
    return pathway_data


def impute_pathway(pathway_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Impute missing reactions and compound SMILES for hsa00010.

    :param pathway_data:
        Raw extracted pathway data.
    :type pathway_data: Dict[str, Any]

    :returns:
        Imputed pathway dictionary.
    :rtype: Dict[str, Any]
    """
    imputer = KEGGImputer()
    fixes = get_hsa00010_fixes()

    imputed = imputer.impute_pathway(
        pathway_data,
        fixes=fixes,
        save_as=str(IMPUTED_JSON),
    )
    return imputed


def build_abstract_reactions(imputed_pathway: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the abstract reaction representation from the imputed pathway.

    :param imputed_pathway:
        Imputed pathway dictionary.
    :type imputed_pathway: Dict[str, Any]

    :returns:
        Abstract reaction dataset.
    :rtype: Dict[str, Any]
    """
    abstract_pathway = AbstractReactionExtractor().build(
        data=imputed_pathway,
        drop_missing_smiles_reactions=True,
        deduplicate=True,
        order="appearance",
        reactant_join="+",
        product_join="+",
        reaction_id_keys=["id"],
        reaction_smiles_keys=["smiles"],
        template_keys=["rule"],
        save_as=str(ABSTRACT_JSON),
    )
    return abstract_pathway


def main() -> Dict[str, Any]:
    """
    Run the full hsa00010 KEGG -> imputed -> abstract workflow.

    :returns:
        Dictionary containing raw, imputed, and abstract outputs.
    :rtype: Dict[str, Any]
    """
    ensure_output_dir()

    pathway_data = extract_pathway()
    imputed_pathway = impute_pathway(pathway_data)
    abstract_pathway = build_abstract_reactions(imputed_pathway)

    return {
        "raw_pathway": pathway_data,
        "imputed_pathway": imputed_pathway,
        "abstract_pathway": abstract_pathway,
        "raw_json": str(RAW_JSON),
        "imputed_json": str(IMPUTED_JSON),
        "abstract_json": str(ABSTRACT_JSON),
    }


if __name__ == "__main__":
    results = main()

    print("Finished hsa00010 workflow")
    print(f"Raw pathway saved to:      {results['raw_json']}")
    print(f"Imputed pathway saved to:  {results['imputed_json']}")
    print(f"Abstract pathway saved to: {results['abstract_json']}")
