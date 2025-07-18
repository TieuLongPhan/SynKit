from rdkit import Chem
from typing import Optional

from synkit.IO.graph_to_mol import GraphToMol
from synkit.Graph.ITS.its_construction import ITSConstruction
from synkit.Graph.ITS.its_decompose import get_rc
from synkit.IO.chem_converter import smiles_to_graph


def _get_partial_aam(smart: str) -> str:
    """Generate a partial atom‐atom mapping (AAM) SMILES string from a reactant
    SMARTS.

    This function:
      1. Parses the forward (“reactant”) and backward (“product”) halves of `smart`.
      2. Builds RDKit graphs (with atom_map indices) for each side.
      3. Constructs the integrated transition state (ITS) graph.
      4. Identifies the reactant core (rc) subgraph.
      5. Zeroes out `atom_map` on any atom not in the reactant core.
      6. Converts the modified graphs to RDKit Mol objects (respecting H‐counts).
      7. Emits SMILES for the “retained” and “partial” graphs, joined with '>>'.

    :param smart: A reaction SMARTS of the form "R>>P".
    :type smart: str
    :returns: An unbalanced, partial‐AAM reaction SMILES "retained>>partial".
    :rtype: str
    :raises RuntimeError:
        If graph decomposition, molecule conversion, or SMILES generation fails.
    """
    # split into reactant/product SMARTS
    r_smi, p_smi = smart.split(">>")
    # build graphs with atom_map indices from the SMILES strings
    r_graph = smiles_to_graph(r_smi, use_index_as_atom_map=True)
    p_graph = smiles_to_graph(p_smi, use_index_as_atom_map=True)

    # construct and decompose ITS
    its = ITSConstruction.ITSGraph(r_graph, p_graph)
    rc = get_rc(its)
    rc_nodes = set(rc.nodes())

    # zero out maps for atoms not in the reactant core
    for node in r_graph.nodes():
        if node not in rc_nodes:
            r_graph.nodes[node]["atom_map"] = 0
    for node in p_graph.nodes():
        if node not in rc_nodes:
            p_graph.nodes[node]["atom_map"] = 0

    converter = GraphToMol()
    try:
        retained_mol = converter.graph_to_mol(r_graph, use_h_count=True)
        partial_mol = converter.graph_to_mol(p_graph, use_h_count=True)
    except Exception as e:
        raise RuntimeError(f"Error converting graphs to molecules: {e}") from e

    try:
        retained_smiles = Chem.MolToSmiles(retained_mol)
        partial_smiles = Chem.MolToSmiles(partial_mol)
    except Exception as e:
        raise RuntimeError(f"Error generating SMILES: {e}") from e

    return f"{retained_smiles}>>{partial_smiles}"


def _remove_small_smiles(smiles: str) -> str:
    """Return the canonical SMILES of the largest fragment from an input
    SMILES.

    This function:
      1. Parses `smiles` to an RDKit Mol without sanitization.
      2. Sanitizes the Mol.
      3. Splits into fragments, picks the one with the most heavy atoms.
      4. Sanitizes that fragment and returns its canonical SMILES.

    :param smiles: The input SMILES string.
    :type smiles: str
    :returns: Canonical SMILES of the largest fragment.
    :rtype: str
    :raises ValueError:
        - If `smiles` is invalid.
        - If sanitization of the whole or fragment fails.
        - If no fragments are found.
    """
    mol: Optional[Chem.Mol] = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: '{smiles}'")

    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        raise ValueError(f"Sanitization failed for '{smiles}': {e}")

    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if not fragments:
        raise ValueError(f"No fragments found for SMILES '{smiles}'")

    largest = max(fragments, key=lambda m: m.GetNumHeavyAtoms())

    try:
        Chem.SanitizeMol(largest)
    except Exception as e:
        raise ValueError(f"Sanitization failed for largest fragment: {e}")

    return Chem.MolToSmiles(largest)


def _create_unbalanced_aam(rsmi: str, side: str = "right") -> str:
    """Produce an unbalanced AAM reaction SMILES by keeping only the largest
    fragment on the specified side(s) of the reaction.

    :param rsmi: A reaction SMILES "reactant>>product".
    :type rsmi: str
    :param side:
        Which side(s) to process:
        - "left" : clean only the reactant side,
        - "right": clean only the product side,
        - "both" : clean both sides.
    :type side: str
    :returns: A new reaction SMILES "reactant>>product" with small fragments removed.
    :rtype: str
    :raises ValueError:
      - If `rsmi` doesn’t contain exactly one ">>".
      - If `side` is not one of "left", "right", "both".
      - If fragment processing fails on the chosen side(s).
    """
    parts = [p.strip() for p in rsmi.split(">>")]
    if len(parts) != 2:
        raise ValueError(f"Expected single '>>' in reaction SMILES: '{rsmi}'")

    r_smi, p_smi = parts
    side_l = side.lower()
    if side_l not in ("left", "right", "both"):
        raise ValueError(f"Invalid side '{side}'; must be 'left', 'right', or 'both'")

    if side_l in ("left", "both"):
        try:
            r_smi = _remove_small_smiles(r_smi)
        except Exception as e:
            raise ValueError(f"Error processing left side: {e}")

    if side_l in ("right", "both"):
        try:
            p_smi = _remove_small_smiles(p_smi)
        except Exception as e:
            raise ValueError(f"Error processing right side: {e}")

    return f"{r_smi}>>{p_smi}"
