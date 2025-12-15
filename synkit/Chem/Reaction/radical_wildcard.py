from rdkit import Chem
from rdkit.Chem import SanitizeFlags
import re
from typing import Tuple, List, Optional, Dict


def clean_wc(
    rsmi: str, invert: bool = False, max_frag: bool = False, wild_card: bool = True
) -> str:
    """
    Clean wildcard-containing fragments from one side of a reaction SMILES,
    optionally selecting the largest remaining fragment.

    :param rsmi: Reaction SMILES string in the form 'R>>P'.
    :type rsmi: str
    :param invert: If True, process the reactant side; otherwise the product side.
    :type invert: bool
    :param max_frag: If True, force fragment selection (implies wild_card=True).
    :type max_frag: bool
    :param wild_card: If True, remove fragments containing '*' before selection.
    :type wild_card: bool
    :returns: The processed reaction SMILES.
    :rtype: str
    :raises ValueError: If input does not split into reactant and product.

    Example
    -------
    >>> clean_wc('A.B>>C.*', invert=False, wild_card=True)
    'A.B>>C'
    >>> clean_wc('A.B>>C.D', invert=False, max_frag=True)
    'A.B>>C'
    """
    # Ensure max_frag implies wild_card
    if max_frag:
        wild_card = True

    # Split into reactant and product
    parts = rsmi.split(">>")
    if len(parts) != 2:
        raise ValueError("Reaction SMILES must contain exactly one '>>'.")
    react, prod = parts

    # Select side to process
    side = react if invert else prod

    processed = side
    if wild_card:
        frags = side.split(".")
        # Filter out fragments containing wildcards
        filtered = [frag for frag in frags if "*" not in frag]
        if len(filtered) > 1:
            # select the longest fragment
            processed = max(filtered, key=len)
        elif len(filtered) == 1:
            processed = filtered[0]
        # if no filtered fragments or single fragment, keep original side

    # Reconstruct and return
    if invert:
        return f"{processed}>>{prod}"
    return f"{react}>>{processed}"


class RadicalWildcardAdder:
    """A utility for adding wildcard dummy atoms ([*]) to radical centers in
    reaction SMILES, with unique incremental atom-map indices and correct
    propagation into products.

    Each reactive radical atom in the reactant block is identified by its unpaired electron count,
    assigned one or more wildcard map indices, and recorded. The same wildcard(s) are then appended
    to the corresponding atom(s) in the product block, ensuring consistent mapping.

    :param start_map: If provided, this integer will be the first atom-map index
                      used for wildcard dummy atoms; subsequent radicals get incremented indices.
                      If None, the next unused index is auto-determined from the input SMILES.
    :type start_map: Optional[int]

    Example
    -------
    >>> adder = RadicalWildcardAdder(start_map=8)
    >>> rxn = "[C:2][OH:4].[O:6][H:7]>>[C:2][O:6].[OH:4][H:7]"
    >>> print(adder.transform(rxn))
    [C:2]([OH:4])[*:8].[O:6]([H:7])[*:9]>>[C:2]([O:6][*:9])[*:8].[OH:4][H:7]
    """

    def __init__(self, start_map: Optional[int] = None) -> None:
        """Initialize the adder with an optional starting map index.

        :param start_map: Starting atom-map index for wildcards or None
            to auto-pick.
        :type start_map: Optional[int]
        """
        self.start_map = start_map

    def __repr__(self) -> str:
        """Official representation."""
        return f"<RadicalWildcardAdder(start_map={self.start_map})>"

    def __str__(self) -> str:
        """User-friendly description."""
        m = self.start_map if self.start_map is not None else "auto"
        return f"RadicalWildcardAdder(start_map={m})"

    def transform(self, rxn_smiles: str) -> str:
        """Append wildcard dummy atoms to each radical center in the reactant
        block and propagate the same wildcards to the matching atoms in the
        product block.

        :param rxn_smiles: Reaction SMILES string, two-component or
            three-component.
        :type rxn_smiles: str
        :returns: Modified reaction SMILES with consistent wildcard
            attachments.
        :rtype: str
        :raises ValueError: If the SMILES is not valid or fragments fail
            to parse.
        """
        # Split into reactants > agents? > products
        react_blk, agents_blk, prod_blk = self._split_reaction(rxn_smiles)

        # Determine first wildcard map index
        existing = [int(n) for n in re.findall(r":(\d+)", rxn_smiles)]
        next_map = (
            self.start_map
            if self.start_map is not None
            else max(existing, default=0) + 1
        )

        # Record mapping: original atom-map -> list of wildcard_maps
        wildcard_map_for: Dict[int, List[int]] = {}

        # Build sanitizeOps mask (skip H-adjustment)
        keep_ops = SanitizeFlags.SANITIZE_ADJUSTHS

        # Process one block (helper)
        def _process(frags: List[str], propagate: bool) -> List[str]:
            nonlocal next_map
            out = []
            for smi in frags:
                if not smi:
                    continue
                # Load unsanitized then re-sanitize to preserve explicit H
                # mol = Chem.MolFromSmiles(smi, sanitize=False)
                # if mol is None:
                #     raise ValueError(f"Cannot parse SMILES fragment: {smi}")
                # Chem.SanitizeMol(mol, sanitizeOps=keep_ops)
                mol = Chem.MolFromSmiles(smi, sanitize=False)
                Chem.SanitizeMol(mol)
                rw = Chem.RWMol(mol)

                atoms = list(rw.GetAtoms())
                changed = False

                for atom in atoms:
                    rad = atom.GetNumRadicalElectrons()
                    orig_map = atom.GetAtomMapNum()
                    if rad > 0:
                        # Initialize list for this orig_map
                        if propagate and orig_map not in wildcard_map_for:
                            wildcard_map_for[orig_map] = []
                        # For each unpaired electron, attach a wildcard
                        for _ in range(rad):
                            if propagate:
                                wm = next_map
                                wildcard_map_for[orig_map].append(wm)
                                next_map += 1
                            else:
                                # in products, use already-recorded wm sequentially
                                wm_list = wildcard_map_for.get(orig_map, [])
                                if not wm_list:
                                    continue
                                wm = wm_list.pop(0)
                            # add dummy wildcard
                            dummy = Chem.Atom(0)
                            dummy.SetAtomMapNum(wm)
                            dummy.SetNoImplicit(True)
                            rw.AddAtom(dummy)
                            rw.AddBond(
                                atom.GetIdx(),
                                rw.GetNumAtoms() - 1,
                                Chem.BondType.SINGLE,
                            )
                            changed = True

                if changed:
                    Chem.SanitizeMol(rw.GetMol(), sanitizeOps=keep_ops)

                out.append(
                    Chem.MolToSmiles(
                        rw.GetMol(), isomericSmiles=True, allHsExplicit=True
                    )
                )
            return out

        react_frags = react_blk.split(".") if react_blk else []
        new_reacts = _process(react_frags, propagate=True)

        prod_frags = prod_blk.split(".") if prod_blk else []
        new_prods = _process(prod_frags, propagate=False)

        react_str = ".".join(new_reacts)
        prod_str = ".".join(new_prods)
        if agents_blk is None:
            return f"{react_str}>>{prod_str}"
        return f"{react_str}>{agents_blk}>{prod_str}"

    @staticmethod
    def _split_reaction(rxn: str) -> Tuple[str, Optional[str], str]:
        """Split a reaction SMILES into reactants, agents (optional), and
        products.

        :param rxn: The reaction SMILES string.
        :type rxn: str
        :returns: Tuple of (reactants_block, agents_block or None,
            products_block).
        :rtype: Tuple[str, Optional[str], str]
        :raises ValueError: If the SMILES does not contain 2 or 3 '>'
            symbols.
        """
        parts = rxn.split(">")
        if len(parts) == 2:
            return parts[0], None, parts[1]
        if len(parts) == 3:
            return parts[0], parts[1], parts[2]
        raise ValueError("Reaction SMILES must contain 2 or 3 '>' symbols")
