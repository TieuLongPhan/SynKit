import re
from rdkit import Chem
from rdkit.Chem import SanitizeFlags
from typing import Tuple, Optional


class RadWC:
    """
    Static utility for appending wildcard dummy atoms ([*]) with atom-map
    indices to all radical centers **in the product block** of a reaction SMILES.

    - Reactant and agent blocks are not modified.
    - Only atoms in the product with unpaired electrons are considered.
    - Each product radical gets a new [*:N] with unique map number (auto or user-supplied).

    Example
    -------
    >>> rxn = '[CH2:1][OH:2]>>[CH2:1][O:2]'
    >>> RadWC.transform(rxn)
    '[CH2:1][OH:2]>>[CH2:1][O:2]'
    >>> rxn2 = '[CH2:1][OH:2]>>[CH:1].[OH:2]'
    >>> RadWC.transform(rxn2)
    '[CH2:1][OH:2]>>[CH:1]([*:3]).[OH:2]'
    """

    @staticmethod
    def transform(rxn_smiles: str, start_map: Optional[int] = None) -> str:
        """
        Add [*] wildcards (with atom-map index) to every radical in the
        product block of the input reaction SMILES.

        :param rxn_smiles: Reaction SMILES, 2 or 3 blocks (R>>P or R>A>P).
        :type  rxn_smiles: str
        :param start_map: Optional; first atom-map index for wildcards.
        :type  start_map: int or None
        :returns: Modified reaction SMILES with product wildcards.
        :rtype: str
        :raises ValueError: On parse error or invalid input.

        Example
        -------
        >>> RadWC.transform('[CH2:1][OH:2]>>[CH:1].[OH:2]')
        '[CH2:1][OH:2]>>[CH:1]([*:3]).[OH:2]'
        """
        react_blk, agents_blk, prod_blk = RadWC._split_reaction(rxn_smiles)
        # Determine atom-map to use for wildcards
        existing = [int(n) for n in re.findall(r":(\d+)", rxn_smiles)]
        next_map = (
            start_map if start_map is not None else (max(existing, default=0) + 1)
        )

        prod_frags = prod_blk.split(".") if prod_blk else []
        new_prod_frags = []

        keep_ops = SanitizeFlags.SANITIZE_ALL & ~SanitizeFlags.SANITIZE_ADJUSTHS

        for smi in prod_frags:
            if not smi:
                continue
            mol = Chem.MolFromSmiles(smi, sanitize=False)
            if mol is None:
                raise ValueError(f"Cannot parse product SMILES fragment: {smi}")
            Chem.SanitizeMol(mol, sanitizeOps=keep_ops)
            rw = Chem.RWMol(mol)
            atoms = list(rw.GetAtoms())
            changed = False
            for atom in atoms:
                rad = atom.GetNumRadicalElectrons()
                if rad > 0:
                    for _ in range(rad):
                        dummy = Chem.Atom(0)
                        dummy.SetAtomMapNum(next_map)
                        dummy.SetNoImplicit(True)
                        rw.AddAtom(dummy)
                        rw.AddBond(
                            atom.GetIdx(), rw.GetNumAtoms() - 1, Chem.BondType.SINGLE
                        )
                        next_map += 1
                        changed = True
            if changed:
                Chem.SanitizeMol(rw.GetMol(), sanitizeOps=keep_ops)
            new_prod_frags.append(
                Chem.MolToSmiles(rw.GetMol(), isomericSmiles=True, allHsExplicit=True)
            )

        prod_str = ".".join(new_prod_frags)
        if agents_blk is None:
            return f"{react_blk}>>{prod_str}"
        return f"{react_blk}>{agents_blk}>{prod_str}"

    @staticmethod
    def _split_reaction(rxn: str) -> Tuple[str, Optional[str], str]:
        """
        Split a reaction SMILES into (reactant, agent or None, product).

        :param rxn: Reaction SMILES string.
        :type  rxn: str
        :returns: (reactant, agent, product) tuple (agent may be None).
        :rtype: Tuple[str, Optional[str], str]
        :raises ValueError: If the SMILES does not contain 2 or 3 '>'s.
        """
        parts = rxn.split(">")
        if len(parts) == 2:
            return parts[0], None, parts[1]
        if len(parts) == 3:
            return parts[0], parts[1], parts[2]
        raise ValueError("Reaction SMILES must contain 2 or 3 '>' symbols")

    @staticmethod
    def describe():
        """
        Print a description and minimal example.
        """
        print(RadWC.__doc__)

    def __repr__(self):
        return "<RadWC: static radical-wildcard utility for product block only>"
