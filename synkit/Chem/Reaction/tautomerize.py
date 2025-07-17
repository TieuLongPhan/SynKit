from typing import List, Dict, Optional
from rdkit import Chem
from fgutils import FGQuery
from joblib import Parallel, delayed


class Tautomerize:
    """Standardize molecules by converting enol and hemiketal tautomers into
    their more stable carbonyl forms, and apply these corrections to individual
    SMILES or collections of reaction data."""

    @staticmethod
    def standardize_enol(smiles: str, atom_indices: Optional[List[int]] = None) -> str:
        """Convert an enol tautomer into its corresponding carbonyl form.

        :param smiles: SMILES string of the enol-containing molecule.
        :type smiles: str
        :param atom_indices: List of three atom indices [C1, C2, O]
            defining the enol. If None, defaults to [0, 1, 2].
        :type atom_indices: List[int] or None
        :returns: SMILES of the molecule after enol→carbonyl conversion,
            or an error message if the input is invalid or indices fail.
        :rtype: str
        """
        if atom_indices is None:
            atom_indices = [0, 1, 2]

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "Invalid SMILES format."
        emol = Chem.EditableMol(mol)

        try:
            c_idxs = [
                i for i in atom_indices if mol.GetAtomWithIdx(i).GetSymbol() == "C"
            ]
            c1_idx, c2_idx = c_idxs[:2]
            o_idx = next(
                i for i in atom_indices if mol.GetAtomWithIdx(i).GetSymbol() == "O"
            )
        except Exception as e:
            return f"Error processing indices: {e}"

        try:
            emol.RemoveBond(c1_idx, c2_idx)
            emol.RemoveBond(c2_idx, o_idx)
            emol.AddBond(c1_idx, c2_idx, Chem.rdchem.BondType.SINGLE)
            emol.AddBond(c2_idx, o_idx, Chem.rdchem.BondType.DOUBLE)
            new_mol = emol.GetMol()
            Chem.SanitizeMol(new_mol)
            return Chem.MolToSmiles(new_mol)
        except Exception as e:
            return f"Error in modifying molecule: {e}"

    @staticmethod
    def standardize_hemiketal(smiles: str, atom_indices: List[int]) -> str:
        """Convert a hemiketal tautomer into its corresponding carbonyl form.

        :param smiles: SMILES string of the hemiketal-containing
            molecule.
        :type smiles: str
        :param atom_indices: List of atom indices [C, O1, O2] defining
            the hemiketal.
        :type atom_indices: List[int]
        :returns: SMILES of the molecule after hemiketal→carbonyl
            conversion, or an error message if the input is invalid.
        :rtype: str
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "Invalid SMILES format."
        emol = Chem.EditableMol(mol)

        try:
            c_idx = next(
                i for i in atom_indices if mol.GetAtomWithIdx(i).GetSymbol() == "C"
            )
            o_idxs = [
                i for i in atom_indices if mol.GetAtomWithIdx(i).GetSymbol() == "O"
            ]
            o1_idx = o_idxs[0]
        except Exception as e:
            return f"Error processing indices: {e}"

        try:
            emol.RemoveBond(c_idx, o1_idx)
            if len(o_idxs) > 1:
                emol.RemoveBond(c_idx, o_idxs[1])
            emol.AddBond(c_idx, o1_idx, Chem.rdchem.BondType.DOUBLE)
            new_mol = emol.GetMol()
            Chem.SanitizeMol(new_mol)
            return Chem.MolToSmiles(new_mol)
        except Exception as e:
            return f"Error in modifying molecule: {e}"

    @staticmethod
    def fix_smiles(smiles: str) -> str:
        """Iteratively apply enol and hemiketal standardizations until no
        further changes, then return the canonical SMILES.

        :param smiles: SMILES string to standardize.
        :type smiles: str
        :returns: Canonical SMILES of the standardized molecule.
        :rtype: str
        """
        query = FGQuery()
        fg = query.get(smiles)
        for item in fg:
            label, indices = item
            if label == "hemiketal":
                smiles = Tautomerize.standardize_hemiketal(smiles, indices)
                fg = query.get(smiles)
            elif label == "enol":
                smiles = Tautomerize.standardize_enol(smiles, indices)
                fg = query.get(smiles)
        return Chem.CanonSmiles(smiles)

    @staticmethod
    def fix_dict(data: Dict[str, str], reaction_column: str) -> Dict[str, str]:
        """Standardize the reactant and product SMILES in a reaction
        dictionary.

        :param data: Dictionary containing a reaction SMILES under `reaction_column`.
        :type data: Dict[str, str]
        :param reaction_column: Key in `data` where the reaction SMILES is stored.
        :type reaction_column: str
        :returns: The same dictionary with standardized reaction SMILES.
        :rtype: Dict[str, str]
        """
        try:
            react, prod = data[reaction_column].split(">>")
            data[reaction_column] = (
                f"{Tautomerize.fix_smiles(react)}>>{Tautomerize.fix_smiles(prod)}"
            )
        except ValueError:
            data[reaction_column] = Tautomerize.fix_smiles(data[reaction_column])
        return data

    @staticmethod
    def fix_dicts(
        data: List[Dict[str, str]],
        reaction_column: str,
        n_jobs: int = 4,
        verbose: int = 0,
    ) -> List[Dict[str, str]]:
        """Standardize multiple reaction dictionaries in parallel.

        :param data: List of dictionaries containing reaction SMILES under `reaction_column`.
        :type data: List[Dict[str, str]]
        :param reaction_column: Key in each dictionary for the reaction SMILES.
        :type reaction_column: str
        :param n_jobs: Number of parallel jobs to run. Defaults to 4.
        :type n_jobs: int
        :param verbose: Verbosity level for the joblib Parallel call. Defaults to 0.
        :type verbose: int
        :returns: List of dictionaries with standardized SMILES.
        :rtype: List[Dict[str, str]]
        """
        results = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(Tautomerize.fix_dict)(d, reaction_column) for d in data
        )
        return results
