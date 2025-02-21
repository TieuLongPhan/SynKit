from typing import List, Dict, Optional
from rdkit import Chem
from fgutils import FGQuery
from joblib import Parallel, delayed


class Tautomerize:
    """
    A class to standardize molecules by converting specific functional groups to their
    more common forms using RDKit for molecule manipulation.
    """

    @staticmethod
    def standardize_enol(smiles: str, atom_indices: Optional[List[int]] = None) -> str:
        """
        Converts an enol form to a carbonyl form based on specified atom indices.

        Parameters:
        - smiles (str): The SMILES string.
        - atom_indices (List[int], optional): List containing indices of two carbons and
        one oxygen involved in the enol formation. Defaults to [0, 1, 2].

        Returns:
        - str: The SMILES string of the molecule after conversion.
                Returns an error message if indices are invalid.
        """
        if atom_indices is None:
            atom_indices = [0, 1, 2]

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "Invalid SMILES format."
        emol = Chem.EditableMol(mol)

        try:
            c1_idx, c2_idx = (
                i for i in atom_indices if mol.GetAtomWithIdx(i).GetSymbol() == "C"
            )
            o_idx = next(
                i for i in atom_indices if mol.GetAtomWithIdx(i).GetSymbol() == "O"
            )
        except Exception as e:
            return f"Error processing indices: {str(e)}"

        try:
            emol.RemoveBond(c1_idx, c2_idx)
            emol.RemoveBond(c2_idx, o_idx)
            emol.AddBond(c1_idx, c2_idx, order=Chem.rdchem.BondType.SINGLE)
            emol.AddBond(c2_idx, o_idx, order=Chem.rdchem.BondType.DOUBLE)
            new_mol = emol.GetMol()
            Chem.SanitizeMol(new_mol)
            return Chem.MolToSmiles(new_mol)
        except Exception as e:
            return f"Error in modifying molecule: {str(e)}"

    @staticmethod
    def standardize_hemiketal(smiles: str, atom_indices: List[int]) -> str:
        """
        Converts a hemiketal form to a carbonyl form based on specified atom indices.

        Parameters:
        - smiles (str): SMILES representation of the original molecule.
        - atom_indices (List[int]): Indices of the carbon and two oxygen atoms
        involved in the transformation.

        Returns:
        - str: SMILES string of the modified molecule if successful,
                otherwise returns an error message.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "Invalid SMILES format."
        emol = Chem.EditableMol(mol)

        try:
            c_idx = next(
                i for i in atom_indices if mol.GetAtomWithIdx(i).GetSymbol() == "C"
            )
            o1_idx, o2_idx = (
                i for i in atom_indices if mol.GetAtomWithIdx(i).GetSymbol() == "O"
            )
            emol.RemoveBond(c_idx, o1_idx)
            emol.RemoveBond(c_idx, o2_idx)
            emol.AddBond(c_idx, o1_idx, order=Chem.rdchem.BondType.DOUBLE)
            new_mol = emol.GetMol()
            Chem.SanitizeMol(new_mol)
            return Chem.MolToSmiles(new_mol)
        except Exception as e:
            return f"Error in modifying molecule: {str(e)}"

    @staticmethod
    def fix_smiles(smiles: str) -> str:
        """
        Performs the standardization process by identifying and converting all relevant
        functional groups to their target forms based on predefined rules and updates the
        SMILES string accordingly.

        Parameters:
        - smiles (str): SMILES string of the original molecule.

        Returns:
        - str: Canonical SMILES string of the standardized molecule.
        """
        query = FGQuery()
        fg = query.get(smiles)
        for item in fg:
            if "hemiketal" in item:
                atom_indices = item[1]
                smiles = Tautomerize.standardize_hemiketal(smiles, atom_indices)
                fg = query.get(smiles)
            elif "enol" in item:
                atom_indices = item[1]
                smiles = Tautomerize.standardize_enol(smiles, atom_indices)
                fg = query.get(smiles)
        return Chem.CanonSmiles(smiles)

    @staticmethod
    def fix_dict(data: Dict[str, str], reaction_column: str) -> Dict[str, str]:
        """
        Updates a dictionary containing reaction data by
        standardizing the SMILES strings of reactants and products.

        Parameters:
        - data (Dict[str, str]): Dictionary containing the reaction data.
        - reaction_column (str): The key in the dictionary where the reaction SMILES
        string is stored.

        Returns:
        - Dict[str, str]: The updated dictionary with standardized SMILES strings.
        """
        try:
            reactants, products = data[reaction_column].split(">>")
            reactants = Tautomerize.fix_smiles(reactants)
            products = Tautomerize.fix_smiles(products)
            data[reaction_column] = f"{reactants}>>{products}"
        except ValueError:
            smiles = data[reaction_column]
            smiles = Tautomerize.fix_smiles(smiles)
            data[reaction_column] = smiles
        return data

    @staticmethod
    def fix_dicts(
        data: List[Dict[str, str]],
        reaction_column: str,
        n_jobs: int = 4,
        verbose: int = 0,
    ) -> List[Dict[str, str]]:
        """
        Standardizes multiple dictionaries containing
        reaction data in parallel.

        Parameters:
        - data (List[Dict[str, str]]): List of dictionaries, each containing reaction
        data.
        - reaction_column (str): The key where the reaction SMILES strings are
        stored in each dictionary.
        - n_jobs (int, optional): Number of jobs to run in parallel. Defaults to 4.
        - verbose (int, optional): The verbosity level. Defaults to 0.

        Returns:
        - List[Dict[str, str]]: A list of updated dictionaries
                            with standardized SMILES strings.
        """
        results = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(Tautomerize.fix_dict)(d, reaction_column) for d in data
        )
        return results
