import re
from typing import List, Set
from rdkit import Chem
from rdkit.Chem import rdChemReactions


class SMARTSGeneralizer:
    """
    Generalizes a list of atom-mapped (reaction) SMARTS into one combinatorial SMARTS
    with element-list placeholders at mapped atom positions.
    Optionally validates output using RDKit.

    :param sanity_check: If True, validate the output SMARTS with RDKit.
    :type sanity_check: bool

    Example
    -------
    >>> input_smarts = [
    ...     '[C:1]-[N:2]>>[N:1]-[C:2]',
    ...     '[N:1]-[N:2]>>[N:1]-[N:2]',
    ...     '[O:1]-[N:2]>>[N:1]-[N:2]'
    ... ]
    >>> gen = SMARTSGeneralizer()
    >>> print(gen.generalize(input_smarts))
    [C,N,O:1]-[N:2]>>[N:1]-[C,N,O:2]
    """

    _atom_pat = re.compile(r"\[([A-Z][a-z]?):(\d+)\]")

    def __init__(self, sanity_check: bool = True):
        """
        Initialize SMARTSGeneralizer.

        :param sanity_check: If True, validate the output SMARTS with RDKit.
        :type sanity_check: bool
        """
        self.sanity_check = sanity_check

    def generalize(self, smarts_list: List[str]) -> str:
        """
        Generalize a list of SMARTS/reaction SMARTS into one combinatorial SMARTS,
        with element-list placeholders per atom-map index and position.

        :param smarts_list: List of atom-mapped SMARTS strings (same topology/order).
        :type smarts_list: list[str]
        :return: Generalized SMARTS with atom-list placeholders.
        :rtype: str
        :raises ValueError: If input list is empty, topology is inconsistent, or output is invalid.
        """
        if not smarts_list:
            raise ValueError("Input list is empty.")

        if len(smarts_list) == 1:
            combined = smarts_list[0]
        else:
            pos_list: List[List[re.Match]] = [
                list(self._atom_pat.finditer(s)) for s in smarts_list
            ]
            n_atoms = len(pos_list[0])
            for pl in pos_list:
                if len(pl) != n_atoms:
                    raise ValueError(
                        "All input SMARTS must have same atom-mapped topology and order."
                    )

            pos2map: List[str] = []
            pos2elems: List[Set[str]] = []
            for i in range(n_atoms):
                mapnum = pos_list[0][i].group(2)
                elems = set(match[i].group(1) for match in pos_list)
                pos2map.append(mapnum)
                pos2elems.append(elems)

            # Template assembly: alternate static and atom-map segments
            first = smarts_list[0]
            atoms = list(self._atom_pat.finditer(first))
            segments = []
            last = 0
            for m in atoms:
                # fmt: off
                segments.append(first[last: m.start()])
                # fmt: on
                segments.append(m.group(2))  # mapnum as marker
                last = m.end()
            segments.append(first[last:])

            # Reconstruct combinatorial SMARTS
            out = []
            idx = 0
            for seg in segments:
                if idx < len(pos2map) and seg == pos2map[idx]:
                    els = sorted(pos2elems[idx])
                    out.append(f"[{','.join(els)}:{seg}]")
                    idx += 1
                else:
                    out.append(seg)
            combined = "".join(out)

        # RDKit validation
        if self.sanity_check:
            if ">>" in combined:
                rxn = rdChemReactions.ReactionFromSmarts(combined)
                if rxn is None or rxn.GetNumProductTemplates() == 0:
                    raise ValueError(f"Invalid reaction SMARTS generated: {combined}")
            else:
                mol = Chem.MolFromSmarts(combined)
                if mol is None:
                    raise ValueError(f"Invalid molecule SMARTS generated: {combined}")

        return combined

    def describe(self) -> None:
        """
        Print usage instructions and an example.

        :return: None
        """
        print(
            "SMARTSGeneralizer: Generalize a list of atom-mapped (reaction) SMARTS into a single "
            "SMARTS with element-list placeholders at each mapped atom position.\n"
            "Usage example:\n"
            "  >>> gen = SMARTSGeneralizer(sanity_check=True)\n"
            "  >>> smarts_list = ['[C:1]-[N:2]', '[O:1]-[N:2]', '[N:1]-[N:2]']\n"
            "  >>> print(gen.generalize(smarts_list))  # [C,N,O:1]-[N:2]"
        )

    def __repr__(self) -> str:
        """
        String representation.

        :return: Description with current sanity_check setting.
        :rtype: str
        """
        return f"SMARTSGeneralizer(sanity_check={self.sanity_check})"
