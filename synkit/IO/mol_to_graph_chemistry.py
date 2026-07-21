"""Chemical bookkeeping helpers mixed into :class:`MolToGraph`."""

from __future__ import annotations

from typing import Any, Dict, Optional

from rdkit import Chem

from synkit.IO.debug import setup_logging

logger = setup_logging()


class MolToGraphChemistryMixin:
    @staticmethod
    def _safe_atom_map(atom: Chem.Atom) -> int:
        """Return atom-map number or ``0``.

        :param atom: RDKit atom.
        :type atom: Chem.Atom
        :returns: Atom-map number.
        :rtype: int
        """
        try:
            return int(atom.GetAtomMapNum())
        except Exception:
            return 0

    @staticmethod
    def _safe_bond_order_sum(atom: Chem.Atom) -> float:
        """Return raw RDKit bond-order sum.

        :param atom: RDKit atom.
        :type atom: Chem.Atom
        :returns: Incident bond-order sum.
        :rtype: float
        """
        try:
            return float(sum(bond.GetBondTypeAsDouble() for bond in atom.GetBonds()))
        except Exception:
            return 0.0

    @staticmethod
    def _safe_valence_electrons(atom: Chem.Atom) -> int:
        """Return outer-shell valence electron count.

        :param atom: RDKit atom.
        :type atom: Chem.Atom
        :returns: Valence electron count, or ``0`` on failure.
        :rtype: int
        """
        try:
            pt = Chem.GetPeriodicTable()
            return int(pt.GetNOuterElecs(atom.GetAtomicNum()))
        except Exception:
            return 0

    @staticmethod
    def _explicit_h_neighbor_count(atom: Chem.Atom) -> int:
        """Count explicit hydrogen neighbors.

        :param atom: RDKit atom.
        :type atom: Chem.Atom
        :returns: Number of explicit hydrogen neighbors.
        :rtype: int
        """
        try:
            return sum(1 for nb in atom.GetNeighbors() if nb.GetAtomicNum() == 1)
        except Exception:
            return 0

    @staticmethod
    def _non_neighbor_h_count(atom: Chem.Atom) -> int:
        """Count hydrogens not represented as explicit neighbors.

        :param atom: RDKit atom.
        :type atom: Chem.Atom
        :returns: Non-neighbor hydrogen count.
        :rtype: int
        """
        try:
            return int(atom.GetTotalNumHs(includeNeighbors=False))
        except TypeError:
            try:
                return int(atom.GetTotalNumHs())
            except Exception:
                return 0
        except Exception:
            return 0

    @staticmethod
    def _total_h_count(atom: Chem.Atom) -> int:
        """Count explicit-neighbor and non-neighbor hydrogens.

        :param atom: RDKit atom.
        :type atom: Chem.Atom
        :returns: Total hydrogen count.
        :rtype: int
        """
        return MolToGraphChemistryMixin._explicit_h_neighbor_count(
            atom
        ) + MolToGraphChemistryMixin._non_neighbor_h_count(atom)

    @staticmethod
    def _heavy_neighbor_count(atom: Chem.Atom) -> int:
        """Count non-hydrogen neighbors.

        :param atom: RDKit atom.
        :type atom: Chem.Atom
        :returns: Heavy-neighbor count.
        :rtype: int
        """
        try:
            return sum(1 for nb in atom.GetNeighbors() if nb.GetAtomicNum() != 1)
        except Exception:
            return 0

    @staticmethod
    def _make_kekule_copy(mol: Chem.Mol) -> Optional[Chem.Mol]:
        """Return a kekulized copy, or ``None`` on failure.

        :param mol: RDKit molecule.
        :type mol: Chem.Mol
        :returns: Kekulized molecule copy or ``None``.
        :rtype: Optional[Chem.Mol]
        """
        try:
            kek = Chem.Mol(mol)
            Chem.Kekulize(kek, clearAromaticFlags=True)
            return kek
        except Exception as exc:
            logger.debug("Failed to create kekulized copy: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Lone-pair estimation
    # ------------------------------------------------------------------

    @classmethod
    def _is_aromatic_lone_pair_donor(cls, atom: Chem.Atom) -> bool:
        """Detect aromatic heteroatoms whose lone pair contributes to aromaticity.

        :param atom: RDKit atom.
        :type atom: Chem.Atom
        :returns: ``True`` if aromatic bonds should be counted as sigma bonds
            for lone-pair bookkeeping.
        :rtype: bool
        """
        try:
            if not atom.GetIsAromatic():
                return False

            atomic_num = int(atom.GetAtomicNum())
            formal_charge = int(atom.GetFormalCharge())
            total_h = cls._total_h_count(atom)
            heavy_degree = cls._heavy_neighbor_count(atom)

            if atomic_num == 7:
                aromatic_bonds = 0
                nonaromatic_heavy_sigma_bonds = 0
                for bond in atom.GetBonds():
                    other = bond.GetOtherAtom(atom)
                    if bond.GetIsAromatic():
                        aromatic_bonds += 1
                    elif (
                        other.GetAtomicNum() != 1
                        and float(bond.GetBondTypeAsDouble()) <= 1.1
                    ):
                        nonaromatic_heavy_sigma_bonds += 1

                if formal_charge <= 0 and aromatic_bonds >= 2:
                    if total_h > 0 and heavy_degree == 2:
                        return True
                    if (
                        total_h == 0
                        and heavy_degree == 3
                        and nonaromatic_heavy_sigma_bonds >= 1
                    ):
                        return True

                if formal_charge < 0 and heavy_degree <= 2:
                    return True

                return False

            if atomic_num in {8, 16, 34, 52}:
                if formal_charge <= 0 and heavy_degree <= 2:
                    return True

            if atomic_num == 15:
                if formal_charge <= 0 and total_h > 0 and heavy_degree == 2:
                    return True

            return False

        except Exception:
            return False

    @classmethod
    def _bond_order_sum_for_lone_pairs(cls, atom: Chem.Atom) -> float:
        """Return bond-order sum used for lone-pair bookkeeping.

        :param atom: RDKit atom.
        :type atom: Chem.Atom
        :returns: Corrected lone-pair bond-order sum.
        :rtype: float
        """
        try:
            if atom.GetIsAromatic():
                # Lone-pair bookkeeping needs the Kekule heavy-atom valence,
                # not presentation bond orders such as three aromatic 1.5 bonds.
                return float(atom.GetTotalValence() - cls._non_neighbor_h_count(atom))

            aromatic_lp_donor = cls._is_aromatic_lone_pair_donor(atom)
            total = 0.0

            for bond in atom.GetBonds():
                try:
                    if aromatic_lp_donor and bond.GetIsAromatic():
                        total += 1.0
                    else:
                        total += float(bond.GetBondTypeAsDouble())
                except Exception:
                    total += 1.0

            return total

        except Exception:
            return 0.0

    @classmethod
    def estimate_lone_pairs(cls, atom: Chem.Atom) -> int:
        """Estimate total lone-pair count.

        :param atom: RDKit atom.
        :type atom: Chem.Atom
        :returns: Estimated total lone-pair count.
        :rtype: int

        .. code-block:: python

            mol = Chem.MolFromSmiles("c1cc[nH]c1")
            n_atom = next(a for a in mol.GetAtoms() if a.GetSymbol() == "N")
            print(MolToGraph.estimate_lone_pairs(n_atom))
        """
        try:
            valence_electrons = float(cls._safe_valence_electrons(atom))
            formal_charge = float(int(atom.GetFormalCharge()))
            radical_electrons = float(int(atom.GetNumRadicalElectrons()))
            bond_order_sum = float(cls._bond_order_sum_for_lone_pairs(atom))
            non_neighbor_h = float(cls._non_neighbor_h_count(atom))

            nonbonding_electrons = (
                valence_electrons
                - formal_charge
                - radical_electrons
                - bond_order_sum
                - non_neighbor_h
            )

            lone_pairs = int((nonbonding_electrons + 1e-8) // 2)
            return max(0, lone_pairs)

        except Exception:
            return 0

    @classmethod
    def estimate_available_lone_pairs(cls, atom: Chem.Atom) -> int:
        """Estimate lone pairs locally available for ``LP-/B+`` donation.

        :param atom: RDKit atom.
        :type atom: Chem.Atom
        :returns: Locally available lone-pair count.
        :rtype: int
        """
        total_lp = cls.estimate_lone_pairs(atom)

        if total_lp <= 0:
            return 0

        try:
            atomic_num = int(atom.GetAtomicNum())
            formal_charge = int(atom.GetFormalCharge())
            total_h = cls._total_h_count(atom)

            if formal_charge > 0:
                return 0

            if atom.GetIsAromatic():
                if atomic_num == 7 and total_h > 0:
                    return 0
                if atomic_num in {8, 16, 34, 52}:
                    return max(0, total_lp - 1)

            return total_lp

        except Exception:
            return total_lp

    # ------------------------------------------------------------------
    # Oxidation-state estimation
    # ------------------------------------------------------------------

    @classmethod
    def _bond_order_for_oxidation_state(
        cls,
        bond: Chem.Bond,
        kek_bond: Optional[Chem.Bond] = None,
        *,
        prefer_kekule: bool = True,
    ) -> float:
        """Return bond order for oxidation-state bookkeeping.

        :param bond: Original RDKit bond.
        :type bond: Chem.Bond
        :param kek_bond: Matching bond from a kekulized copy.
        :type kek_bond: Optional[Chem.Bond]
        :param prefer_kekule: Whether to use ``kek_bond`` when available.
        :type prefer_kekule: bool
        :returns: Bond order.
        :rtype: float
        """
        try:
            if prefer_kekule and kek_bond is not None:
                return float(kek_bond.GetBondTypeAsDouble())
            return float(bond.GetBondTypeAsDouble())
        except Exception:
            return 1.0

    @classmethod
    def estimate_oxidation_states(
        cls,
        mol: Chem.Mol,
        *,
        kek_mol: Optional[Chem.Mol] = None,
        prefer_kekule: bool = True,
        en_tie_threshold: float = 0.05,
    ) -> Dict[int, float]:
        """Estimate atom oxidation states.

        For each bond, bond electrons are assigned to the more electronegative
        atom. Formal charge is used as the starting value.

        :param mol: RDKit molecule.
        :type mol: Chem.Mol
        :param kek_mol: Optional kekulized copy of ``mol``.
        :type kek_mol: Optional[Chem.Mol]
        :param prefer_kekule: Whether to prefer kekulized bond orders.
        :type prefer_kekule: bool
        :param en_tie_threshold: Electronegativity-difference threshold for
            treating a bond as a tie.
        :type en_tie_threshold: float
        :returns: Oxidation states keyed by RDKit atom index.
        :rtype: Dict[int, float]
        """
        ox: Dict[int, float] = {}

        try:
            for atom in mol.GetAtoms():
                ox[atom.GetIdx()] = float(atom.GetFormalCharge())

            for bond in mol.GetBonds():
                a = bond.GetBeginAtom()
                b = bond.GetEndAtom()

                i = a.GetIdx()
                j = b.GetIdx()

                elem_i = a.GetSymbol()
                elem_j = b.GetSymbol()

                en_i = cls.PAULING_EN.get(elem_i)
                en_j = cls.PAULING_EN.get(elem_j)

                if en_i is None or en_j is None:
                    continue

                kek_bond: Optional[Chem.Bond] = None
                if kek_mol is not None:
                    try:
                        kek_bond = kek_mol.GetBondWithIdx(bond.GetIdx())
                    except Exception:
                        kek_bond = None

                order = cls._bond_order_for_oxidation_state(
                    bond,
                    kek_bond=kek_bond,
                    prefer_kekule=prefer_kekule,
                )

                if abs(order) < 1e-12:
                    continue

                diff = float(en_i) - float(en_j)

                if abs(diff) <= en_tie_threshold:
                    continue

                if diff > 0:
                    ox[i] -= order
                    ox[j] += order
                else:
                    ox[i] += order
                    ox[j] -= order

            return ox

        except Exception as exc:
            logger.debug("Oxidation-state estimation failed: %s", exc)
            return ox

    @classmethod
    def oxidation_states_by_atom_map(
        cls,
        mol: Chem.Mol,
        *,
        kek_mol: Optional[Chem.Mol] = None,
        prefer_kekule: bool = True,
        en_tie_threshold: float = 0.05,
    ) -> Dict[int, Dict[str, Any]]:
        """Return oxidation states keyed by non-zero atom-map number.

        :param mol: Mapped RDKit molecule.
        :type mol: Chem.Mol
        :param kek_mol: Optional kekulized copy.
        :type kek_mol: Optional[Chem.Mol]
        :param prefer_kekule: Whether to prefer kekulized bond orders.
        :type prefer_kekule: bool
        :param en_tie_threshold: Electronegativity tie threshold.
        :type en_tie_threshold: float
        :returns: Oxidation-state records keyed by atom-map number.
        :rtype: Dict[int, Dict[str, Any]]
        """
        if kek_mol is None:
            kek_mol = cls._make_kekule_copy(mol)

        ox = cls.estimate_oxidation_states(
            mol,
            kek_mol=kek_mol,
            prefer_kekule=prefer_kekule,
            en_tie_threshold=en_tie_threshold,
        )

        out: Dict[int, Dict[str, Any]] = {}

        for atom in mol.GetAtoms():
            amap = cls._safe_atom_map(atom)
            if amap == 0:
                continue

            out[amap] = {
                "atom_idx": atom.GetIdx(),
                "element": atom.GetSymbol(),
                "charge": atom.GetFormalCharge(),
                "oxidation_state": ox.get(atom.GetIdx(), 0.0),
            }

        return out

    @classmethod
    def reaction_oxidation_state_delta_from_rsmi(
        cls,
        rsmi: str,
        *,
        threshold: float = 0.5,
        prefer_kekule: bool = True,
        en_tie_threshold: float = 0.05,
    ) -> Dict[int, Dict[str, Any]]:
        """Compute oxidation-state changes for mapped reaction SMILES.

        Positive ``delta`` means oxidation; negative ``delta`` means reduction.

        :param rsmi: Mapped reaction SMILES containing ``">>"``.
        :type rsmi: str
        :param threshold: Minimum absolute delta to report.
        :type threshold: float
        :param prefer_kekule: Whether to prefer kekulized bond orders.
        :type prefer_kekule: bool
        :param en_tie_threshold: Electronegativity tie threshold.
        :type en_tie_threshold: float
        :returns: Significant oxidation-state changes keyed by atom map.
        :rtype: Dict[int, Dict[str, Any]]
        :raises ValueError: If ``rsmi`` lacks ``">>"``.

        .. code-block:: python

            rsmi = "[CH3:1][OH:2]>>[CH2:1]=[O:2]"
            print(MolToGraph.reaction_oxidation_state_delta_from_rsmi(rsmi))
        """
        if ">>" not in rsmi:
            raise ValueError("Expected mapped reaction SMILES containing '>>'.")

        reactants_smi, products_smi = rsmi.split(">>", 1)

        def _side_maps(side_smi: str) -> Dict[int, Dict[str, Any]]:
            merged: Dict[int, Dict[str, Any]] = {}

            for smi in side_smi.split("."):
                smi = smi.strip()
                if not smi:
                    continue

                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue

                merged.update(
                    cls.oxidation_states_by_atom_map(
                        mol,
                        prefer_kekule=prefer_kekule,
                        en_tie_threshold=en_tie_threshold,
                    )
                )

            return merged

        r_by_map = _side_maps(reactants_smi)
        p_by_map = _side_maps(products_smi)

        changes: Dict[int, Dict[str, Any]] = {}

        for amap in sorted(set(r_by_map) | set(p_by_map)):
            r = r_by_map.get(amap)
            p = p_by_map.get(amap)

            if r is None or p is None:
                changes[amap] = {
                    "reactant": r,
                    "product": p,
                    "reason": "atom_map_missing_on_one_side",
                }
                continue

            delta = float(p["oxidation_state"]) - float(r["oxidation_state"])

            if abs(delta) >= threshold:
                changes[amap] = {
                    "element": (r["element"], p["element"]),
                    "reactant_os": round(float(r["oxidation_state"]), 3),
                    "product_os": round(float(p["oxidation_state"]), 3),
                    "delta": round(delta, 3),
                    "classification": "oxidized" if delta > 0 else "reduced",
                }

        return changes
