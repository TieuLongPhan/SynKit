from __future__ import annotations

from typing import Any, Dict, List, Optional
import random

import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem

from synkit.IO.debug import setup_logging
from synkit.Chem.Molecule.descriptors import (
    compute_gasteiger_inplace,
    PerMolDescriptors,
)
from synkit.Chem.Molecule.atom_features import AtomFeatureExtractor
from synkit.Chem.Molecule.graph_annotator import GraphAnnotator

logger = setup_logging()


class MolToGraph:
    """Convert an RDKit molecule into a NetworkX molecular graph.

    The converter preserves the public API while adding corrected lone-pair
    bookkeeping for aromatic heteroatoms, especially pyrrolic / ``[nH]``-like
    aromatic nitrogen. RDKit aromatic bonds have order ``1.5``; for aromatic
    lone-pair donor heteroatoms, this class counts aromatic bonds as sigma bonds
    during lone-pair estimation.

    Important node fields are ``estimated_lone_pairs``, ``lone_pairs``
    backward-compatible alias, ``available_lone_pairs``, ``available_lp``,
    ``bond_order_sum``, ``lp_bond_order_sum``, ``valence_electrons``, and
    ``oxidation_state``.

    :param node_attrs: Optional whitelist of node attributes to keep.
    :type node_attrs: Optional[List[str]]
    :param edge_attrs: Optional whitelist of edge attributes to keep.
    :type edge_attrs: Optional[List[str]]
    :param attr_profile: Atom feature profile, either ``"minimal"`` or
        ``"full"``.
    :type attr_profile: str
    :param with_topology: If ``True``, run :class:`GraphAnnotator` on the graph.
    :type with_topology: bool
    :raises ValueError: If ``attr_profile`` is unsupported.

    .. code-block:: python

        from rdkit import Chem
        from synkit.IO.mol_to_graph import MolToGraph

        mol = Chem.MolFromSmiles("c1cc[nH]c1")
        graph = MolToGraph(attr_profile="minimal").transform(mol)

        for node, data in graph.nodes(data=True):
            print(node, data["element"], data["lone_pairs"], data["available_lp"])

    .. code-block:: python

        mol = Chem.MolFromSmiles("[CH3:1][CH2:2][Br:3]")
        graph = MolToGraph(
            node_attrs=["element", "atom_map", "charge", "lone_pairs"],
            edge_attrs=["order", "kekule_order"],
        ).transform(mol, use_index_as_atom_map=True)
    """

    SUPPORTED_PROFILES = ("minimal", "full")

    # Pauling electronegativities used for oxidation-state bookkeeping.
    # Missing elements are skipped instead of guessed.
    PAULING_EN: Dict[str, float] = {
        "H": 2.20,
        "B": 2.04,
        "C": 2.55,
        "N": 3.04,
        "O": 3.44,
        "F": 3.98,
        "P": 2.19,
        "S": 2.58,
        "Cl": 3.16,
        "Br": 2.96,
        "I": 2.66,
        "Se": 2.55,
    }

    def __init__(
        self,
        node_attrs: Optional[List[str]] = None,
        edge_attrs: Optional[List[str]] = None,
        *,
        attr_profile: str = "minimal",
        with_topology: bool = False,
    ) -> None:
        """Initialize the converter.

        :param node_attrs: Optional node-attribute whitelist.
        :type node_attrs: Optional[List[str]]
        :param edge_attrs: Optional edge-attribute whitelist.
        :type edge_attrs: Optional[List[str]]
        :param attr_profile: Feature profile, ``"minimal"`` or ``"full"``.
        :type attr_profile: str
        :param with_topology: Whether to add topology annotations.
        :type with_topology: bool
        :raises ValueError: If ``attr_profile`` is unsupported.
        """
        if attr_profile not in self.SUPPORTED_PROFILES:
            raise ValueError(
                f"Unsupported attr_profile: {attr_profile!r}. "
                f"Supported: {self.SUPPORTED_PROFILES}"
            )

        self.node_attrs: Optional[List[str]] = (
            None if node_attrs is None else list(node_attrs)
        )
        self.edge_attrs: Optional[List[str]] = (
            None if edge_attrs is None else list(edge_attrs)
        )
        self.attr_profile: str = attr_profile
        self.with_topology: bool = bool(with_topology)

        self._graph: Optional[nx.Graph] = None
        self._last_mol: Optional[Chem.Mol] = None

    # ------------------------------------------------------------------
    # Public conversion API
    # ------------------------------------------------------------------

    def transform(
        self,
        mol: Chem.Mol,
        drop_non_aam: bool = False,
        use_index_as_atom_map: bool = False,
    ) -> nx.Graph:
        """Build a NetworkX graph from an RDKit molecule.

        :param mol: RDKit molecule.
        :type mol: Chem.Mol
        :param drop_non_aam: If ``True``, exclude atoms with atom-map ``0``.
        :type drop_non_aam: bool
        :param use_index_as_atom_map: If ``True``, use non-zero atom-map numbers
            as node identifiers; otherwise use ``atom index + 1``.
        :type use_index_as_atom_map: bool
        :returns: Molecular graph with atom and bond attributes.
        :rtype: nx.Graph
        :raises ValueError: If ``drop_non_aam=True`` but
            ``use_index_as_atom_map=False``.

        .. code-block:: python

            mol = Chem.MolFromSmiles("[CH3:1][CH2:2][Br:3]")
            graph = MolToGraph().transform(
                mol,
                drop_non_aam=True,
                use_index_as_atom_map=True,
            )
        """
        if drop_non_aam and not use_index_as_atom_map:
            raise ValueError(
                "drop_non_aam and use_index_as_atom_map must both be True "
                "to drop unmapped atoms."
            )

        self._last_mol = mol

        try:
            compute_gasteiger_inplace(mol)
        except Exception:
            logger.debug("Gasteiger computation failed (best-effort). Continuing.")

        kek_mol: Optional[Chem.Mol] = self._make_kekule_copy(mol)
        oxidation_states = self.estimate_oxidation_states(mol, kek_mol=kek_mol)

        per: Optional[PerMolDescriptors] = None
        if self.attr_profile == "full":
            try:
                per = PerMolDescriptors.compute(mol)
            except Exception as exc:
                logger.debug("PerMolDescriptors.compute failed: %s", exc)
                per = None

        extractor = AtomFeatureExtractor(mol, per=per, profile=self.attr_profile)

        graph = nx.Graph()
        index_to_id: Dict[int, int] = {}

        for atom in mol.GetAtoms():
            atom_map = self._safe_atom_map(atom)
            atom_id = (
                atom_map
                if (use_index_as_atom_map and atom_map != 0)
                else atom.GetIdx() + 1
            )

            if drop_non_aam and atom_map == 0:
                continue

            try:
                props = extractor.build_dict(atom)
                props = self._augment_atom_properties(
                    atom,
                    props,
                    oxidation_state=oxidation_states.get(atom.GetIdx()),
                    profile=self.attr_profile,
                )
            except Exception as exc:
                logger.debug(
                    "AtomFeatureExtractor failed for atom %s: %s",
                    atom.GetIdx(),
                    exc,
                )
                props = self._gather_atom_properties(
                    atom,
                    oxidation_state=oxidation_states.get(atom.GetIdx()),
                    profile=self.attr_profile,
                )

            if self.node_attrs is not None:
                props = {k: v for k, v in props.items() if k in self.node_attrs}

            graph.add_node(atom_id, **props)
            index_to_id[atom.GetIdx()] = atom_id

        for bond in mol.GetBonds():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()

            begin = index_to_id.get(begin_idx)
            end = index_to_id.get(end_idx)
            if begin is None or end is None:
                continue

            kek_bond: Optional[Chem.Bond] = None
            if kek_mol is not None:
                try:
                    kek_bond = kek_mol.GetBondWithIdx(bond.GetIdx())
                except Exception:
                    kek_bond = None

            try:
                bprops = self._gather_bond_properties(bond, kek_bond=kek_bond)
            except Exception as exc:
                logger.debug(
                    "Bond property collection failed for bond %s: %s",
                    bond.GetIdx(),
                    exc,
                )
                original_order = (
                    bond.GetBondTypeAsDouble()
                    if hasattr(bond, "GetBondTypeAsDouble")
                    else 1.0
                )
                bprops = {
                    "order": original_order,
                    "bond_type": (
                        str(bond.GetBondType())
                        if hasattr(bond, "GetBondType")
                        else "UNKNOWN"
                    ),
                    "kekule_order": (
                        kek_bond.GetBondTypeAsDouble()
                        if kek_bond is not None
                        and hasattr(kek_bond, "GetBondTypeAsDouble")
                        else original_order
                    ),
                    "kekule_bond_type": (
                        str(kek_bond.GetBondType())
                        if kek_bond is not None and hasattr(kek_bond, "GetBondType")
                        else (
                            str(bond.GetBondType())
                            if hasattr(bond, "GetBondType")
                            else "UNKNOWN"
                        )
                    ),
                    "aromatic": (
                        bond.GetIsAromatic()
                        if hasattr(bond, "GetIsAromatic")
                        else False
                    ),
                }

            if self.edge_attrs is not None:
                bprops = {k: v for k, v in bprops.items() if k in self.edge_attrs}

            graph.add_edge(begin, end, **bprops)

        if self.with_topology:
            try:
                GraphAnnotator(graph, in_place=True).annotate()
            except Exception as exc:
                logger.debug("GraphAnnotator failed: %s", exc)

        return graph

    def transform_store(
        self,
        mol: Chem.Mol,
        drop_non_aam: bool = False,
        use_index_as_atom_map: bool = False,
    ) -> "MolToGraph":
        """Build, store, and return ``self``.

        :param mol: RDKit molecule.
        :type mol: Chem.Mol
        :param drop_non_aam: If ``True``, exclude atoms with atom-map ``0``.
        :type drop_non_aam: bool
        :param use_index_as_atom_map: If ``True``, use atom maps as node IDs.
        :type use_index_as_atom_map: bool
        :returns: Current converter instance.
        :rtype: MolToGraph
        """
        self._graph = self.transform(
            mol,
            drop_non_aam=drop_non_aam,
            use_index_as_atom_map=use_index_as_atom_map,
        )
        return self

    @property
    def graph(self) -> nx.Graph:
        """Return the graph produced by :meth:`transform_store`.

        :returns: Stored molecular graph.
        :rtype: nx.Graph
        :raises RuntimeError: If no graph has been stored yet.
        """
        if self._graph is None:
            raise RuntimeError(
                "No graph produced yet. Call `transform_store(mol)` first."
            )
        return self._graph

    def __repr__(self) -> str:
        """Return a compact representation.

        :returns: Developer-facing representation string.
        :rtype: str
        """
        try:
            n = self._graph.number_of_nodes() if self._graph is not None else 0
        except Exception:
            n = -1

        return (
            f"{self.__class__.__name__}(profile={self.attr_profile!r}, "
            f"with_topology={self.with_topology}, node_attrs={self.node_attrs!r}, "
            f"edge_attrs={self.edge_attrs!r}, last_nodes={n})"
        )

    @classmethod
    def help(cls) -> str:
        """Return a short usage string.

        :returns: Usage summary.
        :rtype: str
        """
        return (
            "MolToGraph.help() -> str\n\n"
            "Create with MolToGraph(node_attrs=[...], edge_attrs=[...], "
            "attr_profile='minimal'|'full', with_topology=False).\n"
            "Use `.transform(mol)` to get an nx.Graph, or `.transform_store(mol)` "
            "to build and store the graph on the instance."
        )

    # ------------------------------------------------------------------
    # Safe RDKit helpers
    # ------------------------------------------------------------------

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
        return MolToGraph._explicit_h_neighbor_count(
            atom
        ) + MolToGraph._non_neighbor_h_count(atom)

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

    # ------------------------------------------------------------------
    # Atom and bond property collection
    # ------------------------------------------------------------------

    @classmethod
    def _augment_atom_properties(
        cls,
        atom: Chem.Atom,
        props: Dict[str, Any],
        oxidation_state: Optional[float] = None,
        *,
        profile: str = "full",
    ) -> Dict[str, Any]:
        """Add electron-bookkeeping fields to existing atom attributes.

        For both profiles sets ``oxidation_state``, ``radical``,
        ``available_lp``, and the backward-compatible ``lone_pairs`` alias.
        The ``"full"`` profile additionally sets ``bond_order_sum``,
        ``lp_bond_order_sum``, ``valence_electrons``,
        ``estimated_lone_pairs``, and ``available_lone_pairs``.

        :param atom: RDKit atom.
        :type atom: Chem.Atom
        :param props: Existing atom attributes from
            :class:`~synkit.Chem.Molecule.atom_features.AtomFeatureExtractor`.
        :type props: Dict[str, Any]
        :param oxidation_state: Pre-computed oxidation state, or ``None``.
        :type oxidation_state: Optional[float]
        :param profile: Feature profile — ``"minimal"`` or ``"full"``.
        :type profile: str
        :returns: Augmented atom attributes dict.
        :rtype: Dict[str, Any]
        """
        new_props = dict(props)

        estimated_lone_pairs = cls.estimate_lone_pairs(atom)
        available_lone_pairs = cls.estimate_available_lone_pairs(atom)

        new_props["oxidation_state"] = (
            None if oxidation_state is None else round(float(oxidation_state), 3)
        )
        new_props["radical"] = int(atom.GetNumRadicalElectrons())
        new_props["available_lp"] = available_lone_pairs > 0
        # Backward-compatible field used by SynEltra.
        new_props["lone_pairs"] = estimated_lone_pairs
        new_props["valence_electrons"] = cls._safe_valence_electrons(atom)

        if profile == "full":
            new_props["bond_order_sum"] = round(cls._safe_bond_order_sum(atom), 3)
            new_props["lp_bond_order_sum"] = round(
                cls._bond_order_sum_for_lone_pairs(atom), 3
            )
            new_props["estimated_lone_pairs"] = estimated_lone_pairs
            new_props["available_lone_pairs"] = available_lone_pairs

        return new_props

    @staticmethod
    def _gather_atom_properties(
        atom: Chem.Atom,
        oxidation_state: Optional[float] = None,
        *,
        profile: str = "full",
    ) -> Dict[str, Any]:
        """Collect fallback atom-level node attributes.

        Minimal profile keys: ``element``, ``aromatic``, ``hcount``,
        ``charge``, ``radical``, ``isomer``, ``partial_charge``,
        ``hybridization``, ``in_ring``, ``neighbors``, ``atom_map``,
        ``oxidation_state``, ``available_lp``, ``lone_pairs``,
        ``valence_electrons``.

        Full profile additionally includes ``bond_order_sum``,
        ``lp_bond_order_sum``, ``estimated_lone_pairs``,
        ``available_lone_pairs``.

        :param atom: RDKit atom.
        :type atom: Chem.Atom
        :param oxidation_state: Pre-computed oxidation state, or ``None``.
        :type oxidation_state: Optional[float]
        :param profile: Feature profile — ``"minimal"`` or ``"full"``.
        :type profile: str
        :returns: Node attribute dict.
        :rtype: Dict[str, Any]
        """
        try:
            gcharge = (
                round(float(atom.GetProp("_GasteigerCharge")), 3)
                if atom.HasProp("_GasteigerCharge")
                else 0.0
            )
        except Exception:
            gcharge = 0.0

        try:
            neighbors = sorted(nb.GetSymbol() for nb in atom.GetNeighbors())
        except Exception:
            neighbors = []

        atom_map = MolToGraph._safe_atom_map(atom)
        estimated_lone_pairs = MolToGraph.estimate_lone_pairs(atom)
        available_lone_pairs = MolToGraph.estimate_available_lone_pairs(atom)

        props: Dict[str, Any] = {
            "element": atom.GetSymbol(),
            "aromatic": atom.GetIsAromatic(),
            "hcount": atom.GetTotalNumHs(),
            "charge": atom.GetFormalCharge(),
            "radical": atom.GetNumRadicalElectrons(),
            "isomer": MolToGraph.get_stereochemistry(atom),
            "partial_charge": gcharge,
            "hybridization": str(atom.GetHybridization()),
            "in_ring": atom.IsInRing(),
            "neighbors": neighbors,
            "atom_map": atom_map,
            "oxidation_state": (
                None if oxidation_state is None else round(float(oxidation_state), 3)
            ),
            "available_lp": available_lone_pairs > 0,
            "lone_pairs": estimated_lone_pairs,
            "valence_electrons": MolToGraph._safe_valence_electrons(atom),
        }

        if profile == "full":
            props["bond_order_sum"] = round(MolToGraph._safe_bond_order_sum(atom), 3)
            props["lp_bond_order_sum"] = round(
                MolToGraph._bond_order_sum_for_lone_pairs(atom), 3
            )
            props["estimated_lone_pairs"] = estimated_lone_pairs
            props["available_lone_pairs"] = available_lone_pairs

        return props

    @staticmethod
    def _gather_bond_properties(
        bond: Chem.Bond,
        kek_bond: Optional[Chem.Bond] = None,
    ) -> Dict[str, Any]:
        """Collect bond-level edge attributes.

        :param bond: Original RDKit bond.
        :type bond: Chem.Bond
        :param kek_bond: Matching bond from a kekulized copy.
        :type kek_bond: Optional[Chem.Bond]
        :returns: Edge attributes.
        :rtype: Dict[str, Any]
        """
        try:
            order = bond.GetBondTypeAsDouble()
        except Exception:
            order = 1.0

        try:
            bond_type = str(bond.GetBondType())
        except Exception:
            bond_type = "UNKNOWN"

        try:
            ez = MolToGraph.get_bond_stereochemistry(bond)
        except Exception:
            ez = "N"

        try:
            conjugated = bond.GetIsConjugated()
        except Exception:
            conjugated = False

        try:
            in_ring = bond.IsInRing()
        except Exception:
            in_ring = False

        try:
            aromatic = bond.GetIsAromatic()
        except Exception:
            aromatic = False

        try:
            kekule_order = (
                kek_bond.GetBondTypeAsDouble() if kek_bond is not None else order
            )
        except Exception:
            kekule_order = order

        try:
            kekule_bond_type = (
                str(kek_bond.GetBondType()) if kek_bond is not None else bond_type
            )
        except Exception:
            kekule_bond_type = bond_type

        sigma_order, pi_order = MolToGraph._split_sigma_pi_order(kekule_order)

        return {
            "order": order,
            "bond_type": bond_type,
            "aromatic": aromatic,
            "kekule_order": kekule_order,
            "sigma_order": sigma_order,
            "pi_order": pi_order,
            "kekule_bond_type": kekule_bond_type,
            "ez_isomer": ez,
            "conjugated": conjugated,
            "in_ring": in_ring,
        }

    @staticmethod
    def _split_sigma_pi_order(kekule_order: float) -> tuple[float, float]:
        """Split a Kekule bond order into sigma and pi contributions."""
        order = max(0.0, float(kekule_order))
        if order <= 0:
            return 0.0, 0.0
        return 1.0, max(0.0, order - 1.0)

    # ------------------------------------------------------------------
    # Stereochemistry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def get_stereochemistry(atom: Chem.Atom) -> str:
        """Return ``S``, ``R``, or ``N`` from the RDKit chiral tag.

        :param atom: RDKit atom.
        :type atom: Chem.Atom
        :returns: Simple atom stereochemistry label.
        :rtype: str
        """
        chiral_tag = atom.GetChiralTag()
        if chiral_tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
            return "S"
        if chiral_tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
            return "R"
        return "N"

    @staticmethod
    def get_bond_stereochemistry(bond: Chem.Bond) -> str:
        """Return ``E``, ``Z``, or ``N`` for double-bond stereochemistry.

        :param bond: RDKit bond.
        :type bond: Chem.Bond
        :returns: Simple bond stereochemistry label.
        :rtype: str
        """
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            return "N"

        stereo = bond.GetStereo()
        if stereo == Chem.BondStereo.STEREOE:
            return "E"
        if stereo == Chem.BondStereo.STEREOZ:
            return "Z"
        return "N"

    # ------------------------------------------------------------------
    # Mapping and legacy API
    # ------------------------------------------------------------------

    @staticmethod
    def has_atom_mapping(mol: Chem.Mol) -> bool:
        """Return whether any atom has a non-zero atom-map number.

        :param mol: RDKit molecule.
        :type mol: Chem.Mol
        :returns: ``True`` if mapped.
        :rtype: bool
        """
        return any(atom.GetAtomMapNum() != 0 for atom in mol.GetAtoms())

    @staticmethod
    def random_atom_mapping(mol: Chem.Mol) -> Chem.Mol:
        """Assign random atom-map numbers from ``1`` to ``n`` in-place.

        :param mol: RDKit molecule to mutate.
        :type mol: Chem.Mol
        :returns: Same molecule with assigned atom-map numbers.
        :rtype: Chem.Mol
        """
        indices = list(range(1, mol.GetNumAtoms() + 1))
        random.shuffle(indices)
        for atom, idx in zip(mol.GetAtoms(), indices):
            atom.SetAtomMapNum(idx)
        return mol

    @classmethod
    def mol_to_graph(
        cls,
        mol: Chem.Mol,
        drop_non_aam: bool = False,
        light_weight: bool = False,
        use_index_as_atom_map: bool = False,
    ) -> nx.Graph:
        """Backward-compatible graph converter.

        New code should usually prefer :meth:`transform`.

        :param mol: RDKit molecule.
        :type mol: Chem.Mol
        :param drop_non_aam: If ``True``, remove atoms with atom-map ``0``.
        :type drop_non_aam: bool
        :param light_weight: If ``True``, use reduced attributes.
        :type light_weight: bool
        :param use_index_as_atom_map: If ``True``, use atom maps as node IDs.
        :type use_index_as_atom_map: bool
        :returns: Molecular graph.
        :rtype: nx.Graph
        :raises ValueError: If ``drop_non_aam=True`` but
            ``use_index_as_atom_map=False``.

        .. code-block:: python

            mol = Chem.MolFromSmiles("[CH3:1][CH2:2][Br:3]")
            graph = MolToGraph.mol_to_graph(
                mol,
                drop_non_aam=True,
                light_weight=True,
                use_index_as_atom_map=True,
            )
        """
        if drop_non_aam and not use_index_as_atom_map:
            raise ValueError(
                "drop_non_aam and use_index_as_atom_map must be both False or both True."
            )

        if light_weight:
            return cls._create_light_weight_graph(
                mol,
                drop_non_aam=drop_non_aam,
                use_index_as_atom_map=use_index_as_atom_map,
            )

        return cls._create_detailed_graph(
            mol,
            drop_non_aam=drop_non_aam,
            use_index_as_atom_map=use_index_as_atom_map,
        )

    @classmethod
    def _create_light_weight_graph(
        cls,
        mol: Chem.Mol,
        drop_non_aam: bool = False,
        use_index_as_atom_map: bool = False,
    ) -> nx.Graph:
        """Create a lightweight graph with corrected lone-pair fields.

        Node attributes: ``element``, ``aromatic``, ``hcount``, ``charge``,
        ``radical``, ``neighbors``, ``atom_map``, ``oxidation_state``,
        ``available_lp``, ``lone_pairs``, ``valence_electrons``.
        Edge attributes: ``order``, ``bond_type``, ``aromatic``,
        ``kekule_order``, ``sigma_order``, ``pi_order``,
        ``kekule_bond_type``.

        :param mol: RDKit molecule.
        :type mol: Chem.Mol
        :param drop_non_aam: If ``True``, remove atoms with atom-map ``0``.
        :type drop_non_aam: bool
        :param use_index_as_atom_map: If ``True``, use atom-map numbers as node
            IDs for mapped atoms; unmapped atoms fall back to
            ``atom.GetIdx() + 1``.
        :type use_index_as_atom_map: bool
        :returns: Lightweight molecular graph.
        :rtype: nx.Graph
        """
        graph = nx.Graph()
        kek_mol: Optional[Chem.Mol] = cls._make_kekule_copy(mol)
        oxidation_states = cls.estimate_oxidation_states(mol, kek_mol=kek_mol)

        for atom in mol.GetAtoms():
            atom_map = cls._safe_atom_map(atom)
            atom_id = (
                atom_map
                if (use_index_as_atom_map and atom_map != 0)
                else atom.GetIdx() + 1
            )

            if drop_non_aam and atom_map == 0:
                continue

            try:
                neighbors = sorted(nb.GetSymbol() for nb in atom.GetNeighbors())
            except Exception:
                neighbors = []

            estimated_lone_pairs = cls.estimate_lone_pairs(atom)
            available_lone_pairs = cls.estimate_available_lone_pairs(atom)

            graph.add_node(
                atom_id,
                element=atom.GetSymbol(),
                aromatic=atom.GetIsAromatic(),
                hcount=atom.GetTotalNumHs(),
                charge=atom.GetFormalCharge(),
                radical=atom.GetNumRadicalElectrons(),
                neighbors=neighbors,
                atom_map=atom_map,
                oxidation_state=round(
                    float(oxidation_states.get(atom.GetIdx(), 0.0)), 3
                ),
                available_lp=available_lone_pairs > 0,
                lone_pairs=estimated_lone_pairs,
                valence_electrons=cls._safe_valence_electrons(atom),
            )

        for bond in mol.GetBonds():
            begin_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()

            begin_map = cls._safe_atom_map(begin_atom)
            end_map = cls._safe_atom_map(end_atom)

            if drop_non_aam and (begin_map == 0 or end_map == 0):
                continue

            begin_id = (
                begin_map
                if (use_index_as_atom_map and begin_map != 0)
                else begin_atom.GetIdx() + 1
            )
            end_id = (
                end_map
                if (use_index_as_atom_map and end_map != 0)
                else end_atom.GetIdx() + 1
            )

            kek_bond: Optional[Chem.Bond] = None
            if kek_mol is not None:
                try:
                    kek_bond = kek_mol.GetBondWithIdx(bond.GetIdx())
                except Exception:
                    kek_bond = None

            try:
                order = bond.GetBondTypeAsDouble()
            except Exception:
                order = 1.0

            try:
                aromatic = bond.GetIsAromatic()
            except Exception:
                aromatic = False

            try:
                bond_type = str(bond.GetBondType())
            except Exception:
                bond_type = "UNKNOWN"

            try:
                kekule_order = (
                    kek_bond.GetBondTypeAsDouble() if kek_bond is not None else order
                )
            except Exception:
                kekule_order = order

            try:
                kekule_bond_type = (
                    str(kek_bond.GetBondType()) if kek_bond is not None else bond_type
                )
            except Exception:
                kekule_bond_type = bond_type

            graph.add_edge(
                begin_id,
                end_id,
                order=order,
                bond_type=bond_type,
                aromatic=aromatic,
                kekule_order=kekule_order,
                kekule_bond_type=kekule_bond_type,
            )

        return graph

    @classmethod
    def _create_detailed_graph(
        cls,
        mol: Chem.Mol,
        drop_non_aam: bool = False,
        use_index_as_atom_map: bool = False,
    ) -> nx.Graph:
        """Create a detailed graph with fallback atom and bond attributes.

        :param mol: RDKit molecule.
        :type mol: Chem.Mol
        :param drop_non_aam: If ``True``, remove unmapped atoms.
        :type drop_non_aam: bool
        :param use_index_as_atom_map: If ``True``, use atom maps as node IDs.
        :type use_index_as_atom_map: bool
        :returns: Detailed molecular graph.
        :rtype: nx.Graph
        """
        try:
            compute_gasteiger_inplace(mol)
        except Exception:
            logger.debug("Gasteiger compute failed inside _create_detailed_graph.")

        graph = nx.Graph()
        idx_map: Dict[int, int] = {}
        kek_mol: Optional[Chem.Mol] = cls._make_kekule_copy(mol)
        oxidation_states = cls.estimate_oxidation_states(mol, kek_mol=kek_mol)

        for atom in mol.GetAtoms():
            atom_map = cls._safe_atom_map(atom)
            atom_id = (
                atom_map
                if (use_index_as_atom_map and atom_map != 0)
                else atom.GetIdx() + 1
            )

            if drop_non_aam and atom_map == 0:
                continue

            graph.add_node(
                atom_id,
                **cls._gather_atom_properties(
                    atom,
                    oxidation_state=oxidation_states.get(atom.GetIdx()),
                ),
            )
            idx_map[atom.GetIdx()] = atom_id

        for bond in mol.GetBonds():
            begin = idx_map.get(bond.GetBeginAtomIdx())
            end = idx_map.get(bond.GetEndAtomIdx())

            if begin is None or end is None:
                continue

            kek_bond: Optional[Chem.Bond] = None
            if kek_mol is not None:
                try:
                    kek_bond = kek_mol.GetBondWithIdx(bond.GetIdx())
                except Exception:
                    kek_bond = None

            graph.add_edge(
                begin,
                end,
                **cls._gather_bond_properties(bond, kek_bond=kek_bond),
            )

        return graph

    @staticmethod
    def add_partial_charges(mol: Chem.Mol) -> None:
        """Compute Gasteiger partial charges in-place.

        :param mol: RDKit molecule to modify.
        :type mol: Chem.Mol
        :returns: ``None``.
        :rtype: None
        """
        try:
            AllChem.ComputeGasteigerCharges(mol)
        except Exception as exc:
            logger.error("Error computing Gasteiger charges: %s", exc)
