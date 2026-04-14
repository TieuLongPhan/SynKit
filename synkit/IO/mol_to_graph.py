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
    """
    Convert an RDKit molecule into a :class:`networkx.Graph`.

    This class provides a flexible RDKit-to-NetworkX conversion layer with
    support for:

    - selective node and edge attributes
    - minimal or full atom feature profiles
    - optional topology annotation
    - backward-compatible one-shot graph creation
    - chainable graph construction and retrieval

    The default public workflow is :meth:`transform`, which directly returns a
    graph. For chainable usage, :meth:`transform_store` stores the graph on the
    instance and returns ``self``.

    Lone-pair-related information is exposed through heuristic atom-level
    attributes such as ``estimated_lone_pairs``, ``available_lp``,
    ``bond_order_sum``, and ``valence_electrons``. These are intended as useful
    approximate descriptors rather than exact quantum-mechanical quantities.

    Bond-level aromatic systems are represented using both the native RDKit
    aromatic form and an auxiliary kekulized form. As a result, an edge can
    expose:

    - ``order``: bond order from the original RDKit molecule
    - ``bond_type``: bond type from the original RDKit molecule
    - ``aromatic``: aromatic flag from the original RDKit molecule
    - ``kekule_order``: localized single/double bond order from a kekulized copy
    - ``kekule_bond_type``: localized bond type from a kekulized copy

    This allows downstream code to preserve aromaticity while also using an
    explicit resonance assignment as an extra feature.

    :param node_attrs:
        Optional list of node attribute names to retain. If ``None``, all
        computed node attributes are kept.
    :type node_attrs: Optional[List[str]]
    :param edge_attrs:
        Optional list of edge attribute names to retain. If ``None``, all
        computed edge attributes are kept.
    :type edge_attrs: Optional[List[str]]
    :param attr_profile:
        Attribute profile to use. Supported values are ``"minimal"`` and
        ``"full"``. The full profile attempts to compute
        :class:`PerMolDescriptors`.
    :type attr_profile: str
    :param with_topology:
        If ``True``, apply :class:`GraphAnnotator` after graph construction.
    :type with_topology: bool

    :raises ValueError:
        If ``attr_profile`` is not one of the supported profile names.

    .. note::
       Lone-pair estimation is based on simple valence-electron bookkeeping and
       may be inaccurate for hypervalent atoms, strongly delocalized systems,
       organometallics, or unusual valence states.

    **Example**

    .. code-block:: python

        from rdkit import Chem

        mol = Chem.MolFromSmiles("CC(=O)O")

        converter = MolToGraph(
            attr_profile="minimal",
            with_topology=False,
        )
        g = converter.transform(mol)

        print(g.number_of_nodes())
        print(g.number_of_edges())

    **Chainable example**

    .. code-block:: python

        from rdkit import Chem

        mol = Chem.MolFromSmiles("c1ccccc1O")

        converter = MolToGraph(attr_profile="full", with_topology=True)
        converter.transform_store(mol)
        g = converter.graph

        print(converter)
        print(g.nodes(data=True))
    """

    SUPPORTED_PROFILES = ("minimal", "full")

    def __init__(
        self,
        node_attrs: Optional[List[str]] = None,
        edge_attrs: Optional[List[str]] = None,
        *,
        attr_profile: str = "minimal",
        with_topology: bool = False,
    ) -> None:
        """
        Initialize a molecule-to-graph converter.

        :param node_attrs:
            Optional list of node attribute names to keep. If ``None``, all
            available node attributes are retained.
        :type node_attrs: Optional[List[str]]
        :param edge_attrs:
            Optional list of edge attribute names to keep. If ``None``, all
            available edge attributes are retained.
        :type edge_attrs: Optional[List[str]]
        :param attr_profile:
            Feature profile to use. Must be ``"minimal"`` or ``"full"``.
        :type attr_profile: str
        :param with_topology:
            Whether to apply graph topology annotation after graph creation.
        :type with_topology: bool

        :raises ValueError:
            If an unsupported ``attr_profile`` is provided.

        **Example**

        .. code-block:: python

            converter = MolToGraph(
                node_attrs=["element", "charge", "estimated_lone_pairs"],
                edge_attrs=["order", "bond_type", "kekule_order"],
                attr_profile="minimal",
                with_topology=False,
            )
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

    def transform(
        self,
        mol: Chem.Mol,
        drop_non_aam: bool = False,
        use_index_as_atom_map: bool = False,
    ) -> nx.Graph:
        """
        Build and return a :class:`networkx.Graph` from an RDKit molecule.

        This is the main backward-compatible entry point. It converts an RDKit
        molecule into a graph whose nodes represent atoms and whose edges
        represent bonds.

        Aromatic bonds are read from the original molecule, while localized bond
        orders are optionally recovered from a kekulized copy and exposed under
        ``kekule_order``.

        :param mol:
            RDKit molecule to convert.
        :type mol: Chem.Mol
        :param drop_non_aam:
            If ``True``, atoms with atom-map number ``0`` are excluded. This
            requires ``use_index_as_atom_map=True``.
        :type drop_non_aam: bool
        :param use_index_as_atom_map:
            If ``True``, use explicit atom-map numbers when present and non-zero.
            Otherwise, use ``atom index + 1`` as node identifiers.
        :type use_index_as_atom_map: bool

        :returns:
            Constructed molecular graph.
        :rtype: nx.Graph

        :raises ValueError:
            If ``drop_non_aam=True`` while ``use_index_as_atom_map=False``.

        **Example**

        .. code-block:: python

            from rdkit import Chem

            mol = Chem.MolFromSmiles("[CH3:1][OH:2]")
            g = MolToGraph().transform(
                mol,
                drop_non_aam=False,
                use_index_as_atom_map=True,
            )

            print(g.nodes(data=True))
            print(g.edges(data=True))
        """
        if drop_non_aam and not use_index_as_atom_map:
            raise ValueError(
                "drop_non_aam and use_index_as_atom_map must both be True to drop unmapped atoms."
            )

        self._last_mol = mol

        try:
            compute_gasteiger_inplace(mol)
        except Exception:
            logger.debug("Gasteiger computation failed (best-effort). Continuing.")

        kek_mol: Optional[Chem.Mol] = self._make_kekule_copy(mol)

        per: Optional[PerMolDescriptors] = None
        if self.attr_profile == "full":
            try:
                per = PerMolDescriptors.compute(mol)
            except Exception as exc:
                logger.debug("PerMolDescriptors.compute failed: %s", exc)
                per = None

        extractor = AtomFeatureExtractor(mol, per=per, profile=self.attr_profile)

        G = nx.Graph()
        index_to_id: Dict[int, int] = {}

        for atom in mol.GetAtoms():
            try:
                atom_map = atom.GetAtomMapNum()
            except Exception:
                atom_map = 0

            atom_id = (
                atom_map
                if (use_index_as_atom_map and atom_map != 0)
                else atom.GetIdx() + 1
            )

            if drop_non_aam and atom_map == 0:
                continue

            try:
                props = extractor.build_dict(atom)
                props = self._augment_atom_properties(atom, props)
            except Exception:
                props = self._gather_atom_properties(atom)

            if self.node_attrs is not None:
                props = {k: v for k, v in props.items() if k in self.node_attrs}

            G.add_node(atom_id, **props)
            index_to_id[atom.GetIdx()] = atom_id

        for bond in mol.GetBonds():
            b_idx = bond.GetBeginAtomIdx()
            e_idx = bond.GetEndAtomIdx()
            begin = index_to_id.get(b_idx)
            end = index_to_id.get(e_idx)
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
            except Exception:
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

            G.add_edge(begin, end, **bprops)

        if self.with_topology:
            try:
                GraphAnnotator(G, in_place=True).annotate()
            except Exception as exc:
                logger.debug("GraphAnnotator failed: %s", exc)

        return G

    def transform_store(
        self,
        mol: Chem.Mol,
        drop_non_aam: bool = False,
        use_index_as_atom_map: bool = False,
    ) -> "MolToGraph":
        """
        Build a graph from ``mol``, store it internally, and return ``self``.

        This method is useful for fluent or chainable workflows where the
        resulting graph is later accessed through the :attr:`graph` property.

        :param mol:
            RDKit molecule to convert.
        :type mol: Chem.Mol
        :param drop_non_aam:
            If ``True``, atoms with atom-map number ``0`` are excluded.
        :type drop_non_aam: bool
        :param use_index_as_atom_map:
            Whether to use atom-map numbers as node identifiers when available.
        :type use_index_as_atom_map: bool

        :returns:
            The current converter instance.
        :rtype: MolToGraph

        **Example**

        .. code-block:: python

            from rdkit import Chem

            mol = Chem.MolFromSmiles("CCN")

            converter = MolToGraph()
            converter.transform_store(mol)
            g = converter.graph
        """
        self._graph = self.transform(
            mol,
            drop_non_aam=drop_non_aam,
            use_index_as_atom_map=use_index_as_atom_map,
        )
        return self

    @property
    def graph(self) -> nx.Graph:
        """
        Return the last graph produced by :meth:`transform_store`.

        :returns:
            The stored molecular graph.
        :rtype: nx.Graph

        :raises RuntimeError:
            If no graph has been stored yet.

        **Example**

        .. code-block:: python

            converter = MolToGraph()
            converter.transform_store(mol)
            g = converter.graph
        """
        if self._graph is None:
            raise RuntimeError(
                "No graph produced yet. Call `transform_store(mol)` first."
            )
        return self._graph

    def __repr__(self) -> str:
        """
        Return a concise string representation of the converter state.

        :returns:
            String representation including profile, topology option, attribute
            filters, and last graph size.
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
        """
        Return a short machine-readable help string for this class.

        :returns:
            A compact usage summary.
        :rtype: str
        """
        return (
            "MolToGraph.help() -> str\n\n"
            "Create with MolToGraph(node_attrs=[...], edge_attrs=[...], "
            "attr_profile='minimal'|'full', with_topology=False).\n"
            "Use `.transform(mol)` to get an nx.Graph (backwards-compatible),\n"
            "or `.transform_store(mol)` to build and store the graph on the instance\n"
            "and then retrieve it via `.graph` (chainable)."
        )

    @staticmethod
    def _safe_bond_order_sum(atom: Chem.Atom) -> float:
        """
        Return the sum of bond orders around an atom.

        :param atom:
            Atom whose incident bond orders will be summed.
        :type atom: Chem.Atom

        :returns:
            Sum of bond orders.
        :rtype: float
        """
        try:
            return float(sum(bond.GetBondTypeAsDouble() for bond in atom.GetBonds()))
        except Exception:
            return 0.0

    @staticmethod
    def _safe_valence_electrons(atom: Chem.Atom) -> int:
        """
        Return the number of valence electrons for an atom.

        :param atom:
            Atom whose outer-shell electron count will be estimated from the
            periodic table.
        :type atom: Chem.Atom

        :returns:
            Number of outer-shell electrons.
        :rtype: int
        """
        try:
            pt = Chem.GetPeriodicTable()
            return int(pt.GetNOuterElecs(atom.GetAtomicNum()))
        except Exception:
            return 0

    @staticmethod
    def _make_kekule_copy(mol: Chem.Mol) -> Optional[Chem.Mol]:
        """
        Return a kekulized copy of ``mol``.

        The returned copy has aromatic bond flags cleared and explicit localized
        single/double bond assignments. This helper is used to derive
        ``kekule_order`` and ``kekule_bond_type`` while preserving aromatic
        information from the original molecule.

        :param mol:
            Input molecule.
        :type mol: Chem.Mol

        :returns:
            Kekulized copy of the molecule, or ``None`` if kekulization fails.
        :rtype: Optional[Chem.Mol]

        **Example**

        .. code-block:: python

            from rdkit import Chem

            mol = Chem.MolFromSmiles("c1ccccc1")
            kek = MolToGraph._make_kekule_copy(mol)

            if kek is not None:
                for bond in kek.GetBonds():
                    print(bond.GetBondType(), bond.GetBondTypeAsDouble())
        """
        try:
            kek = Chem.Mol(mol)
            Chem.Kekulize(kek, clearAromaticFlags=True)
            return kek
        except Exception as exc:
            logger.debug("Failed to create kekulized copy: %s", exc)
            return None

    @classmethod
    def estimate_lone_pairs(cls, atom: Chem.Atom) -> int:
        """
        Estimate the number of lone pairs on an atom.

        The estimate is based on simple valence bookkeeping:

        .. code-block:: text

            nonbonding_electrons =
                valence_electrons
                - formal_charge
                - radical_electrons
                - bond_order_sum
                - total_hcount

            lone_pairs = floor(nonbonding_electrons / 2)

        :param atom:
            Atom for which lone pairs will be estimated.
        :type atom: Chem.Atom

        :returns:
            Estimated number of lone pairs.
        :rtype: int

        .. note::
           This estimate is heuristic and may be inaccurate for aromatic,
           resonance-delocalized, hypervalent, or otherwise nonclassical atoms.

        **Example**

        .. code-block:: python

            mol = Chem.MolFromSmiles("O")
            atom = mol.GetAtomWithIdx(0)
            lp = MolToGraph.estimate_lone_pairs(atom)
            print(lp)
        """
        try:
            valence_electrons = cls._safe_valence_electrons(atom)
            formal_charge = int(atom.GetFormalCharge())
            radical_electrons = int(atom.GetNumRadicalElectrons())
            bond_order_sum = int(round(cls._safe_bond_order_sum(atom)))
            total_hcount = int(atom.GetTotalNumHs())

            nonbonding_electrons = (
                valence_electrons
                - formal_charge
                - radical_electrons
                - bond_order_sum
                - total_hcount
            )
            return max(0, nonbonding_electrons // 2)
        except Exception:
            return 0

    @classmethod
    def _augment_atom_properties(
        cls,
        atom: Chem.Atom,
        props: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Add lone-pair and electron bookkeeping fields to an atom property dict.

        :param atom:
            Atom used to compute additional properties.
        :type atom: Chem.Atom
        :param props:
            Existing atom property dictionary.
        :type props: Dict[str, Any]

        :returns:
            Enriched atom property dictionary.
        :rtype: Dict[str, Any]
        """
        new_props = dict(props)

        bond_order_sum = cls._safe_bond_order_sum(atom)
        valence_electrons = cls._safe_valence_electrons(atom)
        estimated_lone_pairs = cls.estimate_lone_pairs(atom)

        new_props.setdefault("bond_order_sum", round(bond_order_sum, 3))
        new_props.setdefault("valence_electrons", valence_electrons)
        new_props.setdefault("estimated_lone_pairs", estimated_lone_pairs)
        new_props.setdefault("available_lp", estimated_lone_pairs > 0)
        new_props.setdefault("lone_pairs", estimated_lone_pairs)

        return new_props

    @staticmethod
    def _gather_atom_properties(atom: Chem.Atom) -> Dict[str, Any]:
        """
        Collect a full set of atom-level graph node attributes.

        This is a legacy-compatible helper used as a fallback when the feature
        extractor path is unavailable or fails.

        :param atom:
            Atom to describe.
        :type atom: Chem.Atom

        :returns:
            Dictionary of atom properties.
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

        try:
            atom_map = atom.GetAtomMapNum()
        except Exception:
            atom_map = 0

        bond_order_sum = round(MolToGraph._safe_bond_order_sum(atom), 3)
        valence_electrons = MolToGraph._safe_valence_electrons(atom)
        estimated_lone_pairs = MolToGraph.estimate_lone_pairs(atom)

        return {
            "element": atom.GetSymbol(),
            "aromatic": atom.GetIsAromatic(),
            "hcount": atom.GetTotalNumHs(),
            "charge": atom.GetFormalCharge(),
            "radical": atom.GetNumRadicalElectrons(),
            "isomer": MolToGraph.get_stereochemistry(atom),
            "partial_charge": gcharge,
            "hybridization": str(atom.GetHybridization()),
            "in_ring": atom.IsInRing(),
            "implicit_hcount": atom.GetNumImplicitHs(),
            "neighbors": neighbors,
            "atom_map": atom_map,
            "bond_order_sum": bond_order_sum,
            "valence_electrons": valence_electrons,
            "estimated_lone_pairs": estimated_lone_pairs,
            "available_lp": estimated_lone_pairs > 0,
            "lone_pairs": estimated_lone_pairs,
        }

    @staticmethod
    def _gather_bond_properties(
        bond: Chem.Bond,
        kek_bond: Optional[Chem.Bond] = None,
    ) -> Dict[str, Any]:
        """
        Collect a full set of bond-level graph edge attributes.

        The input ``bond`` is read from the original molecule, so native RDKit
        aromaticity and aromatic bond typing are preserved. If ``kek_bond`` is
        provided, it is assumed to be the corresponding bond from a kekulized
        copy of the same molecule and is used to expose localized bond
        information.

        :param bond:
            Bond from the original molecule.
        :type bond: Chem.Bond
        :param kek_bond:
            Matching bond from a kekulized copy of the molecule.
        :type kek_bond: Optional[Chem.Bond]

        :returns:
            Dictionary of bond properties.
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
            conj = bond.GetIsConjugated()
        except Exception:
            conj = False

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

        return {
            "order": order,
            "bond_type": bond_type,
            "aromatic": aromatic,
            "kekule_order": kekule_order,
            "kekule_bond_type": kekule_bond_type,
            "ez_isomer": ez,
            "conjugated": conj,
            "in_ring": in_ring,
        }

    @staticmethod
    def get_stereochemistry(atom: Chem.Atom) -> str:
        """
        Return a simple atom stereochemistry label.

        :param atom:
            Atom to inspect.
        :type atom: Chem.Atom

        :returns:
            ``"S"``, ``"R"``, or ``"N"`` if not assigned.
        :rtype: str
        """
        ch = atom.GetChiralTag()
        if ch == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
            return "S"
        if ch == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
            return "R"
        return "N"

    @staticmethod
    def get_bond_stereochemistry(bond: Chem.Bond) -> str:
        """
        Return a simple double-bond stereochemistry label.

        :param bond:
            Bond to inspect.
        :type bond: Chem.Bond

        :returns:
            ``"E"``, ``"Z"``, or ``"N"`` if not applicable or not assigned.
        :rtype: str
        """
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            return "N"
        st = bond.GetStereo()
        if st == Chem.BondStereo.STEREOE:
            return "E"
        if st == Chem.BondStereo.STEREOZ:
            return "Z"
        return "N"

    @staticmethod
    def has_atom_mapping(mol: Chem.Mol) -> bool:
        """
        Check whether a molecule contains at least one non-zero atom-map number.

        :param mol:
            Molecule to inspect.
        :type mol: Chem.Mol

        :returns:
            ``True`` if any atom has a non-zero atom-map number.
        :rtype: bool
        """
        return any(atom.GetAtomMapNum() != 0 for atom in mol.GetAtoms())

    @staticmethod
    def random_atom_mapping(mol: Chem.Mol) -> Chem.Mol:
        """
        Assign random atom-map numbers from ``1..n`` to all atoms in a molecule.

        The molecule is modified in place and also returned for convenience.

        :param mol:
            Molecule to modify.
        :type mol: Chem.Mol

        :returns:
            The same molecule instance with updated atom-map numbers.
        :rtype: Chem.Mol

        **Example**

        .. code-block:: python

            mol = Chem.MolFromSmiles("CCO")
            mol = MolToGraph.random_atom_mapping(mol)

            for atom in mol.GetAtoms():
                print(atom.GetIdx(), atom.GetAtomMapNum())
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
        """
        Backward-compatible high-level graph converter.

        This method mirrors the previous public API and provides either a
        lightweight or detailed graph representation.

        :param mol:
            Molecule to convert.
        :type mol: Chem.Mol
        :param drop_non_aam:
            If ``True``, drop atoms whose atom-map number is zero.
        :type drop_non_aam: bool
        :param light_weight:
            If ``True``, build a lightweight graph with reduced attributes.
            Otherwise, build the detailed legacy graph.
        :type light_weight: bool
        :param use_index_as_atom_map:
            Whether to use atom-map numbers as node identifiers when present.
        :type use_index_as_atom_map: bool

        :returns:
            Constructed molecular graph.
        :rtype: nx.Graph

        :raises ValueError:
            If ``drop_non_aam=True`` while ``use_index_as_atom_map=False``.

        **Example**

        .. code-block:: python

            mol = Chem.MolFromSmiles("CCO")

            g1 = MolToGraph.mol_to_graph(mol, light_weight=True)
            g2 = MolToGraph.mol_to_graph(mol, light_weight=False)
        """
        if drop_non_aam and not use_index_as_atom_map:
            raise ValueError(
                "drop_non_aam and use_index_as_atom_map must be both False or both True."
            )
        if light_weight:
            return cls._create_light_weight_graph(
                mol, drop_non_aam, use_index_as_atom_map
            )
        return cls._create_detailed_graph(mol, drop_non_aam, use_index_as_atom_map)

    @classmethod
    def _create_light_weight_graph(
        cls,
        mol: Chem.Mol,
        drop_non_aam: bool = False,
        use_index_as_atom_map: bool = False,
    ) -> nx.Graph:
        """
        Create a lightweight graph with reduced atom and bond attributes.

        This lightweight representation now also exposes ``aromatic`` and
        ``kekule_order`` for bonds when available.

        :param mol:
            Molecule to convert.
        :type mol: Chem.Mol
        :param drop_non_aam:
            Whether to exclude atoms with zero atom-map number.
        :type drop_non_aam: bool
        :param use_index_as_atom_map:
            Whether to prefer atom-map numbers over ``atom index + 1``.
        :type use_index_as_atom_map: bool

        :returns:
            Lightweight graph representation.
        :rtype: nx.Graph
        """
        G = nx.Graph()
        kek_mol: Optional[Chem.Mol] = cls._make_kekule_copy(mol)

        for atom in mol.GetAtoms():
            try:
                atom_map = atom.GetAtomMapNum()
            except Exception:
                atom_map = 0

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

            G.add_node(
                atom_id,
                element=atom.GetSymbol(),
                aromatic=atom.GetIsAromatic(),
                hcount=atom.GetTotalNumHs(),
                charge=atom.GetFormalCharge(),
                neighbors=neighbors,
                atom_map=atom_map,
                bond_order_sum=round(cls._safe_bond_order_sum(atom), 3),
                valence_electrons=cls._safe_valence_electrons(atom),
                estimated_lone_pairs=estimated_lone_pairs,
                available_lp=estimated_lone_pairs > 0,
                lone_pairs=estimated_lone_pairs,
            )

            for bond in atom.GetBonds():
                nbr = bond.GetOtherAtom(atom)
                try:
                    nbr_id = (
                        nbr.GetAtomMapNum()
                        if use_index_as_atom_map and nbr.GetAtomMapNum() != 0
                        else nbr.GetIdx() + 1
                    )
                except Exception:
                    nbr_id = nbr.GetIdx() + 1

                if drop_non_aam and nbr.GetAtomMapNum() == 0:
                    continue

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
                    kekule_order = (
                        kek_bond.GetBondTypeAsDouble()
                        if kek_bond is not None
                        else order
                    )
                except Exception:
                    kekule_order = order

                G.add_edge(
                    atom_id,
                    nbr_id,
                    order=order,
                    aromatic=aromatic,
                    kekule_order=kekule_order,
                )
        return G

    @classmethod
    def _create_detailed_graph(
        cls,
        mol: Chem.Mol,
        drop_non_aam: bool = True,
        use_index_as_atom_map: bool = True,
    ) -> nx.Graph:
        """
        Create a detailed graph with full atom and bond attributes.

        This method implements the older detailed conversion flow used by the
        legacy API.

        :param mol:
            Molecule to convert.
        :type mol: Chem.Mol
        :param drop_non_aam:
            Whether to exclude atoms with zero atom-map number.
        :type drop_non_aam: bool
        :param use_index_as_atom_map:
            Whether to prefer atom-map numbers over ``atom index + 1``.
        :type use_index_as_atom_map: bool

        :returns:
            Detailed graph representation.
        :rtype: nx.Graph
        """
        try:
            compute_gasteiger_inplace(mol)
        except Exception:
            logger.debug("Gasteiger compute failed inside _create_detailed_graph.")

        G = nx.Graph()
        idx_map: Dict[int, int] = {}
        kek_mol: Optional[Chem.Mol] = cls._make_kekule_copy(mol)

        for atom in mol.GetAtoms():
            try:
                atom_map = atom.GetAtomMapNum()
            except Exception:
                atom_map = 0

            atom_id = (
                atom_map
                if (use_index_as_atom_map and atom_map != 0)
                else atom.GetIdx() + 1
            )

            if drop_non_aam and atom_map == 0:
                continue

            G.add_node(atom_id, **cls._gather_atom_properties(atom))
            idx_map[atom.GetIdx()] = atom_id

        for bond in mol.GetBonds():
            b = idx_map.get(bond.GetBeginAtomIdx())
            e = idx_map.get(bond.GetEndAtomIdx())
            if b is None or e is None:
                continue

            kek_bond: Optional[Chem.Bond] = None
            if kek_mol is not None:
                try:
                    kek_bond = kek_mol.GetBondWithIdx(bond.GetIdx())
                except Exception:
                    kek_bond = None

            G.add_edge(b, e, **cls._gather_bond_properties(bond, kek_bond=kek_bond))

        return G

    @staticmethod
    def add_partial_charges(mol: Chem.Mol) -> None:
        """
        Compute and assign Gasteiger partial charges to a molecule in place.

        :param mol:
            Molecule to modify.
        :type mol: Chem.Mol

        :returns:
            ``None``. The molecule is modified in place.
        :rtype: None

        **Example**

        .. code-block:: python

            mol = Chem.MolFromSmiles("CCO")
            MolToGraph.add_partial_charges(mol)

            for atom in mol.GetAtoms():
                if atom.HasProp("_GasteigerCharge"):
                    print(atom.GetProp("_GasteigerCharge"))
        """
        try:
            AllChem.ComputeGasteigerCharges(mol)
        except Exception as e:
            logger.error("Error computing Gasteiger charges: %s", e)
