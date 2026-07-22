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
from synkit.IO.mol_to_graph_chemistry import MolToGraphChemistryMixin

logger = setup_logging()


class MolToGraph(MolToGraphChemistryMixin):
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
    :param include_stereo_descriptors: If ``False``, omit the graph-level
        relative-stereochemistry registry. Defaults to ``True``.
    :type include_stereo_descriptors: bool
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
    AUGMENTED_NODE_ATTRS = frozenset(
        {
            "oxidation_state",
            "available_lp",
            "lone_pairs",
            "valence_electrons",
            "stereo_descriptor",
            "cip_label",
            "chiral_tag",
            "bond_order_sum",
            "lp_bond_order_sum",
            "estimated_lone_pairs",
            "available_lone_pairs",
        }
    )
    DIRECT_NODE_ATTRS = frozenset(
        {
            "element",
            "isotope",
            "aromatic",
            "hcount",
            "charge",
            "radical",
            "neighbors",
            "atom_map",
        }
    )
    KEKULE_EDGE_ATTRS = frozenset(
        {"kekule_order", "sigma_order", "pi_order", "kekule_bond_type"}
    )
    DIRECT_EDGE_ATTRS = frozenset({"order", "aromatic"})

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
        include_stereo_descriptors: bool = True,
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
        :param include_stereo_descriptors: Whether to construct the graph-level
            relative-stereochemistry registry.
        :type include_stereo_descriptors: bool
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
        self.include_stereo_descriptors: bool = bool(include_stereo_descriptors)

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

        needs_partial_charge = (
            self.node_attrs is None or "partial_charge" in self.node_attrs
        )
        if needs_partial_charge:
            try:
                compute_gasteiger_inplace(mol)
            except Exception:
                logger.debug("Gasteiger computation failed (best-effort). Continuing.")

        needs_oxidation_state = (
            self.node_attrs is None or "oxidation_state" in self.node_attrs
        )
        needs_kekule = (
            self.edge_attrs is None
            or bool(self.KEKULE_EDGE_ATTRS.intersection(self.edge_attrs))
            or needs_oxidation_state
        )
        kek_mol: Optional[Chem.Mol] = (
            self._make_kekule_copy(mol) if needs_kekule else None
        )
        oxidation_states = (
            self.estimate_oxidation_states(mol, kek_mol=kek_mol)
            if needs_oxidation_state
            else {}
        )

        per: Optional[PerMolDescriptors] = None
        if self.attr_profile == "full":
            try:
                per = PerMolDescriptors.compute(mol)
            except Exception as exc:
                logger.debug("PerMolDescriptors.compute failed: %s", exc)
                per = None

        extractor = AtomFeatureExtractor(mol, per=per, profile=self.attr_profile)
        needs_augmentation = self.node_attrs is None or bool(
            self.AUGMENTED_NODE_ATTRS.intersection(self.node_attrs)
        )
        requested_node_attrs = frozenset(self.node_attrs or ())
        use_direct_node_attrs = self.node_attrs is not None and (
            requested_node_attrs.issubset(self.DIRECT_NODE_ATTRS)
        )
        requested_edge_attrs = frozenset(self.edge_attrs or ())
        use_direct_edge_attrs = self.edge_attrs is not None and (
            requested_edge_attrs.issubset(self.DIRECT_EDGE_ATTRS)
        )

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

            if use_direct_node_attrs:
                props = self._gather_direct_atom_properties(atom, requested_node_attrs)
            else:
                try:
                    props = extractor.build_dict(atom)
                    if needs_augmentation:
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

            if use_direct_edge_attrs:
                bprops = self._gather_direct_bond_properties(bond, requested_edge_attrs)
            else:
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

        # Relative stereo is registry-based because descriptors depend on
        # multiple atoms. For ordinary unmapped reactor substrates, graph node
        # IDs (RDKit index + 1) provide stable internal references; external
        # atom maps remain reserved for mapped rules and serialized reactions.
        descriptors = {}
        if self.include_stereo_descriptors:
            try:
                from synkit.Graph.Stereo import descriptors_from_rdkit

                atom_maps = [int(atom.GetAtomMapNum()) for atom in mol.GetAtoms()]
                mapped = all(value > 0 for value in atom_maps) and len(
                    set(atom_maps)
                ) == len(atom_maps)
                descriptors = descriptors_from_rdkit(mol, require_atom_maps=mapped)
            except ImportError:
                descriptors = {}
        graph.graph["stereo_descriptors"] = descriptors
        node_by_map = {
            int(attrs.get("atom_map", 0)): node
            for node, attrs in graph.nodes(data=True)
            if int(attrs.get("atom_map", 0) or 0) > 0
        }
        for descriptor_id, descriptor in descriptors.items():
            if descriptor_id.startswith("atom:"):
                node = node_by_map.get(int(descriptor.center))
                if node is not None:
                    graph.nodes[node].setdefault("stereo_descriptor_ids", []).append(
                        descriptor_id
                    )
            else:
                left, right = (int(value) for value in descriptor.atoms[2:4])
                if (
                    left in node_by_map
                    and right in node_by_map
                    and graph.has_edge(node_by_map[left], node_by_map[right])
                ):
                    graph.edges[node_by_map[left], node_by_map[right]].setdefault(
                        "stereo_descriptor_ids", []
                    ).append(descriptor_id)

        if self.with_topology:
            try:
                GraphAnnotator(graph, in_place=True).annotate()
            except Exception as exc:
                logger.debug("GraphAnnotator failed: %s", exc)

        return graph

    @staticmethod
    def _gather_direct_atom_properties(
        atom: Chem.Atom,
        attributes: frozenset[str],
    ) -> Dict[str, Any]:
        """Read a requested subset of inexpensive RDKit atom properties."""
        props: Dict[str, Any] = {}
        if "element" in attributes:
            props["element"] = atom.GetSymbol()
        if "isotope" in attributes:
            props["isotope"] = atom.GetIsotope()
        if "aromatic" in attributes:
            props["aromatic"] = atom.GetIsAromatic()
        if "hcount" in attributes:
            props["hcount"] = atom.GetTotalNumHs()
        if "charge" in attributes:
            props["charge"] = atom.GetFormalCharge()
        if "radical" in attributes:
            props["radical"] = atom.GetNumRadicalElectrons()
        if "neighbors" in attributes:
            props["neighbors"] = sorted(
                neighbor.GetSymbol() for neighbor in atom.GetNeighbors()
            )
        if "atom_map" in attributes:
            props["atom_map"] = atom.GetAtomMapNum()
        return props

    @staticmethod
    def _gather_direct_bond_properties(
        bond: Chem.Bond,
        attributes: frozenset[str],
    ) -> Dict[str, Any]:
        """Read requested bond attributes without building discarded fields."""
        props: Dict[str, Any] = {}
        if "order" in attributes:
            props["order"] = bond.GetBondTypeAsDouble()
        if "aromatic" in attributes:
            props["aromatic"] = bond.GetIsAromatic()
        return props

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
            f"with_topology={self.with_topology}, "
            f"include_stereo_descriptors={self.include_stereo_descriptors}, "
            f"node_attrs={self.node_attrs!r}, "
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
        # v2 keeps relative stereo identity separate from derived display
        # labels.  Sprint 5 fills ``stereo_descriptor`` with an ordered,
        # map-based descriptor; keeping the field now avoids schema churn.
        new_props["stereo_descriptor"] = None
        new_props["cip_label"] = (
            atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else None
        )
        new_props["chiral_tag"] = cls.get_chiral_tag(atom)

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
            "isotope": atom.GetIsotope(),
            "aromatic": atom.GetIsAromatic(),
            "hcount": atom.GetTotalNumHs(),
            "charge": atom.GetFormalCharge(),
            "radical": atom.GetNumRadicalElectrons(),
            "stereo_descriptor": None,
            "cip_label": atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else None,
            "chiral_tag": MolToGraph.get_chiral_tag(atom),
            # Compatibility display field only; do not use as a graph-stereo
            # identity because it historically mapped RDKit CW/CCW to R/S.
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
            "stereo_descriptor": None,
            "ez_label": ez,
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
        """Return the legacy ``R``/``S`` display projection of a chiral tag.

        This compatibility helper is **not** a CIP assignment and must not be
        used as stereochemical identity.  New code should use
        :meth:`get_chiral_tag` until a v2 relative descriptor is available.

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
    def get_chiral_tag(atom: Chem.Atom) -> str:
        """Return RDKit's atom-order-relative tetrahedral tag.

        ``CW`` and ``CCW`` encode parity relative to atom ordering, not an
        absolute CIP label.  ``NONE`` covers atoms without a tetrahedral tag.
        """
        chiral_tag = atom.GetChiralTag()
        if chiral_tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
            return "CCW"
        if chiral_tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
            return "CW"
        return "NONE"

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
