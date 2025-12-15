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
    RDKit -> NetworkX converter with attribute selection and optional topology
    annotation.

    Backwards-compatibility: the primary method ``transform(...)`` returns an
    ``networkx.Graph`` (default behavior used in older pipeline):

        g = MolToGraph().transform(mol)

    Chainable alternative: ``transform_store(...)`` builds and stores the graph
    on the instance, returning ``self``. Retrieve the stored graph via ``.graph``.

    Parameters
    ----------
    node_attrs:
        Optional list of node attribute keys to keep. If ``None`` (default),
        all computed node attributes are retained.
    edge_attrs:
        Optional list of edge attribute keys to keep. If ``None`` (default),
        all computed edge attributes are retained.
    attr_profile:
        ``'minimal'`` (default) or ``'full'``. Full profile computes PerMolDescriptors.
    with_topology:
        If True, run :class:`GraphAnnotator` after building the node/edge data.
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
        if attr_profile not in self.SUPPORTED_PROFILES:
            raise ValueError(
                f"Unsupported attr_profile: {attr_profile!r}. "
                f"Supported: {self.SUPPORTED_PROFILES}"
            )
        # None means "keep all" for backward compatibility
        self.node_attrs: Optional[List[str]] = (
            None if node_attrs is None else list(node_attrs)
        )
        self.edge_attrs: Optional[List[str]] = (
            None if edge_attrs is None else list(edge_attrs)
        )

        self.attr_profile: str = attr_profile
        self.with_topology: bool = bool(with_topology)

        # internal state (used by transform_store / chainable API)
        self._graph: Optional[nx.Graph] = None
        self._last_mol: Optional[Chem.Mol] = None

    # ------------------------------------------------------------------ #
    # Backwards-compatible transform: returns nx.Graph (default)
    # ------------------------------------------------------------------ #
    def transform(
        self,
        mol: Chem.Mol,
        drop_non_aam: bool = False,
        use_index_as_atom_map: bool = False,
    ) -> nx.Graph:
        """
        Build and return a NetworkX graph from ``mol`` (backwards-compatible).

        Parameters
        ----------
        mol:
            RDKit Mol to convert.
        drop_non_aam:
            If True, drop atoms with atom-map number == 0. Must be used with
            ``use_index_as_atom_map=True``.
        use_index_as_atom_map:
            If True, treat non-zero atom-map numbers as explicit IDs; otherwise
            use atom index + 1.

        Returns
        -------
        nx.Graph
            The constructed graph (nodes have attribute dicts).
        """
        if drop_non_aam and not use_index_as_atom_map:
            raise ValueError(
                "drop_non_aam and use_index_as_atom_map must both be True to drop unmapped atoms."
            )

        # Keep a reference for debugging
        self._last_mol = mol

        # Ensure minimal-profile expectations (Gasteiger) — best-effort
        try:
            compute_gasteiger_inplace(mol)
        except Exception:
            logger.debug("Gasteiger computation failed (best-effort). Continuing.")

        # Per-mol descriptors only when full requested
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

        # Nodes
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

            # Try extractor first (preferred), fallback to legacy helper
            try:
                props = extractor.build_dict(atom)
            except Exception:
                props = self._gather_atom_properties(atom)

            # apply selection filter if provided (None means keep all)
            if self.node_attrs is not None:
                props = {k: v for k, v in props.items() if k in self.node_attrs}

            G.add_node(atom_id, **props)
            index_to_id[atom.GetIdx()] = atom_id

        # Edges
        for bond in mol.GetBonds():
            b_idx = bond.GetBeginAtomIdx()
            e_idx = bond.GetEndAtomIdx()
            begin = index_to_id.get(b_idx)
            end = index_to_id.get(e_idx)
            if begin is None or end is None:
                continue
            try:
                bprops = self._gather_bond_properties(bond)
            except Exception:
                bprops = {
                    "order": (
                        bond.GetBondTypeAsDouble()
                        if hasattr(bond, "GetBondTypeAsDouble")
                        else 1.0
                    )
                }
            if self.edge_attrs is not None:
                bprops = {k: v for k, v in bprops.items() if k in self.edge_attrs}
            G.add_edge(begin, end, **bprops)

        # Optional topology annotations (explicit opt-in)
        if self.with_topology:
            try:
                GraphAnnotator(G, in_place=True).annotate()
            except Exception as exc:
                logger.debug("GraphAnnotator failed: %s", exc)

        # Return the graph (default, backwards-compatible behavior)
        return G

    # ------------------------------------------------------------------ #
    # Chainable alternative: build/store and return self
    # ------------------------------------------------------------------ #
    def transform_store(
        self,
        mol: Chem.Mol,
        drop_non_aam: bool = False,
        use_index_as_atom_map: bool = False,
    ) -> "MolToGraph":
        """
        Build the graph and store it on ``self._graph``. Returns ``self`` to
        enable chainable usage. Retrieve the graph via the ``.graph`` property.
        """
        self._graph = self.transform(
            mol, drop_non_aam=drop_non_aam, use_index_as_atom_map=use_index_as_atom_map
        )
        return self

    # ------------------------------------------------------------------ #
    # Accessors / metadata
    # ------------------------------------------------------------------ #
    @property
    def graph(self) -> nx.Graph:
        """
        Return the last graph produced by ``transform_store`` (chainable flow).
        Raises ``RuntimeError`` if none exists.
        """
        if self._graph is None:
            raise RuntimeError(
                "No graph produced yet. Call `transform_store(mol)` first."
            )
        return self._graph

    def __repr__(self) -> str:
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
        """Short machine-readable help describing the converter options."""
        return (
            "MolToGraph.help() -> str\n\n"
            "Create with MolToGraph(node_attrs=[...], edge_attrs=[...],"
            " attr_profile='minimal'|'full', with_topology=False).\n"
            "Use `.transform(mol)` to get an nx.Graph (backwards-compatible),\n"
            "or `.transform_store(mol)` to build and store the graph on the instance\n"
            "and then retrieve it via `.graph` (chainable)."
        )

    # ------------------------------------------------------------------ #
    # Backwards-compatible helpers (kept for API stability)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _gather_atom_properties(atom: Chem.Atom) -> Dict[str, Any]:
        """Collect the full set of atom attributes for graph nodes (legacy helper)."""
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
        }

    @staticmethod
    def _gather_bond_properties(bond: Chem.Bond) -> Dict[str, Any]:
        """Collect the full set of bond attributes for graph edges (legacy helper)."""
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

        return {
            "order": order,
            "bond_type": bond_type,
            "ez_isomer": ez,
            "conjugated": conj,
            "in_ring": in_ring,
        }

    # ------------------------------------------------------------------ #
    # Small compatibility utilities
    # ------------------------------------------------------------------ #
    @staticmethod
    def get_stereochemistry(atom: Chem.Atom) -> str:
        """Return 'S', 'R' or 'N' for chiral atoms."""
        ch = atom.GetChiralTag()
        if ch == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
            return "S"
        if ch == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
            return "R"
        return "N"

    @staticmethod
    def get_bond_stereochemistry(bond: Chem.Bond) -> str:
        """Return 'E', 'Z' or 'N' for double bond stereochemistry."""
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
        """Return True if any atom has a non-zero atom-map number."""
        return any(atom.GetAtomMapNum() != 0 for atom in mol.GetAtoms())

    @staticmethod
    def random_atom_mapping(mol: Chem.Mol) -> Chem.Mol:
        """Assign random atom mapping numbers (1..n) and return the mutated molecule."""
        indices = list(range(1, mol.GetNumAtoms() + 1))
        random.shuffle(indices)
        for atom, idx in zip(mol.GetAtoms(), indices):
            atom.SetAtomMapNum(idx)
        return mol

    # ------------------------------------------------------------------ #
    # Legacy convenience: create light-weight or detailed graph directly
    # ------------------------------------------------------------------ #
    @classmethod
    def mol_to_graph(
        cls,
        mol: Chem.Mol,
        drop_non_aam: bool = False,
        light_weight: bool = False,
        use_index_as_atom_map: bool = False,
    ) -> nx.Graph:
        """
        Backwards-compatible high-level converter returning an nx.Graph.
        Mirrors previous public API.
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
        """Create a lightweight graph with basic atom and bond info."""
        G = nx.Graph()
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
            G.add_node(
                atom_id,
                element=atom.GetSymbol(),
                aromatic=atom.GetIsAromatic(),
                hcount=atom.GetTotalNumHs(),
                charge=atom.GetFormalCharge(),
                neighbors=neighbors,
                atom_map=atom_map,
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
                if not drop_non_aam or nbr.GetAtomMapNum() != 0:
                    try:
                        order = bond.GetBondTypeAsDouble()
                    except Exception:
                        order = 1.0
                    G.add_edge(atom_id, nbr_id, order=order)
        return G

    @classmethod
    def _create_detailed_graph(
        cls,
        mol: Chem.Mol,
        drop_non_aam: bool = True,
        use_index_as_atom_map: bool = True,
    ) -> nx.Graph:
        """Create a detailed graph with full atom and bond attributes (legacy flow)."""
        try:
            compute_gasteiger_inplace(mol)
        except Exception:
            logger.debug("Gasteiger compute failed inside _create_detailed_graph.")

        G = nx.Graph()
        idx_map: Dict[int, int] = {}
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
            if b and e:
                G.add_edge(b, e, **cls._gather_bond_properties(bond))
        return G

    # ------------------------------------------------------------------ #
    # Convenience: partial charges with logging
    # ------------------------------------------------------------------ #
    @staticmethod
    def add_partial_charges(mol: Chem.Mol) -> None:
        """Compute and assign Gasteiger charges to all atoms in the molecule."""
        try:
            AllChem.ComputeGasteigerCharges(mol)
        except Exception as e:
            logger.error("Error computing Gasteiger charges: %s", e)
