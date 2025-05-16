import inspect
import networkx as nx
from rdkit import Chem
from rdkit.Chem import MolFromSmiles as _rdkit_MolFromSmiles
from typing import List, Tuple, Union, Optional

from synkit.Graph.canon_graph import GraphCanonicaliser
from synkit.IO import graph_to_smi, rsmi_to_graph


class CanonRSMI:
    """
    A **pure-Python / pure-NetworkX** utility for canonicalizing reaction SMILES
    by expanding atom-maps and deterministically reindexing reaction graphs.

    Workflow
    --------
    1. **Expand atom-maps** on reactants to ensure each atom has a unique map ID.
    2. **Convert** reaction SMILES to reactant/product NetworkX graphs.
    3. **Canonicalize** the reactant graph using `GraphCanonicaliser` (generic or WL backend).
    4. **Match** atom-map IDs to compute pairwise indices between reactants and products.
    5. **Remap** the product graph to align with the canonical reactant ordering.
    6. **Sync** each node’s `atom_map` attribute to its new graph index.
    7. **Reassemble** the reaction SMILES from the canonical graphs.

    Classes
    -------
    - `CanonRSMI` – Main interface for transforming any `reactants>>products` SMILES
      into a canonicalized form, preserving all node and edge attributes.

    Example
    -------
    >>> from canonical_rsm import CanonRSMI
    >>> canon = CanonRSMI(backend='wl', wl_iterations=5)
    >>> result = canon.canonicalise('[CH3:3][CH2:5][OH:10]>>[CH2:3]=[CH2:5].[OH2:10]')
    >>> print(result.canonical_rsmi)
    [OH:1][CH2:3][CH3:2]>>[CH2:2]=[CH2:3].[OH2:1]
    """

    def __init__(
        self,
        backend: str = "wl",
        wl_iterations: int = 3,
        morgan_radius: int = 3,
        node_attrs: List[str] = ("element", "aromatic", "charge", "hcount"),
    ):
        self._canon = GraphCanonicaliser(
            backend=backend,
            wl_iterations=wl_iterations,
            morgan_radius=morgan_radius,
            node_attrs=node_attrs,
        )
        # internal storage
        self._raw_rsmi: Optional[str] = None
        self._canon_rsmi: Optional[str] = None
        self._raw_reactant_graph: Optional[nx.Graph] = None
        self._raw_product_graph: Optional[nx.Graph] = None
        self._canon_reactant_graph: Optional[nx.Graph] = None
        self._canon_product_graph: Optional[nx.Graph] = None
        self._mapping_pairs: Optional[List[Tuple[int, int]]] = None

    @staticmethod
    def _mol_from_smiles(smi: str) -> Chem.Mol:
        """
        RDKit MolFromSmiles with explicit sanitize step.
        """
        mol = _rdkit_MolFromSmiles(smi, sanitize=False)
        Chem.SanitizeMol(mol)
        return mol

    def expand_aam(self, rsmi: str) -> str:
        """
        Assign new atom-map IDs to unmapped reactant atoms in 'reactants>>products' SMILES.
        New IDs start at max(existing maps)+1.
        """
        try:
            reac_s, prod_s = rsmi.split(">>")
        except ValueError:
            raise ValueError("Input must be in 'reactants>>products' form")

        reac_mols = [self._mol_from_smiles(s) for s in reac_s.split(".")]
        prod_mols = [self._mol_from_smiles(s) for s in prod_s.split(".")]

        existing = {
            atom.GetAtomMapNum()
            for mol in (*reac_mols, *prod_mols)
            for atom in mol.GetAtoms()
            if atom.GetAtomMapNum() > 0
        }
        next_id = max(existing, default=0) + 1

        for mol in reac_mols:
            for atom in mol.GetAtoms():
                if atom.GetAtomMapNum() == 0:
                    atom.SetAtomMapNum(next_id)
                    next_id += 1

        new_reac = ".".join(Chem.MolToSmiles(m, canonical=True) for m in reac_mols)
        new_prod = ".".join(Chem.MolToSmiles(m, canonical=True) for m in prod_mols)
        return f"{new_reac}>>{new_prod}"

    @staticmethod
    def sync_atom_map_with_index(G: nx.Graph) -> None:
        """
        In-place: set each node's 'atom_map' attribute to its node ID.
        """
        nx.set_node_attributes(G, {n: n for n in G.nodes()}, "atom_map")

    @staticmethod
    def get_aam_pairwise_indices(
        G: nx.Graph, H: nx.Graph, aam_key: str = "atom_map"
    ) -> List[Tuple[int, int]]:
        """
        Return sorted list of (G_node, H_node) for shared atom-map IDs.
        """
        gmap = {
            data[aam_key]: n
            for n, data in G.nodes(data=True)
            if data.get(aam_key, 0) > 0
        }
        hmap = {
            data[aam_key]: n
            for n, data in H.nodes(data=True)
            if data.get(aam_key, 0) > 0
        }
        common = sorted(gmap.keys() & hmap.keys())
        return [(gmap[k], hmap[k]) for k in common]

    @staticmethod
    def remap_graph(
        G: nx.Graph, node_map: Union[List[int], List[Tuple[int, int]]]
    ) -> nx.Graph:
        """
        Remap a product graph to match a canonical reactant ordering:

        :param G: reactant graph
        :type G: nx.Graph
        :param mapping:
            mapping from old product node IDs to new IDs
        :type mapping: dict[int,int]
        :returns:
            remapped product graph
        :rtype: nx.Graph
        """

        if not node_map:
            raise ValueError("node_map must be non-empty")
        if isinstance(node_map[0], int):
            mapping = {old: new for new, old in enumerate(node_map, start=1)}
        else:
            mapping = {old: new for new, old in node_map}
        missing = set(mapping) - set(G.nodes())
        if missing:
            raise KeyError(f"Mappings refer to nodes not in G: {missing}")
        return nx.relabel_nodes(G, mapping, copy=True)

    def canonicalise(self, rsmi: str) -> "CanonRSMI":
        """
        Full pipeline returning self with properties populated:
          - raw_rsmi
          - raw_reactant_graph, raw_product_graph
          - mapping_pairs
          - canonical_reactant_graph, canonical_product_graph
          - canonical_rsmi
        """
        # store raw
        self._raw_rsmi = rsmi
        # expand and build graphs
        expanded = self.expand_aam(rsmi)
        self._raw_reactant_graph, self._raw_product_graph = rsmi_to_graph(expanded)
        # canonicalise reactants
        self._canon_reactant_graph = self._canon.canonicalise_graph(
            self._raw_reactant_graph
        ).canonical_graph
        # map products
        self._mapping_pairs = self.get_aam_pairwise_indices(
            self._canon_reactant_graph, self._raw_product_graph
        )
        self._canon_product_graph = self.remap_graph(
            self._raw_product_graph, self._mapping_pairs
        )
        # sync maps
        self.sync_atom_map_with_index(self._canon_reactant_graph)
        self.sync_atom_map_with_index(self._canon_product_graph)
        # assemble output
        self._canon_rsmi = (
            f"{graph_to_smi(self._canon_reactant_graph)}>>"
            f"{graph_to_smi(self._canon_product_graph)}"
        )
        return self

    __call__ = canonicalise

    @property
    def raw_rsmi(self) -> Optional[str]:
        """Original SMILES before canonicalisation."""
        return self._raw_rsmi

    @property
    def canonical_rsmi(self) -> Optional[str]:
        """Canonical SMILES after processing."""
        return self._canon_rsmi

    @property
    def raw_reactant_graph(self) -> Optional[nx.Graph]:
        """NetworkX graph of raw reactants."""
        return self._raw_reactant_graph

    @property
    def raw_product_graph(self) -> Optional[nx.Graph]:
        """NetworkX graph of raw products."""
        return self._raw_product_graph

    @property
    def canonical_reactant_graph(self) -> Optional[nx.Graph]:
        """NetworkX graph of canonicalised reactants."""
        return self._canon_reactant_graph

    @property
    def canonical_product_graph(self) -> Optional[nx.Graph]:
        """NetworkX graph of canonicalised products."""
        return self._canon_product_graph

    @property
    def canonical_hash(self) -> Optional[str]:
        """Reaction-level hash combining reactant and product canonical hashes."""
        if not self._canon_reactant_graph or not self._canon_product_graph:
            return None
        h_reac = self._canon.canonical_signature(self._canon_reactant_graph)
        h_prod = self._canon.canonical_signature(self._canon_product_graph)
        return f"{h_reac}>>{h_prod}"

    @property
    def mapping_pairs(self) -> Optional[List[Tuple[int, int]]]:
        """List of atom-map index pairs between reactants and products."""
        return self._mapping_pairs

    def help(self) -> None:  # pragma: no cover
        """
        Pretty-print the class doc and public methods with signatures.
        """
        print(inspect.getdoc(self.__class__))
        for meth in (
            "expand_aam",
            "canonicalise",
            "get_aam_pairwise_indices",
            "remap_graph",
        ):
            print(f"  • {meth}{inspect.signature(getattr(self, meth))}")

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<CanonRSMI backend={self._canon.backend!r} "
            f"wl_iterations={self._canon._wl_k}>"
        )
