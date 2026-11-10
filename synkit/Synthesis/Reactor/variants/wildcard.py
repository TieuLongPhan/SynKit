"""Typed wildcard application with Python and native C++ backends."""

from dataclasses import dataclass, field, fields
import re
from typing import Any, Mapping

import networkx as nx
from rdkit import Chem

from synkit.Graph.Morphism.constraints import WildcardConstraint
from synkit.Graph.Morphism.llg import LewisLabelledGraph
from synkit.IO import rsmi_to_its
from synkit.IO.chem_converter import ITSFormat
from synkit.Rule import SynRule

from ..core.engine import SynReactor


def _refuse_isotope_smiles(value: str) -> None:
    """Check before the default molecular converter can erase isotope labels."""
    for endpoint in value.split(">>"):
        if not re.search(r"\[\d", endpoint):
            continue
        molecule = Chem.MolFromSmiles(endpoint, sanitize=False)
        if molecule is not None and any(a.GetIsotope() for a in molecule.GetAtoms()):
            raise ValueError("Isotope-aware typed wildcard application is not in this profile.")


@dataclass
class WildcardEngine(SynReactor):
    """Apply tuple rules with injective wildcard bindings and explicit ports.

    Bare ``*`` vertices are query atoms. Typed ``attachment_port`` vertices
    bind distinct neighboring host atoms and preserve their external context.
    Missing rule components are completed symbolically, with persistent ports
    recording unknown external single bonds. True radicals are preserved.

    ``partial=True`` enumerates every nonempty component subset, including
    full matches. ``partial=False`` requires a full rule match. Explicit
    search limits raise rather than silently returning a truncated population.
    """

    partial: bool = True
    explicit_h: bool = False
    template_format: ITSFormat = "tuple"
    automorphism: bool = False
    wildcard_mode: str = "typed"
    wildcards: Mapping[int, WildcardConstraint] = field(default_factory=dict)
    backend: str = "python"
    library_path: str | None = None
    first_application: bool = False
    _native_cache: dict | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.wildcard_mode != "typed":
            raise ValueError("WildcardEngine requires wildcard_mode='typed'.")
        if self.backend not in {"python", "cpp"}:
            raise ValueError("backend must be 'python' or 'cpp'.")
        if self.first_application and (self.backend != "cpp" or self.partial):
            raise ValueError("first_application requires backend='cpp' and partial=False.")

    def _native_result(self) -> dict:
        if self._native_cache is None:
            from .._native.runtime import execute

            result = execute(self)
            order = sorted(
                range(len(result["mappings"])),
                key=lambda i: self._mapping_atom_map_alignment(
                    self.rule.left.raw, self._matching_host_graph(), result["mappings"][i]
                ),
                reverse=True,
            )
            result["mappings"] = [result["mappings"][i] for i in order]
            result["graphs"] = [result["graphs"][i] for i in order]
            self._native_cache = result
        return self._native_cache

    @property
    def native_statistics(self) -> dict:
        return dict(self._native_result()["statistics"]) if self.backend == "cpp" else {}

    @property
    def mappings(self):
        if self.backend == "python":
            return super().mappings
        return self._native_result()["mappings"]

    @property
    def its_list(self):
        if self.backend == "python":
            return super().its_list
        if self._its is None:
            result = self._native_result()
            graphs = result["graphs"]
            for index, (graph, mapping) in enumerate(zip(graphs, result["mappings"])):
                graph.graph["application_provenance"] = {
                    "application_index": index,
                    "mapping_index": index,
                    "rewrite_index": 0,
                    "stereo_branch_index": 0,
                    "mapping": self._mapping_provenance(mapping, self.rule.left.raw, self._matching_host_graph()),
                    "port_assignment": (),
                    "backend": "cpp",
                }
            if self.dedup_its and self.product_deduplication == "structural":
                graphs = self._deduplicate_structural_its(graphs)
            self._its = graphs
        return self._its

    @classmethod
    def from_smiles(
        cls,
        smiles: str,
        template: str | nx.Graph | SynRule,
        **kwargs: Any,
    ) -> "WildcardEngine":
        """Use the same typed defaults as the primary constructor."""
        return cls(smiles, template, **kwargs)

    @classmethod
    def from_lewis_graph(
        cls,
        graph: LewisLabelledGraph | nx.Graph,
        template: str | nx.Graph | SynRule,
        **kwargs: Any,
    ) -> "WildcardEngine":
        """Apply directly to a checked Lewis host, preserving electron state.

        NetworkX inputs are checked under the electron LLG schema. Frozen
        LLG inputs retain their schema and must be total concrete states.
        Both Python and C++ backends use the graph directly without a SMILES
        round trip. Symbolic hosts still require an explicit unifier.
        """
        if isinstance(graph, nx.Graph):
            graph = LewisLabelledGraph.from_electron_graph(graph)
        if not isinstance(graph, LewisLabelledGraph):
            raise TypeError("from_lewis_graph expects a LewisLabelledGraph or NetworkX graph.")
        return cls(graph, template, **kwargs)

    def _wrap_input(self, obj):
        if isinstance(obj, str):
            _refuse_isotope_smiles(obj)
        if isinstance(obj, LewisLabelledGraph):
            if obj.partial:
                raise ValueError("Lewis reactor hosts must be total concrete states.")
            obj = LewisLabelledGraph.from_electron_graph(
                obj.to_networkx(),
                schema=obj.schema,
                name=obj.name,
            ).to_networkx()
        return super()._wrap_input(obj)

    def _wrap_template(self, template: str | nx.Graph | SynRule) -> SynRule:
        if isinstance(template, str):
            _refuse_isotope_smiles(template)
        if not self.wildcards:
            return super()._wrap_template(template)
        if isinstance(template, str):
            graph = rsmi_to_its(template, format=self.template_format)
        elif isinstance(template, SynRule):
            graph = template.rc.raw.copy()
        elif isinstance(template, nx.Graph):
            graph = template.copy()
        else:
            raise TypeError(f"Unsupported template type: {type(template)}")
        by_map = {}
        for node, attrs in graph.nodes(data=True):
            atom_map = attrs.get("atom_map", node)
            if isinstance(atom_map, tuple) and len(atom_map) == 2:
                atom_map = atom_map[0]
            by_map[atom_map] = node
        for atom_map, constraint in self.wildcards.items():
            if not isinstance(constraint, WildcardConstraint):
                raise TypeError("wildcards values must be WildcardConstraint objects.")
            if atom_map not in by_map:
                raise ValueError(f"Wildcard atom map {atom_map} is absent from the rule.")
            attrs = graph.nodes[by_map[atom_map]]
            if attrs.get("element") not in {"*", ("*", "*")}:
                raise ValueError(f"Atom map {atom_map} must denote a wildcard on both sides.")
            attrs["wildcard_role"] = constraint.role.value
            attrs.update(
                {
                    item.name: getattr(constraint, item.name)
                    for item in fields(constraint)
                    if item.name != "role" and getattr(constraint, item.name) is not None
                }
            )
        # Use SynRule's reversal so typed owner/side contracts are transported.
        rule = SynRule(
            graph,
            canonicaliser=self.canonicaliser,
            implicit_h=template._implicit_h if isinstance(template, SynRule) else True,
        )
        return super()._wrap_template(rule)
