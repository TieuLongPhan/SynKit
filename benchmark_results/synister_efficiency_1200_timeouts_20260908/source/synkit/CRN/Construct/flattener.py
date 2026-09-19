from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import networkx as nx


@dataclass
class ReactionDeltaFlattener:
    graph: nx.DiGraph
    skip_no_change: bool = True
    allow_empty_side: bool = False
    deduplicate: bool = True
    _cache: List[Dict[str, Any]] = field(default_factory=list, init=False)

    def build(self) -> "ReactionDeltaFlattener":
        self._cache = self._flatten()
        return self

    @property
    def reactions(self) -> List[Dict[str, Any]]:
        return list(self._cache)

    def _collect_in(self, eid: int) -> List[str]:
        out: List[str] = []
        for u, _, ed in self.graph.in_edges(eid, data=True):
            if ed.get("role") != "reactant":
                continue
            if self.graph.nodes[u].get("kind") != "species":
                continue
            stoich = int(ed.get("stoich", 1))
            out.extend([self.graph.nodes[u].get("smiles", str(u))] * stoich)
        return out

    def _collect_out(self, eid: int) -> List[str]:
        out: List[str] = []
        for _, v, ed in self.graph.out_edges(eid, data=True):
            if ed.get("role") != "product":
                continue
            if self.graph.nodes[v].get("kind") != "species":
                continue
            stoich = int(ed.get("stoich", 1))
            out.extend([self.graph.nodes[v].get("smiles", str(v))] * stoich)
        return out

    def _nz(self, x: Optional[int]) -> int:
        return 10**9 if x is None else int(x)

    def _flatten(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen: Set[str] = set()

        for eid, nd in self.graph.nodes(data=True):
            if nd.get("kind") != "rule":
                continue

            r_all = sorted(self._collect_in(eid))
            p_all = sorted(self._collect_out(eid))

            r_counter = Counter(r_all)
            p_counter = Counter(p_all)
            common = r_counter & p_counter

            rchg = sorted(list((r_counter - common).elements()))
            pchg = sorted(list((p_counter - common).elements()))

            if self.skip_no_change and not rchg and not pchg:
                continue
            if (not self.allow_empty_side) and (not rchg or not pchg):
                continue

            rxn_smiles = f"{'.'.join(rchg)}>>{'.'.join(pchg)}"
            if self.deduplicate and rxn_smiles in seen:
                continue
            seen.add(rxn_smiles)

            out.append(
                {
                    "rule_id": eid,
                    "label": nd.get("label"),
                    "step": nd.get("step"),
                    "rule_index": nd.get("rule_index"),
                    "app_index": nd.get("app_index"),
                    "reactants": rchg,
                    "products": pchg,
                    "rule_smiles": rxn_smiles,
                }
            )

        out.sort(
            key=lambda r: (
                self._nz(r.get("step")),
                self._nz(r.get("rule_index")),
                self._nz(r.get("app_index")),
                r["rule_id"],
            )
        )
        return out
