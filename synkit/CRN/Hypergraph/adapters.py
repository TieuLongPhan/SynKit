# # CRN/Topo/adapters.py
# from __future__ import annotations
# from typing import Tuple, List
# from dataclasses import dataclass

# from ..core import CRNNetwork, CRNSpecies, CRNReaction
# from .hypergraph import CRNHyperGraph


# def hypergraph_to_crnnetwork(H: CRNHyperGraph) -> CRNNetwork:
#     """
#     Convert a CRNHyperGraph to legacy CRNNetwork for reuse of existing props.

#     :returns: CRNNetwork with same species order (sorted) and irreversible reactions.
#     """
#     species_names = H.species_list()
#     idx = {s: i for i, s in enumerate(species_names)}
#     species: List[CRNSpecies] = [CRNSpecies(name=s) for s in species_names]

#     reactions: List[CRNReaction] = []
#     for e in H.edge_list():
#         reactants = {idx[s]: float(c) for s, c in e.reactants.items()}
#         products = {idx[s]: float(c) for s, c in e.products.items()}
#         reactions.append(
#             CRNReaction(
#                 reactants=reactants,
#                 products=products,
#                 reversible=False,
#                 metadata={"rule": e.rule, "edge_id": e.id},
#             )
#         )

#     return CRNNetwork(species=species, reactions=reactions)


# def crnnetwork_to_hypergraph(net: CRNNetwork) -> CRNHyperGraph:
#     """
#     Convert a legacy CRNNetwork to CRNHyperGraph.
#     """
#     H = CRNHyperGraph()
#     names = [s.name for s in net.species]
#     for j, rxn in enumerate(net.reactions):
#         lhs = {names[i]: int(v) for i, v in rxn.reactants.items()}
#         rhs = {names[i]: int(v) for i, v in rxn.products.items()}
#         rule = str(rxn.metadata.get("rule", "r"))
#         H.add_rxn(lhs, rhs, rule=rule, edge_id=f"r_{j+1}")
#         if rxn.reversible:
#             # keep explicit reverse to stay faithful to CRNNetwork semantics
#             H.add_rxn(rhs, lhs, rule=rule, edge_id=f"r_{j+1}_rev")
#     return H
