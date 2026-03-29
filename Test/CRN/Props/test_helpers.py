from __future__ import annotations

import unittest

import networkx as nx

from synkit.CRN.Props import helper

# ---------------------------------------------------------------------------
# Small helper wrapper classes for _as_graph tests
# ---------------------------------------------------------------------------


class WrapperWithG:
    def __init__(self, G: nx.Graph) -> None:
        self.G = G


class WrapperWithNxGraph:
    def __init__(self, G: nx.Graph) -> None:
        self.nx_graph = G


class WrapperWithBipartiteGraph:
    def __init__(self, G: nx.Graph) -> None:
        self.bipartite_graph = G


class WrapperWithBipartite:
    def __init__(self, G: nx.Graph) -> None:
        self.bipartite = G


class WrapperWithToDigraph:
    def __init__(self, G: nx.Graph) -> None:
        self._G = G

    def to_digraph(self):
        return self._G


class WrapperWithToDigraphIncludeRule:
    def __init__(self, G: nx.Graph) -> None:
        self._G = G

    def to_digraph(self, include_rule: bool = False):
        if include_rule:
            return self._G
        return self._G


class WrapperWithBadToDigraph:
    def to_digraph(self):
        return "not a graph"


class WrapperWithTypeErrorThenWorks:
    def __init__(self, G: nx.Graph) -> None:
        self._G = G

    def to_digraph(self, include_rule):  # requires one positional/keyword arg
        if include_rule:
            return self._G
        return self._G


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------


def build_test_graph() -> nx.DiGraph:
    G = nx.DiGraph()

    G.add_node("A", kind="species", label="A")
    G.add_node("B", kind="species", label="B")
    G.add_node("r1", kind="rule", label="r1", step=0, rule_index=1, app_index=0)

    G.add_edge("A", "r1", role="reactant", stoich=2.0)
    G.add_edge("r1", "B", role="product", stoich=1.0)
    return G


# ---------------------------------------------------------------------------
# Tests for _as_graph
# ---------------------------------------------------------------------------


class TestAsGraph(unittest.TestCase):
    def test_as_graph_accepts_networkx_graph_directly(self) -> None:
        G = build_test_graph()
        out = helper._as_graph(G)
        self.assertIs(out, G)

    def test_as_graph_accepts_wrapper_with_G(self) -> None:
        G = build_test_graph()
        obj = WrapperWithG(G)
        out = helper._as_graph(obj)
        self.assertIs(out, G)

    def test_as_graph_accepts_wrapper_with_nx_graph(self) -> None:
        G = build_test_graph()
        obj = WrapperWithNxGraph(G)
        out = helper._as_graph(obj)
        self.assertIs(out, G)

    def test_as_graph_accepts_wrapper_with_bipartite_graph(self) -> None:
        G = build_test_graph()
        obj = WrapperWithBipartiteGraph(G)
        out = helper._as_graph(obj)
        self.assertIs(out, G)

    def test_as_graph_accepts_wrapper_with_bipartite(self) -> None:
        G = build_test_graph()
        obj = WrapperWithBipartite(G)
        out = helper._as_graph(obj)
        self.assertIs(out, G)

    def test_as_graph_accepts_wrapper_with_to_digraph(self) -> None:
        G = build_test_graph()
        obj = WrapperWithToDigraph(G)
        out = helper._as_graph(obj)
        self.assertIs(out, G)

    def test_as_graph_accepts_wrapper_with_to_digraph_include_rule(self) -> None:
        G = build_test_graph()
        obj = WrapperWithToDigraphIncludeRule(G)
        out = helper._as_graph(obj)
        self.assertIs(out, G)

    def test_as_graph_handles_first_to_digraph_signature_failing_then_second_working(
        self,
    ) -> None:
        G = build_test_graph()
        obj = WrapperWithTypeErrorThenWorks(G)
        out = helper._as_graph(obj)
        self.assertIs(out, G)

    def test_as_graph_raises_when_to_digraph_returns_non_graph(self) -> None:
        obj = WrapperWithBadToDigraph()
        with self.assertRaises(TypeError):
            helper._as_graph(obj)

    def test_as_graph_raises_when_no_graph_found(self) -> None:
        class NoGraph:
            pass

        with self.assertRaises(TypeError):
            helper._as_graph(NoGraph())


# ---------------------------------------------------------------------------
# Tests for node-kind helpers
# ---------------------------------------------------------------------------


class TestNodeKindHelpers(unittest.TestCase):
    def test_node_kind_normalization(self) -> None:
        self.assertEqual(helper._node_kind({"kind": "species"}), "species")
        self.assertEqual(helper._node_kind({"kind": " Species "}), "species")
        self.assertEqual(helper._node_kind({"kind": "RULE"}), "rule")
        self.assertEqual(helper._node_kind({}), "")

    def test_is_species_node(self) -> None:
        self.assertTrue(helper._is_species_node({"kind": "species"}))
        self.assertTrue(helper._is_species_node({"kind": " Species "}))
        self.assertFalse(helper._is_species_node({"kind": "rule"}))
        self.assertFalse(helper._is_species_node({}))

    def test_is_rule_node(self) -> None:
        self.assertTrue(helper._is_rule_node({"kind": "rule"}))
        self.assertTrue(helper._is_rule_node({"kind": " RULE "}))
        self.assertFalse(helper._is_rule_node({"kind": "species"}))
        self.assertFalse(helper._is_rule_node({}))


# ---------------------------------------------------------------------------
# Tests for role normalization
# ---------------------------------------------------------------------------


class TestNormalizeRole(unittest.TestCase):
    def test_normalize_role_reactant_aliases(self) -> None:
        for role in ["reactant", "Reactant", " lhs ", "educt", "substrate"]:
            self.assertEqual(helper._normalize_role(role), "reactant")

    def test_normalize_role_product_aliases(self) -> None:
        for role in ["product", "Product", " rhs "]:
            self.assertEqual(helper._normalize_role(role), "product")

    def test_normalize_role_invalid_or_none(self) -> None:
        self.assertIsNone(helper._normalize_role(None))
        self.assertIsNone(helper._normalize_role("catalyst"))
        self.assertIsNone(helper._normalize_role("agent"))
        self.assertIsNone(helper._normalize_role(""))


# ---------------------------------------------------------------------------
# Tests for edge coefficient parsing
# ---------------------------------------------------------------------------


class TestEdgeCoeff(unittest.TestCase):
    def test_edge_coeff_prefers_stoich(self) -> None:
        data = {"stoich": 2.0, "coeff": 5.0, "coefficient": 7.0}
        self.assertEqual(helper._edge_coeff(data), 2.0)

    def test_edge_coeff_falls_back_to_coeff(self) -> None:
        data = {"coeff": 3.0, "coefficient": 7.0}
        self.assertEqual(helper._edge_coeff(data), 3.0)

    def test_edge_coeff_falls_back_to_coefficient(self) -> None:
        data = {"coefficient": 4.0}
        self.assertEqual(helper._edge_coeff(data), 4.0)

    def test_edge_coeff_defaults_to_one(self) -> None:
        self.assertEqual(helper._edge_coeff({}), 1.0)

    def test_edge_coeff_casts_to_float(self) -> None:
        self.assertEqual(helper._edge_coeff({"stoich": "2"}), 2.0)


# ---------------------------------------------------------------------------
# Tests for node sorting
# ---------------------------------------------------------------------------


class TestNodeSortKey(unittest.TestCase):
    def test_species_sort_before_rule_and_unknown(self) -> None:
        items = [
            ("r1", {"kind": "rule", "label": "r1"}),
            ("A", {"kind": "species", "label": "A"}),
            ("u1", {"kind": "other", "label": "u1"}),
        ]
        out = [node for node, _data in sorted(items, key=helper._node_sort_key)]
        self.assertEqual(out, ["A", "r1", "u1"])

    def test_rule_sort_uses_step_rule_index_app_index(self) -> None:
        items = [
            ("r3", {"kind": "rule", "step": 1, "rule_index": 0, "app_index": 0}),
            ("r2", {"kind": "rule", "step": 0, "rule_index": 1, "app_index": 0}),
            ("r1", {"kind": "rule", "step": 0, "rule_index": 0, "app_index": 0}),
        ]
        out = [node for node, _data in sorted(items, key=helper._node_sort_key)]
        self.assertEqual(out, ["r1", "r2", "r3"])

    def test_sort_uses_label_smiles_repr_as_tiebreakers(self) -> None:
        items = [
            ("node2", {"kind": "species", "label": "B", "smiles": "CC"}),
            ("node1", {"kind": "species", "label": "A", "smiles": "CC"}),
        ]
        out = [node for node, _data in sorted(items, key=helper._node_sort_key)]
        self.assertEqual(out, ["node1", "node2"])


# ---------------------------------------------------------------------------
# Tests for species/rule ordering
# ---------------------------------------------------------------------------


class TestSpeciesAndRuleOrder(unittest.TestCase):
    def test_species_and_rule_order_basic(self) -> None:
        G = build_test_graph()

        species_nodes, rule_nodes, species_index, rule_index = (
            helper._species_and_rule_order(G)
        )

        self.assertEqual(species_nodes, ["A", "B"])
        self.assertEqual(rule_nodes, ["r1"])
        self.assertEqual(species_index, {"A": 0, "B": 1})
        self.assertEqual(rule_index, {"r1": 0})

    def test_species_and_rule_order_ignores_other_kinds(self) -> None:
        G = build_test_graph()
        G.add_node("x1", kind="other", label="x1")

        species_nodes, rule_nodes, species_index, rule_index = (
            helper._species_and_rule_order(G)
        )

        self.assertEqual(species_nodes, ["A", "B"])
        self.assertEqual(rule_nodes, ["r1"])
        self.assertNotIn("x1", species_index)
        self.assertNotIn("x1", rule_index)

    def test_species_and_rule_order_stable_rule_sort(self) -> None:
        G = nx.DiGraph()

        G.add_node("B", kind="species", label="B")
        G.add_node("A", kind="species", label="A")

        G.add_node("r2", kind="rule", label="r2", step=1, rule_index=0, app_index=0)
        G.add_node("r1", kind="rule", label="r1", step=0, rule_index=0, app_index=0)
        G.add_node("r3", kind="rule", label="r3", step=1, rule_index=1, app_index=0)

        species_nodes, rule_nodes, species_index, rule_index = (
            helper._species_and_rule_order(G)
        )

        self.assertEqual(species_nodes, ["A", "B"])
        self.assertEqual(rule_nodes, ["r1", "r2", "r3"])
        self.assertEqual(species_index["A"], 0)
        self.assertEqual(species_index["B"], 1)
        self.assertEqual(rule_index["r1"], 0)
        self.assertEqual(rule_index["r2"], 1)
        self.assertEqual(rule_index["r3"], 2)

    def test_species_and_rule_order_accepts_wrapper_object(self) -> None:
        G = build_test_graph()
        wrapped = WrapperWithG(G)

        species_nodes, rule_nodes, species_index, rule_index = (
            helper._species_and_rule_order(wrapped)
        )

        self.assertEqual(species_nodes, ["A", "B"])
        self.assertEqual(rule_nodes, ["r1"])
        self.assertEqual(species_index["A"], 0)
        self.assertEqual(rule_index["r1"], 0)


if __name__ == "__main__":
    unittest.main()
