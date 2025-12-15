import unittest

import networkx as nx

from synkit.Graph.Matcher.automorphism import Automorphism


class TestAutomorphismOrbits(unittest.TestCase):
    def test_asymmetric_graph_singleton_orbits(self) -> None:
        """Path graph on 4 nodes has two reflection orbits: {0,3} and {1,2}."""
        G = nx.path_graph(4)  # 0-1-2-3
        auto = Automorphism(G)

        orbits = auto.orbits
        covered = {n for orb in orbits for n in orb}
        self.assertEqual(covered, set(G.nodes()))

        # expected orbit partition for path_graph(4)
        expected = {frozenset({0, 3}), frozenset({1, 2})}
        self.assertEqual(set(orbits), expected)
        self.assertEqual(len(auto), len(expected))

    def test_cycle_graph_single_orbit(self) -> None:
        """4-cycle with no attributes: all nodes are symmetric."""
        G = nx.cycle_graph(4)
        auto = Automorphism(G)

        orbits = auto.orbits
        self.assertEqual(len(orbits), 1)
        self.assertEqual(orbits[0], frozenset(G.nodes()))
        self.assertEqual(len(auto), 1)

    def test_star_graph_central_vs_leaves(self) -> None:
        """Star graph: center in its own orbit, leaves in another."""
        G = nx.star_graph(3)  # nodes: 0 is center, 1-3 are leaves
        auto = Automorphism(G)

        orbits = auto.orbits
        self.assertEqual(len(orbits), 2)

        center_orbit = [orb for orb in orbits if 0 in orb][0]
        leaf_orbit = [orb for orb in orbits if 0 not in orb][0]

        self.assertEqual(center_orbit, frozenset({0}))
        self.assertEqual(leaf_orbit, frozenset({1, 2, 3}))

    def test_node_attributes_break_symmetry(self) -> None:
        """
        Changing an attribute on one node should split its orbit.

        For a 4-cycle where node 0 is distinct, expected orbits are:
          {0}, {2}, and {1,3}
        (node 2 is the opposite node and remains fixed under the reflection
         that swaps 1 and 3).
        """
        G = nx.cycle_graph(4)
        # Assign all nodes the same element
        for n in G:
            G.nodes[n]["element"] = "C"
        # Break symmetry at node 0
        G.nodes[0]["element"] = "O"

        auto = Automorphism(G)
        orbits = auto.orbits

        # Node 0 should be alone in its orbit
        orbit_0 = [orb for orb in orbits if 0 in orb][0]
        self.assertEqual(orbit_0, frozenset({0}))

        # Node 2 (opposite of 0) should also be fixed
        orbit_2 = [orb for orb in orbits if 2 in orb][0]
        self.assertEqual(orbit_2, frozenset({2}))

        # Nodes 1 and 3 should form the remaining orbit
        orbit_13 = [orb for orb in orbits if 1 in orb][0]
        self.assertEqual(orbit_13, frozenset({1, 3}))


class TestAutomorphismDeduplicate(unittest.TestCase):
    def test_empty_mapping_list(self) -> None:
        """Deduplicating an empty list should return an empty list."""
        G = nx.cycle_graph(3)
        auto = Automorphism(G)

        unique = auto.deduplicate([])
        self.assertEqual(unique, [])

    def test_equivalent_mappings_are_merged(self) -> None:
        """Mappings that only differ by automorphisms should be merged."""
        G = nx.cycle_graph(4)
        auto = Automorphism(G)

        # Three mappings that differ only by rotation of the cycle
        mappings = [
            {"a": 0, "b": 1},
            {"a": 1, "b": 2},
            {"a": 2, "b": 3},
        ]
        original_copy = list(mappings)

        unique = auto.deduplicate(mappings)

        # There should be only one representative
        self.assertEqual(len(unique), 1)
        self.assertIn(unique[0], original_copy)

    def test_inequivalent_mappings_are_kept(self) -> None:
        """Mappings hitting different orbits must not be collapsed."""
        # Star graph: center vs leaves are different orbits
        G = nx.star_graph(3)  # 0 center, 1-3 leaves
        auto = Automorphism(G)

        # Map pattern node 'x' to center vs a leaf
        mappings = [
            {"x": 0},
            {"x": 1},
        ]

        unique = auto.deduplicate(mappings)

        # Center orbit index != leaf orbit index → both mappings remain
        self.assertEqual(len(unique), 2)
        self.assertCountEqual(unique, mappings)

    def test_repr_contains_basic_info(self) -> None:
        """__repr__ should contain number of orbits and nodes."""
        G = nx.path_graph(3)
        auto = Automorphism(G)
        rep = repr(auto)

        self.assertIn("Automorphism", rep)
        self.assertIn("orbits=", rep)
        self.assertIn("nodes=", rep)


if __name__ == "__main__":
    unittest.main()
