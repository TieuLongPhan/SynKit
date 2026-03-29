from __future__ import annotations

import unittest

from synkit.CRN.Pathway._adapter import (
    _assert_unique_tokens,
    _build_reaction_token_map,
    _build_species_token_map,
    _build_tokenized_edges,
    _invert_token_map,
    _reaction_token,
    _species_token,
    _tokenize_reaction_multiset,
    tokenize_syncrn_incidence,
)
from synkit.CRN.Petrinet.net import extract_syncrn_incidence
from synkit.CRN.Structure import SynCRN


class TestPathwayAdapterFromSynCRN(unittest.TestCase):
    """Unit tests for SynCRN incidence tokenization helpers."""

    @staticmethod
    def _build_syn() -> SynCRN:
        """
        Build the standard SynCRN testcase.

        :returns: Parsed SynCRN instance.
        :rtype: SynCRN
        """
        rxns = [
            "2A>>B+3C",
            "2B>>D",
            "D+C>>E",
            "D+3C>>F",
            "E+2C>>F",
            "3B>>G",
            "G+3C>>H",
            "B+C>>I",
            "I+C>>J",
            "E+I>>K",
            "K+C>>H",
        ]
        return SynCRN.from_reaction_strings(rxns)

    def _incidence(self):
        """
        Extract canonical incidence from the SynCRN testcase.

        :returns: Canonical SynCRN incidence object.
        :rtype: object
        """
        return extract_syncrn_incidence(self._build_syn())

    @staticmethod
    def _find_species_id_by_label(incidence, label: str) -> str:
        """
        Resolve an internal species id by species label.

        :param incidence: Canonical incidence object.
        :type incidence: object
        :param label: Species label to resolve.
        :type label: str
        :returns: Internal species id.
        :rtype: str
        """
        matches = [
            sid for sid, lab in incidence.species_labels.items() if str(lab) == label
        ]
        if len(matches) != 1:
            raise AssertionError(
                f"Expected exactly one species with label {label!r}, got {matches}"
            )
        return matches[0]

    @staticmethod
    def _find_reaction_id_by_stoich(
        incidence,
        tail_labels: dict[str, int],
        head_labels: dict[str, int],
    ) -> str:
        """
        Resolve an internal reaction id by exact stoichiometry in label space.

        :param incidence: Canonical incidence object.
        :type incidence: object
        :param tail_labels: Reactant multiset in species-label space.
        :type tail_labels: dict[str, int]
        :param head_labels: Product multiset in species-label space.
        :type head_labels: dict[str, int]
        :returns: Internal reaction id.
        :rtype: str
        """
        matches = []
        for rid in incidence.reaction_order:
            tail = {
                incidence.species_labels[sid]: int(v)
                for sid, v in incidence.pre.get(rid, {}).items()
            }
            head = {
                incidence.species_labels[sid]: int(v)
                for sid, v in incidence.post.get(rid, {}).items()
            }
            if tail == tail_labels and head == head_labels:
                matches.append(rid)

        if len(matches) != 1:
            raise AssertionError(
                f"Expected exactly one reaction for {tail_labels} >> {head_labels}, got {matches}"
            )
        return matches[0]

    def test_species_token_modes(self) -> None:
        """
        ``_species_token`` should support id, label, and source modes.
        """
        incidence = self._incidence()
        sid = self._find_species_id_by_label(incidence, "A")

        self.assertEqual(_species_token(incidence, sid, "id"), sid)
        self.assertEqual(_species_token(incidence, sid, "label"), "A")
        self.assertEqual(
            _species_token(incidence, sid, "source"),
            str(incidence.species_source_node_ids.get(sid, sid)),
        )

    def test_species_token_invalid_mode_raises(self) -> None:
        """
        ``_species_token`` should reject unsupported tokenization modes.
        """
        incidence = self._incidence()
        sid = self._find_species_id_by_label(incidence, "A")

        with self.assertRaises(ValueError):
            _species_token(incidence, sid, "bad-mode")

    def test_reaction_token_modes(self) -> None:
        """
        ``_reaction_token`` should support id, label, and source modes.
        """
        incidence = self._incidence()
        rid = self._find_reaction_id_by_stoich(
            incidence,
            {"A": 2},
            {"B": 1, "C": 3},
        )

        self.assertEqual(_reaction_token(incidence, rid, "id"), rid)
        self.assertEqual(
            _reaction_token(incidence, rid, "label"),
            incidence.reaction_labels.get(rid, rid),
        )
        self.assertEqual(
            _reaction_token(incidence, rid, "source"),
            str(incidence.reaction_source_node_ids.get(rid, rid)),
        )

    def test_reaction_token_invalid_mode_raises(self) -> None:
        """
        ``_reaction_token`` should reject unsupported tokenization modes.
        """
        incidence = self._incidence()
        rid = self._find_reaction_id_by_stoich(
            incidence,
            {"A": 2},
            {"B": 1, "C": 3},
        )

        with self.assertRaises(ValueError):
            _reaction_token(incidence, rid, "bad-mode")

    def test_assert_unique_tokens_accepts_unique_mapping(self) -> None:
        """Unique token mappings should pass validation."""
        _assert_unique_tokens({"s1": "A", "s2": "B"}, kind="species")

    def test_assert_unique_tokens_rejects_duplicates(self) -> None:
        """Duplicate exported tokens should raise a ValueError."""
        with self.assertRaises(ValueError):
            _assert_unique_tokens({"s1": "A", "s2": "A"}, kind="species")

        with self.assertRaises(ValueError):
            _assert_unique_tokens({"r1": "x", "r2": "x"}, kind="reaction")

    def test_build_species_token_map_label_mode(self) -> None:
        """
        Species token map in label mode should map internal ids to human-readable
        species labels.
        """
        incidence = self._incidence()
        species_map = _build_species_token_map(incidence, species="label")

        self.assertEqual(len(species_map), 11)
        self.assertEqual(
            set(species_map.values()),
            {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"},
        )

        sid_a = self._find_species_id_by_label(incidence, "A")
        sid_k = self._find_species_id_by_label(incidence, "K")
        self.assertEqual(species_map[sid_a], "A")
        self.assertEqual(species_map[sid_k], "K")

    def test_build_reaction_token_map_id_mode(self) -> None:
        """
        Reaction token map in id mode should be identity on internal reaction ids.
        """
        incidence = self._incidence()
        reaction_map = _build_reaction_token_map(incidence, reaction="id")

        self.assertEqual(len(reaction_map), 11)
        for rid in incidence.reaction_order:
            self.assertEqual(reaction_map[rid], rid)

    def test_tokenize_reaction_multiset_converts_species_ids_to_tokens(self) -> None:
        """
        ``_tokenize_reaction_multiset`` should convert one stoichiometric side
        from internal species ids to exported species tokens.
        """
        incidence = self._incidence()
        species_map = _build_species_token_map(incidence, species="label")
        rid = self._find_reaction_id_by_stoich(
            incidence,
            {"A": 2},
            {"B": 1, "C": 3},
        )

        tail = _tokenize_reaction_multiset(incidence.pre[rid], species_map)
        head = _tokenize_reaction_multiset(incidence.post[rid], species_map)

        self.assertEqual(tail, {"A": 2})
        self.assertEqual(head, {"B": 1, "C": 3})

    def test_build_tokenized_edges_label_species_id_reactions(self) -> None:
        """
        Tokenized edges should preserve stoichiometry while using the requested
        public-facing species and reaction tokens.
        """
        incidence = self._incidence()
        species_map = _build_species_token_map(incidence, species="label")
        reaction_map = _build_reaction_token_map(incidence, reaction="id")

        edges = _build_tokenized_edges(incidence, species_map, reaction_map)

        self.assertEqual(len(edges), 11)

        rid1 = self._find_reaction_id_by_stoich(
            incidence,
            {"A": 2},
            {"B": 1, "C": 3},
        )
        rid10 = self._find_reaction_id_by_stoich(
            incidence,
            {"E": 1, "I": 1},
            {"K": 1},
        )

        self.assertEqual(edges[rid1], ({"A": 2}, {"B": 1, "C": 3}))
        self.assertEqual(edges[rid10], ({"E": 1, "I": 1}, {"K": 1}))

    def test_invert_token_map(self) -> None:
        """Inverting a token map should swap keys and values."""
        token_map = {"s1": "A", "s2": "B", "s3": "C"}
        inv = _invert_token_map(token_map)
        self.assertEqual(inv, {"A": "s1", "B": "s2", "C": "s3"})

    def test_tokenize_syncrn_incidence_label_species_id_reactions(self) -> None:
        """
        The main tokenizer should return ordered tokenized species, tokenized
        reaction edges, reverse token maps, and the original incidence object.
        """
        syn = self._build_syn()

        vertices, edges, species_token_to_id, reaction_token_to_id, incidence = (
            tokenize_syncrn_incidence(
                syn,
                species="label",
                reaction="id",
            )
        )

        self.assertEqual(
            vertices, ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]
        )
        self.assertEqual(len(edges), 11)
        self.assertEqual(
            set(species_token_to_id),
            {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"},
        )
        self.assertEqual(len(reaction_token_to_id), 11)

        sid_a = species_token_to_id["A"]
        sid_h = species_token_to_id["H"]
        self.assertEqual(incidence.species_labels[sid_a], "A")
        self.assertEqual(incidence.species_labels[sid_h], "H")

        rid1 = self._find_reaction_id_by_stoich(
            incidence,
            {"A": 2},
            {"B": 1, "C": 3},
        )
        rid11 = self._find_reaction_id_by_stoich(
            incidence,
            {"K": 1, "C": 1},
            {"H": 1},
        )

        self.assertEqual(edges[rid1], ({"A": 2}, {"B": 1, "C": 3}))
        self.assertEqual(edges[rid11], ({"K": 1, "C": 1}, {"H": 1}))
        self.assertEqual(reaction_token_to_id[rid1], rid1)
        self.assertEqual(reaction_token_to_id[rid11], rid11)

    def test_tokenize_syncrn_incidence_id_species_id_reactions(self) -> None:
        """
        With ``species='id'`` and ``reaction='id'``, returned vertices and edges
        should be keyed entirely by internal ids.
        """
        syn = self._build_syn()

        vertices, edges, species_token_to_id, reaction_token_to_id, incidence = (
            tokenize_syncrn_incidence(
                syn,
                species="id",
                reaction="id",
            )
        )

        self.assertEqual(vertices, incidence.species_order)
        self.assertEqual(len(edges), len(incidence.reaction_order))

        for sid in incidence.species_order:
            self.assertIn(sid, species_token_to_id)
            self.assertEqual(species_token_to_id[sid], sid)

        for rid in incidence.reaction_order:
            self.assertIn(rid, reaction_token_to_id)
            self.assertEqual(reaction_token_to_id[rid], rid)

    def test_tokenize_syncrn_incidence_source_mode_roundtrip(self) -> None:
        """
        Source-mode tokenization should round-trip through the reported source
        node identifiers.
        """
        syn = self._build_syn()

        vertices, edges, species_token_to_id, reaction_token_to_id, incidence = (
            tokenize_syncrn_incidence(
                syn,
                species="source",
                reaction="source",
            )
        )

        self.assertEqual(len(vertices), 11)
        self.assertEqual(len(edges), 11)

        expected_species_tokens = {
            str(incidence.species_source_node_ids.get(sid, sid))
            for sid in incidence.species_order
        }
        expected_reaction_tokens = {
            str(incidence.reaction_source_node_ids.get(rid, rid))
            for rid in incidence.reaction_order
        }

        self.assertEqual(set(vertices), expected_species_tokens)
        self.assertEqual(set(species_token_to_id), expected_species_tokens)
        self.assertEqual(set(reaction_token_to_id), expected_reaction_tokens)

    def test_tokenize_syncrn_incidence_invalid_species_mode_raises(self) -> None:
        """
        Invalid species tokenization mode should raise a ValueError.
        """
        syn = self._build_syn()
        with self.assertRaises(ValueError):
            tokenize_syncrn_incidence(
                syn,
                species="bad-mode",
                reaction="id",
            )

    def test_tokenize_syncrn_incidence_invalid_reaction_mode_raises(self) -> None:
        """
        Invalid reaction tokenization mode should raise a ValueError.
        """
        syn = self._build_syn()
        with self.assertRaises(ValueError):
            tokenize_syncrn_incidence(
                syn,
                species="label",
                reaction="bad-mode",
            )


if __name__ == "__main__":
    unittest.main()
