"""
Unit and integration tests for RBLEngine.

Covers:
- Integration examples (ester synthesis, transesterification)
- API contracts (invalid input, repr, diagnostics, quick-check short-circuit)
- Quick-check return type
- wl_min_score stored and forwarded to WLSel
- non-wildcard ITS early-stop collection (mode="early_stop")
- Matcher hoisted once per process() call
- ApproxMCSMatcher backend importable and usable
- matcher_kwargs stored and forwarded correctly
- Code-quality guards (dead code removed)
"""

import functools
import unittest
import unittest.mock
from unittest.mock import patch

from synkit.Graph.Matcher.approx_mcs import ApproxMCSMatcher
from synkit.Graph.Matcher.mcs_matcher import MCSMatcher
from synkit.IO import its_to_rsmi, rsmi_to_its
from synkit.Synthesis.Reactor.rbl_engine import RBLEngine

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DummyQuickCheckEngine(RBLEngine):
    """Subclass that returns a fixed quick-check result for a marker reaction."""

    def _quick_check(self, rsmi: str, template) -> "str | None":
        if rsmi == "A>>B":
            return "A*>>B*"
        return None


def _minimal_its_graph(element: str = "C"):
    """Minimal ITS-like nx.Graph with no wildcard nodes."""
    import networkx as nx

    G = nx.Graph()
    G.add_node(
        1,
        element=element,
        typesGH=(("C", False, 3, 0, []), ("C", False, 3, 0, [])),
    )
    G.add_node(
        2,
        element="O",
        typesGH=(("O", False, 1, 0, []), ("O", False, 1, 0, [])),
    )
    G.add_edge(1, 2, order=1)
    return G


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestRBLEngineExamples(unittest.TestCase):
    """Integration-style tests using standard RBL reaction examples."""

    def _run(self, rsmi: str, template: str, *, replace_wc: bool = True) -> list:
        engine = RBLEngine(mode="full")
        engine.process(rsmi, template, replace_wc=replace_wc)
        return engine.fused_rsmis

    def test_ester_synthesis(self) -> None:
        """Alcohol + acid → ester; template built via ITS round-trip."""
        raw_template = (
            "[CH3:1][C:2](=[O:3])[OH:4]."
            "[CH3:5][O:6][H:7]>>"
            "[CH3:1][C:2](=[O:3])[O:6][CH3:5]."
            "[H:7][OH:4]"
        )
        its_template = rsmi_to_its(raw_template, core=True)
        template = its_to_rsmi(its_template)

        result = self._run("CCC(=O)(O)>>CCC(=O)OC", template)

        self.assertTrue(result, "Engine produced no results for ester synthesis")
        for r in result:
            self.assertIn(">>", r)

    def test_transesterification(self) -> None:
        """Simple acyl transfer pattern."""
        template = "[C:1][O:2].[O:3][H:4]>>[C:1][O:3].[O:2][H:4]"
        result = self._run("CCC(=O)OC>>CCC(=O)OCC", template)

        self.assertTrue(result, "Engine produced no results for transesterification")
        self.assertIn(">>", result[0])


# ---------------------------------------------------------------------------
# API tests
# ---------------------------------------------------------------------------


class TestRBLEngineAPI(unittest.TestCase):

    def test_invalid_reaction_string_raises(self) -> None:
        engine = RBLEngine()
        with self.assertRaises(ValueError):
            engine.process(
                "no_arrow_here",
                "[C:1][O:2].[O:3][H:4]>>[C:1][O:3].[O:2][H:4]",
            )

    def test_repr_contains_key_info(self) -> None:
        engine = RBLEngine(wildcard_element="X*", node_attrs=["element"])
        rep = repr(engine)
        self.assertIn("RBLEngine", rep)
        self.assertIn("wildcard_element='X*'", rep)
        self.assertIn("reactor_cls=", rep)

    def test_result_dict_has_expected_keys(self) -> None:
        engine = RBLEngine(mode="full")
        engine.process(
            "CCC(=O)OC>>CCC(=O)OCC",
            "[C:1][O:2].[O:3][H:4]>>[C:1][O:3].[O:2][H:4]",
        )
        result = engine.result
        for key in (
            "fused_rsmis",
            "mode",
            "reason",
            "metadata",
            "n_forward_its",
            "n_backward_its",
            "n_fused_its",
        ):
            self.assertIn(key, result)

    def test_quick_check_short_circuits_pipeline(self) -> None:
        engine = _DummyQuickCheckEngine()
        engine.process("A>>B", "[C:1]>>[C:1]")
        self.assertEqual(engine.fused_rsmis, ["A*>>B*"])
        self.assertEqual(engine.fused_its, [])


# ---------------------------------------------------------------------------
# Quick-check return type
# ---------------------------------------------------------------------------


class TestQuickCheckReturnType(unittest.TestCase):
    def test_fused_rsmis_items_are_strings(self):
        """Each element of fused_rsmis must be a str, not a list."""
        engine = RBLEngine()
        engine.process("[CH3:1][OH:2]>>[CH3:1][OH:2]", "[C:1][OH:2]>>[C:1][OH:2]")
        for item in engine.fused_rsmis:
            self.assertIsInstance(
                item, str, f"Expected str, got {type(item)}: {item!r}"
            )


# ---------------------------------------------------------------------------
# wl_min_score
# ---------------------------------------------------------------------------


class TestWlMinScore(unittest.TestCase):
    def test_default_is_0_8(self):
        self.assertAlmostEqual(RBLEngine().wl_min_score, 0.8)

    def test_custom_value_stored(self):
        self.assertAlmostEqual(RBLEngine(wl_min_score=0.3).wl_min_score, 0.3)

    def test_forwarded_to_wlsel(self):
        engine = RBLEngine(mode="full", wl_min_score=0.42)
        with patch(
            "synkit.Synthesis.Reactor.rbl_engine.WLSel",
            wraps=__import__("synkit.Graph.Matcher.wl_sel", fromlist=["WLSel"]).WLSel,
        ) as mock_wlsel:
            engine.process(
                "CCC(=O)OC>>CCC(=O)OCC",
                "[C:1][O:2].[O:3][H:4]>>[C:1][O:3].[O:2][H:4]",
            )
            self.assertIsNotNone(mock_wlsel.call_args, "WLSel was never instantiated")
            _, kwargs = mock_wlsel.call_args
            self.assertAlmostEqual(kwargs.get("min_score", 0.8), 0.42)

    def test_zero_score_does_not_crash(self):
        engine = RBLEngine(mode="full", wl_min_score=0.0)
        engine.process(
            "CCC(=O)OC>>CCC(=O)OCC",
            "[C:1][O:2].[O:3][H:4]>>[C:1][O:3].[O:2][H:4]",
        )


# ---------------------------------------------------------------------------
# early_stop / non-wildcard collection
# ---------------------------------------------------------------------------


class TestEarlyStopNonWildcard(unittest.TestCase):

    def test_all_candidates_collected_when_early_stop_false(self):
        engine = RBLEngine()
        g1, g2 = _minimal_its_graph("C"), _minimal_its_graph("N")
        engine._last_reaction = "[CH3:1][OH:2]>>[CH3:1][OH:2]"
        engine._postprocess_single = lambda g, *, replace_wc, rw_adder: f"rsmi_{id(g)}"
        engine._validate_candidate_rsmi = lambda **kw: True

        result = engine._early_stop_on_nonwildcard(
            fw_its=[g1, g2],
            bw_its=[],
            replace_wc=True,
            early_stop=False,
        )
        self.assertTrue(result)
        self.assertEqual(len(engine.fused_rsmis), 2)

    def test_only_first_candidate_kept_when_early_stop_true(self):
        engine = RBLEngine()
        g1, g2 = _minimal_its_graph("C"), _minimal_its_graph("N")
        # Use a balanced, org-rxn-satisfying SMILES so _is_balanced and
        # _org_rxn_satisfied both pass — required since the early-stop path checks both.
        engine._last_reaction = "[CH3:1][OH:2]>>[CH3:1][OH:2]"
        engine._postprocess_single = (
            lambda g, *, replace_wc, rw_adder: "[CH3:1][OH:2]>>[CH3:1][OH:2]"
        )
        engine._validate_candidate_rsmi = lambda **kw: True

        result = engine._early_stop_on_nonwildcard(
            fw_its=[g1, g2],
            bw_its=[],
            replace_wc=True,
            early_stop=True,
        )
        self.assertTrue(result)
        self.assertEqual(len(engine.fused_rsmis), 1)


# ---------------------------------------------------------------------------
# Matcher hoisted once per process()
# ---------------------------------------------------------------------------


class TestMatcherHoisted(unittest.TestCase):
    def test_build_matcher_called_once_per_process(self):
        engine = RBLEngine(mode="full")
        call_count = 0
        original = engine._build_matcher

        def counting_build_matcher():
            nonlocal call_count
            call_count += 1
            return original()

        engine._build_matcher = counting_build_matcher
        engine.process(
            "CCC(=O)OC>>CCC(=O)OCC",
            "[C:1][O:2].[O:3][H:4]>>[C:1][O:3].[O:2][H:4]",
        )
        self.assertEqual(call_count, 1, f"Expected 1, got {call_count}")


# ---------------------------------------------------------------------------
# ApproxMCSMatcher backend
# ---------------------------------------------------------------------------


class TestApproxMCSBackend(unittest.TestCase):

    def test_importable_from_rbl_engine_module(self):
        import synkit.Synthesis.Reactor.rbl_engine as _mod

        self.assertTrue(hasattr(_mod, "ApproxMCSMatcher"))

    def test_accepted_as_matcher_cls(self):
        self.assertIs(
            RBLEngine(matcher_cls=ApproxMCSMatcher).matcher_cls, ApproxMCSMatcher
        )

    def test_produces_valid_rsmis(self):
        engine = RBLEngine(matcher_cls=ApproxMCSMatcher, mode="full")
        engine.process(
            "CCC(=O)OC>>CCC(=O)OCC",
            "[C:1][O:2].[O:3][H:4]>>[C:1][O:3].[O:2][H:4]",
        )
        self.assertTrue(engine.fused_rsmis)
        for rsmi in engine.fused_rsmis:
            self.assertIn(">>", rsmi)

    def test_default_backend_is_mcs_matcher(self):
        self.assertIs(RBLEngine().matcher_cls, MCSMatcher)

    def test_build_matcher_returns_approx_instance(self):
        self.assertIsInstance(
            RBLEngine(matcher_cls=ApproxMCSMatcher)._build_matcher(), ApproxMCSMatcher
        )

    def test_build_matcher_returns_mcs_instance_by_default(self):
        self.assertIsInstance(RBLEngine()._build_matcher(), MCSMatcher)


# ---------------------------------------------------------------------------
# matcher_kwargs forwarding
# ---------------------------------------------------------------------------


class TestMatcherKwargs(unittest.TestCase):

    def test_default_is_empty_dict(self):
        self.assertEqual(RBLEngine().matcher_kwargs, {})

    def test_custom_kwargs_stored(self):
        self.assertEqual(
            RBLEngine(matcher_kwargs={"use_wl": True, "wl_max_iter": 5}).matcher_kwargs,
            {"use_wl": True, "wl_max_iter": 5},
        )

    def test_none_stored_as_empty(self):
        self.assertEqual(RBLEngine(matcher_kwargs=None).matcher_kwargs, {})

    def test_constructor_kwargs_forwarded(self):
        engine = RBLEngine(
            matcher_cls=ApproxMCSMatcher,
            matcher_kwargs={"use_wl": True, "wl_max_iter": 2},
        )
        m = engine._build_matcher()
        self.assertTrue(m.use_wl)
        self.assertEqual(m.wl_max_iter, 2)

    def test_rc_call_params_excluded_from_constructor(self):
        engine = RBLEngine(
            matcher_cls=ApproxMCSMatcher,
            matcher_kwargs={"max_seeds": 8, "max_steps": 64},
        )
        try:
            engine._build_matcher()
        except TypeError as exc:
            self.fail(f"_build_matcher raised TypeError: {exc}")

    def test_max_seeds_forwarded_to_find_rc_mapping(self):
        engine = RBLEngine(
            matcher_cls=ApproxMCSMatcher,
            mode="full",
            matcher_kwargs={"max_seeds": 4, "max_steps": 32},
        )
        call_log = []
        original_frc = ApproxMCSMatcher.find_rc_mapping

        @functools.wraps(original_frc)
        def spy(self_m, *args, **kwargs):
            call_log.append(kwargs)
            return original_frc(self_m, *args, **kwargs)

        with unittest.mock.patch.object(ApproxMCSMatcher, "find_rc_mapping", spy):
            engine.process(
                "CCC(=O)OC>>CCC(=O)OCC",
                "[C:1][O:2].[O:3][H:4]>>[C:1][O:3].[O:2][H:4]",
            )

        self.assertTrue(call_log, "find_rc_mapping was never called")
        for kw in call_log:
            self.assertEqual(kw.get("max_seeds"), 4)
            self.assertEqual(kw.get("max_steps"), 32)

    def test_max_seeds_not_forwarded_to_mcs_matcher(self):
        engine = RBLEngine(
            matcher_cls=MCSMatcher, mode="full", matcher_kwargs={"max_seeds": 4}
        )
        try:
            engine.process(
                "CCC(=O)OC>>CCC(=O)OCC",
                "[C:1][O:2].[O:3][H:4]>>[C:1][O:3].[O:2][H:4]",
            )
        except TypeError as exc:
            self.fail(f"max_seeds leaked into MCSMatcher: {exc}")

    def test_use_wl_true_produces_results(self):
        engine = RBLEngine(
            matcher_cls=ApproxMCSMatcher,
            mode="full",
            matcher_kwargs={"use_wl": True},
        )
        engine.process(
            "CCC(=O)OC>>CCC(=O)OCC",
            "[C:1][O:2].[O:3][H:4]>>[C:1][O:3].[O:2][H:4]",
        )
        self.assertTrue(engine.fused_rsmis)

    def test_approx_early_stop_true_does_not_crash(self):
        engine = RBLEngine(
            matcher_cls=ApproxMCSMatcher,
            matcher_kwargs={"max_seeds": 8, "max_steps": 64},
        )
        engine.process(
            "CCC(=O)OC>>CCC(=O)OCC",
            "[C:1][O:2].[O:3][H:4]>>[C:1][O:3].[O:2][H:4]",
        )


# ---------------------------------------------------------------------------
# Code-quality guards
# ---------------------------------------------------------------------------


class TestCodeQuality(unittest.TestCase):

    def test_approx_mcs_no_legacy_block_comment(self):
        import pathlib

        src = pathlib.Path("synkit/Graph/Matcher/approx_mcs.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn(
            '# from __future__ import annotations\n\n# """\n# approx_mcs_matcher',
            src,
        )

    def test_approx_mcs_starts_with_live_code(self):
        import pathlib

        first = (
            pathlib.Path("synkit/Graph/Matcher/approx_mcs.py")
            .read_text(encoding="utf-8")
            .splitlines()[0]
        )
        self.assertEqual(first, "from __future__ import annotations")


if __name__ == "__main__":
    unittest.main()
