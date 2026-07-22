import unittest

from synkit.IO import its_to_rsmi, rsmi_to_its
from synkit.Synthesis.Reactor.rbl_engine import RBLEngine
from synkit.Synthesis.Reactor.fusion_validation import validate_fusion_rsmi


class TestRBLEngineExamples(unittest.TestCase):
    """Integration-style tests using legacy RBL examples."""

    def _run_engine_to_rsmi(
        self,
        rsmi: str,
        template: str,
        *,
        replace_wc: bool = True,
    ) -> list[str]:
        """
        Helper: run RBLEngine.process and return fused RSMIs.

        We explicitly set early_stop=False to ensure the full
        forward/backward + fusion pipeline is exercised rather than
        the quick-check path.
        """
        engine = RBLEngine(early_stop=False)
        engine.process(rsmi, template, replace_wc=replace_wc)
        return engine.fused_rsmis

    def test_example1(self) -> None:
        """
        Example 1 from the original RBL tests.

        Alcohol + acid → ester; template is provided as a fully mapped
        reaction, internally converted to ITS and back into a canonical
        RSMI form before use.
        """
        rsmi = "CCC(=O)(O)>>CCC(=O)OC"
        raw_template = (
            "[CH3:1][C:2](=[O:3])[OH:4]."
            "[CH3:5][O:6][H:7]>>"
            "[CH3:1][C:2](=[O:3])[O:6][CH3:5]."
            "[H:7][OH:4]"
        )

        # Mirror the original behaviour: template was built as ITS and then
        # serialized again to RSMI before passing to the engine.
        its_template = rsmi_to_its(raw_template, core=True)
        template = its_to_rsmi(its_template)

        result = self._run_engine_to_rsmi(rsmi, template, replace_wc=True)

        self.assertTrue(result)
        self.assertTrue(all(">>" in candidate for candidate in result))
        self.assertTrue(
            all(validate_fusion_rsmi(candidate).valid for candidate in result)
        )

    def test_example2(self) -> None:
        """
        Example 2 from the original RBL tests.

        Simple acyl transfer / transesterification pattern.
        """
        rsmi = "CCC(=O)OC>>CCC(=O)OCC"
        template = "[C:1][O:2].[O:3][H:4]>>[C:1][O:3].[O:2][H:4]"

        result = self._run_engine_to_rsmi(rsmi, template, replace_wc=True)

        self.assertTrue(result)
        self.assertTrue(all(">>" in candidate for candidate in result))
        self.assertTrue(
            all(validate_fusion_rsmi(candidate).valid for candidate in result)
        )


class DummyQuickCheckRBLEngine(RBLEngine):
    """
    Minimal subclass to test the quick-check path deterministically
    without mocks/patches.

    We override ``_quick_check`` to return a fixed solution whenever
    early_stop=True. This avoids depending on the actual SynReactor
    behaviour in this specific test.
    """

    def _quick_check(  # type: ignore[override]
        self,
        rsmi: str,
        template: str | object,
    ) -> str | None:
        # Only pretend to succeed for a specific marker reaction so
        # the behaviour is well-defined in the test.
        if rsmi == "[CH3:1]>>[CH3:1]":
            return "[CH3:1]>>[CH3:1]"
        return None


class TestRBLEngineAPI(unittest.TestCase):
    """Light API / error-handling tests for RBLEngine."""

    def test_invalid_reaction_string_raises(self) -> None:
        """Reaction strings without '>>' should raise ValueError."""
        engine = RBLEngine()
        with self.assertRaises(ValueError):
            engine.process(
                "no_arrow_here",
                "[C:1][O:2].[O:3][H:4]>>[C:1][O:3].[O:2][H:4]",
            )

    def test_repr_contains_key_info(self) -> None:
        """__repr__ should expose key configuration fields."""
        engine = RBLEngine(wildcard_element="X*", node_attrs=["element"])
        rep = repr(engine)
        self.assertIn("RBLEngine", rep)
        self.assertIn("wildcard_element='X*'", rep)
        self.assertIn("reactor_cls=", rep)

    def test_diagnostics_are_grouped_by_reactor_stage(self) -> None:
        engine = RBLEngine(early_stop=False, electron_diagnostics=True)
        engine.process(
            "CCC(=O)OC>>CCC(=O)OCC",
            "[C:1][O:2].[O:3][H:4]>>[C:1][O:3].[O:2][H:4]",
        )

        self.assertEqual(
            set(engine.diagnostics),
            {"forward", "backward", "quick_check", "fusion"},
        )
        self.assertTrue(engine.diagnostics["forward"])
        self.assertTrue(engine.diagnostics["backward"])
        self.assertIn("diagnostics", engine.result)

    def test_quick_check_short_circuits_pipeline(self) -> None:
        """
        When early_stop=True and _quick_check succeeds, process()
        should:

        - populate fused_rsmis with the quick-check solution only,
        - leave fused_its empty,
        - skip the full forward/backward + fusion pipeline.
        """
        engine = DummyQuickCheckRBLEngine(early_stop=True)

        engine.process("[CH3:1]>>[CH3:1]", "[C:1]>>[C:1]")

        self.assertEqual(engine.fused_rsmis, ["[CH3:1]>>[CH3:1]"])
        self.assertEqual(engine.fused_its, [])


if __name__ == "__main__":
    unittest.main()
