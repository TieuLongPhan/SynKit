import unittest
import logging
from pathlib import Path

from rdkit import Chem

from synkit.Chem.Reaction.standardize import Standardize
from synkit.IO import load_database
from synkit.IO.chem_converter import detect_its_format, rsmi_to_its
from synkit.Synthesis.Reactor.syn_reactor import SynReactor


class TestSynReactorRealCases(unittest.TestCase):
    REAL_CASE_BATCH_SIZE = 100
    PROGRESS_STEPS = 10
    ERROR_LOG = Path("error.txt")

    @classmethod
    def setUpClass(cls):
        cls.data = load_database("./Data/smart.json.gz")
        cls.standardizer = Standardize()

    @staticmethod
    def _canonical_fragments(side: str) -> list[str]:
        fragments = []
        for fragment in side.split("."):
            mol = Chem.MolFromSmiles(fragment)
            if mol is None:
                raise AssertionError(f"Could not parse fragment: {fragment}")
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)
            fragments.append(Chem.MolToSmiles(Chem.RemoveHs(mol)))
        return sorted(fragments)

    def _round_trip_tuple_rc(self, index: int) -> None:
        smart = self.data[index]["smart"]
        rsmi = self.standardizer.fit(smart)
        reactants, products = rsmi.split(">>")
        rc = rsmi_to_its(smart, core=True, format="tuple")

        forward = SynReactor(
            substrate=reactants,
            template=rc,
            implicit_temp=False,
            explicit_h=False,
        )
        backward = SynReactor(
            substrate=products,
            template=rc,
            implicit_temp=False,
            explicit_h=False,
            invert=True,
        )

        forward_smis = [
            self.standardizer.fit(item, remove_aam=True) for item in forward.smarts
        ]
        backward_smis = [
            self.standardizer.fit(item, remove_aam=True) for item in backward.smarts
        ]

        self.assertEqual(detect_its_format(rc), "tuple")
        self.assertIn(rsmi, forward_smis)
        self.assertIn(rsmi, backward_smis)

    def _write_case_error(self, index: int, exc: BaseException) -> None:
        """Append one reproducible failed real-case record to ``error.txt``."""
        entry = self.data[index]
        smart = entry["smart"]
        try:
            rsmi = self.standardizer.fit(smart)
        except Exception as standardize_exc:
            rsmi = f"<standardization failed: {standardize_exc!r}>"

        with self.ERROR_LOG.open("a", encoding="utf-8") as handle:
            handle.write(
                "\n".join(
                    [
                        "=" * 88,
                        f"index: {index}",
                        f"reaction_id: {entry.get('R-id')}",
                        f"error_type: {type(exc).__name__}",
                        f"error: {exc}",
                        "smart:",
                        smart,
                        "standardized_rsmi:",
                        rsmi,
                        "",
                    ]
                )
            )

    def test_first_fixture_runs_through_tuple_template(self):
        smart = self.data[0]["smart"]
        substrate, expected_product = smart.split(">>")

        reactor = SynReactor(
            substrate,
            smart,
            explicit_h=False,
            template_format="tuple",
        )

        self.assertEqual(detect_its_format(reactor.rule.rc.raw), "tuple")
        self.assertTrue(reactor.its_list)
        self.assertEqual(len(reactor.smarts), 1)
        actual_product = reactor.smarts[0].split(">>")[1]
        self.assertEqual(
            self._canonical_fragments(actual_product),
            self._canonical_fragments(expected_product),
        )

    def test_first_fixture_round_trips_with_tuple_rc(self):
        self._round_trip_tuple_rc(0)

    def test_curated_tuple_rc_matrix_round_trips(self):
        for index in (0, 1, 2, 10, 25, 100):
            with self.subTest(index=index, reaction_id=self.data[index]["R-id"]):
                self._round_trip_tuple_rc(index)

    def test_backward_role_regression_from_index_33(self):
        self._round_trip_tuple_rc(33)

    def test_tuple_rc_round_trip_batch(self):
        logger = logging.getLogger(__name__)
        total = self.REAL_CASE_BATCH_SIZE
        progress_every = max(1, total // self.PROGRESS_STEPS)
        self.ERROR_LOG.unlink(missing_ok=True)

        for index in range(total):
            completed = index + 1
            if completed == 1 or completed % progress_every == 0 or completed == total:
                logger.info(
                    "tuple RC real-case progress: %d/%d (%.0f%%)",
                    completed,
                    total,
                    completed / total * 100,
                )
            with self.subTest(index=index, reaction_id=self.data[index]["R-id"]):
                try:
                    self._round_trip_tuple_rc(index)
                except Exception as exc:
                    self._write_case_error(index, exc)
                    raise


if __name__ == "__main__":
    unittest.main()
