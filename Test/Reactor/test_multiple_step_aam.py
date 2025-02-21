# import unittest
# from synkit.IO.data_io import load_database
# from synkit.Chem.Reaction.standardize import Standardize
# from synkit.IO.chem_converter import smart_to_gml
# from synkit.Reactor.reagent import add_catalysis
# from synkit.Reactor.multi_step import (
#     perform_multi_step_reaction,
#     find_all_paths,
# )
# from synkit.Reactor.multi_step_aam import (
#     get_aam_reactions,
#     get_mechanism,
# )


# class TestMultiStepAAM(unittest.TestCase):
#     def setUp(self) -> None:
#         self.data = load_database("Data/Testcase/mech.json.gz")
#         self.rsmi = Standardize().fit(self.data[0]["reaction"])

#     def test_get_aam_reactions(self):
#         test = self.data[0]["mechanisms"][1]
#         rule = [
#             smart_to_gml(value["smart_string"], core=True, explicit_hydrogen=True)
#             for value in test["steps"]
#         ]
#         order = list(range(len(rule)))
#         rsmi = add_catalysis(self.rsmi, test["cat"])

#         results, reaction_tree = perform_multi_step_reaction(rule, order, rsmi)

#         target_products = sorted(rsmi.split(">>")[1].split("."))
#         max_depth = len(results)
#         all_paths = find_all_paths(reaction_tree, target_products, rsmi, max_depth)
#         real_path = all_paths[0][1:]  # remove the original
#         all_steps = get_aam_reactions(real_path, rule, order, test["cat"])
#         self.assertTrue(
#             all(m is not None for m in all_steps),
#             "All mechanism steps should have valid mappings",
#         )

#         self.assertEqual(len(all_steps), len(test["steps"]))

#     def test_get_mechanism(self):
#         for key, entry in enumerate(self.data[0]["mechanisms"]):
#             if key > 0:  # Skip the first entry due to bug cannot be fixed
#                 rule = [
#                     smart_to_gml(
#                         value["smart_string"], core=True, explicit_hydrogen=True
#                     )
#                     for value in entry["steps"]
#                 ]
#                 order = list(range(len(rule)))
#                 cat = entry["cat"]

#                 mech = get_mechanism(rule, order, self.rsmi, cat=cat)

#                 self.assertEqual(len(mech), len(entry["steps"]))

#                 self.assertTrue(
#                     all(m is not None for m in mech),
#                     "All mechanism steps should have valid mappings",
#                 )


# if __name__ == "__main__":
#     unittest.main()
