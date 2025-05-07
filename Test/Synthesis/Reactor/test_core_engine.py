# import os
# import unittest
# import tempfile
# from synkit.Synthesis.Reactor.core_engine import CoreEngine


# class TestCoreEngine(unittest.TestCase):
#     def setUp(self):
#         # Create a temporary directory
#         self.temp_dir = tempfile.TemporaryDirectory()

#         # Path for the rule file
#         self.rule_file_path = os.path.join(self.temp_dir.name, "test_rule.gml")

#         # Define rule content
#         self.rule_content = """
#         rule [
#            ruleID "1"
#            left [
#               edge [ source 1 target 2 label "=" ]
#               edge [ source 3 target 4 label "-" ]
#            ]
#            context [
#               node [ id 1 label "C" ]
#               node [ id 2 label "C" ]
#               node [ id 3 label "H" ]
#               node [ id 4 label "H" ]
#            ]
#            right [
#               edge [ source 1 target 2 label "-" ]
#               edge [ source 1 target 3 label "-" ]
#               edge [ source 2 target 4 label "-" ]
#            ]
#         ]
#         """

#         # Write rule content to the temporary file
#         with open(self.rule_file_path, "w") as rule_file:
#             rule_file.write(self.rule_content)

#         # Initialize SMILES strings for testing
#         self.initial_smiles_fw = ["CC=CC", "[HH]"]
#         self.initial_smiles_bw = ["CCCC"]

#     def tearDown(self):
#         # Clean up temporary directory
#         self.temp_dir.cleanup()

#     def test_perform_reaction_forward(self):
#         # Test the perform_reaction method with forward reaction type
#         result = CoreEngine._inference(
#             rule_file_path=self.rule_file_path,
#             initial_smiles=self.initial_smiles_fw,
#             prediction_type="forward",
#             print_results=False,
#             verbosity=0,
#         )
#         print(result)
#         # Check if result is a list of strings and has content
#         self.assertIsInstance(
#             result, list, "Expected a list of reaction SMILES strings."
#         )
#         self.assertTrue(
#             len(result) > 0, "Result should contain reaction SMILES strings."
#         )

#         self.assertEqual(result[0], "CC=CC.[HH]>>CCCC")

#         # Check if the result SMILES format matches expected output format
#         for reaction_smiles in result:
#             self.assertIn(">>", reaction_smiles, "Reaction SMILES format is incorrect.")
#             parts = reaction_smiles.split(">>")
#             self.assertEqual(
#                 parts[0],
#                 ".".join(self.initial_smiles_fw),
#                 "Base SMILES are not correctly formatted.",
#             )
#             self.assertTrue(len(parts[1]) > 0, "Product SMILES should be non-empty.")

#     def test_perform_reaction_backward(self):
#         # Test the perform_reaction method with backward reaction type
#         result = CoreEngine._inference(
#             rule_file_path=self.rule_file_path,
#             initial_smiles=self.initial_smiles_bw,
#             prediction_type="backward",
#             print_results=False,
#             verbosity=0,
#         )
#         # Check if result is a list of strings and has content
#         self.assertIsInstance(
#             result, list, "Expected a list of reaction SMILES strings."
#         )
#         self.assertTrue(
#             len(result) > 0, "Result should contain reaction SMILES strings."
#         )
#         self.assertEqual(result[0], "C=CCC.[H][H]>>CCCC")
#         self.assertEqual(result[1], "[H][H].C(C)=CC>>CCCC")

#         # Check if the result SMILES format matches expected output format
#         for reaction_smiles in result:
#             self.assertIn(">>", reaction_smiles, "Reaction SMILES format is incorrect.")
#             parts = reaction_smiles.split(">>")
#             self.assertTrue(len(parts[0]) > 0, "Product SMILES should be non-empty.")
#             self.assertEqual(
#                 parts[1],
#                 ".".join(self.initial_smiles_bw),
#                 "Base SMILES are not correctly formatted.",
#             )


# if __name__ == "__main__":
#     unittest.main()
