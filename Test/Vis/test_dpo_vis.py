# import unittest
# import os
# import time
# import shutil
# from synkit.Vis.dpo_vis import DPOVis


# class TestDPOVis(unittest.TestCase):
#     def setUp(self):
#         """
#         Set up the test case, initialize required data, and clean up the 'out' directory.
#         """
#         self.rsmi = "[CH2:1]=[CH:2][CH:3]=[CH2:4].[CH2:5]=[CH2:6]>>[CH2:1]1[CH:2]=[CH:3][CH2:4][CH2:5][CH2:6]1"
#         self.vis = DPOVis()

#         # Clean up the 'out' directory before each test
#         self._remove_out_directory()

#     def tearDown(self):
#         """
#         Clean up any leftover files after each test run.
#         """
#         self._remove_out_directory()

#     def _remove_out_directory(self):
#         """
#         Helper function to clean the 'out' directory by force removing its contents
#         and the directory itself.
#         """
#         out_dir = "out"
#         if os.path.exists(out_dir):
#             print(f"Attempting to clean up the {out_dir} directory...")

#             # First, attempt to clean all files inside the directory
#             for filename in os.listdir(out_dir):
#                 file_path = os.path.join(out_dir, filename)
#                 try:
#                     if os.path.isdir(file_path):
#                         shutil.rmtree(file_path)  # Remove directory and its contents
#                     else:
#                         os.remove(file_path)  # Remove individual files
#                 except Exception as e:
#                     print(f"Error removing {file_path}: {e}")

#             # After cleanup, attempt to remove the directory itself
#             try:
#                 shutil.rmtree(out_dir)
#                 print(f"{out_dir} directory cleaned and deleted.")
#             except OSError as e:
#                 print(f"Failed to delete {out_dir}: {e}. Retrying...")
#                 time.sleep(1)  # Wait before retrying
#                 try:
#                     shutil.rmtree(out_dir)
#                     print(f"{out_dir} directory cleaned and deleted on retry.")
#                 except Exception as e:
#                     print(f"Error during second attempt to delete {out_dir}: {e}")
#         else:
#             print(f"No {out_dir} directory found to clean.")

#     def test_dpo_vis_rsmi(self):
#         """
#         Test initialization of DPOVis with RSMI input, ensuring that the 'out' directory
#         is not empty after execution.
#         """
#         self.vis._vis(self.rsmi, compile=False, clean=False)

#         # Check that the 'out' directory exists and is not empty
#         out_dir = "out"
#         self.assertTrue(os.path.exists(out_dir), "The 'out' directory was not created.")

#         # Ensure the 'out' directory is not empty
#         files_in_out = os.listdir(out_dir)
#         self.assertGreater(
#             len(files_in_out), 0, "The 'out' directory is empty after running _vis."
#         )
