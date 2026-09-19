import os,subprocess,sys
from pathlib import Path
r=Path(__file__).resolve().parent
env=dict(os.environ,SYNKIT_TEST_NATIVE_LIBRARY=(r/sys.argv[1]).read_text().strip())
files=[
"Test/Chem/Mapper/module/"+name+".py" for name in (
"test_exact_distance","test_exact_symmetry","test_graph_automorphism",
"test_mapper_analysis","test_mapper_spectrum","test_distance_profile_bounds",
"test_dynamic_profile_search","test_exact_seed","test_seed_fragments",
"test_seed_relaxation","test_structure_code_cache","test_tight_profile_propagation",
"test_two_sided_symmetry")]
files.append("Test/Graph/Canon/test_exact_ir.py")
subprocess.run(["taskset","-c","12","/home/labhhc4/anaconda3/envs/synfrag/bin/pytest","-q",*files],env=env,check=True)
