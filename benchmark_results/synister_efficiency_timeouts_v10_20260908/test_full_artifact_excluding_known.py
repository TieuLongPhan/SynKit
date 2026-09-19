import os,subprocess,sys
from pathlib import Path
r=Path(__file__).resolve().parent
env=dict(os.environ,SYNKIT_TEST_NATIVE_LIBRARY=(r/sys.argv[1]).read_text().strip())
files=["Test/Chem/Mapper/module","Test/Graph/Canon/test_exact_ir.py"]
subprocess.run(["taskset","-c","12","/home/labhhc4/anaconda3/envs/synfrag/bin/pytest","-q","-k","not test_frozen_alternative_its_case_is_bound_to_current_implementation",*files],env=env,check=True)
