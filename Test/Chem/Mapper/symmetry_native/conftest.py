"""An isolated native build verifies owned modules without root build hooks."""

import ctypes as ct
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def symmetry_library(tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("C++17 compiler unavailable")
    root = Path(__file__).resolve().parents[4]
    cpp = root / "synkit/Chem/Mapper/solvers/native/cpp"
    destination = tmp_path_factory.mktemp("native-symmetry")
    shim = destination / "shim.cpp"
    shim.write_text("""
#include "synkit_mapper/core/runtime.h"
extern "C" int symmetry_test_cpabs(const SynkitMapperProblem *p,
 const SynkitMapperRequest *r, SynkitMapperCallback callback,void *data,
 SynkitMapperStatistics *statistics) {
 try {
  synkit_mapper::validate(*p,*r);
  if(r->backend==SYNKIT_ASTAR) return synkit_mapper::solve_astar(*p,*r,callback,data,*statistics);
  return synkit_mapper::solve_cpabs(*p,*r,callback,data,*statistics);
 } catch(...) { return -1; }
}
extern "C" int synkit_mapper_solve(const SynkitMapperProblem *p,
 const SynkitMapperRequest *r, SynkitMapperCallback callback,void *data,
 SynkitMapperStatistics *statistics) {
 return symmetry_test_cpabs(p,r,callback,data,statistics);
}
extern "C" int synkit_mapper_solve_domains(const SynkitMapperProblem *p,
 const SynkitMapperRequest *r,const uint8_t *allowed,SynkitMapperCallback callback,
 void *data,SynkitMapperStatistics *statistics) {
 try {
  synkit_mapper::validate(*p,*r);
  if(!allowed) return -1;
  for(int i=0;i<p->n*p->n;++i) if(allowed[i]>1) return -1;
  if(r->backend!=SYNKIT_KBB) {synkit_mapper::initialize(*statistics,*r);return synkit_mapper::unsupported(*statistics);}
  return synkit_mapper::solve_abb(*p,*r,allowed,callback,data,*statistics);
 } catch(...) {return -1;}
}
extern "C" int synkit_mapper_core_abi() {return 1;}
extern "C" uint64_t synkit_mapper_capabilities() {return (uint64_t{1}<<6)|(uint64_t{1}<<5)|(uint64_t{1}<<SYNKIT_KBB);}
extern "C" const char *synkit_mapper_backend_version(int) {return "isolated-test";}
extern "C" const char *synkit_mapper_core_error() {return "isolated test shim";}
extern "C" void *synkit_mapper_cancel_create() {return new synkit_mapper::Cancellation;}
extern "C" void synkit_mapper_cancel_request(void *p) {
 static_cast<synkit_mapper::Cancellation*>(p)->requested.store(true);
}
extern "C" void synkit_mapper_cancel_free(void *p) {delete static_cast<synkit_mapper::Cancellation*>(p);}
""")
    library = destination / "libsymmetry_test.so"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-pedantic",
            "-fPIC",
            "-shared",
            "-I",
            str(cpp / "include"),
            str(shim),
            str(cpp / "src/symmetry/classification.cpp"),
            str(cpp / "src/symmetry/classification_pipeline.cpp"),
            str(cpp / "src/symmetry/cpabs.cpp"),
            str(cpp / "src/pabs/bridge.cpp"),
            str(cpp / "src/pabs/sparse_embedding.cpp"),
            str(cpp / "src/core/pmcd_witness.cpp"),
            str(cpp / "src/symmetry/edge_deck.cpp"),
            str(cpp / "src/slap.cpp"),
            str(cpp / "src/astar/bound.cpp"),
            str(cpp / "src/astar/astar.cpp"),
            str(cpp / "src/assignment/abb.cpp"),
            str(cpp / "src/hydrogen/hydrogen.cpp"),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    ct.CDLL(str(library))
    return library
