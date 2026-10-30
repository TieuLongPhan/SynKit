"""An isolated native build verifies owned modules without root build hooks."""

import ctypes as ct
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def symmetry_library(tmp_path_factory):
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake unavailable for the isolated C++17 symmetry fixture")
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
    (destination / "CMakeLists.txt").write_text("""
cmake_minimum_required(VERSION 3.16)
project(synkit_symmetry_test LANGUAGES CXX)
add_library(symmetry_test SHARED
  shim.cpp
  "${MAPPER_CPP}/src/symmetry/classification.cpp"
  "${MAPPER_CPP}/src/symmetry/classification_pipeline.cpp"
  "${MAPPER_CPP}/src/symmetry/cpabs.cpp"
  "${MAPPER_CPP}/src/pabs/bridge.cpp"
  "${MAPPER_CPP}/src/pabs/sparse_embedding.cpp"
  "${MAPPER_CPP}/src/core/pmcd_witness.cpp"
  "${MAPPER_CPP}/src/symmetry/edge_deck.cpp"
  "${MAPPER_CPP}/src/slap.cpp"
  "${MAPPER_CPP}/src/astar/bound.cpp"
  "${MAPPER_CPP}/src/astar/astar.cpp"
  "${MAPPER_CPP}/src/assignment/abb.cpp"
  "${MAPPER_CPP}/src/hydrogen/hydrogen.cpp")
target_include_directories(symmetry_test PRIVATE "${MAPPER_CPP}/include")
target_compile_features(symmetry_test PRIVATE cxx_std_17)
set_target_properties(symmetry_test PROPERTIES
  CXX_EXTENSIONS OFF POSITION_INDEPENDENT_CODE ON WINDOWS_EXPORT_ALL_SYMBOLS ON)
if(MSVC)
  target_compile_options(symmetry_test PRIVATE /W4)
  set_property(TARGET symmetry_test PROPERTY MSVC_RUNTIME_LIBRARY
    "MultiThreaded$<$<CONFIG:Debug>:Debug>")
else()
  target_compile_options(symmetry_test PRIVATE -Wall -Wextra -Wpedantic)
  if(MINGW)
    target_link_options(symmetry_test PRIVATE -static -static-libgcc -static-libstdc++)
  endif()
endif()
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/$<CONFIG>/symmetry-library.txt"
  CONTENT "$<TARGET_FILE:symmetry_test>")
""")
    build = destination / "build"
    commands = [
        [cmake, "-S", str(destination), "-B", str(build),
         "-DCMAKE_BUILD_TYPE=Release", f"-DMAPPER_CPP={cpp.as_posix()}"],
        [cmake, "--build", str(build), "--config", "Release", "--parallel", "2"],
    ]
    for command in commands:
        subprocess.run(command, check=True, capture_output=True, text=True)
    # CMake supplies the platform suffix and the generator's configuration path.
    library = Path((build / "Release/symmetry-library.txt").read_text().strip())
    ct.CDLL(str(library))
    return library
