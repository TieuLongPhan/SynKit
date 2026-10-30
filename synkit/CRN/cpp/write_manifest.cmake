# A CMake-built CRN library has the same sidecar contract as build_native().
file(READ "${BUILD_METADATA}" metadata)
string(JSON metadata SET "${metadata}" abi 1)
if(EXACT_ENABLED)
  string(JSON metadata SET "${metadata}" semiflows_abi 1)
  string(JSON metadata SET "${metadata}" capabilities
    "[\"closure\",\"structural_sets\",\"structural_unlimited\",\"exhaustive_siphons\",\"exact_semiflows\",\"exact_algebra\",\"crnt\",\"pathway\",\"symmetry\"]")
else()
  string(JSON metadata SET "${metadata}" semiflows_abi null)
  string(JSON metadata SET "${metadata}" capabilities
    "[\"closure\",\"structural_sets\",\"structural_unlimited\",\"exhaustive_siphons\"]")
endif()
string(JSON metadata SET "${metadata}" source_sha256 "{}")
file(GLOB sources
  "${SOURCE_DIR}/*.cpp" "${SOURCE_DIR}/*.h"
  "${SOURCE_DIR}/*.cmake" "${SOURCE_DIR}/CMakeLists.txt")
foreach(source IN LISTS sources)
  file(SHA256 "${source}" digest)
  get_filename_component(name "${source}" NAME)
  string(JSON metadata SET "${metadata}" source_sha256 "${name}" "\"${digest}\"")
endforeach()
string(SHA256 digest "${metadata}")
string(JSON metadata SET "${metadata}" build_sha256 "\"${digest}\"")
file(SHA256 "${LIBRARY_FILE}" digest)
string(JSON metadata SET "${metadata}" library_sha256 "\"${digest}\"")
get_filename_component(directory "${LIBRARY_FILE}" DIRECTORY)
get_filename_component(stem "${LIBRARY_FILE}" NAME_WE)
file(WRITE "${directory}/${stem}.json" "${metadata}\n")
