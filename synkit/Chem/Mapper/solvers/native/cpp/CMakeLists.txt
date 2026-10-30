cmake_minimum_required(VERSION 3.16)
project(synkit_mapper VERSION 1.6.5 LANGUAGES C CXX)

add_library(synkit_mapper SHARED
  src/slap.cpp src/core/dispatch.cpp src/core/pmcd_witness.cpp
  src/pabs/bridge.cpp src/pabs/sparse_embedding.cpp src/symmetry/edge_deck.cpp
  src/assignment/abb.cpp
  src/supporting/dfs.cpp src/supporting/edit_support.cpp src/sepdp/sepdp.cpp
  src/mces/mces.cpp
  src/hydrogen/hydrogen.cpp
  src/astar/astar.cpp src/astar/bound.cpp
  src/symmetry/classification.cpp src/symmetry/classification_pipeline.cpp src/symmetry/cpabs.cpp)
target_compile_definitions(synkit_mapper PRIVATE SYNKIT_MAPPER_WITH_ASTAR SYNKIT_MAPPER_WITH_CPABS SYNKIT_MAPPER_WITH_ASSIGNMENT SYNKIT_MAPPER_WITH_SUPPORTING SYNKIT_MAPPER_WITH_MCES)
target_compile_features(synkit_mapper PUBLIC cxx_std_17)
target_include_directories(synkit_mapper PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

# Only the sparse canonical-labeling engine and its transitive dependencies
# are vendored. Keep third-party compiler settings separate from our C++ core.
set(SYNKIT_NAUTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/symmetry/vendor/nauty)
option(SYNKIT_MAPPER_WITH_NAUTY "Enable optional sparse edge-deck certificates" ${UNIX})
if(SYNKIT_MAPPER_WITH_NAUTY)
  add_library(synkit_mapper_nauty STATIC
    ${SYNKIT_NAUTY_DIR}/nauty.c ${SYNKIT_NAUTY_DIR}/nautil.c
    ${SYNKIT_NAUTY_DIR}/nausparse.c ${SYNKIT_NAUTY_DIR}/naugraph.c
    ${SYNKIT_NAUTY_DIR}/schreier.c ${SYNKIT_NAUTY_DIR}/naurng.c)
  set_target_properties(synkit_mapper_nauty PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_include_directories(synkit_mapper PRIVATE ${SYNKIT_NAUTY_DIR})
  target_link_libraries(synkit_mapper PRIVATE synkit_mapper_nauty)
  target_compile_definitions(synkit_mapper PRIVATE SYNKIT_MAPPER_WITH_NAUTY)
endif()

option(SYNKIT_MAPPER_TEST_HOOKS "Build private mathematical-oracle entry points" OFF)
if(SYNKIT_MAPPER_TEST_HOOKS)
  target_compile_definitions(synkit_mapper PRIVATE SYNKIT_MAPPER_TEST_HOOKS)
endif()
if(MSVC)
  target_compile_options(synkit_mapper PRIVATE /W4)
else()
  target_compile_options(synkit_mapper PRIVATE -Wall -Wextra -Wpedantic)
endif()

option(SYNKIT_MAPPER_WITH_RDKIT "Build standalone native reaction preparation/rendering" OFF)
if(SYNKIT_MAPPER_WITH_RDKIT)
  find_package(rdkit CONFIG REQUIRED)
  add_library(synkit_mapper_reaction SHARED src/reaction/reaction.cpp)
  # The optional RDKit 2026 SDK uses constexpr destructors in public headers.
  # Keep the dependency-free mapper core on C++17.
  target_compile_features(synkit_mapper_reaction PUBLIC cxx_std_20)
  target_include_directories(synkit_mapper_reaction PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
  target_link_libraries(synkit_mapper_reaction PRIVATE synkit_mapper
    RDKit::GraphMol RDKit::SmilesParse RDKit::RDGeneral)
  add_executable(mapper_reaction examples/map_reaction.cpp)
  target_link_libraries(mapper_reaction PRIVATE synkit_mapper_reaction)
endif()
set_target_properties(synkit_mapper PROPERTIES
  CXX_EXTENSIONS OFF
  WINDOWS_EXPORT_ALL_SYMBOLS ON
  POSITION_INDEPENDENT_CODE ON)

option(SYNKIT_MAPPER_WITH_ORTOOLS "Build CP-SAT, SCIP and certified PABS PDLP bounds" OFF)
if(SYNKIT_MAPPER_WITH_ORTOOLS)
  find_package(ortools CONFIG REQUIRED)
  target_sources(synkit_mapper PRIVATE src/solvers/ortools.cpp)
  target_link_libraries(synkit_mapper PRIVATE ortools::ortools)
  target_compile_definitions(synkit_mapper PRIVATE SYNKIT_MAPPER_WITH_ORTOOLS)
endif()
set(SYNKIT_MAPPER_OPENWBO_EXECUTABLE "" CACHE FILEPATH "Qualified native Open-WBO executable")
set(SYNKIT_MAPPER_EVALMAXSAT_EXECUTABLE "" CACHE FILEPATH "Qualified native EvalMaxSAT executable")
if(SYNKIT_MAPPER_OPENWBO_EXECUTABLE OR SYNKIT_MAPPER_EVALMAXSAT_EXECUTABLE)
  if(SYNKIT_MAPPER_OPENWBO_EXECUTABLE AND NOT EXISTS "${SYNKIT_MAPPER_OPENWBO_EXECUTABLE}")
    message(FATAL_ERROR "Open-WBO executable does not exist")
  endif()
  target_sources(synkit_mapper PRIVATE src/solvers/maxsat.cpp)
  target_compile_definitions(synkit_mapper PRIVATE SYNKIT_MAPPER_WITH_MAXSAT)
  if(SYNKIT_MAPPER_OPENWBO_EXECUTABLE)
    target_compile_definitions(synkit_mapper PRIVATE
      SYNKIT_MAPPER_OPENWBO_EXECUTABLE="${SYNKIT_MAPPER_OPENWBO_EXECUTABLE}")
  endif()
  if(SYNKIT_MAPPER_EVALMAXSAT_EXECUTABLE)
    if(NOT EXISTS "${SYNKIT_MAPPER_EVALMAXSAT_EXECUTABLE}")
      message(FATAL_ERROR "EvalMaxSAT executable does not exist")
    endif()
    target_compile_definitions(synkit_mapper PRIVATE
      SYNKIT_MAPPER_EVALMAXSAT_EXECUTABLE="${SYNKIT_MAPPER_EVALMAXSAT_EXECUTABLE}")
  endif()
endif()

option(SYNKIT_MAPPER_BUILD_EXAMPLES "Build standalone C++ mapper examples" OFF)
if(SYNKIT_MAPPER_BUILD_EXAMPLES)
  add_executable(slap_graph examples/slap_graph.cpp)
  target_link_libraries(slap_graph PRIVATE synkit_mapper)
endif()

include(GNUInstallDirs)
option(SYNKIT_MAPPER_WHEEL "Install only the runtime library inside the Python wheel" OFF)
if(SYNKIT_MAPPER_WHEEL)
  if(MSVC AND NOT SYNKIT_MAPPER_WITH_ORTOOLS)
    # Standard wheels must not depend on a separately installed MSVC C++ runtime.
    set_property(TARGET synkit_mapper PROPERTY MSVC_RUNTIME_LIBRARY
      "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  endif()
  install(TARGETS synkit_mapper
    LIBRARY DESTINATION synkit/Chem/Mapper/solvers/native/lib
    RUNTIME DESTINATION synkit/Chem/Mapper/solvers/native/lib)
else()
  install(TARGETS synkit_mapper
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
  install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  if(SYNKIT_MAPPER_WITH_RDKIT)
    install(TARGETS synkit_mapper_reaction mapper_reaction
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  endif()
endif()
