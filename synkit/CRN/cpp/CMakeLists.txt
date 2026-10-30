cmake_minimum_required(VERSION 3.20)
project(synkit_crn VERSION 1.6.5 LANGUAGES CXX)

option(SYNKIT_CRN_EXACT "Build exact Boost-backed CRN analysis" ON)
if(SYNKIT_CRN_EXACT)
  find_path(SYNKIT_CRN_BOOST_INCLUDE_DIR boost/multiprecision/cpp_int.hpp
    HINTS "$ENV{BOOST_INCLUDE_DIR}" "$ENV{BOOST_ROOT}"
    PATH_SUFFIXES include)
  if(NOT SYNKIT_CRN_BOOST_INCLUDE_DIR)
    message(FATAL_ERROR "CRN exact analysis requires Boost headers. Set BOOST_INCLUDE_DIR to the directory containing boost/.")
  endif()
endif()

add_library(synkit_crn SHARED closure.cpp structural.cpp)
if(SYNKIT_CRN_EXACT)
  target_sources(synkit_crn PRIVATE semiflows.cpp algebra.cpp pathway.cpp symmetry.cpp)
  target_include_directories(synkit_crn PRIVATE "${SYNKIT_CRN_BOOST_INCLUDE_DIR}")
endif()
target_compile_features(synkit_crn PRIVATE cxx_std_17)
set_target_properties(synkit_crn PROPERTIES
  CXX_EXTENSIONS OFF
  WINDOWS_EXPORT_ALL_SYMBOLS ON
  POSITION_INDEPENDENT_CODE ON)
set_property(TARGET synkit_crn APPEND PROPERTY LINK_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt"
  "${CMAKE_CURRENT_SOURCE_DIR}/write_manifest.cmake")
if(MSVC)
  target_compile_definitions(synkit_crn PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
  set_property(TARGET synkit_crn PROPERTY MSVC_RUNTIME_LIBRARY
    "MultiThreaded$<$<CONFIG:Debug>:Debug>")
elseif(MINGW)
  target_link_options(synkit_crn PRIVATE -static)
endif()

set(boost_version null)
if(SYNKIT_CRN_EXACT)
  file(STRINGS "${SYNKIT_CRN_BOOST_INCLUDE_DIR}/boost/version.hpp" boost_version
    REGEX "^#define BOOST_VERSION [0-9]+$")
  string(REGEX REPLACE ".* ([0-9]+)$" "\\1" boost_version "${boost_version}")
endif()
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/crn-build.json" CONTENT
  "{\"compiler\":\"${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}\",\"compiler_path\":\"${CMAKE_CXX_COMPILER}\",\"boost_version\":${boost_version}}\n")

add_custom_command(TARGET synkit_crn POST_BUILD
  COMMAND "${CMAKE_COMMAND}"
    "-DLIBRARY_FILE=$<TARGET_FILE:synkit_crn>"
    "-DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
    "-DBUILD_METADATA=${CMAKE_CURRENT_BINARY_DIR}/crn-build.json"
    "-DEXACT_ENABLED=${SYNKIT_CRN_EXACT}"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/write_manifest.cmake"
  VERBATIM)

install(TARGETS synkit_crn
  LIBRARY DESTINATION synkit/CRN/lib
  RUNTIME DESTINATION synkit/CRN/lib)
install(FILES
  "$<TARGET_FILE_DIR:synkit_crn>/$<TARGET_FILE_PREFIX:synkit_crn>$<TARGET_FILE_BASE_NAME:synkit_crn>.json"
  DESTINATION synkit/CRN/lib)
