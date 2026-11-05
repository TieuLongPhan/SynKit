cmake_minimum_required(VERSION 3.20)
project(synkit_crn_expand VERSION 1.6.5 LANGUAGES CXX)

# Native rule-based CRN expansion against an RDKit C++ SDK.
find_package(rdkit CONFIG REQUIRED)
find_package(Boost CONFIG REQUIRED)

add_executable(synkit_crn_expand expand.cpp)
# RDKit 2026 public headers use constexpr destructors (C++20).
target_compile_features(synkit_crn_expand PRIVATE cxx_std_20)
set_target_properties(synkit_crn_expand PROPERTIES
  CXX_EXTENSIONS OFF
  BUILD_RPATH_USE_ORIGIN ON)
find_package(Threads REQUIRED)
target_link_libraries(synkit_crn_expand PRIVATE
  RDKit::ChemReactions RDKit::SubstructMatch RDKit::SmilesParse
  RDKit::GraphMol RDKit::RDGeneral Boost::headers Threads::Threads)
target_compile_definitions(synkit_crn_expand PRIVATE BOOST_JSON_NO_LIB BOOST_CONTAINER_NO_LIB)
if(NOT MSVC)
  target_compile_options(synkit_crn_expand PRIVATE -Wall -Wextra)
endif()
