#ifndef SYNKIT_CHEMISTRY_RDKIT_HPP
#define SYNKIT_CHEMISTRY_RDKIT_HPP
#include "configured.hpp"
namespace RDKit { class ROMol; }
// Optional synkit_science library, using the public RDKit C++ SDK.
synkit::stereo::Stereograph prepare_rdkit_molecule(const RDKit::ROMol &,bool allow_unknown=false);
std::vector<synkit::stereo::Stereograph> prepare_rdkit_family(const RDKit::ROMol &,bool resonance);
std::vector<synkit::stereo::Stereograph> prepare_rdkit_smiles(const std::string &,bool resonance);
#endif
