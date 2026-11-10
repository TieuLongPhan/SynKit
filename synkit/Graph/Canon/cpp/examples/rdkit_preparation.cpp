// Source-declared chemistry preparation through RDKit's public C++ API.
// No third-party stereo matcher, canonicalizer or permutation table is copied.
#include "rdkit_preparation.hpp"
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/Chirality.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/PeriodicTable.h>
#include <GraphMol/Resonance.h>
#include <nlohmann/json.hpp>
#include <cmath>
#include <set>
#include <numeric>

using namespace synkit::stereo;
namespace {
int lone_pairs(const RDKit::Atom &a) {
  double bonds=0;
  if (a.getIsAromatic()) bonds=a.getTotalValence()-a.getTotalNumHs();
  else for (const auto b:a.getOwningMol().atomBonds(&a)) bonds+=b->getBondTypeAsDouble();
  const double electrons=RDKit::PeriodicTable::getTable()->getNouterElecs(a.getAtomicNum())-
    a.getFormalCharge()-a.getNumRadicalElectrons()-bonds-a.getTotalNumHs();
  return std::max(0,static_cast<int>(std::floor((electrons+1e-8)/2)));
}
Reference missing(const RDKit::Atom &a) {
  if (a.getTotalNumHs()>0) return Reference(a.getIdx(),Resource::hydrogen);
  if (lone_pairs(a)>0) return Reference(a.getIdx(),Resource::lone_pair);
  throw std::invalid_argument("Missing ligand is neither represented H nor a lone pair");
}
std::vector<Reference> neighbors(const RDKit::ROMol &m,const RDKit::Atom &a,int exclude=-1) {
  std::vector<Reference> refs;
  for (const auto v:m.atomNeighbors(&a)) if (static_cast<int>(v->getIdx())!=exclude) refs.emplace_back(v->getIdx());
  return refs;
}
Stereograph convert(RDKit::ROMol &mol,bool allow_unknown=false,bool extract_stereo=true) {
  if (!mol.getStereoGroups().empty()) throw std::invalid_argument("Enhanced stereo groups require a population contract");
  RDKit::MolOps::assignStereochemistry(mol,false,true);
  Stereograph g; g.profile="rdkit-source/lewis_state/1";
  for (const auto a:mol.atoms()) {
    nlohmann::json color={a->getAtomicNum(),a->getIsotope(),a->getFormalCharge(),
      a->getNumRadicalElectrons(),a->getIsAromatic(),a->getTotalNumHs(),lone_pairs(*a)};
    g.colors.push_back("rdkit-atom:"+color.dump());
    if (!extract_stereo) continue;
    auto tag=a->getChiralTag(); Configuration d;
    if (tag==RDKit::Atom::CHI_TETRAHEDRAL_CW || tag==RDKit::Atom::CHI_TETRAHEDRAL_CCW) {
      d.geometry=Geometry::tetrahedral; d.parity=tag==RDKit::Atom::CHI_TETRAHEDRAL_CW?1:-1;
      auto refs=neighbors(mol,*a); if (refs.size()==3) refs.push_back(missing(*a));
      // A two-coordinate axial tag is not a four-slot tetrahedral descriptor.
      if (refs.size()!=4) continue;
      d.frame.emplace_back(a->getIdx()); d.frame.insert(d.frame.end(),refs.begin(),refs.end());
      g.configurations.push_back(d);
    } else if (tag==RDKit::Atom::CHI_SQUAREPLANAR || tag==RDKit::Atom::CHI_TRIGONALBIPYRAMIDAL || tag==RDKit::Atom::CHI_OCTAHEDRAL) {
      const int count=tag==RDKit::Atom::CHI_SQUAREPLANAR?4:(tag==RDKit::Atom::CHI_TRIGONALBIPYRAMIDAL?5:6);
      d.geometry=count==4?Geometry::square_planar:(count==5?Geometry::trigonal_bipyramidal:Geometry::octahedral);
      d.parity=count==4?0:1;
      auto refs=neighbors(mol,*a); std::vector<int> bonds;
      for (const auto b:mol.atomBonds(a)) bonds.push_back(b->getIdx());
      if (refs.size()+a->getTotalNumHs()!=static_cast<size_t>(count) && refs.size()!=static_cast<size_t>(count))
        throw std::invalid_argument("Unresolved non-tetrahedral coordination");
      while (refs.size()<static_cast<size_t>(count)) { refs.emplace_back(a->getIdx(),Resource::hydrogen); bonds.push_back(-1); }
      std::vector<int> order(count); std::iota(order.begin(),order.end(),0); bool found=false;
      do {
        RDKit::INT_LIST probe; for (int i:order) probe.push_back(bonds[i]);
        if (RDKit::Chirality::getChiralPermutation(a,probe)!=1) continue;
        // Only the documented permutation-1 positional convention is needed;
        // the public RDKit API performs numbering transport for all variants.
        d.frame.emplace_back(a->getIdx());
        if (count==4) for (int i:order) d.frame.push_back(refs[i]);
        else {
          d.frame.push_back(refs[order.front()]); d.frame.push_back(refs[order.back()]);
          for (int i=1;i<count-1;++i) d.frame.push_back(refs[order[i]]);
        }
        found=true; break;
      } while (std::next_permutation(order.begin(),order.end()));
      if (!found) {
        if (!allow_unknown) throw std::invalid_argument("Unspecified or unsupported non-tetrahedral configuration");
        d.fixed=false; d.frame={Reference(a->getIdx())};
        d.frame.insert(d.frame.end(),refs.begin(),refs.end());
      }
      g.configurations.push_back(d);
    }
  }
  for (const auto b:mol.bonds()) {
    nlohmann::json color={b->getBondTypeAsDouble(),static_cast<int>(b->getBondType()),b->getIsAromatic()};
    g.bonds.push_back({static_cast<int32_t>(b->getBeginAtomIdx()),static_cast<int32_t>(b->getEndAtomIdx()),"rdkit-bond:"+color.dump()});
    if (!extract_stereo) continue;
    auto stereo=b->getStereo(); Configuration d;
    const int left=b->getBeginAtomIdx(),right=b->getEndAtomIdx();
    bool axial=stereo==RDKit::Bond::STEREOATROPCW || stereo==RDKit::Bond::STEREOATROPCCW;
    bool planar=stereo==RDKit::Bond::STEREOE || stereo==RDKit::Bond::STEREOZ ||
      stereo==RDKit::Bond::STEREOCIS || stereo==RDKit::Bond::STEREOTRANS;
    if (!axial && !planar) continue;
    auto l=neighbors(mol,*mol.getAtomWithIdx(left),right),r=neighbors(mol,*mol.getAtomWithIdx(right),left);
    if (l.size()==1) l.push_back(missing(*mol.getAtomWithIdx(left)));
    if (r.size()==1) r.push_back(missing(*mol.getAtomWithIdx(right)));
    if (l.size()!=2 || r.size()!=2) throw std::invalid_argument("Invalid bond stereo references");
    if (planar) {
      const auto &selected=b->getStereoAtoms(); if (selected.size()!=2) throw std::invalid_argument("Missing E/Z reference pair");
      if (l[0].vertex!=selected[0]) std::swap(l[0],l[1]);
      if (r[0].vertex!=selected[1]) std::swap(r[0],r[1]);
      if (stereo==RDKit::Bond::STEREOE || stereo==RDKit::Bond::STEREOTRANS) std::swap(r[0],r[1]);
    }
    d.geometry=axial?Geometry::atrop_bond:Geometry::planar_bond;
    d.parity=axial?(stereo==RDKit::Bond::STEREOATROPCW?1:-1):0;
    d.frame={l[0],l[1],left,right,r[0],r[1]}; g.configurations.push_back(d);
  }
  return g;
}
std::string drawing(const RDKit::ROMol &m) {
  std::string key;
  for (const auto a:m.atoms()) key+=std::to_string(a->getFormalCharge())+",";
  key+=';'; for (const auto b:m.bonds()) key+=std::to_string(static_cast<int>(b->getBondType()))+",";
  return key;
}
} // namespace
Stereograph prepare_rdkit_molecule(const RDKit::ROMol &source,bool allow_unknown) {
  RDKit::ROMol copy(source); return convert(copy,allow_unknown);
}
std::vector<Stereograph> prepare_rdkit_smiles(const std::string &smiles,bool resonance) {
  std::unique_ptr<RDKit::RWMol> mol(RDKit::SmilesToMol(smiles));
  if (!mol) throw std::invalid_argument("SMILES parse failure");
  return prepare_rdkit_family(*mol,resonance);
}
std::vector<Stereograph> prepare_rdkit_family(const RDKit::ROMol &source,bool resonance) {
  auto mol=std::make_unique<RDKit::RWMol>(source);
  auto original=convert(*mol);
  if (!resonance) return {original};
  const unsigned max_forms=4096;
  RDKit::ResonanceMolSupplier supplier(*mol,RDKit::ResonanceMolSupplier::ALLOW_CHARGE_SEPARATION | RDKit::ResonanceMolSupplier::KEKULE_ALL,max_forms);
  if (supplier.length()>=max_forms) throw std::invalid_argument("Resonance supplier reached safety cap");
  std::vector<std::unique_ptr<RDKit::RWMol>> pending;
  for (unsigned i=0;i<supplier.length();++i) {
    std::unique_ptr<RDKit::ROMol> form(supplier[i]); pending.push_back(std::make_unique<RDKit::RWMol>(*form));
  }
  if (pending.empty()) pending.push_back(std::make_unique<RDKit::RWMol>(*mol));
  std::set<std::string> seen; std::vector<Stereograph> result;
  while (!pending.empty()) {
    auto form=std::move(pending.back()); pending.pop_back();
    if (!seen.insert(drawing(*form)).second) continue;
    if (seen.size()>=max_forms) throw std::invalid_argument("Resonance closure reached safety cap");
    auto aromatic=*form; RDKit::MolOps::setAromaticity(aromatic);
    auto graph=convert(aromatic,false,false); graph.configurations=original.configurations; result.push_back(std::move(graph));
    // The explicitly declared terminal phosphate charge/double-bond move.
    for (const auto p:form->atoms()) if (p->getAtomicNum()==15 && p->getFormalCharge()==0 &&
        p->getDegree()==4 && !p->getNumRadicalElectrons() && !p->getTotalNumHs()) {
      std::vector<int> doubles,singles;
      for (const auto oxygen:form->atomNeighbors(p)) if (oxygen->getAtomicNum()==8 && oxygen->getDegree()==1 &&
          !oxygen->getTotalNumHs() && !oxygen->getNumRadicalElectrons()) {
        const auto b=form->getBondBetweenAtoms(p->getIdx(),oxygen->getIdx());
        if (oxygen->getFormalCharge()==0 && b->getBondType()==RDKit::Bond::DOUBLE) doubles.push_back(oxygen->getIdx());
        if (oxygen->getFormalCharge()==-1 && b->getBondType()==RDKit::Bond::SINGLE) singles.push_back(oxygen->getIdx());
      }
      for (int l:doubles) for (int r:singles) {
        auto alternate=std::make_unique<RDKit::RWMol>(*form);
        alternate->getAtomWithIdx(l)->setFormalCharge(-1); alternate->getAtomWithIdx(r)->setFormalCharge(0);
        alternate->getBondBetweenAtoms(p->getIdx(),l)->setBondType(RDKit::Bond::SINGLE);
        alternate->getBondBetweenAtoms(p->getIdx(),r)->setBondType(RDKit::Bond::DOUBLE);
        RDKit::MolOps::sanitizeMol(*alternate); pending.push_back(std::move(alternate));
      }
    }
  }
  return result;
}
