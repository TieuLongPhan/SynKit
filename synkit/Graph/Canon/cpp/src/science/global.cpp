#include "synkit_canon/science.hpp"
#include "rdkit_preparation.hpp"
#include <GraphMol/ROMol.h>
#include <chrono>
#include <set>

namespace synkit { namespace science {
namespace {
using namespace stereo;
void topology_profile(const RDKit::ROMol &m,Stereograph &g) {
  g.profile="acs_topology/element-isotope-hydrogen-connectivity";
  for (auto a:m.atoms()) g.colors.at(a->getIdx())=Json::array({a->getAtomicNum(),a->getIsotope(),a->getTotalNumHs()}).dump();
  for (auto &b:g.bonds) b.color="bond";
}
} // namespace
Json source_mirror_identity(const RDKit::ROMol &m,const std::string &profile,const Options &options) {
  const auto start=std::chrono::steady_clock::now();
  auto request=[&]() { ConfiguredOptions o; o.seconds=options.seconds; if (o.seconds>=0) { o.seconds-=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count(); if (o.seconds<=0) throw std::runtime_error("science_time_limit"); } return o; };
  if (profile!="chemical" && profile!="lewis_state" && profile!="acs_topology") throw std::invalid_argument("Unknown molecular identity profile");
  auto family=prepare_rdkit_family(m,profile=="chemical");
  if (profile=="acs_topology") for (auto &g:family) topology_profile(m,g);
  Json out={{"identity_profile",profile},{"declared_resonance_drawings",family.size()},
    {"descriptor_count",family.at(0).configurations.size()},{"whole_molecule_annotation_completeness_claimed",false}};
  auto a=canonicalize_family(family,request());
  if (!a.complete) { out["complete"]=false; out["reason"]=a.reason; return out; }
  for (auto &g:family) g=mirror(g);
  auto b=canonicalize_family(family,request());
  out["complete"]=b.complete; out["reason"]=b.reason;
  if (b.complete) { out["status"]=a.code==b.code?"achiral":"chiral"; out["full_certificate_bytes"]=a.code.size()+b.code.size(); }
  return out;
}
Json analyze_global_support(const RDKit::ROMol &m,const stereo::Stereograph &source,const Options &options) {
  using namespace stereo; auto g=source; topology_profile(m,g); g.configurations.clear();
  std::set<int> represented; for (const auto &d:source.configurations) if (d.geometry==Geometry::tetrahedral && !d.frame.empty()) represented.insert(d.frame[0].vertex);
  Configuration framework; framework.geometry=Geometry::framework;
  std::vector<int> completed;
  for (auto a:m.atoms()) {
    int center=a->getIdx(); if (represented.count(center) || a->getHybridization()!=RDKit::Atom::SP3) continue;
    std::vector<Reference> refs; for (auto v:m.atomNeighbors(a)) refs.emplace_back(v->getIdx());
    if (refs.size()==3 && a->getTotalNumHs()==1) refs.emplace_back(center,Resource::hydrogen);
    else if (refs.size()!=4) continue;
    CoupledFrame f; f.center=center; std::copy(refs.begin(),refs.end(),f.references.begin());
    framework.frames.push_back(f); completed.push_back(center);
  }
  Json result={{"complete",true},{"state","potential"},{"necessarily_chiral",false},
    {"completed_frame_centers",completed},{"orientation_selected",false},{"identity_profile",g.profile}};
  if (framework.frames.empty()) return result;
  std::set<int> support{completed.front()}; std::vector<int> pending{completed.front()};
  for (size_t i=0;i<pending.size();++i) for (auto a:m.atomNeighbors(m.getAtomWithIdx(pending[i]))) if (support.insert(a->getIdx()).second) pending.push_back(a->getIdx());
  if (!std::all_of(completed.begin(),completed.end(),[&](int c){return support.count(c);}) || support.size()>256 || framework.frames.size()>64) {
    result["state"]="unsupported"; result["complete"]=false; result["reason"]="framework_support_or_component_limit"; return result;
  }
  framework.support.assign(support.begin(),support.end()); g.configurations={framework};
  ConfiguredOptions limit; limit.seconds=options.seconds;
  auto mirror_result=classify_mirror(g,limit);
  result["complete"]=mirror_result.complete; result["reason"]=mirror_result.reason;
  if (mirror_result.complete) {
    result["necessarily_chiral"]=!mirror_result.achiral;
    result["state"]=mirror_result.achiral?"potential":"necessarily_chiral";
    result["mirror_mapping"]=mirror_result.mapping;
  }
  return result;
}
Json topology_completion_identity(const RDKit::ROMol &m,const stereo::Stereograph &source,const Options &options) {
  using namespace stereo; auto g=source; topology_profile(m,g);
  std::set<int> represented;
  for (const auto &d:g.configurations) if (d.geometry==Geometry::tetrahedral && !d.frame.empty()) represented.insert(d.frame[0].vertex);
  std::vector<int> added;
  for (auto a:m.atoms()) {
    if (represented.count(a->getIdx()) || a->getHybridization()!=RDKit::Atom::SP3) continue;
    Configuration d; d.frame={Reference(a->getIdx())};
    for (auto v:m.atomNeighbors(a)) d.frame.emplace_back(v->getIdx());
    if (d.frame.size()==4 && a->getTotalNumHs()==1) d.frame.emplace_back(a->getIdx(),Resource::hydrogen);
    if (d.frame.size()!=5) continue;
    g.configurations.push_back(d); added.push_back(a->getIdx());
  }
  ConfiguredOptions limit; limit.seconds=options.seconds; auto result=classify_mirror(g,limit);
  Json out={{"complete",result.complete},{"reason",result.reason},{"completed_centers",added},
    {"identity_profile",g.profile},{"orientation_policy","arbitrary common probe orientation; no recovery of missing physical configuration"}};
  if (result.complete) { out["status"]=result.achiral?"achiral":"chiral"; out["mapping"]=result.mapping; }
  return out;
}
}} // namespace
