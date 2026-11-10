#include "synkit_canon/science.hpp"
#include "rdkit_preparation.hpp"
#include <GraphMol/ROMol.h>
#include <GraphMol/Substruct/SubstructMatch.h>
#include <set>
#include <chrono>

namespace synkit { namespace science {
namespace {
using namespace stereo;
Configuration relabel(Configuration d,const std::vector<int> &mapping) {
  for (auto &r:d.frame) r.vertex=mapping.at(r.vertex);
  for (auto &v:d.path) v=mapping.at(v);
  return d;
}
std::vector<uint8_t> pinned_key(const Stereograph &source,const std::vector<Configuration> &registry,const Options &o) {
  auto g=source; g.configurations=registry;
  for (size_t i=0;i<g.colors.size();++i) g.colors[i]=Json::array({g.colors[i],i}).dump();
  ConfiguredOptions limit; limit.seconds=o.seconds;
  auto result=canonicalize(g,limit);
  if (!result.complete) throw std::runtime_error("coordinate_transport_key_resource_stop");
  return result.code;
}
} // namespace
std::vector<stereo::Configuration> coordinate_assistance(const RDKit::ROMol &m,const stereo::Stereograph &source,
    const RDKit::ROMol &coordinate,const std::vector<std::string> &units,Json &audit,const Options &options) {
  using namespace stereo;
  audit={{"complete",true},{"matches",0},{"added",0},{"target_labels_used",false}};
  if (m.getNumAtoms()!=coordinate.getNumAtoms() || m.getNumBonds()!=coordinate.getNumBonds()) return {};
  const auto start=std::chrono::steady_clock::now();
  auto current=[&]() { auto o=options; if (o.seconds>=0) { o.seconds-=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count(); if (o.seconds<=0) throw std::runtime_error("science_time_limit"); } return o; };
  RDKit::SubstructMatchParameters params; params.uniquify=false; params.useChirality=false; params.maxMatches=10001;
  auto raw=RDKit::SubstructMatch(coordinate,m,params);
  if (raw.size()>10000) throw std::runtime_error("unresolved_coordinate_transport:10000-map-limit");
  auto atoms=[](const RDKit::Atom &a) { return Json::array({a.getAtomicNum(),a.getIsotope(),a.getFormalCharge(),a.getTotalNumHs(),a.getNumRadicalElectrons(),a.getIsAromatic()}); };
  std::vector<std::vector<int>> maps;
  for (const auto &match:raw) {
    current(); std::vector<int> mapping(m.getNumAtoms(),-1); bool exact=true;
    for (auto [a,b]:match) mapping.at(a)=b;
    for (unsigned a=0;a<m.getNumAtoms();++a) if (mapping[a]<0 || atoms(*m.getAtomWithIdx(a))!=atoms(*coordinate.getAtomWithIdx(mapping[a]))) exact=false;
    if (!exact) continue;
    for (auto b:m.bonds()) { auto other=coordinate.getBondBetweenAtoms(mapping[b->getBeginAtomIdx()],mapping[b->getEndAtomIdx()]); if (!other || b->getBondType()!=other->getBondType() || b->getIsAromatic()!=other->getIsAromatic()) exact=false; }
    if (exact) maps.push_back(std::move(mapping));
  }
  audit["matches"]=maps.size(); if (maps.empty()) return {};
  std::set<std::string> requested;
  for (const auto &unit:units) {
    if (unit=="CT") requested.insert("double_bond");
    if (unit=="CT4") requested.insert("extended_cis_trans");
    if (unit=="HE") requested.insert("helical");
    if (unit=="TH3" || unit=="TH5") requested.insert("cumulene_axis");
  }
  std::vector<Configuration> sources;
  if (std::find(units.begin(),units.end(),"AT")!=units.end()) {
    const auto inventory=perceive(coordinate,prepare_rdkit_molecule(coordinate,true),true,current());
    std::vector<Configuration> axes;
    for (const auto &e:inventory["elements"]) if (e["family"]=="atrop_axis" && e["carrier_status"]=="confirmed") axes.push_back(configuration_from_json(e["support"]));
    if (axes.size()==1) try { sources.push_back(configure_geometry(coordinate,axes.front())); } catch (const std::invalid_argument &) {}
  }
  if (!requested.empty()) {
    auto inventory=perceive(m,source,true,current());
    for (const auto &e:inventory["elements"]) {
      if (!requested.count(e["family"].get<std::string>()) || e["carrier_status"]!="confirmed") continue;
      auto d=relabel(configuration_from_json(e["support"]),maps.front());
      if (d.geometry==Geometry::cumulene_axis) {
        bool material=true; for (int i:{0,1,4,5}) material=material && d.frame.at(i).kind==Resource::atom;
        auto tag=coordinate.getAtomWithIdx(d.path.at(d.path.size()/2))->getChiralTag();
        if (!material && tag!=RDKit::Atom::CHI_TETRAHEDRAL_CW && tag!=RDKit::Atom::CHI_TETRAHEDRAL_CCW) continue;
      }
      try { sources.push_back(configure_geometry(coordinate,d)); } catch (const std::invalid_argument &) {}
    }
  }
  if (sources.empty()) return {};
  // Compare registries with every target atom pinned. Agreement under an
  // arbitrary target automorphism would not establish unanimous transport.
  std::vector<Configuration> agreed; std::vector<uint8_t> key; bool first=true;
  for (const auto &mapping:maps) {
    std::vector<int> inverse(mapping.size()); for (size_t i=0;i<mapping.size();++i) inverse[mapping[i]]=i;
    std::vector<Configuration> registry; for (const auto &d:sources) registry.push_back(relabel(d,inverse));
    auto candidate=pinned_key(source,registry,current());
    if (first) { first=false; agreed=registry; key=std::move(candidate); }
    else if (key!=candidate) { audit["unanimous"]=false; return {}; }
  }
  audit["unanimous"]=true;
  std::set<std::vector<uint8_t>> seen;
  std::vector<Configuration> unique;
  for (const auto &d:agreed) if (seen.insert(pinned_key(source,{d},current())).second) unique.push_back(d);
  audit["added"]=unique.size(); return unique;
}
}} // namespace
