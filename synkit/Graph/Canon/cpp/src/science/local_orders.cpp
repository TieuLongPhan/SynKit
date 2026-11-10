#include "synkit_canon/science.hpp"
#include "../stereo_groups.h"
#include <GraphMol/ROMol.h>
#include <set>
#include <map>

namespace synkit { namespace science {
namespace {
using namespace stereo;
std::string locus(const Configuration &d) {
  const int family=static_cast<int>(d.geometry);
  if (family==11) return configuration_json(d).dump();
  if (family<=4) return "atom:"+std::to_string(family)+":"+std::to_string(d.frame.at(0).vertex);
  auto p=d.path;
  if (family<=6) p={d.frame.at(2).vertex,d.frame.at(3).vertex};
  auto reversed=p; std::reverse(reversed.begin(),reversed.end());
  return std::to_string(family)+":"+Json(std::min(p,reversed)).dump();
}
std::string formal_key(const Configuration &d) {
  const int family=static_cast<int>(d.geometry);
  if (family==11) return configuration_json(d).dump();
  if (family<=8) {
    auto group=synkit_stereo_detail::group(family==7?6:(family==8?5:family)); std::string result;
    for (int action=0;action<group.size;++action) {
      auto image=d;
      for (int i=0;i<group.arity;++i) image.frame[i]=d.frame[group.images[action][i]];
      if (family>=7 && group.images[action][2]==3) std::reverse(image.path.begin(),image.path.end());
      auto key=configuration_json(image).dump(); if (result.empty() || key<result) result=std::move(key);
    }
    return result;
  }
  std::string result;
  auto path=d.path;
  for (int reverse=0;reverse<2;++reverse) {
    auto p=path; if (reverse) std::reverse(p.begin(),p.end());
    for (size_t offset=0;offset<((family==10 || d.cyclic)?p.size():1);++offset) {
      auto image=d; image.path=p; std::rotate(image.path.begin(),image.path.begin()+offset,image.path.end());
      if (family==10 && reverse) image.parity=-image.parity;
      auto key=configuration_json(image).dump(); if (result.empty() || key<result) result=std::move(key);
    }
  }
  return result;
}
} // namespace
Json exhaustive_local_orders(const RDKit::ROMol &m,const stereo::Stereograph &source,const Options &options) {
  using namespace stereo; auto base=source; base.configurations.clear();
  auto perception_options=options; perception_options.include_extended_ring_axes=false;
  auto inventory=perceive(m,base,false,perception_options);
  std::map<std::string,Configuration> seeds;
  for (const auto &d:source.configurations) seeds.emplace(locus(d),d);
  for (const auto &element:inventory.at("elements")) {
    auto d=configuration_from_json(element.at("support"));
    if (d.geometry==Geometry::planar_bond && d.frame.empty()) {
      int l=d.path.at(0),r=d.path.at(1); std::vector<Reference> left,right;
      auto frame=[&](int owner,int other,std::vector<Reference> &refs) {
        auto a=m.getAtomWithIdx(owner); for (auto v:m.atomNeighbors(a)) if (static_cast<int>(v->getIdx())!=other) refs.emplace_back(v->getIdx());
        if (refs.size()==1) {
          if (a->getTotalNumHs()) refs.emplace_back(owner,Resource::hydrogen);
          else if ((a->getAtomicNum()==7 || a->getAtomicNum()==15) && a->getFormalCharge()<=0) refs.emplace_back(owner,Resource::lone_pair);
        }
      };
      frame(l,r,left); frame(r,l,right);
      if (left.size()!=2 || right.size()!=2) throw std::invalid_argument("Local planar carrier has unresolved ligand slots");
      d.frame={left[0],left[1],l,r,right[0],right[1]};
    }
    seeds.emplace(locus(d),d);
  }
  Json records=Json::array(); size_t raw_count=0; bool passed=true;
  for (const auto &[id,seed]:seeds) {
    auto arrangements=local_arrangements(seed);
    std::map<std::vector<uint8_t>,int> classes; std::map<std::string,int> formal_classes;
    std::map<int,int> formal_to_global;
    std::vector<Configuration> representatives; std::vector<int> labels,formal_labels;
    bool invariant=true,direct_agreement=true;
    ConfiguredOptions limit; limit.seconds=options.seconds;
    for (const auto &d:arrangements) {
      auto g=base; g.configurations={d}; auto code=canonicalize(g,limit);
      if (!code.complete) throw std::runtime_error("local_order_canonicalization_resource_stop");
      auto at=classes.emplace(code.code,static_cast<int>(classes.size())); int observed=at.first->second;
      if (at.second) representatives.push_back(d);
      auto formal=formal_classes.emplace(formal_key(d),static_cast<int>(formal_classes.size()));
      auto check=formal_to_global.emplace(formal.first->second,observed); invariant=invariant && check.first->second==observed;
      labels.push_back(observed); formal_labels.push_back(formal.first->second);
      if (static_cast<int>(d.geometry)<=6) {
        for (size_t i=0;i<representatives.size();++i) {
          auto h=base; h.configurations={representatives[i]}; auto comparison=compare(g,h,limit);
          if (!comparison.complete()) throw std::runtime_error("local_order_direct_check_resource_stop");
          direct_agreement=direct_agreement && comparison.isomorphic()==(static_cast<int>(i)==observed);
        }
      }
    }
    raw_count+=arrangements.size(); passed=passed && invariant && direct_agreement;
    records.push_back({{"locus",id},{"seed",configuration_json(seed)},{"geometry",static_cast<int>(seed.geometry)},
      {"raw_orders",arrangements.size()},{"formal_classes",formal_classes.size()},{"observed_classes",classes.size()},
      {"partition",labels},{"formal_partition",formal_labels},{"formal_orbit_invariance",invariant},
      {"direct_comparison_agreement",static_cast<int>(seed.geometry)<=6?Json(direct_agreement):Json(nullptr)}});
  }
  return {{"complete",true},{"passed",passed},{"carriers",records},{"raw_orders",raw_count},
    {"reference_scope","native formal-orbit and direct/canonical consistency; historical count regression is separate from independent truth"}};
}
}} // namespace
