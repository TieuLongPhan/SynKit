#include "synkit_canon/configured.hpp"
#include <iostream>
#include <numeric>
#include <set>
using namespace synkit::stereo;
namespace {
int checks=0;
void require(bool v,const char *message) { ++checks; if (!v) throw std::runtime_error(message); }
Stereograph star(Geometry geometry,int ligands) {
  Stereograph g; Configuration d; d.geometry=geometry;
  d.parity=geometry==Geometry::square_planar?0:1;
  for (int i=0;i<=ligands;++i) {
    g.colors.push_back("atom:"+std::to_string(i)); d.frame.emplace_back(i);
    if (i) g.bonds.push_back({0,i,"bond"});
  }
  g.configurations.push_back(d); return g;
}
Stereograph renamed(Stereograph g) {
  const int n=static_cast<int>(g.colors.size());
  auto rename=[&](int v) { return n-1-v; };
  std::reverse(g.colors.begin(),g.colors.end());
  for (auto &b:g.bonds) { b.source=rename(b.source); b.target=rename(b.target); }
  std::reverse(g.bonds.begin(),g.bonds.end());
  for (auto &d:g.configurations) {
    for (auto &r:d.frame) r.vertex=rename(r.vertex);
    for (auto &v:d.path) v=rename(v);
    for (auto &v:d.support) v=rename(v);
    if (d.pilot>=0) d.pilot=rename(d.pilot);
    for (auto &f:d.frames) { f.center=rename(f.center); for (auto &r:f.references) r.vertex=rename(r.vertex); }
  }
  std::reverse(g.configurations.begin(),g.configurations.end()); return g;
}
void test(Stereograph g,bool achiral) {
  auto a=canonicalize(g),b=canonicalize(renamed(g));
  require(a.complete && b.complete,"Canonical search stopped");
  require(a.code==b.code,"Renumbering changed configured code");
  require(a.atom_order.size()==g.colors.size(),"Atom order projection incomplete");
  auto result=compare(g,renamed(g)); require(result.complete() && result.isomorphic(),"Renamed graphs differ");
  result=compare(g,renamed(g),{},false); require(result.isomorphic(),"Faithful fallback differs");
  auto mirrored=classify_mirror(g); require(mirrored.complete && mirrored.achiral==achiral,"Incorrect native mirror decision");
  auto opposite=canonicalize(mirror(g)); require(opposite.complete && (a.code==opposite.code)==achiral,"Mirror code/iso disagreement");
  std::set<std::vector<uint8_t>> local_classes;
  for (const auto &drawing:local_arrangements(g.configurations.at(0))) {
    auto state=g; state.configurations={drawing}; auto code=canonicalize(state);
    require(code.complete,"Raw arrangement stopped"); local_classes.insert(code.code);
  }
  require(!local_classes.empty(),"Raw arrangement domain is empty");
}
} // namespace
int main() {
  try {
    test(star(Geometry::tetrahedral,4),false);
    test(star(Geometry::square_planar,4),true);
    test(star(Geometry::trigonal_bipyramidal,5),false);
    test(star(Geometry::octahedral,6),false);
    auto symmetric=star(Geometry::tetrahedral,4);
    for (size_t i=1;i<symmetric.colors.size();++i) symmetric.colors[i]="ligand";
    auto group=symmetry(symmetric,true);
    require(group.complete && group.group_order=="12" && group.automorphisms.size()==12,"Incorrect complete tetrahedral group");
    symmetric=star(Geometry::square_planar,4);
    for (size_t i=1;i<symmetric.colors.size();++i) symmetric.colors[i]="ligand";
    group=symmetry(symmetric,true);
    require(group.complete && group.group_order=="8" && group.automorphisms.size()==8,"Incorrect complete square group");
    auto virtual_h=star(Geometry::tetrahedral,4);
    virtual_h.colors.pop_back(); virtual_h.bonds.pop_back();
    virtual_h.configurations[0].frame[4]=Reference(0,Resource::hydrogen); test(virtual_h,false);
    auto lp=virtual_h; lp.configurations[0].frame[4].kind=Resource::lone_pair;
    require(canonicalize(lp).code!=canonicalize(virtual_h).code,"H/LP types collapsed");
    for (auto geometry:{Geometry::planar_bond,Geometry::atrop_bond,Geometry::cumulene_axis,Geometry::extended_cis_trans}) {
      Stereograph g; Configuration d; d.geometry=geometry;
      int n=geometry==Geometry::cumulene_axis?7:(geometry==Geometry::extended_cis_trans?8:6);
      for (int i=0;i<n;++i) g.colors.push_back("a:"+std::to_string(i));
      d.parity=geometry==Geometry::planar_bond || geometry==Geometry::extended_cis_trans?0:1;
      for (int i=0;i<6;++i) d.frame.emplace_back(i);
      g.bonds={{0,2,"b"},{1,2,"b"},{3,4,"b"},{3,5,"b"}};
      if (n==6) g.bonds.push_back({2,3,"b"});
      else {
        d.path={2,6}; if (n==8) d.path.push_back(7); d.path.push_back(3);
        for (size_t i=1;i<d.path.size();++i) g.bonds.push_back({d.path[i-1],d.path[i],"b"});
      }
      g.configurations.push_back(d); test(g,d.parity==0);
    }
    for (auto geometry:{Geometry::helical,Geometry::planar_chirality}) {
      Stereograph g; Configuration d; d.geometry=geometry; d.path={0,1,2,3,4};
      for (int i=0;i<5;++i) { g.colors.push_back("a:"+std::to_string(i)); if (i) g.bonds.push_back({i-1,i,"b"}); }
      if (geometry==Geometry::planar_chirality) {
        d.pilot=5; g.colors.push_back("pilot"); g.bonds.push_back({4,0,"b"}); g.bonds.push_back({0,5,"b"});
      }
      g.configurations.push_back(d); test(g,false);
      if (geometry==Geometry::planar_chirality) { for (int i=0;i<5;++i) g.colors[i]="plane"; test(g,true); }
    }
    auto framework=star(Geometry::tetrahedral,4);
    auto local=framework.configurations[0]; framework.configurations.clear();
    Configuration d; d.geometry=Geometry::framework; d.support={0,1,2,3,4};
    CoupledFrame f; f.center=0; std::copy(local.frame.begin()+1,local.frame.end(),f.references.begin()); d.frames.push_back(f);
    framework.configurations.push_back(d); test(framework,false);
    ConfiguredOptions options; options.max_expanded_vertices=2;
    auto stopped=canonicalize(framework,options);
    require(!stopped.complete && stopped.reason==3 && stopped.code.empty(),"Expansion cap produced a key");
    options={}; options.seconds=0; stopped=canonicalize(framework,options);
    require(!stopped.complete && stopped.reason==1 && stopped.atom_order.empty(),"Deadline produced a key");
    bool rejected=false; framework.configurations[0].frames[0].references[0]=Reference(4,Resource::hydrogen);
    try { canonicalize(framework); } catch (const std::invalid_argument &) { rejected=true; }
    require(rejected,"Invalid owner accepted");
    std::cout<<checks<<" configured C++ checks passed\n"; return 0;
  } catch (const std::exception &e) { std::cerr<<e.what()<<'\n'; return 1; }
}
