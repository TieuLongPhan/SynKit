// SynKit carrier perception: constitutional stabilizers and explicit supports.
#include "synkit_canon/science.hpp"
#include <GraphMol/ROMol.h>
#include <GraphMol/Chirality.h>
#include <GraphMol/RingInfo.h>
#include <chrono>
#include <set>
#include <queue>
#include <numeric>
#include <optional>
#include <cmath>

namespace synkit { namespace science {
namespace {
using namespace stereo;
using Clock=std::chrono::steady_clock;
using Adj=std::vector<std::vector<int>>;
struct Budget {
  Options options; Clock::time_point start=Clock::now();
  ConfiguredOptions request() const {
    ConfiguredOptions result;
    result.seconds=options.seconds<0?-1:options.seconds-std::chrono::duration<double>(Clock::now()-start).count();
    if (options.seconds>=0 && result.seconds<=0) throw std::runtime_error("science_time_limit");
    return result;
  }
  bool iso(const Stereograph &a,const Stereograph &b) const {
    auto result=compare(a,b,request());
    if (!result.complete()) throw std::runtime_error("science_isomorphism_resource_stop");
    return result.isomorphic();
  }
};
Json ref_json(Reference r) {
  return r.kind==Resource::atom?Json(r.vertex):Json(std::string(r.kind==Resource::hydrogen?"@H:":"@LP:")+std::to_string(r.vertex));
}
Reference ref_read(const Json &j) {
  if (j.is_number_integer()) return Reference(j.get<int>());
  const auto s=j.get<std::string>(); auto split=s.find(':');
  if (split==std::string::npos || (s.substr(0,split)!="@H" && s.substr(0,split)!="@LP")) throw std::invalid_argument("Invalid reference token");
  size_t used=0; int owner=std::stoi(s.substr(split+1),&used);
  if (split+1+used!=s.size()) throw std::invalid_argument("Invalid virtual owner token");
  return Reference(owner,s.substr(0,split)=="@H"?Resource::hydrogen:Resource::lone_pair);
}
std::vector<Reference> neighbors(const RDKit::ROMol &m,int atom,int exclude=-1) {
  std::vector<Reference> refs;
  for (auto a:m.atomNeighbors(m.getAtomWithIdx(atom))) if (static_cast<int>(a->getIdx())!=exclude) refs.emplace_back(a->getIdx());
  return refs;
}
int lp(const RDKit::Atom &a) {
  const int z=a.getAtomicNum();
  return (z==7 || z==15 || z==16) && a.getDegree()+a.getTotalNumHs()==3 && a.getFormalCharge()<=0?1:0;
}
bool tetra(const RDKit::ROMol &m,const RDKit::Atom &a) {
  if (a.getIsAromatic() || a.getDegree()+a.getTotalNumHs()+lp(a)!=4) return false;
  int z=a.getAtomicNum();
  if (z==7 || z==15 || z==16) return a.getHybridization()==RDKit::Atom::SP2 || a.getHybridization()==RDKit::Atom::SP3;
  if (a.getHybridization()!=RDKit::Atom::SP3) return false;
  for (auto b:m.atomBonds(&a)) if (b->getBondType()!=RDKit::Bond::SINGLE) return false;
  return true;
}
Stereograph constitution(const RDKit::ROMol &m) {
  Stereograph g; g.profile="synkit-native/constitutional-carrier/1";
  for (auto a:m.atoms()) g.colors.push_back(Json::array({a->getAtomicNum(),a->getIsotope(),a->getFormalCharge(),a->getNumRadicalElectrons(),a->getTotalNumHs(),a->getIsAromatic()}).dump());
  for (auto b:m.bonds()) g.bonds.push_back({static_cast<int>(b->getBeginAtomIdx()),static_cast<int>(b->getEndAtomIdx()),Json::array({static_cast<int>(b->getBondType()),b->getIsAromatic()}).dump()});
  return g;
}
void pin(Stereograph &g,int v,int marker) { g.colors.at(v)=Json::array({g.colors.at(v),marker}).dump(); }
bool equivalent(Stereograph g,int a,int b,const Budget &budget) {
  if (g.colors.at(a)!=g.colors.at(b)) return false;
  auto h=g; pin(g,a,-1); pin(h,b,-1); return budget.iso(g,h);
}
std::optional<std::vector<Reference>> terminals(const RDKit::ROMol &m,int end,int next) {
  auto refs=neighbors(m,end,next); auto a=m.getAtomWithIdx(end);
  for (unsigned h=0;h<a->getTotalNumHs();++h) refs.emplace_back(end,Resource::hydrogen);
  if (refs.size()==1 && a->getAtomicNum()==15) refs.emplace_back(end,Resource::lone_pair);
  if (refs.size()!=2 || refs[0]==refs[1]) return {};
  return refs;
}
Configuration axis(const RDKit::ROMol &m,std::vector<int> path,Geometry geometry) {
  Configuration d; d.geometry=geometry; d.fixed=false; d.path.assign(path.begin(),path.end());
  auto l=terminals(m,path.front(),path[1]),r=terminals(m,path.back(),path[path.size()-2]);
  if (!l || !r) throw std::invalid_argument("Unavailable terminal references");
  d.frame={(*l)[0],(*l)[1],path.front(),path.back(),(*r)[0],(*r)[1]}; return d;
}
bool axis_confirmed(Stereograph g,const Configuration &d,const Budget &budget) {
  for (size_t i=0;i<d.path.size();++i) pin(g,d.path[i],static_cast<int>(i));
  for (int at:{0,4}) if (d.frame[at].kind==Resource::atom && d.frame[at+1].kind==Resource::atom && equivalent(g,d.frame[at].vertex,d.frame[at+1].vertex,budget)) return false;
  return true;
}
std::vector<std::vector<int>> components(const Adj &a) {
  std::vector<std::vector<int>> result; std::vector<bool> seen(a.size());
  for (size_t root=0;root<a.size();++root) if (!seen[root]) {
    std::vector<int> c{static_cast<int>(root)}; seen[root]=true;
    for (size_t i=0;i<c.size();++i) for (int v:a[c[i]]) if (!seen[v]) { seen[v]=true; c.push_back(v); }
    result.push_back(std::move(c));
  }
  return result;
}
std::vector<int> path(const Adj &a,int start,int end,bool unique=false) {
  std::vector<int> distance(a.size(),-1),parent(a.size(),-1),ways(a.size());
  std::queue<int> queue; queue.push(start); distance[start]=0; ways[start]=1;
  while (!queue.empty()) {
    int u=queue.front(); queue.pop();
    for (int v:a[u]) {
      if (distance[v]<0) { distance[v]=distance[u]+1; parent[v]=u; queue.push(v); }
      if (distance[v]==distance[u]+1) ways[v]=std::min(2,ways[v]+ways[u]);
    }
  }
  if (distance[end]<0 || (unique && ways[end]!=1)) return {};
  std::vector<int> result; for (int v=end;v>=0;v=parent[v]) result.push_back(v);
  std::reverse(result.begin(),result.end()); return result;
}
std::vector<Configuration> helices(const RDKit::ROMol &m,const Adj &molecular) {
  std::vector<std::vector<int>> rings;
  for (auto ring:m.getRingInfo()->atomRings()) if (ring.size()==6 && std::all_of(ring.begin(),ring.end(),[&](int v){return m.getAtomWithIdx(v)->getIsAromatic();})) rings.push_back(ring);
  Adj fusion(rings.size()); std::map<std::pair<int,int>,std::vector<int>> edges;
  for (size_t i=0;i<rings.size();++i) for (size_t j=i+1;j<rings.size();++j) {
    std::vector<int> shared; for (int v:rings[i]) if (std::find(rings[j].begin(),rings[j].end(),v)!=rings[j].end()) shared.push_back(v);
    if (shared.size()==2 && m.getBondBetweenAtoms(shared[0],shared[1])) { fusion[i].push_back(j); fusion[j].push_back(i); edges[{i,j}]=shared; edges[{j,i}]=shared; }
  }
  std::set<std::vector<int>> found;
  for (const auto &component:components(fusion)) {
    if (component.size()<5) continue;
    std::vector<int> ends; size_t sum=0; bool linear=true;
    for (int v:component) { sum+=fusion[v].size(); if (fusion[v].size()>2) linear=false; if (fusion[v].size()==1) ends.push_back(v); }
    if (!linear || ends.size()!=2 || sum!=2*(component.size()-1)) continue;
    std::sort(ends.begin(),ends.end()); auto order=path(fusion,ends[0],ends[1]); bool angular=true;
    auto edge_index=[](const std::vector<int> &ring,const std::vector<int> &edge) {
      for (size_t i=0;i<ring.size();++i) if ((ring[i]==edge[0] && ring[(i+1)%ring.size()]==edge[1]) || (ring[i]==edge[1] && ring[(i+1)%ring.size()]==edge[0])) return static_cast<int>(i);
      throw std::logic_error("Absent fused ring edge");
    };
    for (size_t i=1;i+1<order.size();++i) {
      const auto &l=edges.at({order[i-1],order[i]}),&r=edges.at({order[i],order[i+1]});
      int gap=std::abs(edge_index(rings[order[i]],l)-edge_index(rings[order[i]],r));
      if (std::min(gap,6-gap)!=2) angular=false;
      for (int v:l) if (std::find(r.begin(),r.end(),v)!=r.end()) angular=false;
    }
    if (!angular) continue;
    auto candidates=[&](int ring,const std::vector<int> &edge) {
      Adj perimeter(molecular.size()); std::set<int> allowed(rings[ring].begin(),rings[ring].end());
      for (int u:allowed) for (int v:molecular[u]) if (allowed.count(v) && !((u==edge[0] && v==edge[1]) || (u==edge[1] && v==edge[0]))) perimeter[u].push_back(v);
      auto free=path(perimeter,edge[0],edge[1]); return free.size()==6?std::vector<int>{free[1],free[4]}:std::vector<int>{};
    };
    auto l=candidates(order.front(),edges.at({order[0],order[1]})),r=candidates(order.back(),edges.at({order[order.size()-2],order.back()}));
    size_t best=SIZE_MAX; int count=0,left=-1,right=-1;
    for (int a:l) for (int b:r) { auto p=path(molecular,a,b); if (p.empty()) continue; if (p.size()<best) { best=p.size(); count=0; left=a; right=b; } if (p.size()==best) ++count; }
    if (count!=1) continue;
    auto p=path(molecular,left,right,true); if (p.size()<component.size()+1) continue;
    auto reverse=p; std::reverse(reverse.begin(),reverse.end()); found.insert(std::min(p,reverse));
  }
  std::vector<Configuration> result;
  for (const auto &p:found) { Configuration d; d.geometry=Geometry::helical; d.path.assign(p.begin(),p.end()); d.fixed=false; result.push_back(std::move(d)); }
  return result;
}
} // namespace
Json configuration_json(const stereo::Configuration &d) {
  Json refs=Json::array(); for (auto r:d.frame) refs.push_back(ref_json(r));
  Json frames=Json::array();
  for (const auto &f:d.frames) { Json ports=Json::array(); for (auto r:f.references) ports.push_back(ref_json(r)); frames.push_back({{"center",f.center},{"references",ports},{"relation",f.relation}}); }
  return {{"geometry",static_cast<int>(d.geometry)},{"parity",d.parity},{"fixed",d.fixed},{"frame",refs},{"path",d.path},{"cyclic",d.cyclic},
    {"support",d.support},{"pilot",d.pilot},{"coupling",d.coupling},{"frames",frames}};
}
stereo::Configuration configuration_from_json(const Json &j) {
  stereo::Configuration d; d.geometry=static_cast<stereo::Geometry>(j.at("geometry").get<int>());
  d.parity=j.value("parity",1); d.fixed=j.value("fixed",true); d.cyclic=j.value("cyclic",false);
  for (const auto &r:j.value("frame",Json::array())) d.frame.push_back(ref_read(r));
  d.path=j.value("path",std::vector<int32_t>{}); d.support=j.value("support",std::vector<int32_t>{});
  d.pilot=j.value("pilot",-1); d.coupling=j.value("coupling",std::string("null"));
  for (const auto &f:j.value("frames",Json::array())) {
    stereo::CoupledFrame frame; frame.center=f.at("center"); frame.relation=f.value("relation",1);
    if (f.at("references").size()!=4) throw std::invalid_argument("Invalid framework frame arity");
    for (int i=0;i<4;++i) frame.references[i]=ref_read(f.at("references").at(i)); d.frames.push_back(frame);
  }
  return d;
}
Json perceive(const RDKit::ROMol &m,const stereo::Stereograph &source,bool retain_neighbors,const Options &options) {
  using namespace stereo; Budget budget{options}; auto base=constitution(m);
  if (!std::isfinite(options.seconds) || (options.seconds<0 && options.seconds!=-1)) throw std::invalid_argument("Invalid perception deadline");
  Adj doubles(m.getNumAtoms()),molecular(m.getNumAtoms());
  for (auto b:m.bonds()) { int l=b->getBeginAtomIdx(),r=b->getEndAtomIdx(); molecular[l].push_back(r); molecular[r].push_back(l); if (b->getBondType()==RDKit::Bond::DOUBLE) { doubles[l].push_back(r); doubles[r].push_back(l); } }
  auto constraints=source.configurations;
  for (const auto &component:components(doubles)) {
    std::vector<int> ends; size_t sum=0; bool linear=true;
    for (int v:component) { sum+=doubles[v].size(); if (doubles[v].size()>2) linear=false; if (doubles[v].size()==1) ends.push_back(v); }
    if (!linear || ends.size()!=2 || sum<4 || sum%4!=0 || sum!=2*(component.size()-1)) continue;
    std::sort(ends.begin(),ends.end()); auto p=path(doubles,ends[0],ends[1]); int midpoint=p.size()/2;
    auto atom=m.getAtomWithIdx(p[midpoint]); auto tag=atom->getChiralTag();
    if (atom->getDegree()!=2 || (tag!=RDKit::Atom::CHI_TETRAHEDRAL_CW && tag!=RDKit::Atom::CHI_TETRAHEDRAL_CCW)) continue;
    if (!terminals(m,p.front(),p[1]) || !terminals(m,p.back(),p[p.size()-2])) continue;
    auto local=neighbors(m,atom->getIdx()); if (p[midpoint-1]!=local.front().vertex) std::reverse(p.begin(),p.end());
    auto d=axis(m,p,Geometry::cumulene_axis); d.fixed=true;
    if (tag==RDKit::Atom::CHI_TETRAHEDRAL_CCW) std::swap(d.frame[0],d.frame[1]); constraints.push_back(d);
  }
  Json evidence=Json::array(),elements=Json::array();
  auto emit=[&](const std::string &family,const Configuration &d,bool confirmed,const std::string &status="",const std::string &state="unspecified") {
    elements.push_back({{"family",family},{"support",configuration_json(d)},{"carrier_status",confirmed?"confirmed":"symmetry_related"},
      {"carrier_reason",confirmed?Json(nullptr):Json(family=="atrop_axis"?"symmetry_related_terminal_paths":"duplicate_terminal_ligands")},
      {"constitution_status",status},{"configuration_state",state}});
  };
  for (auto a:m.atoms()) if (tetra(m,*a)) {
    int center=a->getIdx(); auto refs=neighbors(m,center);
    for (unsigned h=0;h<a->getTotalNumHs();++h) refs.emplace_back(center,Resource::hydrogen);
    if (lp(*a)) refs.emplace_back(center,Resource::lone_pair);
    auto anchored=base; pin(anchored,center,0); bool singleton=true;
    for (size_t i=0;i<refs.size();++i) for (size_t j=i+1;j<refs.size();++j)
      if (refs[i]==refs[j] || (refs[i].kind==Resource::atom && refs[j].kind==Resource::atom && equivalent(anchored,refs[i].vertex,refs[j].vertex,budget))) singleton=false;
    bool repeated=std::set<Reference>(refs.begin(),refs.end()).size()!=refs.size();
    Configuration focal; focal.frame={Reference(center)}; focal.frame.insert(focal.frame.end(),refs.begin(),refs.end());
    auto orientation_distinct=[&](bool neighbors_retained) {
      if (repeated) return false;
      auto x=anchored;
      if (neighbors_retained) for (const auto &d:constraints) {
        if (!d.fixed || (d.geometry==Geometry::tetrahedral && !d.frame.empty() && d.frame[0].vertex==center)) continue;
        if (d.geometry==Geometry::tetrahedral || d.geometry==Geometry::planar_bond || d.geometry==Geometry::cumulene_axis) x.configurations.push_back(d);
      }
      x.configurations.push_back(focal); auto y=x; y.configurations.back().parity=-1; return !budget.iso(x,y);
    };
    bool confirmed=singleton || orientation_distinct(false); std::string status=confirmed?(singleton?"constitutionally_distinct":"constitutionally_orientation_distinct"):"symmetry_related";
    if (!confirmed && retain_neighbors && !constraints.empty() && orientation_distinct(true)) { confirmed=true; status="stereo_dependent_distinct"; }
    std::string state="unspecified";
    if (a->getChiralTag()==RDKit::Atom::CHI_TETRAHEDRAL_CW || a->getChiralTag()==RDKit::Atom::CHI_TETRAHEDRAL_CCW) state="specified";
    evidence.push_back({{"center",center},{"status",status},{"confirmed",confirmed},{"singleton_ligand_orbits",singleton},{"configuration_state",state}});
    if (confirmed) emit("tetrahedral",focal,true,status,state);
  }
  std::set<int> cumulated;
  for (const auto &component:components(doubles)) {
    size_t sum=0; bool linear=true; std::vector<int> ends;
    for (int v:component) { sum+=doubles[v].size(); if (doubles[v].size()>2) linear=false; if (doubles[v].size()==1) ends.push_back(v); }
    if (sum>=4) for (int v:component) for (int w:doubles[v]) cumulated.insert(m.getBondBetweenAtoms(v,w)->getIdx());
    if (!linear || ends.size()!=2 || sum<4 || sum!=2*(component.size()-1)) continue;
    std::sort(ends.begin(),ends.end()); auto p=path(doubles,ends[0],ends[1]);
    for (size_t i=1;i+1<p.size();++i) if (m.getAtomWithIdx(p[i])->getDegree()!=2) linear=false;
    if (!linear || !terminals(m,p.front(),p[1]) || !terminals(m,p.back(),p[p.size()-2])) continue;
    bool even=(sum/2)%2==0; auto d=axis(m,p,even?Geometry::cumulene_axis:Geometry::extended_cis_trans);
    emit(even?"cumulene_axis":"extended_cis_trans",d,axis_confirmed(base,d,budget));
  }
  std::set<int> perceived;
  for (const auto &info:RDKit::Chirality::findPotentialStereo(m)) if (info.type==RDKit::Chirality::StereoType::Bond_Double && !cumulated.count(info.centeredOn)) {
    auto b=m.getBondWithIdx(info.centeredOn); int l=b->getBeginAtomIdx(),r=b->getEndAtomIdx();
    Configuration d; d.geometry=Geometry::planar_bond; d.fixed=false; d.path={l,r};
    if (terminals(m,l,r) && terminals(m,r,l)) d=axis(m,{l,r},Geometry::planar_bond);
    emit("double_bond",d,true,"",info.specified==RDKit::Chirality::StereoSpecified::Specified?"specified":(info.specified==RDKit::Chirality::StereoSpecified::Unknown?"unknown":"unspecified")); perceived.insert(info.centeredOn);
  }
  for (auto b:m.bonds()) {
    int l=b->getBeginAtomIdx(),r=b->getEndAtomIdx();
    if (b->getBondType()==RDKit::Bond::DOUBLE && !cumulated.count(b->getIdx()) && !perceived.count(b->getIdx()) && terminals(m,l,r) && terminals(m,r,l)) {
      auto d=axis(m,{l,r},Geometry::planar_bond); bool isotope=false;
      for (int i:{0,1,4,5}) if (d.frame[i].kind==Resource::atom && m.getAtomWithIdx(d.frame[i].vertex)->getIsotope()) isotope=true;
      if (isotope && axis_confirmed(base,d,budget)) emit("double_bond",d,true);
    }
    bool standard=!b->getIsAromatic() && b->getBeginAtom()->getIsAromatic() && b->getEndAtom()->getIsAromatic();
    bool extended=false;
    if (options.include_extended_ring_axes) {
      bool large_ring=false;
      for (const auto &ring:m.getRingInfo()->bondRings()) if (ring.size()>6 && std::find(ring.begin(),ring.end(),b->getIdx())!=ring.end()) large_ring=true;
      auto lh=b->getBeginAtom()->getHybridization(),rh=b->getEndAtom()->getHybridization();
      bool linked=!b->getIsAromatic() && (m.getRingInfo()->numAtomRings(l) || m.getRingInfo()->numAtomRings(r)) &&
        (lh==RDKit::Atom::SP2 || rh==RDKit::Atom::SP2) &&
        (lh==RDKit::Atom::SP2 || lh==RDKit::Atom::SP3) && (rh==RDKit::Atom::SP2 || rh==RDKit::Atom::SP3);
      extended=(b->getIsAromatic() && b->getBeginAtom()->getIsAromatic() && b->getEndAtom()->getIsAromatic() && large_ring) || linked;
    }
    if ((b->getBondType()==RDKit::Bond::SINGLE || b->getBondType()==RDKit::Bond::AROMATIC) && (standard || extended)) {
      auto left=neighbors(m,l,r),right=neighbors(m,r,l);
      if (left.size()==2 && right.size()==2) { Configuration d; d.geometry=Geometry::atrop_bond; d.fixed=false; d.path={l,r}; d.frame={left[0],left[1],l,r,right[0],right[1]}; emit("atrop_axis",d,axis_confirmed(base,d,budget)); }
    }
  }
  for (const auto &d:helices(m,molecular)) emit("helical",d,true);
  return {{"complete",true},{"tetrahedral_evidence",evidence},{"elements",elements},{"retain_neighbor_orientation",retain_neighbors}};
}
}} // namespace
