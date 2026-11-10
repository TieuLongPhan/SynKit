#include "synkit_canon/science.hpp"
#include <GraphMol/ROMol.h>
#include <GraphMol/PeriodicTable.h>
#include <GraphMol/RingInfo.h>
#include <chrono>
#include <cmath>
#include <cctype>
#include <set>
#include <numeric>
#include <optional>

namespace synkit { namespace science {
namespace {
using namespace stereo;
using Signature=std::vector<std::pair<double,double>>;
using Clock=std::chrono::steady_clock;
const char *rules[]={"1a_atomic_number","1b_duplicate_distance","2_isotope_mass",
                     "3_sequence_geometry","4_stereogenic_unit","5_reflection_variant"};
struct Limit {
  Options options; Clock::time_point start=Clock::now(); int64_t occurrences=0;
  explicit Limit(const Options &o):options(o) {
    if (!std::isfinite(o.seconds) || (o.seconds<0 && o.seconds!=-1) || o.max_depth<0 || o.max_occurrences<1)
      throw std::invalid_argument("Invalid native science limits");
  }
  void check(int64_t count=0) {
    occurrences+=count;
    if (occurrences>options.max_occurrences) throw std::runtime_error("science_occurrence_limit");
    if (options.seconds>=0 && std::chrono::duration<double>(Clock::now()-start).count()>=options.seconds)
      throw std::runtime_error("science_time_limit");
  }
};
Json reference(Reference r) {
  if (r.kind==Resource::atom) return r.vertex;
  return std::string(r.kind==Resource::hydrogen?"@H:":"@LP:")+std::to_string(r.vertex);
}
std::set<int> dependencies(const Configuration &d) {
  std::set<int> out(d.path.begin(),d.path.end()); out.insert(d.support.begin(),d.support.end());
  for (auto r:d.frame) out.insert(r.vertex);
  if (d.pilot>=0) out.insert(d.pilot);
  for (const auto &f:d.frames) { out.insert(f.center); for (auto r:f.references) out.insert(r.vertex); }
  return out;
}
bool supported(Geometry g) {
  return g==Geometry::tetrahedral || g==Geometry::planar_bond || g==Geometry::atrop_bond ||
    g==Geometry::cumulene_axis || g==Geometry::extended_cis_trans || g==Geometry::helical;
}
double mass(const RDKit::Atom &a) {
  const auto *table=RDKit::PeriodicTable::getTable();
  return a.getIsotope()?table->getMassForIsotope(a.getAtomicNum(),a.getIsotope()):table->getAtomicWeight(a.getAtomicNum());
}
struct Node { double number,mass; int duplicate_depth=-1; };
struct Evidence {
  std::vector<std::vector<Node>> spheres;
  std::map<int,int> depths;
  bool unsupported_mancude=false,truncated=false;
};
struct Occurrence { int atom,parent; std::vector<int> path; int depth; };
struct Comparison {
  int order=0,rule=-1,depth=-1; bool unsupported=false;
  Json left,right; std::string reason;
  Json json() const {
    return {{"outcome",unsupported?"unsupported":(order>0?"left_higher":(order<0?"right_higher":"tie"))},
      {"deciding_rule",rule>=0?Json(rules[rule]):Json(nullptr)},{"depth",depth<0?Json(nullptr):Json(depth)},
      {"left_witness",left},{"right_witness",right},{"reason",reason}};
  }
};
struct Ranking {
  std::vector<Reference> refs,ordered;
  std::vector<Comparison> comparisons;
  bool complete=false;
  Json json() const {
    Json supplied=Json::array(),order=Json::array(),evidence=Json::array();
    for (auto r:refs) supplied.push_back(reference(r));
    for (auto r:ordered) order.push_back(reference(r));
    for (const auto &c:comparisons) evidence.push_back(c.json());
    return {{"references",supplied},{"ordered_references",order},{"comparisons",evidence},{"complete",complete}};
  }
};
struct Ranker {
  const RDKit::ROMol &mol; const Stereograph &graph; const std::map<int,std::string> &labels;
  int focal; Limit &limit; int max_depth;
  std::map<std::pair<int,Reference>,Evidence> cache;
  Ranker(const RDKit::ROMol &m,const Stereograph &g,const std::map<int,std::string> &l,int f,Limit &budget)
    :mol(m),graph(g),labels(l),focal(f),limit(budget),max_depth(budget.options.max_depth?budget.options.max_depth:std::max(4,static_cast<int>(m.getNumAtoms())*2+2)) {}
  Node node(int v,int duplicate=-1) const {
    const auto *a=mol.getAtomWithIdx(v); return {static_cast<double>(a->getAtomicNum()),mass(*a),duplicate};
  }
  std::optional<Node> mancude(int v,int depth) {
    std::vector<int> neighbors;
    for (const auto b:mol.atomBonds(mol.getAtomWithIdx(v))) if (b->getIsAromatic()) neighbors.push_back(b->getOtherAtomIdx(v));
    int rings=0,length=0;
    for (const auto &ring:mol.getRingInfo()->atomRings()) {
      if (std::find(ring.begin(),ring.end(),v)==ring.end()) continue;
      bool aromatic=true; for (int a:ring) aromatic=aromatic && mol.getAtomWithIdx(a)->getIsAromatic();
      if (aromatic) { ++rings; length=static_cast<int>(ring.size()); }
    }
    if (neighbors.size()!=2 || rings!=1 || length%2) return {};
    auto a=node(neighbors[0]),b=node(neighbors[1]); return Node{(a.number+b.number)/2,(a.mass+b.mass)/2,depth};
  }
  const Evidence &build(int center,Reference r) {
    limit.check(); auto key=std::make_pair(center,r); auto at=cache.find(key); if (at!=cache.end()) return at->second;
    if (center<0 || static_cast<unsigned>(center)>=mol.getNumAtoms() || r.vertex<0 || static_cast<unsigned>(r.vertex)>=mol.getNumAtoms())
      throw std::invalid_argument("Absent CIP reference");
    Evidence e;
    if (r.kind!=Resource::atom) {
      if (r.vertex!=center || (r.kind!=Resource::hydrogen && r.kind!=Resource::lone_pair)) throw std::invalid_argument("Invalid virtual CIP owner");
      const double hydrogen=RDKit::PeriodicTable::getTable()->getAtomicWeight(1);
      e.spheres.push_back({{r.kind==Resource::hydrogen?1.0:0.0,r.kind==Resource::hydrogen?hydrogen:0.0,-1}});
      return cache.emplace(key,std::move(e)).first->second;
    }
    if (!mol.getBondBetweenAtoms(center,r.vertex)) throw std::invalid_argument("Nonadjacent CIP reference");
    e.spheres.push_back({node(r.vertex)}); e.depths[r.vertex]=1;
    std::vector<Occurrence> active{{r.vertex,center,{center,r.vertex},1}};
    while (!active.empty()) {
      int depth=active.front().depth+1;
      if (depth>max_depth) { e.truncated=true; break; }
      std::vector<Node> nodes; std::vector<Occurrence> next;
      for (const auto &o:active) {
        limit.check(1); const auto *a=mol.getAtomWithIdx(o.atom);
        for (unsigned h=0;h<a->getNumExplicitHs()+a->getNumImplicitHs();++h)
          nodes.push_back({1,RDKit::PeriodicTable::getTable()->getAtomicWeight(1),-1});
        if (a->getIsAromatic()) {
          auto average=mancude(o.atom,depth);
          if (average) nodes.push_back(*average); else e.unsupported_mancude=true;
        }
        for (const auto b:mol.atomBonds(a)) {
          int v=b->getOtherAtomIdx(o.atom); if (v==o.parent) continue;
          const auto found=std::find(o.path.begin(),o.path.end(),v);
          int duplicate=found==o.path.end()?depth:static_cast<int>(found-o.path.begin());
          if (found!=o.path.end()) nodes.push_back(node(v,duplicate));
          else {
            nodes.push_back(node(v)); auto path=o.path; path.push_back(v);
            next.push_back({v,o.atom,std::move(path),depth}); e.depths.emplace(v,depth);
          }
          int multiplicity=b->getIsAromatic()?1:std::max(1,static_cast<int>(std::round(b->getBondTypeAsDouble())));
          for (int k=1;k<multiplicity;++k) nodes.push_back(node(v,duplicate));
        }
      }
      if (!nodes.empty()) e.spheres.push_back(std::move(nodes));
      active=std::move(next);
    }
    return cache.emplace(key,std::move(e)).first->second;
  }
  Signature signature(const Evidence &e,size_t depth,int rule) const {
    Signature result; if (depth>=e.spheres.size()) return result;
    for (const auto &n:e.spheres[depth]) {
      if (rule==1 && n.duplicate_depth<0) continue;
      result.emplace_back(n.number,rule==0?0:(rule==1?-n.duplicate_depth:n.mass));
    }
    std::sort(result.begin(),result.end(),std::greater<std::pair<double,double>>()); return result;
  }
  struct Unit { std::string label; int depth; };
  std::vector<Unit> units(const Evidence &e) const {
    std::vector<Unit> result;
    for (size_t i=0;i<graph.configurations.size();++i) {
      if (static_cast<int>(i)==focal || !supported(graph.configurations[i].geometry)) continue;
      int depth=INT32_MAX;
      for (int v:dependencies(graph.configurations[i])) { auto at=e.depths.find(v); if (at!=e.depths.end()) depth=std::min(depth,at->second); }
      if (depth==INT32_MAX) continue;
      auto at=labels.find(static_cast<int>(i)); result.push_back({at==labels.end()?"":at->second,depth});
    }
    return result;
  }
  Comparison compare_evidence(const Evidence &a,const Evidence &b) const {
    Comparison result;
    for (int rule=0;rule<3;++rule) {
      if (rule && (a.truncated || b.truncated)) break; // never prefer a lower rule before constitutional exhaustion
      for (size_t i=0;i<std::max(a.spheres.size(),b.spheres.size());++i) {
        auto left=signature(a,i,rule),right=signature(b,i,rule);
        const size_t width=std::max(left.size(),right.size()); left.resize(width,{0,0}); right.resize(width,{0,0});
        if (left==right) continue;
        result.order=left>right?1:-1; result.rule=rule; result.depth=static_cast<int>(i)+1;
        result.left=left; result.right=right; result.reason="first_exhaustive_sphere_difference"; return result;
      }
    }
    auto left=units(a),right=units(b);
    auto family=[](char c) { c=static_cast<char>(std::toupper(c)); return c=='R'||c=='S'?0:(c=='M'||c=='P'?1:2); };
    auto high=[](char c) { return std::string("RMZrmz").find(c)!=std::string::npos; };
    if (!a.truncated && !b.truncated && left.size()==1 && right.size()==1 &&
        !left[0].label.empty() && !right[0].label.empty() && left[0].depth==right[0].depth) {
      char l=left[0].label[0],r=right[0].label[0];
      if (family(l)==family(r) && high(l)!=high(r)) {
        if (family(l)==2 && std::isupper(l) && std::isupper(r)) result.rule=3;
        else if (std::islower(l) && std::islower(r)) result.rule=4;
        else if (std::isupper(l) && std::isupper(r)) result.rule=5;
        if (result.rule>=0) {
          result.order=high(l)?1:-1; result.left={-left[0].depth,family(l),high(l)};
          result.right={-right[0].depth,family(r),high(r)}; result.reason="single_corresponding_stereogenic_pair"; return result;
        }
      }
    }
    if (a.truncated || b.truncated) { result.unsupported=true; result.reason="explicit_depth_cap"; }
    else if (a.unsupported_mancude || b.unsupported_mancude) { result.unsupported=true; result.reason="unsupported_mancude_duplicate_model"; }
    else if (!left.empty() || !right.empty()) { result.unsupported=true; result.rule=4; result.reason="unsupported_multi_unit_stereogenic_ordering"; }
    else result.reason="constitutionally_and_isotopically_indistinguishable";
    return result;
  }
  Comparison compare(int center,Reference a,Reference b) { return compare_evidence(build(center,a),build(center,b)); }
  Ranking rank(int center,const std::vector<Reference> &refs) {
    Ranking out; out.refs=refs; std::set<Reference> unique(refs.begin(),refs.end());
    if (unique.size()!=refs.size()) throw std::invalid_argument("Repeated CIP reference");
    std::vector<int> wins(refs.size(),0); bool decided=true;
    for (size_t i=0;i<refs.size();++i) for (size_t j=i+1;j<refs.size();++j) {
      auto c=compare(center,refs[i],refs[j]);
      if (c.unsupported || c.order==0) decided=false; else ++wins[c.order>0?i:j]; out.comparisons.push_back(c);
    }
    if (!decided || std::set<int>(wins.begin(),wins.end()).size()!=refs.size()) return out;
    std::vector<int> indices(refs.size()); std::iota(indices.begin(),indices.end(),0);
    std::sort(indices.begin(),indices.end(),[&](int a,int b){return wins[a]>wins[b];});
    for (int i:indices) out.ordered.push_back(refs[i]); out.complete=true; return out;
  }
};
int permutation_sign(const std::vector<Reference> &source,const std::vector<Reference> &ordered) {
  int sign=1;
  for (size_t i=0;i<ordered.size();++i) for (size_t j=i+1;j<ordered.size();++j)
    if (std::find(source.begin(),source.end(),ordered[i])>std::find(source.begin(),source.end(),ordered[j])) sign=-sign;
  return sign;
}
bool rule5(const std::vector<Ranking> &rankings,const std::vector<Comparison> &other={}) {
  for (const auto &r:rankings) for (const auto &c:r.comparisons) if (c.rule==5) return true;
  for (const auto &c:other) if (c.rule==5) return true; return false;
}
std::map<int,std::string> reflected(std::map<int,std::string> labels) {
  for (auto &[id,label]:labels) {
    (void)id;
    if (label=="R") label="S"; else if (label=="S") label="R";
    else if (label=="M") label="P"; else if (label=="P") label="M";
  }
  return labels;
}
struct Label { std::string label,status="unresolved_priority",reason; std::vector<Ranking> rankings; std::vector<Comparison> extra; Json witness; };
Label project(const Configuration &d,Ranker &ranker,bool reflection=true) {
  Label out;
  if (!d.fixed) { out.status="unspecified_configuration"; return out; }
  if (d.geometry==Geometry::helical) {
    if ((d.parity!=1 && d.parity!=-1) || d.path.size()<4) throw std::invalid_argument("Invalid helical CIP configuration");
    out.label=d.parity==1?"P":"M"; out.status="assigned"; out.witness={d.parity}; return out;
  }
  if (!supported(d.geometry)) { out.status="unsupported_descriptor"; return out; }
  const size_t required=d.geometry==Geometry::tetrahedral?5:6;
  if (d.frame.size()!=required || (d.geometry!=Geometry::planar_bond &&
      d.geometry!=Geometry::extended_cis_trans && d.parity!=1 && d.parity!=-1))
    throw std::invalid_argument("Invalid CIP configuration frame or parity");
  if (d.geometry==Geometry::tetrahedral) {
    std::vector<Reference> refs(d.frame.begin()+1,d.frame.end()); out.rankings.push_back(ranker.rank(d.frame.at(0).vertex,refs));
    if (out.rankings[0].complete) {
      int sign=d.parity*permutation_sign(refs,out.rankings[0].ordered); out.label=sign>0?"R":"S"; out.witness={d.parity,sign};
    }
  } else {
    std::vector<Reference> l(d.frame.begin(),d.frame.begin()+2),r(d.frame.begin()+4,d.frame.end());
    out.rankings={ranker.rank(d.frame.at(2).vertex,l),ranker.rank(d.frame.at(3).vertex,r)};
    if (out.rankings[0].complete && out.rankings[1].complete) {
      int a=l[0]==out.rankings[0].ordered[0]?1:-1,b=r[0]==out.rankings[1].ordered[0]?1:-1;
      bool planar=d.geometry==Geometry::planar_bond || d.geometry==Geometry::extended_cis_trans;
      if (planar) { out.label=a==b?"Z":"E"; out.witness={a,b}; }
      else {
        // The near-to-far high-priority twist is invariant under end exchange.
        // Comparing the two ends introduced a spurious sign and wrongly
        // refused symmetric, configurationally chiral allenes.
        int sign=d.parity*a*b; out.label=sign>0?"P":"M"; out.witness={d.parity,a,b,sign};
      }
    }
  }
  if (out.label.empty()) {
    for (const auto &r:out.rankings) for (const auto &c:r.comparisons) if (c.unsupported) out.status="unsupported_sequence_rule";
    for (const auto &c:out.extra) if (c.unsupported) out.status="unsupported_sequence_rule";
    return out;
  }
  out.status="assigned";
  if (reflection && rule5(out.rankings,out.extra)) {
    auto labels=reflected(ranker.labels); Ranker rr(ranker.mol,ranker.graph,labels,ranker.focal,ranker.limit);
    auto reversed=d;
    if (d.geometry!=Geometry::planar_bond && d.geometry!=Geometry::extended_cis_trans) reversed.parity=-reversed.parity;
    auto other=project(reversed,rr,false);
    bool planar=d.geometry==Geometry::planar_bond || d.geometry==Geometry::extended_cis_trans;
    if (!other.label.empty() && (planar?other.label!=out.label:other.label==out.label))
      out.label[0]=static_cast<char>(std::tolower(out.label[0]));
  }
  return out;
}
} // namespace
Json rank_ligands(const RDKit::ROMol &m,int center,const std::vector<Reference> &refs,const Stereograph &g,const Options &o) {
  Limit limit(o); const std::map<int,std::string> labels; Ranker ranker(m,g,labels,-1,limit);
  return ranker.rank(center,refs).json();
}
Json assign_labels(const RDKit::ROMol &m,const Stereograph &g,const Options &o) {
  Limit limit(o); std::map<int,std::string> labels; std::vector<Label> assignments;
  for (size_t iteration=0;iteration<=g.configurations.size();++iteration) {
    assignments.clear(); std::map<int,std::string> updated;
    for (size_t i=0;i<g.configurations.size();++i) {
      limit.check(); Ranker ranker(m,g,labels,static_cast<int>(i),limit);
      assignments.push_back(project(g.configurations[i],ranker));
      if (!assignments.back().label.empty()) updated[static_cast<int>(i)]=assignments.back().label;
    }
    if (updated==labels) break; labels=std::move(updated);
  }
  Json out=Json::array();
  for (size_t i=0;i<assignments.size();++i) {
    const auto &a=assignments[i]; Json rankings=Json::array(),extra=Json::array(); std::set<std::string> required;
    for (const auto &r:a.rankings) { rankings.push_back(r.json()); for (const auto &c:r.comparisons) if (c.rule>=0) required.insert(rules[c.rule]); }
    for (const auto &c:a.extra) { extra.push_back(c.json()); if (c.rule>=0) required.insert(rules[c.rule]); }
    out.push_back({{"configuration_index",i},{"geometry",static_cast<int>(g.configurations[i].geometry)},
      {"label",a.label.empty()?Json(nullptr):Json(a.label)},{"status",a.status},{"reason",a.reason},
      {"rankings",rankings},{"supplemental_comparisons",extra},{"required_rules",required},{"projection_witness",a.witness}});
  }
  return out;
}
}} // namespace
