#include "synkit_canon/configured.hpp"
#include "stereo_groups.h"
#include <chrono>
#include <cmath>
#include <map>
#include <set>
#include <tuple>

namespace synkit { namespace stereo {
namespace {
using Clock=std::chrono::steady_clock;
struct Stop { int reason; };
std::string token(std::initializer_list<std::string> parts) {
  std::string out;
  for (const auto &part:parts) out+=std::to_string(part.size())+":"+part;
  return out;
}
void invalid(const char *message) { throw std::invalid_argument(message); }
struct Budget {
  const ConfiguredOptions &o;
  Clock::time_point start=Clock::now();
  explicit Budget(const ConfiguredOptions &opts):o(opts) {
    if (!std::isfinite(o.seconds) || (o.seconds<0 && o.seconds!=-1) ||
        o.max_expanded_vertices<=0 || o.max_expanded_edges<=0 || o.max_depth<0 || o.max_depth>1024)
      invalid("Invalid configured limits");
    for (auto cap:{o.max_nodes,o.workspace_bytes,o.output_bytes})
      if (cap==0 || cap< -1) invalid("Configured limits must be positive or -1");
  }
  double elapsed() const { return std::chrono::duration<double>(Clock::now()-start).count(); }
  void check() const {
    if (o.cancel_token && synkit_canon_cancelled(o.cancel_token)) throw Stop{5};
    if (o.seconds>=0 && elapsed()>=o.seconds) throw Stop{1};
  }
  double remaining() const { check(); return o.seconds<0 ? -1 : std::max(0.0,o.seconds-elapsed()); }
};
struct Builder {
  const Stereograph &g;
  const ConfiguredOptions &options;
  Budget &budget;
  Expansion out;
  std::map<std::pair<int,int>,int> bonds;
  std::set<std::string> loci;
  std::set<int> framework_atoms;
  bool validation_only;
  Builder(const Stereograph &graph,const ConfiguredOptions &opts,Budget &b,bool light=false):g(graph),options(opts),budget(b),validation_only(light) {
    if (g.colors.size()>static_cast<size_t>(INT32_MAX) || g.profile.empty()) invalid("Invalid prepared graph");
    out.atoms=static_cast<int32_t>(g.colors.size());
    for (const auto &c:g.colors) node(token({"atom",c}));
    for (const auto &e:g.bonds) {
      atom(e.source); atom(e.target);
      if (e.source==e.target) invalid("Self bond");
      const auto key=std::minmax(e.source,e.target);
      if (bonds.count(key)) invalid("Duplicate bond");
      int v=node(token({"bond",e.color})); bonds[key]=v;
      edge(e.source,v); edge(e.target,v);
    }
  }
  void atom(int v) const { if (v<0 || v>=out.atoms) invalid("Absent material atom/owner"); }
  int node(const std::string &color) {
    budget.check();
    if (out.colors.size()>=static_cast<size_t>(options.max_expanded_vertices) ||
        out.colors.size()>=static_cast<size_t>(INT32_MAX)) throw Stop{3};
    out.colors.push_back(color); return static_cast<int>(out.colors.size()-1);
  }
  void edge(int a,int b) {
    budget.check();
    if (out.edges.size()>=static_cast<size_t>(options.max_expanded_edges)) throw Stop{3};
    out.edges.emplace_back(a,b);
  }
  int bond(int a,int b) const {
    auto at=bonds.find(std::minmax(a,b));
    if (at==bonds.end()) invalid("Frame/path reference is not bonded to its owner");
    return at->second;
  }
  int resource(int owner,Reference r) {
    atom(owner); atom(r.vertex);
    if (r.kind==Resource::atom) return bond(owner,r.vertex);
    if (r.vertex!=owner || (r.kind!=Resource::hydrogen && r.kind!=Resource::lone_pair))
      invalid("Invalid virtual resource ownership/type");
    int v=node(token({"virtual",r.kind==Resource::hydrogen?"H":"LP"}));
    edge(owner,v); return v;
  }
  int port(int locus,int owner,Reference ref,int geometry) {
    int v=node(token({"port",std::to_string(geometry)}));
    edge(locus,v); edge(v,resource(owner,ref)); return v;
  }
  void path(const std::vector<int32_t> &p,bool cyclic) const {
    if (p.size()<(cyclic?3U:2U)) invalid("Stereo support is too short");
    std::set<int> seen;
    for (int v:p) { atom(v); if (!seen.insert(v).second) invalid("Repeated path/plane atom"); }
    for (size_t i=1;i<p.size();++i) bond(p[i-1],p[i]);
    if (cyclic) bond(p.back(),p.front());
  }
  std::string path_key(const std::vector<int32_t> &p,bool cyclic) const {
    std::vector<int32_t> best=p;
    for (int flip=0;flip<2;++flip) {
      auto q=p; if (flip) std::reverse(q.begin(),q.end());
      for (size_t shift=0;shift<(cyclic?p.size():1);++shift) {
        best=std::min(best,q); if (!q.empty()) std::rotate(q.begin(),q.begin()+1,q.end());
      }
    }
    std::string key; for (int v:best) key+=std::to_string(v)+",";
    return key;
  }
  void unique(const std::string &key) { if (!loci.insert(key).second) invalid("Duplicate stereo locus"); }
  void relation(int locus,int geometry,const std::vector<int> &targets) {
    int tuple=node(token({"tuple",std::to_string(geometry)})); edge(locus,tuple);
    for (size_t i=0;i<targets.size();++i) {
      int slot=node(token({"slot",std::to_string(geometry),std::to_string(i)}));
      edge(tuple,slot); edge(slot,targets[i]);
    }
  }
  void local(const Configuration &d,int geometry,int locus,bool axis=false) {
    int group_geometry=geometry;
    if (geometry==7) group_geometry=6;
    if (geometry==8) group_geometry=5;
    const auto group=synkit_stereo_detail::group(group_geometry);
    const bool zero=group_geometry==2 || group_geometry==5;
    if (d.frame.size()!=static_cast<size_t>(group.arity) ||
        (zero?d.parity!=0:(d.parity!=1 && d.parity!=-1))) invalid("Invalid fixed local frame/parity");
    std::set<Reference> seen(d.frame.begin(),d.frame.end());
    if (seen.size()!=d.frame.size()) invalid("Repeated stereo reference");
    auto frame=d.frame;
    if (d.parity==-1) {
      if (group_geometry==3) std::swap(frame[4],frame[5]);
      else if (group_geometry==6) std::swap(frame[0],frame[1]);
      else std::swap(frame[1],frame[2]);
    }
    std::map<Reference,int> targets;
    std::vector<int> slots;
    if (geometry<=4) {
      if (frame[0].kind!=Resource::atom) invalid("Virtual center");
      int center=frame[0].vertex; atom(center);
      unique("atom:"+std::to_string(center)); edge(locus,center);
      targets[frame[0]]=center;
      for (int i=1;i<group.arity;++i) targets[frame[i]]=port(locus,center,frame[i],geometry);
      for (int i=1;i<group.arity;++i) slots.push_back(i);
    } else {
      if (frame[2].kind!=Resource::atom || frame[3].kind!=Resource::atom) invalid("Virtual bond/axis owner");
      int left=frame[2].vertex,right=frame[3].vertex; atom(left); atom(right);
      if (axis) {
        path(d.path,false);
        if (d.path.size()<(geometry==7?3U:4U) ||
            (d.path.size()%2)!=(geometry==7?1U:0U) || d.path.front()!=left || d.path.back()!=right)
          invalid("Invalid cumulene path/endpoints");
        unique("axis:"+std::to_string(geometry)+":"+path_key(d.path,false));
        for (size_t i=1;i<d.path.size();++i) edge(locus,bond(d.path[i-1],d.path[i]));
      } else {
        unique("bond:"+path_key({left,right},false)); edge(locus,bond(left,right));
      }
      targets[frame[2]]=left; targets[frame[3]]=right;
      for (int i:{0,1,4,5}) {
        if (axis && frame[i].kind==Resource::atom &&
            std::find(d.path.begin(),d.path.end(),frame[i].vertex)!=d.path.end())
          invalid("Terminal ligand lies on axis");
        targets[frame[i]]=port(locus,i<2?left:right,frame[i],geometry);
      }
      slots={0,1,2,3,4,5};
    }
    if (validation_only) return;
    for (int p=0;p<group.size;++p) {
      std::vector<int> t;
      for (int i:slots) t.push_back(targets.at(frame[group.images[p][i]]));
      relation(locus,geometry,t);
    }
  }
  void framework(const Configuration &d,int locus) {
    std::set<int> support(d.support.begin(),d.support.end());
    if (support.empty() || support.size()!=d.support.size() || d.frames.empty()) invalid("Invalid framework support");
    std::vector<int> reached{*support.begin()}; std::set<int> visited;
    while (!reached.empty()) {
      budget.check(); int v=reached.back(); reached.pop_back();
      if (!visited.insert(v).second) continue;
      for (const auto &b:g.bonds) {
        int w=b.source==v?b.target:(b.target==v?b.source:-1);
        if (support.count(w) && !visited.count(w)) reached.push_back(w);
      }
    }
    if (visited!=support) invalid("Disconnected framework support");
    for (int v:support) {
      atom(v); if (!framework_atoms.insert(v).second) invalid("Overlapping framework supports");
      int s=node(token({"framework_support"})); edge(locus,s); edge(s,v);
    }
    std::set<int> centers;
    const auto group=synkit_stereo_detail::group(1);
    for (const auto &f:d.frames) {
      if (!support.count(f.center) || !centers.insert(f.center).second ||
          (f.relation!=1 && f.relation!=-1)) invalid("Invalid coupled frame center/relation");
      std::set<Reference> refs(f.references.begin(),f.references.end());
      if (refs.size()!=4) invalid("Repeated coupled frame reference");
      std::array<Reference,5> frame; frame[0]=Reference(f.center);
      std::copy(f.references.begin(),f.references.end(),frame.begin()+1);
      for (const auto &r:f.references)
        if (r.kind==Resource::atom && (!support.count(r.vertex) || r.vertex==f.center)) invalid("Framework reference outside support");
      if (f.relation*d.parity==-1) std::swap(frame[1],frame[2]);
      int fv=node(token({"framework_frame"})); edge(locus,fv); edge(fv,f.center);
      for (int p=0;p<group.size;++p) {
        int rep=node(token({"framework_representation"})); edge(fv,rep);
        for (int i=0;i<4;++i) {
          int slot=node(token({"framework_slot",std::to_string(i)})); edge(rep,slot);
          edge(slot,resource(f.center,frame[group.images[p][i+1]]));
        }
      }
    }
  }
  Expansion run() {
    for (const auto &d:g.configurations) {
      budget.check(); if (!d.fixed) invalid("Unspecified configuration");
      int geometry=static_cast<int>(d.geometry);
      if (geometry<1 || geometry>11) invalid("Unsupported configured geometry");
      if (geometry>=9 && d.parity!=1 && d.parity!=-1) invalid("Invalid extended orientation");
      std::string color=token({"locus",std::to_string(geometry)});
      if (geometry==9) color=token({"locus","9",std::to_string(d.parity),d.cyclic?"1":"0",d.coupling});
      int locus=node(color);
      if (geometry<=8) local(d,geometry,locus,geometry>=7);
      else if (geometry==11) framework(d,locus);
      else {
        bool plane=geometry==10;
        path(d.path,plane || d.cyclic);
        unique(token({plane?"plane":"path",path_key(d.path,plane || d.cyclic),plane?std::to_string(d.pilot):d.coupling}));
        if (plane) {
          atom(d.pilot);
          if (std::find(d.path.begin(),d.path.end(),d.pilot)!=d.path.end()) invalid("Pilot belongs to plane");
        }
        const bool cyclic=plane || d.cyclic;
        for (int flip=0;flip<(plane?1:2);++flip) {
          auto p=d.path;
          if (plane?d.parity==-1:flip!=0) std::reverse(p.begin(),p.end());
          for (size_t shift=0;shift<(cyclic?p.size():1);++shift) {
            std::vector<int> refs; if (plane) refs.push_back(d.pilot);
            refs.insert(refs.end(),p.begin(),p.end()); relation(locus,geometry,refs);
            std::rotate(p.begin(),p.begin()+1,p.end());
          }
        }
      }
    }
    return std::move(out);
  }
};
std::vector<std::string> palette(const Expansion &a,const Expansion *b=nullptr) {
  auto tokens=a.colors;
  if (b) tokens.insert(tokens.end(),b->colors.begin(),b->colors.end());
  std::sort(tokens.begin(),tokens.end()); tokens.erase(std::unique(tokens.begin(),tokens.end()),tokens.end());
  return tokens;
}
Graph ranked(const Expansion &e,const std::vector<std::string> &tokens) {
  Graph g;
  for (const auto &c:e.colors) g.colors.push_back(static_cast<int32_t>(std::lower_bound(tokens.begin(),tokens.end(),c)-tokens.begin()));
  for (const auto &b:e.edges) g.edges.push_back({b.first,b.second,0});
  return g;
}
void append64(std::vector<uint8_t> &out,uint64_t value) {
  for (int i=0;i<8;++i) out.push_back(static_cast<uint8_t>(value>>(8*i)));
}
void append_string(std::vector<uint8_t> &out,const std::string &s) {
  append64(out,s.size()); out.insert(out.end(),s.begin(),s.end());
}
} // namespace

Expansion expand(const Stereograph &g,const ConfiguredOptions &o) {
  Budget budget(o);
  try { budget.check(); return Builder(g,o,budget).run(); }
  catch (const Stop &s) { throw std::runtime_error("Configured expansion stopped: "+std::to_string(s.reason)); }
}
Stereograph mirror(const Stereograph &g) {
  auto result=g;
  for (auto &d:result.configurations) {
    int k=static_cast<int>(d.geometry);
    if (k!=2 && k!=5 && k!=8 && d.fixed) d.parity=-d.parity;
  }
  return result;
}
ConfiguredCanonicalResult canonicalize(const Stereograph &g,const ConfiguredOptions &o) {
  Budget budget(o); ConfiguredCanonicalResult result;
  try {
    budget.check(); auto e=Builder(g,o,budget).run(); auto tokens=palette(e);
    auto graph=ranked(e,tokens); detail::Packed packed(graph);
    std::vector<uint8_t> prefix;
    append_string(prefix,"synkit.cpp-configured-stereograph/2"); append_string(prefix,g.profile);
    append64(prefix,tokens.size()); for (const auto &t:tokens) append_string(prefix,t);
    int64_t cap=o.output_bytes;
    if (cap>=0) { if (prefix.size()>=static_cast<size_t>(cap)) throw Stop{4}; cap-=static_cast<int64_t>(prefix.size()); }
    result.preparation_seconds=budget.elapsed();
    SynkitCanonRequest r{1,sizeof(SynkitCanonRequest),packed.view.n,packed.view.arcs,packed.view.offsets,
      packed.view.neighbors,packed.view.vertex_colors,budget.remaining(),o.max_nodes,o.workspace_bytes,cap,o.max_depth,o.cancel_token};
    SynkitCanonResult *raw=nullptr;
    if (synkit_canon_search(&r,&raw)!=0) throw std::invalid_argument(synkit_canon_error());
    std::unique_ptr<SynkitCanonResult,decltype(&synkit_canon_free)> owned(raw,synkit_canon_free);
    result.statistics=*synkit_canon_statistics(raw);
    if (!result.statistics.complete) { result.reason=result.statistics.reason; result.total_seconds=budget.elapsed(); return result; }
    int64_t size=0; const auto *bytes=synkit_canon_bytes(raw,&size);
    result.code=std::move(prefix); result.code.insert(result.code.end(),bytes,bytes+size);
    const auto *order=synkit_canon_order(raw,&size);
    for (int64_t i=0;i<size;++i) if (order[i]<e.atoms) result.atom_order.push_back(order[i]);
    budget.check(); result.complete=true;
  } catch (const Stop &s) { result.reason=s.reason; result.statistics.complete=0; result.statistics.reason=s.reason; result.code.clear(); result.atom_order.clear(); }
  catch (const std::bad_alloc &) { result.reason=3; result.code.clear(); result.atom_order.clear(); }
  result.total_seconds=budget.elapsed(); return result;
}
ConfiguredCanonicalResult canonicalize_family(const std::vector<Stereograph> &drawings,const ConfiguredOptions &o) {
  Budget budget(o); ConfiguredCanonicalResult result;
  if (drawings.empty()) invalid("Empty declared drawing family");
  try {
    std::set<std::vector<uint8_t>> codes; uint64_t bytes=0;
    for (const auto &g:drawings) {
      if (g.profile!=drawings[0].profile) invalid("Family drawing profiles differ");
      auto opts=o; opts.seconds=budget.remaining(); auto member=canonicalize(g,opts);
      if (!member.complete) throw Stop{member.reason};
      result.preparation_seconds+=member.preparation_seconds;
      result.statistics.nodes+=member.statistics.nodes;
      result.statistics.leaves+=member.statistics.leaves;
      result.statistics.search_seconds+=member.statistics.search_seconds;
      result.statistics.serialization_seconds+=member.statistics.serialization_seconds;
      result.statistics.validation_seconds+=member.statistics.validation_seconds;
      result.statistics.peak_workspace_bytes=std::max(result.statistics.peak_workspace_bytes,member.statistics.peak_workspace_bytes);
      if (codes.insert(std::move(member.code)).second) {
        bytes=0; for (const auto &c:codes) bytes+=8+c.size();
        if (o.output_bytes>=0 && bytes>static_cast<uint64_t>(o.output_bytes)) throw Stop{4};
      }
    }
    append_string(result.code,"synkit.cpp-declared-resonance-family/1"); append64(result.code,codes.size());
    for (const auto &c:codes) { append64(result.code,c.size()); result.code.insert(result.code.end(),c.begin(),c.end()); }
    if (o.output_bytes>=0 && result.code.size()>static_cast<size_t>(o.output_bytes)) throw Stop{4};
    budget.check(); result.complete=true; result.statistics.complete=1;
  } catch (const Stop &s) { result.reason=s.reason; result.code.clear(); }
  catch (const std::bad_alloc &) { result.reason=3; result.code.clear(); }
  result.total_seconds=budget.elapsed(); return result;
}
Result compare(const Stereograph &a,const Stereograph &b,const ConfiguredOptions &o,bool direct_local) {
  Budget budget(o); Result result; result.statistics.isomorphic=-1;
  try {
    budget.check();
    const bool local=direct_local && std::all_of(a.configurations.begin(),a.configurations.end(),[](const Configuration &d){return static_cast<int>(d.geometry)<=6;}) &&
      std::all_of(b.configurations.begin(),b.configurations.end(),[](const Configuration &d){return static_cast<int>(d.geometry)<=6;});
    if (a.profile!=b.profile) invalid("Prepared identity profiles differ");
    if (!local) {
      // Expanded supports can exceed the direct matcher's assignment-depth
      // cap. Canonical IR depth counts individualizations instead of atoms.
      auto remaining=o; remaining.seconds=budget.remaining(); auto ca=canonicalize(a,remaining);
      if (!ca.complete) throw Stop{ca.reason};
      remaining.seconds=budget.remaining(); auto cb=canonicalize(b,remaining);
      if (!cb.complete) throw Stop{cb.reason};
      budget.check(); result.statistics.complete=1;
      result.statistics.isomorphic=ca.code==cb.code?1:0;
      result.statistics.nodes=ca.statistics.nodes+cb.statistics.nodes;
      result.statistics.leaves=ca.statistics.leaves+cb.statistics.leaves;
      if (result.isomorphic()) {
        result.mapping.resize(a.colors.size());
        for (size_t i=0;i<ca.atom_order.size();++i) result.mapping[ca.atom_order[i]]=cb.atom_order[i];
      }
      return result;
    }
    // Validate local incidence without constructing its full orbit expansion.
    auto ea=Builder(a,o,budget,local).run(),eb=Builder(b,o,budget,local).run();
    Graph ga,gb;
    if (local) {
      auto prepare=[](const Stereograph &g) {
        Expansion e; e.atoms=static_cast<int32_t>(g.colors.size());
        for (const auto &c:g.colors) e.colors.push_back(token({"atom",c}));
        return e;
      };
      ea=prepare(a); eb=prepare(b);
      // Bond colors are retained through exact edge palette ranks.
      std::vector<std::string> edge_tokens;
      for (const auto *g:{&a,&b}) for (const auto &e:g->bonds) edge_tokens.push_back(e.color);
      std::sort(edge_tokens.begin(),edge_tokens.end()); edge_tokens.erase(std::unique(edge_tokens.begin(),edge_tokens.end()),edge_tokens.end());
      auto locals=[&](const Stereograph &g,Expansion &e,Graph &out) {
        std::map<Reference,int> virtuals;
        for (const auto &d:g.configurations) {
          SynkitStereoFrame frame{}; frame.geometry=static_cast<int>(d.geometry); frame.parity=d.parity;
          for (size_t i=0;i<d.frame.size();++i) {
            const auto ref=d.frame[i]; int v=ref.vertex;
            if (ref.kind!=Resource::atom) {
              auto at=virtuals.find(ref);
              if (at==virtuals.end()) {
                v=static_cast<int>(e.colors.size()); virtuals.emplace(ref,v);
                e.colors.push_back(token({"virtual",ref.kind==Resource::hydrogen?"H":"LP"}));
                // A virtual edge has its own exact token, disjoint from material bonds.
                out.edges.push_back({v,ref.vertex,static_cast<int32_t>(edge_tokens.size())});
              } else v=at->second;
            }
            frame.vertices[i]=v;
          }
          out.frames.push_back(frame);
        }
        for (const auto &e:g.bonds) out.edges.push_back({e.source,e.target,
          static_cast<int32_t>(std::lower_bound(edge_tokens.begin(),edge_tokens.end(),e.color)-edge_tokens.begin())});
      };
      locals(a,ea,ga); locals(b,eb,gb);
      auto tokens=palette(ea,&eb); ga.colors=ranked(ea,tokens).colors; gb.colors=ranked(eb,tokens).colors;
    } else {
      auto tokens=palette(ea,&eb); ga=ranked(ea,tokens); gb=ranked(eb,tokens);
    }
    Options opts=o; opts.seconds=budget.remaining(); result=isomorphism(ga,gb,opts);
    budget.check();
    if (result.complete() && result.isomorphic()) {
      result.mapping.resize(a.colors.size());
      for (int v:result.mapping) if (v<0 || static_cast<size_t>(v)>=b.colors.size()) throw std::logic_error("Atom projection changed vertex role");
    }
  } catch (const Stop &s) { result.statistics.complete=0; result.statistics.reason=s.reason; result.statistics.isomorphic=-1; result.mapping.clear(); }
  catch (const std::bad_alloc &) { result.statistics.complete=0; result.statistics.reason=3; result.statistics.isomorphic=-1; result.mapping.clear(); }
  return result;
}
ConfiguredMirrorResult classify_mirror(const Stereograph &g,const ConfiguredOptions &o) {
  Budget budget(o); ConfiguredMirrorResult result;
  try {
    auto reflected=mirror(g); auto opts=o; opts.seconds=budget.remaining();
    const auto comparison=compare(g,reflected,opts);
    budget.check(); result.complete=comparison.complete(); result.reason=comparison.statistics.reason;
    if (result.complete) { result.achiral=comparison.isomorphic(); result.mapping=comparison.mapping; }
  } catch (const Stop &s) { result.reason=s.reason; }
  catch (const std::bad_alloc &) { result.reason=3; }
  return result;
}
ConfiguredSymmetryResult symmetry(const Stereograph &g,bool full_group,const ConfiguredOptions &o) {
  Budget budget(o); ConfiguredSymmetryResult result;
  try {
    budget.check(); Builder(g,o,budget,true).run();
    const int n=static_cast<int>(g.colors.size());
    int64_t bytes=0;
    auto charge=[&]() {
      const int64_t amount=static_cast<int64_t>(n)*4;
      if (amount>INT64_MAX-bytes || (o.output_bytes>=0 && amount>o.output_bytes-bytes)) throw Stop{4};
      bytes+=amount;
    };
    std::string order="1";
    auto multiply=[&](int factor) {
      int64_t carry=0;
      for (size_t k=order.size();k>0;--k) {
        int64_t value=static_cast<int64_t>(order[k-1]-'0')*factor+carry; order[k-1]=static_cast<char>('0'+value%10); carry=value/10;
      }
      while (carry) { order.insert(order.begin(),static_cast<char>('0'+carry%10)); carry/=10; }
    };
    for (int base=0;base<n;++base) {
      budget.check(); std::vector<int> parent(n); std::iota(parent.begin(),parent.end(),0);
      auto root=[&](int v) { while (parent[v]!=v) { parent[v]=parent[parent[v]]; v=parent[v]; } return v; };
      auto rebuild=[&]() {
        for (const auto &p:result.generators) {
          bool fixes=true; for (int i=0;i<base;++i) if (p[i]!=i) { fixes=false; break; }
          if (!fixes) continue;
          for (int i=base;i<n;++i) { int a=root(i),b=root(p[i]); parent[b]=a; }
        }
      };
      rebuild();
      for (int image=base+1;image<n;++image) {
        budget.check();
        if (g.colors[base]!=g.colors[image] || root(base)==root(image)) continue;
        auto source=g,target=g;
        for (int v=0;v<n;++v) {
          int left=v<base?v+1:(v==base?base+1:0);
          int right=v<base?v+1:(v==image?base+1:0);
          source.colors[v]=token({g.colors[v],std::to_string(left)});
          target.colors[v]=token({g.colors[v],std::to_string(right)});
        }
        auto opts=o; opts.seconds=budget.remaining();
        auto mapping=compare(source,target,opts); ++result.comparisons;
        if (!mapping.complete()) throw Stop{mapping.statistics.reason};
        if (mapping.isomorphic()) {
          charge(); result.generators.push_back(std::move(mapping.mapping)); rebuild();
        }
      }
      int orbit=0; for (int i=base;i<n;++i) orbit+=root(i)==root(base);
      multiply(orbit);
    }
    result.group_order=order;
    if (full_group) {
      std::set<std::vector<int32_t>> seen;
      std::vector<int32_t> identity(n); std::iota(identity.begin(),identity.end(),0);
      charge(); seen.insert(identity); result.automorphisms.push_back(identity);
      for (size_t i=0;i<result.automorphisms.size();++i) for (const auto &gen:result.generators) {
        budget.check(); std::vector<int32_t> composition(n);
        for (int v=0;v<n;++v) composition[v]=gen[result.automorphisms[i][v]];
        if (seen.find(composition)==seen.end()) { charge(); seen.insert(composition); result.automorphisms.push_back(std::move(composition)); }
      }
      if (std::to_string(result.automorphisms.size())!=order) throw std::logic_error("Stabilizer order and full closure disagree");
    }
    budget.check(); result.complete=true;
  } catch (const Stop &s) { result.reason=s.reason; result.group_order.clear(); result.generators.clear(); result.automorphisms.clear(); }
  catch (const std::bad_alloc &) { result.reason=3; result.group_order.clear(); result.generators.clear(); result.automorphisms.clear(); }
  return result;
}
}} // namespace
