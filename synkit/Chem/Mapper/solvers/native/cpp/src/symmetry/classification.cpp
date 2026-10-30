#include "synkit_mapper/symmetry/classification.h"
#include "synkit_mapper/core/runtime.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
using Perm = std::vector<int32_t>;
using Partition = std::vector<Perm>;
thread_local std::string error;

// FIPS 180-4 SHA256; only a compact index, never used for class equality.
std::string sha256(const std::string &message) {
  static constexpr uint32_t k[64] = {
      0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
      0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
      0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
      0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
      0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
      0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
      0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
      0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
  auto rotate = [](uint32_t x, int n) { return (x >> n) | (x << (32 - n)); };
  std::vector<unsigned char> data(message.begin(), message.end());
  uint64_t bits = static_cast<uint64_t>(data.size()) * 8;
  data.push_back(0x80);
  while (data.size() % 64 != 56) data.push_back(0);
  for (int i=7;i>=0;--i) data.push_back(static_cast<unsigned char>(bits >> (8*i)));
  std::array<uint32_t,8> h = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                             0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
  for (size_t offset=0;offset<data.size();offset+=64) {
    uint32_t w[64];
    for (int i=0;i<16;++i) {
      w[i]=0;
      for (int j=0;j<4;++j) w[i]=(w[i]<<8)|data[offset+4*i+j];
    }
    for (int i=16;i<64;++i) {
      uint32_t x=w[i-15],y=w[i-2];
      w[i]=w[i-16]+(rotate(x,7)^rotate(x,18)^(x>>3))+w[i-7]+(rotate(y,17)^rotate(y,19)^(y>>10));
    }
    auto v=h;
    for (int i=0;i<64;++i) {
      uint32_t t1=v[7]+(rotate(v[4],6)^rotate(v[4],11)^rotate(v[4],25))+
                  ((v[4]&v[5])^(~v[4]&v[6]))+k[i]+w[i];
      uint32_t t2=(rotate(v[0],2)^rotate(v[0],13)^rotate(v[0],22))+
                  ((v[0]&v[1])^(v[0]&v[2])^(v[1]&v[2]));
      for (int j=7;j>0;--j) v[j]=v[j-1];
      v[4]+=t1;v[0]=t1+t2;
    }
    for (int i=0;i<8;++i) h[i]+=v[i];
  }
  std::ostringstream out;
  out<<std::hex<<std::setfill('0');
  for (uint32_t x:h) out<<std::setw(8)<<x;
  return out.str();
}

struct Budget {
  Clock::time_point start=Clock::now();
  double seconds;
  int64_t max_nodes,nodes=0;
  int reason=0;
  void *cancel_token=nullptr;
  bool check(bool visit=false) {
    if(reason) return false;
    if(cancel_token && static_cast<synkit_mapper::Cancellation*>(cancel_token)->requested.load(std::memory_order_relaxed)) reason=5;
    else if(seconds>=0 && std::chrono::duration<double>(Clock::now()-start).count()>=seconds) reason=1;
    else if(visit && max_nodes>=0 && nodes>=max_nodes) reason=2;
    if(reason) return false;
    if(visit) ++nodes;
    return true;
  }
};

// Exact individualization/refinement. Signature, target-cell and leaf order
// match exact/native's existing canonicalizer. No WL equality shortcuts.
struct Canonical {
  int n;
  const Perm &colors,&edges;
  Budget &budget;
  Perm best,best_key,twins;
  std::vector<Perm> automorphisms;
  Canonical(int size,const Perm &c,const Perm &e,Budget &b)
      :n(size),colors(c),edges(e),budget(b),twins(size) {
    std::map<Perm,int> rows;
    for(int i=0;i<n;++i) {
      Perm row{colors[i]};
      row.insert(row.end(),edges.begin()+i*n,edges.begin()+(i+1)*n);
      auto inserted=rows.emplace(std::move(row),i);
      twins[i]=inserted.first->second;
    }
  }
  Partition refine(Partition p) {
    while(true) {
      Perm ci(n);
      for(size_t k=0;k<p.size();++k) for(int i:p[k]) ci[i]=static_cast<int>(k);
      Partition next;
      bool split=false;
      for(const auto &cell:p) {
        std::map<Perm,Perm> buckets;
        for(int x:cell) {
          std::map<std::pair<int,int>,int> counts;
          for(int y=0;y<n;++y) if(edges[x*n+y]) ++counts[{ci[y],edges[x*n+y]}];
          Perm signature;
          int last=-1;
          for(const auto &entry:counts) {
            int id=entry.first.first;
            if(id!=last) {
              if(last>=0) signature.push_back(0);
              signature.push_back(-id);last=id;
            }
            signature.push_back(entry.first.second);signature.push_back(entry.second);
          }
          if(last>=0) signature.push_back(0);
          buckets[signature].push_back(x);
        }
        split|=buckets.size()>1;
        for(auto &entry:buckets) next.push_back(std::move(entry.second));
      }
      p=std::move(next);
      if(!split) return p;
    }
  }
  void visit(Partition p) {
    if(!budget.check(true)) return;
    p=refine(std::move(p));
    int target=-1;
    while(true) {
      target=-1;
      for(size_t k=0;k<p.size();++k)
        if(p[k].size()>1 && (target<0 || p[k].size()<p[target].size())) target=static_cast<int>(k);
      if(target<0) break;
      bool equal=true;
      for(int i:p[target]) equal&=twins[i]==twins[p[target][0]];
      if(!equal) break;
      Partition next;
      for(size_t k=0;k<p.size();++k) {
        if(static_cast<int>(k)==target) for(int i:p[k]) next.push_back(Perm{i});
        else next.push_back(p[k]);
      }
      p=std::move(next);
    }
    if(target<0) {
      Perm order,key;
      for(const auto &cell:p) order.push_back(cell[0]);
      for(int x:order) key.push_back(colors[x]);
      for(int i=0;i<n;++i) for(int j=i;j<n;++j) if(edges[order[i]*n+order[j]]) {
        key.push_back(-(i*n+j));key.push_back(edges[order[i]*n+order[j]]);
      }
      if(best.empty() || key<best_key) {best=order;best_key=key;}
      else if(key==best_key) {
        Perm g(n);
        for(int i=0;i<n;++i) g[best[i]]=order[i];
        automorphisms.push_back(std::move(g));
      }
      return;
    }
    Perm ci(n),explored;
    for(size_t k=0;k<p.size();++k) for(int x:p[k]) ci[x]=static_cast<int>(k);
    for(int chosen:p[target]) {
      // Every witness comes from identical exact leaves. Orbit closure uses
      // only witnesses stabilizing this ordered partition, so is safe even
      // when group discovery is incomplete.
      std::set<int> orbit{chosen};
      bool changed=true;
      while(changed) {
        changed=false;
        for(const auto &g:automorphisms) {
          bool stabilizes=true;
          for(int i=0;i<n;++i) if(ci[i]!=ci[g[i]]) {stabilizes=false;break;}
          if(stabilizes) {
            Perm old(orbit.begin(),orbit.end());
            for(int x:old) changed|=orbit.insert(g[x]).second;
          }
        }
      }
      bool seen=false;
      for(int x:explored) if(orbit.count(x)) {seen=true;break;}
      if(seen) continue;
      Partition child=p;
      Perm rest;
      for(int x:p[target]) if(x!=chosen) rest.push_back(x);
      child[target]=Perm{chosen};child.insert(child.begin()+target+1,std::move(rest));
      visit(std::move(child));
      if(budget.reason) return;
      explored.push_back(chosen);
    }
  }
  Perm run() {
    std::map<int,Perm> buckets;
    for(int i=0;i<n;++i) buckets[colors[i]].push_back(i);
    Partition p;
    for(auto &entry:buckets) p.push_back(std::move(entry.second));
    visit(std::move(p));
    return best;
  }
};

void append16(std::string &out,int value) {
  out.push_back(static_cast<char>(value&255));out.push_back(static_cast<char>((value>>8)&255));
}
void append32(std::string &out,uint32_t value) {
  for(int i=0;i<4;++i) out.push_back(static_cast<char>((value>>(8*i))&255));
}
std::string escape_repr(const std::string &token) {
  std::string out;
  for(char c:token) {if(c=='\\'||c=='\'') out.push_back('\\');out.push_back(c);}
  return out;
}
struct Certificate {std::string packed,identifier;};
// Packed SKC1 bytes remain the exact equality key. Build the legacy dense
// identifier only after that key has been found to be a new accepted class.
std::string packed_certificate(const SynkitClassProblem &p,const Perm &colors,const Perm &edges,const Perm &order) {
  std::set<int> used;
  for(int x:order) used.insert(colors[x]);
  for(int i=0;i<p.n;++i) for(int j=i+1;j<p.n;++j) if(edges[i*p.n+j]) used.insert(edges[i*p.n+j]-1);
  Perm local(p.palette_size,-1);
  std::string palette="[";
  int index=0;
  for(int rank:used) {
    if(index) palette+=',';
    palette+=p.palette_json[rank];local[rank]=index++;
  }
  palette+=']';
  std::string packed="SKC1";
  append32(packed,static_cast<uint32_t>(palette.size()));append16(packed,p.n);packed+=palette;
  for(int i=0;i<p.n;++i) append16(packed,local[colors[order[i]]]);
  for(int i=0;i<p.n;++i) for(int j=i;j<p.n;++j) {
    const int edge=edges[order[i]*p.n+order[j]];
    if(edge) {append16(packed,i);append16(packed,j);append16(packed,local[edge-1]);}
  }
  return packed;
}
std::string legacy_identifier(const SynkitClassProblem &p,const Perm &colors,const Perm &edges,const Perm &order) {
  std::string dense="'[[\"directed\",0],[\"nodes\",[";
  for(int i=0;i<p.n;++i) {
    const int rank=colors[order[i]];
    if(i) dense+=',';
    dense+=escape_repr(p.palette_json[rank]);
  }
  dense+="]],[\"adjacency\",[";
  bool first=true;
  for(int i=0;i<p.n;++i) for(int j=i;j<p.n;++j) {
    if(!first) dense+=',';
    first=false;
    const int edge=edges[order[i]*p.n+order[j]];
    if(edge) dense+="[\"edge\","+escape_repr(p.palette_json[edge-1])+"]";
    else dense+="[\"absent\"]";
  }
  dense+="]]]'";
  return sha256(p.n ? dense : "('empty_colored_graph',)");
}
void validate(const SynkitClassProblem &p) {
  if(p.n<0||p.n>512||p.palette_size<0||p.palette_size>=65536||p.edge_states_r<1||p.edge_states_p<1)
    throw std::invalid_argument("invalid class problem dimensions");
  if(p.n && (!p.node_pair_colors||!p.edges_r||!p.edges_p||!p.edge_pair_colors||!p.palette_json))
    throw std::invalid_argument("null class problem buffer");
  for(int i=0;i<p.palette_size;++i) if(!p.palette_json[i]) throw std::invalid_argument("null palette token");
  for(int i=0;i<p.n;++i) for(int j=0;j<p.n;++j) {
    int at=i*p.n+j;
    if(p.node_pair_colors[at]<0||p.node_pair_colors[at]>=p.palette_size||
       p.edges_r[at]<0||p.edges_r[at]>=p.edge_states_r||p.edges_p[at]<0||p.edges_p[at]>=p.edge_states_p||
       p.edges_r[at]!=p.edges_r[j*p.n+i]||p.edges_p[at]!=p.edges_p[j*p.n+i])
      throw std::invalid_argument("invalid class palette or asymmetric endpoint state");
  }
  int64_t pairs=static_cast<int64_t>(p.edge_states_r)*p.edge_states_p;
  for(int64_t i=0;i<pairs;++i) if(p.edge_pair_colors[i]<0||p.edge_pair_colors[i]>p.palette_size)
    throw std::invalid_argument("invalid edge pair color");
}
struct Record {Perm mapping;int64_t cost,count;Certificate key;int64_t published=0;};
} // namespace

extern "C" int synkit_mapper_classification_abi() {return 1;}
extern "C" int synkit_mapper_classification_max_vertices() {return 512;}
extern "C" const char *synkit_mapper_classification_error() {return error.c_str();}
static int classify_batch_impl(
    const SynkitClassProblem *problem,int64_t count,const int32_t *mappings,
    const int64_t *costs2,double seconds,int64_t max_nodes,int64_t max_classes,void *cancel_token,
    SynkitClassCallback callback,void *userdata,SynkitClassStatistics *statistics,
    bool updates,bool packed_identifier) {
  const auto invocation_start=Clock::now();
  try {
    error.clear();
    if(!problem||!statistics||!callback||count<0||(!mappings&&count&&problem->n)||(!costs2&&count)||
       !std::isfinite(seconds)||(seconds<0&&seconds!=-1)||(max_nodes<1&&max_nodes!=-1)||(max_classes<1&&max_classes!=-1))
      throw std::invalid_argument("invalid classification request");
    *statistics={};validate(*problem);
    const auto &p=*problem;
    Budget budget{invocation_start,seconds,max_nodes,0,0,cancel_token};
    budget.check();
    std::map<std::string,size_t> seen;
    // Exact fixed-label states share the same canonical certificate. Bound this
    // invocation-local accelerator; no approximate/WL/hash-only equality is used.
    constexpr size_t labelled_cache_bytes_limit=8*1024*1024;
    constexpr size_t labelled_cache_entries_limit=4096;
    size_t labelled_cache_bytes=0;
    std::map<std::string,size_t> labelled_seen;
    std::vector<Record> records;
    auto publish=[&](Record &record) {
      // Publication shares the original deadline. Node/class caps interrupt
      // construction, but may still leave time to deliver completed evidence.
      auto interrupted=[&]() {
        if(budget.seconds>=0 && std::chrono::duration<double>(Clock::now()-budget.start).count()>=budget.seconds) {
          if(!budget.reason) budget.reason=1;
          return true;
        }
        if(cancel_token && static_cast<synkit_mapper::Cancellation*>(cancel_token)->requested.load(std::memory_order_relaxed)) {
          if(!budget.reason) budget.reason=5;
          return true;
        }
        return false;
      };
      if(interrupted()) return false;
      if(callback(record.mapping.data(),record.cost,record.count,
          reinterpret_cast<const unsigned char*>(record.key.packed.data()),
          static_cast<int64_t>(record.key.packed.size()),record.key.identifier.c_str(),userdata)) {
        if(!budget.reason) budget.reason=4;
        return false;
      }
      record.published=record.count;
      return !interrupted();
    };
    for(int64_t k=0;k<count && budget.check();++k) {
      Perm mapping(p.n),taken(p.n,0),colors(p.n),edges(p.n*p.n,0);
      for(int i=0;i<p.n;++i) {
        int image=mappings[k*p.n+i];
        if(image<0||image>=p.n||taken[image]++) throw std::invalid_argument("mapping is not a bijection");
        mapping[i]=image;colors[i]=p.node_pair_colors[i*p.n+image];
      }
      if(costs2[k]<0) throw std::invalid_argument("negative objective cost");
      for(int i=0;i<p.n;++i) for(int j=i+1;j<p.n;++j)
        edges[i*p.n+j]=edges[j*p.n+i]=p.edge_pair_colors[
          p.edges_r[i*p.n+j]*p.edge_states_p+p.edges_p[mapping[i]*p.n+mapping[j]]];
      std::string labelled;
      labelled.reserve(static_cast<size_t>(p.n)*(p.n+1)*2);
      for(int color:colors) append32(labelled,static_cast<uint32_t>(color));
      for(int i=0;i<p.n;++i) for(int j=i+1;j<p.n;++j)
        append32(labelled,static_cast<uint32_t>(edges[i*p.n+j]));
      auto duplicate=labelled_seen.find(labelled);
      if(duplicate!=labelled_seen.end()) {
        ++records[duplicate->second].count;
        ++statistics->candidates;
        continue;
      }
      Canonical canonical(p.n,colors,edges,budget);
      Perm order=p.n?canonical.run():Perm{};
      if(budget.reason) break;
      auto packed=packed_certificate(p,colors,edges,order);
      auto old=seen.find(packed);
      bool discovered=old==seen.end();
      size_t record_index=discovered?records.size():old->second;
      if(old!=seen.end()) ++records[old->second].count;
      else {
        if(max_classes>=0 && static_cast<int64_t>(records.size())>=max_classes) {budget.reason=3;break;}
        auto identifier=packed_identifier?sha256(packed):legacy_identifier(p,colors,edges,order);
        Certificate key{std::move(packed),std::move(identifier)};
        seen.emplace(key.packed,records.size());records.push_back({std::move(mapping),costs2[k],1,std::move(key)});
      }
      if(labelled_seen.size()<labelled_cache_entries_limit &&
         labelled.size()<=labelled_cache_bytes_limit-labelled_cache_bytes) {
        labelled_cache_bytes+=labelled.size();
        labelled_seen.emplace(std::move(labelled),record_index);
      }
      ++statistics->candidates;
      // The additive update interface publishes first discovery immediately.
      // Legacy batch callbacks still receive exactly one final prefix count.
      if(updates && discovered && !publish(records.back())) break;
    }
    if(budget.reason!=4) for(auto &record:records)
      if((!updates || record.published!=record.count) && !publish(record)) break;
    statistics->classes=static_cast<int64_t>(records.size());statistics->canonical_nodes=budget.nodes;
    statistics->complete=budget.reason==0;statistics->reason=budget.reason;
    statistics->elapsed_seconds=std::chrono::duration<double>(Clock::now()-budget.start).count();
    return 0;
  } catch(const std::exception &exception) {error=exception.what();return -1;}
  catch(...) {error="unknown native classification error";return -1;}
}

extern "C" int synkit_mapper_classify_batch_with_cancel(
    const SynkitClassProblem *problem,int64_t count,const int32_t *mappings,
    const int64_t *costs2,double seconds,int64_t max_nodes,int64_t max_classes,void *cancel_token,
    SynkitClassCallback callback,void *userdata,SynkitClassStatistics *statistics) {
  return classify_batch_impl(problem,count,mappings,costs2,seconds,max_nodes,max_classes,
      cancel_token,callback,userdata,statistics,false,false);
}

extern "C" int synkit_mapper_classify_batch_updates_with_cancel(
    const SynkitClassProblem *problem,int64_t count,const int32_t *mappings,
    const int64_t *costs2,double seconds,int64_t max_nodes,int64_t max_classes,void *cancel_token,
    SynkitClassCallback callback,void *userdata,SynkitClassStatistics *statistics) {
  return classify_batch_impl(problem,count,mappings,costs2,seconds,max_nodes,max_classes,
      cancel_token,callback,userdata,statistics,true,false);
}

extern "C" int synkit_mapper_classify_batch_ex(
    const SynkitClassProblem *problem,int64_t count,const int32_t *mappings,
    const int64_t *costs2,double seconds,int64_t max_nodes,int64_t max_classes,void *cancel_token,
    SynkitClassCallback callback,void *userdata,SynkitClassStatistics *statistics,
    int32_t updates,int32_t packed_identifier) {
  if((updates!=0&&updates!=1)||(packed_identifier!=0&&packed_identifier!=1)) {
    error="classification option flags must be zero or one";return -1;
  }
  return classify_batch_impl(problem,count,mappings,costs2,seconds,max_nodes,max_classes,
      cancel_token,callback,userdata,statistics,updates!=0,packed_identifier!=0);
}

extern "C" int synkit_mapper_classify_batch(
    const SynkitClassProblem *problem,int64_t count,const int32_t *mappings,
    const int64_t *costs2,double seconds,int64_t max_nodes,int64_t max_classes,
    SynkitClassCallback callback,void *userdata,SynkitClassStatistics *statistics) {
  return synkit_mapper_classify_batch_with_cancel(problem,count,mappings,costs2,seconds,
      max_nodes,max_classes,nullptr,callback,userdata,statistics);
}


// The streaming context shares the same exact canonicalizer, certificate
// routines, cache units and cumulative-record contract as batch classification.
// It retains state while raw search is suspended inside its native callback.
namespace synkit_mapper::detail {
struct ClassStream {
  const SynkitClassProblem &p;
  ::Budget budget;
  int64_t max_classes,candidates=0;
  SynkitClassCallback callback;
  void *userdata;
  bool packed_identifier;
  double processing_seconds=0;
  std::map<std::string,size_t> seen,labelled_seen;
  size_t labelled_bytes=0;
  std::vector<::Record> records;
  ClassStream(const SynkitClassProblem &problem,double seconds,int64_t nodes,
      int64_t classes,void *cancel,SynkitClassCallback cb,void *data,bool packed):
      p(problem),budget{::Clock::now(),seconds,nodes,0,0,cancel},
      max_classes(classes),callback(cb),userdata(data),packed_identifier(packed) {
    if(!std::isfinite(seconds)||(seconds<0&&seconds!=-1)||
       (nodes<1&&nodes!=-1)||(classes<1&&classes!=-1)||!callback)
      throw std::invalid_argument("invalid streaming classification request");
    ::validate(p);
    processing_seconds=std::chrono::duration<double>(::Clock::now()-budget.start).count();
  }
  bool publish(::Record &record) {
    auto interrupted=[&]() {
      if(budget.seconds>=0&&std::chrono::duration<double>(::Clock::now()-budget.start).count()>=budget.seconds) {
        if(!budget.reason)budget.reason=1;
        return true;
      }
      if(budget.cancel_token&&static_cast<Cancellation*>(budget.cancel_token)->requested.load(std::memory_order_relaxed)) {
        if(!budget.reason)budget.reason=5;
        return true;
      }
      return false;
    };
    if(interrupted())return false;
    if(callback(record.mapping.data(),record.cost,record.count,
        reinterpret_cast<const unsigned char*>(record.key.packed.data()),
        record.key.packed.size(),record.key.identifier.c_str(),userdata)) {
      if(!budget.reason)budget.reason=4;
      return false;
    }
    record.published=record.count;
    return !interrupted();
  }
  int push(const int32_t *input,int64_t cost) {
    auto start=::Clock::now();
    struct Charge {
      ClassStream &self;::Clock::time_point start;
      ~Charge(){self.processing_seconds+=std::chrono::duration<double>(::Clock::now()-start).count();}
    } charge{*this,start};
    if(!budget.check())return budget.reason;
    ::Perm mapping(p.n),taken(p.n,0),colors(p.n),edges(p.n*p.n,0);
    for(int i=0;i<p.n;++i) {
      int image=input[i];
      if(image<0||image>=p.n||taken[image]++)throw std::invalid_argument("mapping is not a bijection");
      mapping[i]=image;colors[i]=p.node_pair_colors[i*p.n+image];
    }
    if(cost<0)throw std::invalid_argument("negative objective cost");
    for(int i=0;i<p.n;++i)for(int j=i+1;j<p.n;++j)
      edges[i*p.n+j]=edges[j*p.n+i]=p.edge_pair_colors[
        p.edges_r[i*p.n+j]*p.edge_states_p+p.edges_p[mapping[i]*p.n+mapping[j]]];
    std::string labelled;labelled.reserve(static_cast<size_t>(p.n)*(p.n+1)*2);
    for(int color:colors)append32(labelled,static_cast<uint32_t>(color));
    for(int i=0;i<p.n;++i)for(int j=i+1;j<p.n;++j)
      append32(labelled,static_cast<uint32_t>(edges[i*p.n+j]));
    auto duplicate=labelled_seen.find(labelled);
    if(duplicate!=labelled_seen.end()) {
      ++records[duplicate->second].count;++candidates;
      return 0;
    }
    ::Canonical canonical(p.n,colors,edges,budget);
    ::Perm order=p.n?canonical.run(): ::Perm{};
    if(budget.reason)return budget.reason;
    auto packed=packed_certificate(p,colors,edges,order);
    auto old=seen.find(packed);
    bool discovered=old==seen.end();
    size_t index=discovered?records.size():old->second;
    if(!discovered)++records[index].count;
    else {
      if(max_classes>=0&&static_cast<int64_t>(records.size())>=max_classes) {
        budget.reason=3;return 3;
      }
      auto identifier=packed_identifier?sha256(packed):legacy_identifier(p,colors,edges,order);
      ::Certificate key{std::move(packed),std::move(identifier)};
      seen.emplace(key.packed,records.size());
      records.push_back({std::move(mapping),cost,1,std::move(key)});
    }
    if(labelled_seen.size()<4096&&labelled.size()<=8*1024*1024-labelled_bytes) {
      labelled_bytes+=labelled.size();labelled_seen.emplace(std::move(labelled),index);
    }
    ++candidates;
    if(discovered&&!publish(records.back()))return budget.reason;
    return 0;
  }
  void finish(SynkitClassStatistics &statistics) {
    auto start=::Clock::now();
    if(budget.reason!=4)for(auto &record:records)
      if(record.published!=record.count&&!publish(record))break;
    budget.check();
    processing_seconds+=std::chrono::duration<double>(::Clock::now()-start).count();
    statistics={};statistics.candidates=candidates;statistics.classes=records.size();
    statistics.canonical_nodes=budget.nodes;statistics.complete=budget.reason==0;
    statistics.reason=budget.reason;statistics.elapsed_seconds=processing_seconds;
  }
};
ClassStream *class_stream_create(const SynkitClassProblem &p,double seconds,
    int64_t nodes,int64_t classes,void *cancel,SynkitClassCallback cb,
    void *data,bool packed) {
  ::error.clear();
  try{return new ClassStream(p,seconds,nodes,classes,cancel,cb,data,packed);}
  catch(const std::exception &exception){::error=exception.what();throw;}
}
int class_stream_push(ClassStream *context,const int32_t *mapping,int64_t cost) {
  try{return context->push(mapping,cost);}
  catch(const std::exception &exception){::error=exception.what();throw;}
}
void class_stream_finish(ClassStream *context,SynkitClassStatistics &statistics) {
  try{context->finish(statistics);}
  catch(const std::exception &exception){::error=exception.what();throw;}
}
void class_stream_delete(ClassStream *context) {delete context;}
}
