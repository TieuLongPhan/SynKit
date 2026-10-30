#include "synkit_mapper/symmetry/cpabs.h"
#include "synkit_mapper/core/runtime.h"
#include "synkit_mapper/pabs/seed.h"
#include "synkit_mapper/astar/bound.h"

#include <functional>
#include <limits>
#include <numeric>
#include <set>
#include <utility>

namespace synkit_mapper {
namespace {
using Perm=std::vector<int32_t>;
using Matching=std::vector<int32_t>; // sorted r*n+p pairs; arbitrary source set
using Group=std::vector<Perm>;
thread_local SynkitCPABSProfile profile;
thread_local SynkitCPABSProfileV2 detailed;

// Close only admitted generators. If adding one would exceed the explicit
// quotient budget, retain the previous closed subgroup, never an open prefix.
Group bounded_subgroup(const Group &witnesses,int n,size_t cap,Budget &budget) {
  Perm identity(n);std::iota(identity.begin(),identity.end(),0);
  Group group{identity},generators;
  std::set<Perm> known{identity};
  for(const auto &witness:witnesses) {
    budget.check();if(known.count(witness)) continue;
    Group candidate{identity},basis=generators;basis.push_back(witness);
    std::set<Perm> seen{identity};bool over=false;
    for(size_t k=0;k<candidate.size()&&!over;++k) for(const auto &g:basis) {
      budget.check();Perm image(n);for(int i=0;i<n;++i) image[i]=g[candidate[k][i]];
      if(seen.insert(image).second) {candidate.push_back(std::move(image));if(candidate.size()>cap) {over=true;break;}}
    }
    if(!over) {group=std::move(candidate);known=std::move(seen);generators=std::move(basis);}
  }
  return group;
}

// Exact twin symmetric factors are represented implicitly. The quotient group
// is enumerated only on the blocks; exceeding a discovery cap retains all
// twin factors and retains a closed bounded quotient subgroup.
struct EndpointSymmetry {
  std::vector<Perm> blocks;
  Perm block_of;
  Group quotient;
  bool fallback=false;
  int64_t order=1;
};
int64_t multiply_order(int64_t a,int64_t b) {
  if(a<0||b<0||a>std::numeric_limits<int64_t>::max()/b) return -1;
  return a*b;
}
EndpointSymmetry endpoint_group(const Problem &p,bool product,Budget &budget) {
  auto weights=product?p.b2:p.a2;
  auto types=product?p.types_p:p.types_r;
  auto colors=product?p.node_colors_p:p.node_colors_r;
  auto edges=product?p.edge_colors_p:p.edge_colors_r;
  EndpointSymmetry result;result.block_of.assign(p.n,-1);
  auto twins=[&](int i,int j) {
    if(types[i]!=types[j]||(colors&&colors[i]!=colors[j])||(edges&&edges[i*p.n+i]!=edges[j*p.n+j])) return false;
    for(int k=0;k<p.n;++k) if(k!=i&&k!=j &&
      (weights[i*p.n+k]!=weights[j*p.n+k] ||
       (edges&&edges[i*p.n+k]!=edges[j*p.n+k]))) return false;
    return true;
  };
  for(int i=0;i<p.n;++i) if(result.block_of[i]<0) {
    budget.check();
    int id=static_cast<int>(result.blocks.size());Perm block{i};result.block_of[i]=id;
    for(int j=i+1;j<p.n;++j) if(result.block_of[j]<0&&twins(i,j)) {block.push_back(j);result.block_of[j]=id;}
    for(size_t k=2;k<=block.size();++k) result.order=multiply_order(result.order,k);
    result.blocks.push_back(std::move(block));
  }
  int m=static_cast<int>(result.blocks.size());
  std::vector<Perm> choices(m);
  std::vector<std::vector<int64_t>> initial(m);
  for(int i=0;i<m;++i) {
    int atom=result.blocks[i][0],second=result.blocks[i].size()>1?result.blocks[i][1]:atom;
    initial[i]={static_cast<int64_t>(result.blocks[i].size()),types[atom],colors?colors[atom]:0,
                weights[atom*p.n+second],edges?edges[atom*p.n+second]:0};
  }
  Perm refined(m);
  auto intern=[&](const std::vector<std::vector<int64_t>> &signatures) {
    std::map<std::vector<int64_t>,int> ids;
    for(const auto &signature:signatures) ids.emplace(signature,0);
    int next=0;for(auto &entry:ids) entry.second=next++;
    Perm labels;for(const auto &signature:signatures) labels.push_back(ids.at(signature));return labels;
  };
  refined=intern(initial);
  for(int round=0;round<m;++round) {
    budget.check();auto signatures=initial;
    for(int i=0;i<m;++i) {
      std::vector<std::vector<int64_t>> neighbors;
      for(int j=0;j<m;++j) if(i!=j) {
        int x=result.blocks[i][0],y=result.blocks[j][0];
        neighbors.push_back({refined[j],weights[x*p.n+y],edges?edges[x*p.n+y]:0});
      }
      std::sort(neighbors.begin(),neighbors.end());
      signatures[i].push_back(refined[i]);
      for(const auto &neighbor:neighbors) signatures[i].insert(signatures[i].end(),neighbor.begin(),neighbor.end());
    }
    Perm next=intern(signatures);if(next==refined) break;refined=std::move(next);
  }
  for(int i=0;i<m;++i) for(int j=0;j<m;++j) if(refined[i]==refined[j]) choices[i].push_back(j);
  Perm mapping(m,-1),used(m,0);
  int64_t nodes=0;bool over=false;
  std::function<void(int)> visit=[&](int depth) {
    budget.check();if(++nodes>200000||result.quotient.size()>256) {over=true;return;}
    if(depth==m) {result.quotient.push_back(mapping);return;}
    int chosen=-1;Perm valid_choices;
    for(int i=0;i<m;++i) if(mapping[i]<0) {
      Perm valid;
      for(int j:choices[i]) if(!used[j]) {
        bool compatible=true;
        for(int k=0;k<m;++k) if(mapping[k]>=0) {
          int x=result.blocks[i][0],y=result.blocks[k][0],u=result.blocks[j][0],v=result.blocks[mapping[k]][0];
          if(weights[x*p.n+y]!=weights[u*p.n+v]||(edges&&edges[x*p.n+y]!=edges[u*p.n+v])) {compatible=false;break;}
        }
        if(compatible) valid.push_back(j);
      }
      if(valid.empty()) return;
      if(chosen<0||valid.size()<valid_choices.size()) {chosen=i;valid_choices=std::move(valid);}
    }
    for(int j:valid_choices) {
      mapping[chosen]=j;used[j]=1;visit(depth+1);mapping[chosen]=-1;used[j]=0;if(over) return;
    }
  };
  visit(0);profile.discovery_nodes+=nodes;
  if(over||result.quotient.size()>256) {
    result.fallback=true;result.quotient=bounded_subgroup(result.quotient,m,256,budget);
  }
  if(result.quotient.empty()) throw std::logic_error("Endpoint quotient identity missing");
  result.order=multiply_order(result.order,result.quotient.size());
  return result;
}
EndpointSymmetry identity_group(int n) {
  EndpointSymmetry result;result.block_of.resize(n);Perm id(n);
  for(int i=0;i<n;++i) {result.blocks.push_back(Perm{i});result.block_of[i]=id[i]=i;}
  result.quotient={id};return result;
}

// An orbit of the two independent twin factors is exactly its block-flow
// matrix. Greedy earliest source/target vertices give its lexicographic minimum.
Matching twin_canonical(const Matching &matching,const EndpointSymmetry &r,
                        const EndpointSymmetry &p,const Perm &rg,const Perm &pg,int n) {
  std::map<std::pair<int,int>,int> cells;
  std::vector<int> row_counts(r.blocks.size(),0),row_used(r.blocks.size(),0),column_used(p.blocks.size(),0);
  for(int pair:matching) {
    int a=rg[r.block_of[pair/n]],b=pg[p.block_of[pair%n]];
    ++cells[{a,b}];++row_counts[a];
  }
  Perm selected;
  for(size_t a=0;a<r.blocks.size();++a) for(int k=0;k<row_counts[a];++k) selected.push_back(r.blocks[a][k]);
  std::sort(selected.begin(),selected.end());Matching result;
  for(int source:selected) {
    int a=r.block_of[source],chosen=-1,target=n;
    for(const auto &entry:cells) if(entry.first.first==a&&entry.second) {
      int b=entry.first.second,next=p.blocks[b][column_used[b]];
      if(next<target) {target=next;chosen=b;}
    }
    if(chosen<0) throw std::logic_error("Invalid twin flow");
    result.push_back(source*n+target);--cells[{a,chosen}];++column_used[chosen];++row_used[a];
  }
  return result;
}

struct Traversal {
  const Problem &p;
  const Request &request;
  Callback callback;
  void *userdata;
  Statistics &stats;
  Budget budget;
  EndpointSymmetry rows,columns;
  int64_t best=std::numeric_limits<int64_t>::max();
  int64_t threshold=std::numeric_limits<int64_t>::max();
  bool optimization=false;
  int64_t pending=0;
  Matching base;
  Perm variable_sources,base_source;
  std::vector<Perm> prefix_counts,next_block_possible;
  Traversal(const Problem &problem,const Request &r,Callback cb,void *data,Statistics &s)
      :p(problem),request(r),callback(cb),userdata(data),stats(s),budget(r) {}
  // Singleton hard types force these pairs in every full compatible map.
  // Both endpoint groups fix them pointwise, including bounded subgroups.
  void prepare_root() {
    std::map<int32_t,Perm> source_types,target_types;
    for(int i=0;i<p.n;++i) {
      budget.check();source_types[p.types_r[i]].push_back(i);
      target_types[p.types_p[i]].push_back(i);
    }
    base_source.assign(p.n,0);
    for(int i=0;i<p.n;++i) {
      if(source_types[p.types_r[i]].size()==1) {
        const auto &targets=target_types.at(p.types_r[i]);
        if(targets.size()!=1) throw std::logic_error("Incompatible forced type");
        base.push_back(i*p.n+targets.front());base_source[i]=1;
      } else variable_sources.push_back(i);
    }
    prefix_counts.push_back(Perm(rows.blocks.size(),0));
    for(int pair:base) ++prefix_counts.back()[rows.block_of[pair/p.n]];
    for(int i:variable_sources) {
      budget.check();prefix_counts.push_back(prefix_counts.back());
      ++prefix_counts.back()[rows.block_of[i]];
    }
    next_block_possible.assign(variable_sources.size(),Perm(rows.blocks.size(),-1));
  }
  bool prefix_domain(const Matching &matching) const {
    size_t next=0;
    for(int pair:matching) if(!base_source[pair/p.n]) {
      if(next>=variable_sources.size()||pair/p.n!=variable_sources[next++]) return false;
    }
    return matching.size()==base.size()+next;
  }
  // Necessary domain-orbit test only. Twins realize every subset of a block
  // with the specified cardinality; quotient permutations move whole blocks.
  // Full canonicalization and its actual prefix test remain mandatory.
  bool possible_prefix_extension(size_t matched,int block) {
    const size_t depth=matched-base.size();
    auto &answer=next_block_possible.at(depth).at(block);
    if(answer>=0) return answer;
    Perm counts=prefix_counts.at(depth);++counts[block];
    const auto &wanted=prefix_counts.at(depth+1);
    for(const auto &g:rows.quotient) {
      budget.check();bool equal=true;
      for(size_t a=0;a<counts.size();++a) if(counts[a]!=wanted[g[a]]) {equal=false;break;}
      if(equal) {answer=1;return true;}
    }
    answer=0;return false;
  }
  Matching canonical(const Matching &matching) {
    budget.check();
    auto start=Clock::now();++profile.canonical_calls;
    Matching best_matching=matching,image;image.reserve(matching.size());
    for(const auto &row:rows.quotient) for(const auto &column:columns.quotient) {
      budget.check();image=twin_canonical(matching,rows,columns,row,column,p.n);
      if(image<best_matching) best_matching=image;
    }
    profile.canonical_seconds+=elapsed(start);
    return best_matching;
  }
  int64_t partial_cost(const Matching &matching) {
    int64_t value=0;
    for(size_t i=0;i<matching.size();++i) for(size_t j=i+1;j<matching.size();++j)
      value+=std::abs(p.a2[(matching[i]/p.n)*p.n+matching[j]/p.n]-
                      p.b2[(matching[i]%p.n)*p.n+matching[j]%p.n]);
    return value;
  }
  int64_t lower_bound(const Matching &matching) {
    auto start=Clock::now();++detailed.bound_calls;
    Perm mapping(p.n,-1);for(int pair:matching) mapping[pair/p.n]=pair%p.n;
    int64_t value=astar::partial_lower_bound2(p.n,p.a2,p.b2,p.types_r,p.types_p,mapping,[&](){budget.check();});
    detailed.bound_seconds+=elapsed(start);return value;
  }
  void seed() {
    auto started=Clock::now();
    // The controlled variant shares PABS's complete input-only seed policy.
    // This changes only a feasible optimization upper bound, never the
    // canonical-parent extension tree or numeric exact/at-most enumeration.
    std::vector<int32_t> shared_mapping;
    int64_t shared_incumbent=-1,shared_lower=-1;
    auto retain_shared=[&]() {
      if(static_cast<int>(shared_mapping.size())!=p.n||shared_incumbent<0) return;
      std::vector<bool> seen(p.n,false);
      for(int i=0;i<p.n;++i) {
        const int j=shared_mapping[i];
        if(j<0||j>=p.n||seen[j]||p.types_r[i]!=p.types_p[j])
          throw std::logic_error("cPABS shared seed is not a typed bijection");
        seen[j]=true;
      }
      if(mapping_cost2(p,shared_mapping)!=shared_incumbent)
        throw std::logic_error("cPABS shared seed rescore mismatch");
      if(shared_lower>shared_incumbent)
        throw std::logic_error("cPABS shared lower bound exceeds feasible seed");
      best=std::min(best,shared_incumbent);stats.incumbent2=best;
      stats.lower_bound2=std::max(stats.lower_bound2,shared_lower);
      detailed.seed_cost2=best;detailed.seed_seconds=elapsed(started);
    };
    try {
      if(pabs::prepare_seed(p,budget,true,shared_mapping,shared_incumbent,shared_lower)) {
        if(static_cast<int>(shared_mapping.size())!=p.n||shared_incumbent<0)
          throw std::logic_error("cPABS shared seed helper returned no feasible map");
        retain_shared();return;
      }
    } catch(const Stop &) {
      // The helper publishes a valid incumbent before checking its deadline.
      retain_shared();detailed.seed_seconds=elapsed(started);throw;
    }
    // Keep cPABS's former seed policy outside the shared PABS helper domain.
    Perm source(p.n),target(p.n);std::iota(source.begin(),source.end(),0);target=source;
    std::vector<int64_t> left(p.n),right(p.n);
    for(int i=0;i<p.n;++i) for(int j=0;j<p.n;++j) {left[i]+=p.a2[i*p.n+j];right[i]+=p.b2[i*p.n+j];}
    std::stable_sort(source.begin(),source.end(),[&](int i,int j){return std::make_pair(p.types_r[i],left[i])<std::make_pair(p.types_r[j],left[j]);});
    std::stable_sort(target.begin(),target.end(),[&](int i,int j){return std::make_pair(p.types_p[i],right[i])<std::make_pair(p.types_p[j],right[j]);});
    Perm mapping(p.n);for(int k=0;k<p.n;++k) mapping[source[k]]=target[k];
    auto improve=[&](Perm candidate) {
    int64_t candidate_cost=mapping_cost2(p,candidate);
    best=std::min(best,candidate_cost);stats.incumbent2=best;
    for(int pass=0;pass<8;++pass) {
      budget.check();bool improved=false;
      for(int i=0;i<p.n;++i) for(int j=i+1;j<p.n;++j) if(p.types_r[i]==p.types_r[j]) {
        budget.check();int64_t delta=0;
        for(int k=0;k<p.n;++k) if(k!=i&&k!=j) {
          delta+=std::abs(p.a2[i*p.n+k]-p.b2[candidate[j]*p.n+candidate[k]])+std::abs(p.a2[j*p.n+k]-p.b2[candidate[i]*p.n+candidate[k]])-
                 std::abs(p.a2[i*p.n+k]-p.b2[candidate[i]*p.n+candidate[k]])-std::abs(p.a2[j*p.n+k]-p.b2[candidate[j]*p.n+candidate[k]]);
        }
        if(delta<0) {
          std::swap(candidate[i],candidate[j]);candidate_cost+=delta;improved=true;
          best=std::min(best,candidate_cost);stats.incumbent2=best;
        }
      }
      if(!improved) break;
    }
    if(mapping_cost2(p,candidate)!=candidate_cost) throw std::logic_error("cPABS seed rescore mismatch");
    };
    improve(mapping);
    // Same typed profile-LAP relaxation as the PABS starting assignment.
    // Keep the old degree seed as a second candidate, so the feasible upper
    // bound cannot regress. Neither seed uses an annotation or narrows search.
    improve(astar::profile_assignment_seed(p.n,p.a2,p.b2,p.types_r,p.types_p,
                                          [&](){budget.check();}));
    stats.incumbent2=best;
    detailed.seed_cost2=best;detailed.seed_seconds=elapsed(started);
  }
  void leaf(const Matching &matching,int64_t cost) {
    ++stats.leaves;
    if(optimization) {
      best=std::min(best,cost);stats.incumbent2=best;return;
    }
    if((request.query==SYNKIT_EXACT_DISTANCE||request.query==SYNKIT_MINIMUM_CLASSES)&&cost!=threshold) return;
    if(cost>threshold) return;
    if(request.max_candidates>=0&&stats.candidate_count>=request.max_candidates) throw Stop{SYNKIT_STOP_CANDIDATES};
    Perm mapping(p.n,-1);
    for(int pair:matching) mapping[pair/p.n]=pair%p.n;
    if(mapping_cost2(p,mapping)!=cost) throw std::logic_error("cPABS independent rescore mismatch");
    stats.incumbent2=stats.incumbent2<0?cost:std::min(stats.incumbent2,cost);
    ++stats.candidate_count;
    auto started=Clock::now();
    int stop=callback(mapping.data(),cost,userdata);
    stats.callback_seconds+=elapsed(started);
    if(stop) throw Stop{SYNKIT_STOP_CALLBACK};
    budget.check();
  }
  void visit(const Matching &matching,int64_t known_bound=-1,int64_t known_cost=-1) {
    budget.check();
    if(request.max_states>=0&&stats.nodes>=request.max_states) throw Stop{SYNKIT_STOP_STATES};
    ++stats.nodes;
    const int64_t cost=known_cost<0?partial_cost(matching):known_cost;
    const int64_t limit=optimization?best:threshold;
    const int64_t bound=known_bound<0?lower_bound(matching):known_bound;
    if(matching.size()==base.size()) stats.lower_bound2=std::max(stats.lower_bound2,bound);
    if(bound>limit||(optimization&&bound>=best)) {++profile.bound_rejections;return;}
    if(static_cast<int>(matching.size())==p.n) {leaf(matching,cost);return;}
    Perm assigned(p.n,0),used(p.n,0);
    for(int pair:matching) {assigned[pair/p.n]=1;used[pair%p.n]=1;}
    // Unused vertices in one exact twin block can be swapped while fixing
    // the current matching pointwise. One representative per block is enough;
    // retain every eligible source/target block pair, including coupled orbits.
    Perm targets;
    std::vector<bool> target_seen(columns.blocks.size(),false),source_seen(rows.blocks.size(),false);
    for(int j=0;j<p.n;++j) if(!used[j]&&!target_seen[columns.block_of[j]]) {
      targets.push_back(j);target_seen[columns.block_of[j]]=true;
    }
    // A negative bound marks an exactly considered but rejected canonical key.
    // This deduplicates before expensive LAPs, including keys failing a bound.
    std::map<Matching,std::pair<int64_t,int64_t>> children;
    int64_t accepted=0;
    for(int i=0;i<p.n;++i) if(!assigned[i]&&!source_seen[rows.block_of[i]]) {
      source_seen[rows.block_of[i]]=true;
      if(!possible_prefix_extension(matching.size(),rows.block_of[i])) continue;
      for(int j:targets) if(p.types_r[i]==p.types_p[j]) {
        budget.check();++profile.augmentations;
        int64_t child_cost=cost;
        for(int pair:matching) child_cost+=std::abs(p.a2[i*p.n+pair/p.n]-p.b2[j*p.n+pair%p.n]);
        if(child_cost>limit||(optimization&&child_cost>=best)) {++profile.bound_rejections;continue;}
        Matching child=matching;child.push_back(i*p.n+j);std::sort(child.begin(),child.end());
        child=canonical(child);
        if(!prefix_domain(child)) {++profile.canonical_rejections;continue;}
        auto inserted=children.emplace(std::move(child),std::make_pair(-1,child_cost));
        if(!inserted.second) continue;
        const auto &key=inserted.first->first;
        Matching parent=key;
        int remove=static_cast<int>(parent.size())-1;
        while(remove>=0&&base_source[parent[remove]/p.n]) --remove;
        if(remove<0) throw std::logic_error("Non-base extension missing");
        parent.erase(parent.begin()+remove);parent=canonical(parent);
        if(parent!=matching) {++profile.canonical_rejections;continue;}
        const int64_t child_bound=lower_bound(key);
        if(child_bound>limit||(optimization&&child_bound>=best)) {++profile.bound_rejections;continue;}
        inserted.first->second.first=child_bound;++accepted;
      }
    }
    pending+=accepted;
    stats.peak_frontier=std::max(stats.peak_frontier,pending);
    if(request.max_frontier>=0&&pending>request.max_frontier) throw Stop{SYNKIT_STOP_FRONTIER};
    while(!children.empty()) {
      auto node=children.extract(children.begin());
      if(node.mapped().first<0) continue;
      --pending;visit(node.key(),node.mapped().first,node.mapped().second);
    }
  }
  int run() {
    if(request.classify||request.max_classes!=-1||request.memory_limit_bytes!=-1) return unsupported(stats);
    if(p.n>256) return unsupported(stats);
    profile={};detailed={};detailed.seed_cost2=-1;stats.peak_frontier_bytes=-1;
    auto start=Clock::now();
    if(request.symmetry) {
      rows=endpoint_group(p,false,budget);columns=endpoint_group(p,true,budget);
      if(rows.quotient.size()*columns.quotient.size()>16384) {
        if(columns.order>=0) columns.order/=static_cast<int64_t>(columns.quotient.size());
        columns.quotient=bounded_subgroup(columns.quotient,columns.blocks.size(),16384/rows.quotient.size(),budget);
        columns.order=multiply_order(columns.order,columns.quotient.size());columns.fallback=true;
      }
      profile.identity_fallback_r=rows.fallback;profile.identity_fallback_p=columns.fallback;
    } else {rows=identity_group(p.n);columns=identity_group(p.n);}
    profile.subgroup_order=multiply_order(rows.order,columns.order);
    detailed.twin_blocks_r=rows.blocks.size();detailed.twin_blocks_p=columns.blocks.size();
    detailed.quotient_order_r=rows.quotient.size();detailed.quotient_order_p=columns.quotient.size();
    profile.discovery_seconds=elapsed(start);stats.model_seconds=profile.discovery_seconds;
    start=Clock::now();prepare_root();stats.model_seconds+=elapsed(start);
    try {
      if(request.query==SYNKIT_MINIMUM_CLASSES||request.query==SYNKIT_MINIMUM_PROOF) {
        optimization=true;start=Clock::now();seed();visit(base);stats.optimization_seconds=elapsed(start);
        stats.minimum_proven=1;stats.minimum2=stats.incumbent2=stats.lower_bound2=best;
        if(request.query==SYNKIT_MINIMUM_PROOF) {stats.search_complete=1;stats.elapsed_seconds=budget.seconds();return 0;}
        threshold=best;optimization=false;
      } else threshold=request.target2;
      start=Clock::now();visit(base);stats.enumeration_seconds=elapsed(start);stats.search_complete=1;
    } catch(const Stop &stop) {
      if(optimization) stats.optimization_seconds=budget.seconds()-stats.model_seconds;
      else stats.enumeration_seconds=budget.seconds()-stats.model_seconds-stats.optimization_seconds;
      stopped(stats,stop.reason);
    }
    if(stats.minimum_proven==0&&stats.lower_bound2<0) stats.lower_bound2=0;
    // Raw output counts subgroup orbits, not reaction classes or labeled maps.
    stats.labeled_count=-1;stats.class_count=-1;
    stats.elapsed_seconds=budget.seconds();return 0;
  }
};
} // namespace

int solve_cpabs(const Problem &p,const Request &r,Callback callback,void *userdata,Statistics &statistics) {
  initialize(statistics,r);
  if(!callback&&r.query!=SYNKIT_MINIMUM_PROOF) throw std::invalid_argument("cPABS enumeration requires a callback");
  Traversal search(p,r,callback,userdata,statistics);
  try {return search.run();}
  catch(const Stop &stop) {stopped(statistics,stop.reason);statistics.elapsed_seconds=search.budget.seconds();return 0;}
}
} // namespace synkit_mapper

extern "C" void synkit_mapper_cpabs_profile(SynkitCPABSProfile *out) {
  if(out) *out=synkit_mapper::profile;
}
extern "C" void synkit_mapper_cpabs_profile_v2(SynkitCPABSProfileV2 *out) {
  if(out) {*out=synkit_mapper::detailed;out->base=synkit_mapper::profile;}
}
