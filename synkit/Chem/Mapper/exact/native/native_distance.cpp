// Exact integer-distance candidate enumeration for two-sided symmetry analysis.
// ABI 3: all matrices and indices are validated by the Python wrapper.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <map>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <unordered_set>
#include <vector>
using Perm=std::vector<int>;
using Group=std::vector<Perm>;
using Clock=std::chrono::steady_clock;
using Callback=int (*)(const int*, void*);
constexpr int INF=100000000;
// Hash only an initial branching prefix: each surviving subtree belongs to
// exactly one shard. Leaves with fewer choices are partitioned at the leaf.
constexpr int SHARD_BRANCHES=24;
struct VectorHash {
 std::size_t operator()(const Perm& p) const noexcept {
  std::size_t h=0; for(int x:p) h=h*1315423911u+static_cast<unsigned>(x+1); return h;
 }
};
static Perm compose(const Perm&a,const Perm&b){
 Perm p(a.size());for(size_t i=0;i<a.size();++i)p[i]=b[a[i]];return p;
}
static std::shared_ptr<Group> stabilize(std::shared_ptr<Group> gs,int point,int n,bool*complete=nullptr){
 bool fixed=true;for(const auto&g:*gs)if(g[point]!=point){fixed=false;break;}
 if(fixed)return gs;
 Perm identity(n);std::iota(identity.begin(),identity.end(),0);
 std::vector<Perm> trans(n),inv(n);std::vector<int> orbit{point};trans[point]=identity;
 int work=0;
 for(size_t k=0;k<orbit.size();++k)for(const auto&g:*gs){
  if(++work>100000){if(complete)*complete=false;return std::make_shared<Group>();}
  int x=orbit[k],y=g[x];
  if(trans[y].empty()){trans[y]=compose(trans[x],g);orbit.push_back(y);}
 }
 for(int x:orbit){inv[x].resize(n);for(int i=0;i<n;++i)inv[x][trans[x][i]]=i;}
 auto out=std::make_shared<Group>();std::unordered_set<Perm,VectorHash> seen;
 for(int x:orbit)for(const auto&g:*gs){
  Perm h=compose(compose(trans[x],g),inv[g[x]]);
  if(h!=identity&&seen.insert(h).second){
   out->push_back(std::move(h));if(out->size()>512){if(complete)*complete=false;return std::make_shared<Group>();}
  }
 }
 return out;
}
struct Solver {
 int n,nt,nl,target;double seconds;int64_t cap;int shard_index=0,shards=1;
 const int *a,*b,*ca,*cb,*order,*pred; // pred is reserved for ABI compatibility.
 std::vector<int> mapping,used,cross,ha,hb,levels,row_order,row_min;
 std::vector<char> permitted;
 Callback callback;void*userdata;Clock::time_point started;
 // Resumable jobs name a subtree by its deterministic sequence of images.
 // Counting only nodes below that prefix guarantees progress even for tiny
 // slices and long chains of forced assignments.
 Perm prefix,path;std::vector<Perm> frontier;
 struct PrefixNode {std::map<int,int> children;bool terminal=false;};
 std::vector<PrefixNode> requests;bool batch_mode=false,balance_slice=false;
 int64_t replay_nodes=0,new_nodes=0;
 int64_t slice_nodes=0,work_nodes=0;Clock::time_point slice_started{};
 int status=0;int64_t nodes=0,leaves=0,accepted=0,pruned=0;
 Solver(int nn,int types,int lev,int tar,double sec,int64_t limit,
        const int*aa,const int*bb,const int*cca,const int*ccb,const int*ord,
        const int*pre,Callback call,void*data):
  n(nn),nt(types),nl(lev),target(tar),seconds(sec),cap(limit),
  a(aa),b(bb),ca(cca),cb(ccb),order(ord),pred(pre),
  mapping(n,-1),used(n,0),cross(n*n,0),ha(n*nt*nl),hb(n*nt*nl),row_order(ord,ord+n),row_min(n,-1),permitted(n*n,1),
  callback(call),userdata(data),started(Clock::now()){
   std::vector<int> values(a,a+n*n);values.insert(values.end(),b,b+n*n);
   std::sort(values.begin(),values.end());values.erase(std::unique(values.begin(),values.end()),values.end());
   if(static_cast<int>(values.size())!=nl+1)throw std::runtime_error("levels");
   levels=values;
   for(int i=0;i<n;++i)for(int j=0;j<n;++j)for(int k=0;k<nl;++k){
    if(a[i*n+j]>=levels[k+1])++ha[(i*nt+ca[j])*nl+k];
    if(b[i*n+j]>=levels[k+1])++hb[(i*nt+cb[j])*nl+k];
   }
 }
 bool expired(){return std::chrono::duration<double>(Clock::now()-started).count()>=seconds;}
 void update(int atom,int image,int sign,int depth,const Perm&columns){
  // row_order[depth+1:] and columns minus image are exactly the unassigned
  // vertices, both on entry and after the recursive row-order restoration.
  for(int pos=depth+1;pos<n;++pos){
   int i=row_order[pos];
   for(int k=0;k<nl;++k)if(a[i*n+atom]>=levels[k+1])
    ha[(i*nt+ca[atom])*nl+k]-=sign;
   for(int j:columns)if(j!=image)
    cross[i*n+j]+=sign*2*std::abs(a[i*n+atom]-b[j*n+image]);
  }
  for(int j:columns)if(j!=image)for(int k=0;k<nl;++k)
   if(b[j*n+image]>=levels[k+1])hb[(j*nt+cb[image])*nl+k]-=sign;
 }
 // Dense Hungarian assignment with integer duals. Infeasibility fails closed.
 static int assignment(const std::vector<int>&cost,int m,Perm&match,Perm&u,Perm&v,
                       const Perm*seed_v=nullptr,const Perm*seed_match=nullptr){
  Perm p(m+1),way(m+1);u.assign(m+1,0);v.assign(m+1,0);
  // Inherited potentials are hints. Repair every row against CURRENT finite
  // edges; never assume the parent dual remains feasible after a cost change.
  const bool warm=seed_v&&static_cast<int>(seed_v->size())==m;
  if(warm)
   for(int j=1;j<=m;++j)v[j]=std::clamp((*seed_v)[j-1],-1000000,1000000);
  // Feasible row/column dual reductions, followed by a complementary
  // zero-reduced-cost partial matching. Augment only unmatched rows.
  for(int i=1;i<=m;++i){
   int best=INF;
   if(warm){
    for(int j=1;j<=m;++j)if(cost[(i-1)*m+j-1]<INF/2)
     best=std::min(best,cost[(i-1)*m+j-1]-v[j]);
   }else{
    // Preserve the simple vectorizable reduction for fresh small problems.
    for(int j=1;j<=m;++j)best=std::min(best,cost[(i-1)*m+j-1]);
   }
   if(best>=INF/2)return INF;
   u[i]=best;
  }
  for(int j=1;j<=m;++j){
   int best=INF;for(int i=1;i<=m;++i)if(cost[(i-1)*m+j-1]<INF/2)
    best=std::min(best,cost[(i-1)*m+j-1]-u[i]);
   if(best>=INF/2)return INF;
   v[j]=best;
  }
  std::vector<char>matched(m+1,0);
  // Keep only complementary, finite, disjoint inherited matching edges.
  if(seed_match&&static_cast<int>(seed_match->size())==m)
   for(int i=1;i<=m;++i){
    int column=(*seed_match)[i-1];if(column<0||column>=m)continue;
    int j=column+1;
    if(!p[j]&&cost[(i-1)*m+j-1]<INF/2&&
       cost[(i-1)*m+j-1]==u[i]+v[j]){p[j]=i;matched[i]=1;}
   }
  for(int i=1;i<=m;++i)if(!matched[i])for(int j=1;j<=m;++j)
   if(!p[j]&&cost[(i-1)*m+j-1]<INF/2&&cost[(i-1)*m+j-1]==u[i]+v[j]){
    p[j]=i;matched[i]=1;break;
   }
  for(int i=1;i<=m;++i){
   if(matched[i])continue;
   p[0]=i;int j0=0;Perm minv(m+1,INF);std::vector<char>seen(m+1,0);
   do{
    seen[j0]=1;int i0=p[j0],delta=INF,j1=0;
    for(int j=1;j<=m;++j)if(!seen[j]){
     int cur=cost[(i0-1)*m+j-1]-u[i0]-v[j];
     if(cur<minv[j]){minv[j]=cur;way[j]=j0;}
     if(minv[j]<delta){delta=minv[j];j1=j;}
    }
    if(delta>=INF/2)return INF;
    for(int j=0;j<=m;++j)if(seen[j]){u[p[j]]+=delta;v[j]-=delta;}else minv[j]-=delta;
    j0=j1;
   }while(p[j0]!=0);
   do{int j1=way[j0];p[j0]=p[j1];j0=j1;}while(j0);
  }
  match.resize(m);int total=0;
  for(int j=1;j<=m;++j){match[p[j]-1]=j-1;int value=cost[(p[j]-1)*m+j-1];if(value>=INF/2)return INF;total+=value;}
  return total;
 }
 static int forced_paths(const Perm&cost,int m,const Perm&match,const Perm&u,const Perm&v,
                         Perm&dist,Perm&component,bool zero_only=false){
  dist.assign(m*m,INF);
  for(int i=0;i<m;++i)for(int j=0;j<m;++j)if(cost[i*m+match[j]]<INF/2)
   dist[i*m+j]=cost[i*m+match[j]]-u[i+1]-v[match[j]+1];
  // Contract strongly connected zero-reduced-cost components. Movement
  // within each component is free, so shortest-path distances are exactly
  // the distances in this smaller weighted quotient graph.
  component.resize(m);std::iota(component.begin(),component.end(),0);int dim=m;
  if(m>4||zero_only){
   std::vector<char>seen(m,0);Perm finished;
   auto forward=[&](auto&&self,int x)->void{
    seen[x]=1;
    for(int y=0;y<m;++y)if(dist[x*m+y]==0&&!seen[y])self(self,y);
    finished.push_back(x);
   };
   for(int x=0;x<m;++x)if(!seen[x])forward(forward,x);
   std::fill(seen.begin(),seen.end(),0);dim=0;
   auto reverse=[&](auto&&self,int x)->void{
    seen[x]=1;component[x]=dim;
    for(int y=0;y<m;++y)if(dist[y*m+x]==0&&!seen[y])self(self,y);
   };
   for(auto it=finished.rbegin();it!=finished.rend();++it)if(!seen[*it]){
    reverse(reverse,*it);++dim;
   }
   if(zero_only){
    // At zero slack a forced edge is feasible precisely when its reduced
    // cost is zero and its endpoints share a zero-cost SCC. Nonnegative
    // reduced costs make every other alternating cycle strictly positive.
    dist.assign(dim*dim,INF);
    for(int i=0;i<dim;++i)dist[i*dim+i]=0;
    return dim;
   }
   if(dim<m){
    std::vector<int>quotient(dim*dim,INF);
    for(int i=0;i<m;++i)for(int j=0;j<m;++j){
     int pos=component[i]*dim+component[j];
     quotient[pos]=std::min(quotient[pos],dist[i*m+j]);
    }
    dist.swap(quotient);
   }else std::iota(component.begin(),component.end(),0);
  }
  for(int k=0;k<dim;++k)for(int i=0;i<dim;++i){
   // A finite alternating path can exceed INF/2 in the diagnostic domain.
   int first=dist[i*dim+k];if(first>=INF)continue;
   for(int j=0;j<dim;++j)dist[i*dim+j]=std::min(dist[i*dim+j],first+dist[k*dim+j]);
  }
  return dim;
 }
 static uint64_t shard_mix(uint64_t x){
  x^=x>>30;x*=0xbf58476d1ce4e5b9ULL;x^=x>>27;x*=0x94d049bb133111ebULL;return x^(x>>31);
 }
 struct AssignmentSeed {Perm columns,images;};
 void visit(int depth,int committed,std::shared_ptr<Group> generators,std::shared_ptr<Group> reactant_generators,
            int branch_level=0,uint64_t branch_hash=0,int request=0,const AssignmentSeed*parent_seed=nullptr){
  if(status)return;
  if(shards>1&&branch_level==SHARD_BRANCHES&&shard_mix(branch_hash)%shards!=static_cast<unsigned>(shard_index))return;
  bool inside=batch_mode?(request<0||requests[request].terminal):depth>=static_cast<int>(prefix.size());
  if(batch_mode&&inside)request=-1;
  if(slice_nodes>0&&inside){
   if(work_nodes==0)slice_started=Clock::now();
   // Leaf classification cost varies widely; a node budget alone can strand
   // most workers behind a long final batch. Yield only at subtree boundaries,
   // preserving the same disjoint complete frontier, with guaranteed progress.
   bool wall_slice=balance_slice&&work_nodes>=64&&
    std::chrono::duration<double>(Clock::now()-slice_started).count()>=0.05;
   if(work_nodes>=slice_nodes||wall_slice){frontier.push_back(path);return;}
   ++work_nodes;
  }
  ++nodes;
  if(inside)++new_nodes;else ++replay_nodes;
  if(expired()){status=1;return;}
  if(committed>target){++pruned;return;}
  if(depth==n){
   if(shards>1&&branch_level<SHARD_BRANCHES&&shard_mix(branch_hash)%shards!=static_cast<unsigned>(shard_index))return;
   if(!inside)return;
   ++leaves;
   if(committed==target){
    ++accepted;
    if(callback(mapping.data(),userdata)){status=3;return;}
    if(cap>0&&accepted>=cap)status=2;
   }
   return;
  }
  int m=n-depth,atom=row_order[depth];Perm columns;columns.reserve(m);for(int j=0;j<n;++j)if(!used[j])columns.push_back(j);
  Perm candidates;std::array<int,256> forced_images;int chosen=-1;bool all_forced=true;
  // A singleton hard domain is forced in every feasible continuation.
  // Assigning it without a relaxation only defers pruning, never removes a
  // feasible completion. Apply the usual two-sided domain updates below.
  for(int i=0;i<m;++i){
   int x=row_order[depth+i],count=0,last=-1;
   for(int y:columns)if(ca[x]==cb[y]&&y>row_min[x]&&permitted[x*n+y]
                       &&committed+cross[x*n+y]<=target){
    ++count;last=y;
    // This guard only distinguishes empty, singleton, and nonsingleton
    // domains. Two witnesses settle the last case; exact LAP domains below
    // still inspect every allowed edge before any pruning or branching.
    if(count==2)break;
   }
   if(count==0){++pruned;return;}
   if(count==1){forced_images[i]=last;if(chosen<0){chosen=i;candidates.push_back(last);}}
   else all_forced=false;
  }
  auto close_forced=[&]()->bool{
   std::vector<char> images_seen(n,0);bool fixed=true;
   for(int i=0;i<m;++i){
    int x=row_order[depth+i],y=forced_images[i];
    if(images_seen[y]){++pruned;return true;}images_seen[y]=1;
    for(const auto&g:*reactant_generators)if(g[x]!=x){fixed=false;break;}
    for(const auto&g:*generators)if(g[y]!=y){fixed=false;break;}
   }
   if(fixed){
    // There is one bijective hard-domain completion and every future
    // stabilizer/domain update is inert. Verify its exact cost directly.
    int exact=committed;
    for(int i=0;i<m;++i){
     int x=row_order[depth+i],y=forced_images[i];exact+=cross[x*n+y];
     for(int j=0;j<i;++j)
      exact+=2*std::abs(a[x*n+row_order[depth+j]]-b[y*n+forced_images[j]]);
    }
    if(exact!=target){++pruned;return true;}
    int child_request=request;
    for(int i=0;i<m;++i){
     int y=forced_images[i];
     if(batch_mode&&child_request>=0){
      if(requests[child_request].terminal)child_request=-1;
      else{
       auto found=requests[child_request].children.find(y);
       if(found==requests[child_request].children.end())return true;
       child_request=found->second;
      }
     }else if(!batch_mode&&depth+i<static_cast<int>(prefix.size())&&y!=prefix[depth+i])return true;
    }
    for(int i=0;i<m;++i){mapping[row_order[depth+i]]=forced_images[i];path.push_back(forced_images[i]);}
    visit(n,exact,generators,reactant_generators,branch_level,branch_hash,child_request);
    for(int i=0;i<m;++i){mapping[row_order[depth+i]]=-1;path.pop_back();}
    return true;
   }
   return false;
  };
  if(all_forced&&m>1&&close_forced())return;
  AssignmentSeed next_seed;
  const AssignmentSeed*child_seed=parent_seed;
  if(chosen<0){
  Perm active_types;std::vector<char> type_seen(nt,0);
  for(int i=depth;i<n;++i)if(!type_seen[ca[row_order[i]]]){
   type_seen[ca[row_order[i]]]=1;active_types.push_back(ca[row_order[i]]);
  }
  std::vector<int> cost(m*m,INF);
  int64_t row_lower=0;int remaining_budget=target-committed;
  for(int i=0;i<m;++i){
   int x=row_order[depth+i],minimum=row_min[x],row_best=INF;
   for(int j=0;j<m;++j){
    int y=columns[j];if(ca[x]!=cb[y]||y<=minimum||!permitted[x*n+y])continue;
    int value=cross[x*n+y];
    if(value>remaining_budget)continue;
    for(int t:active_types){
     for(int k=0;k<nl;++k)
      value+=(levels[k+1]-levels[k])*std::abs(ha[(x*nt+t)*nl+k]-hb[(y*nt+t)*nl+k]);
     if(value>remaining_budget)break;
    }
    if(value>remaining_budget)continue;
    cost[i*m+j]=value;row_best=std::min(row_best,value);
   }
   // Each row contributes one nonnegative assignment cost. The partial
   // sum of row minima is already a valid lower bound for the entire LAP.
   if(row_best>=INF/2||row_lower+row_best>remaining_budget){++pruned;return;}
   row_lower+=row_best;
  }
  Perm match,u,v,seed_v,seed_match;
#ifndef SYNKIT_COLD_ASSIGNMENT
  if(parent_seed&&m>=16){
   Perm indices(n,-1);for(int j=0;j<m;++j)indices[columns[j]]=j;
   for(int j:columns)seed_v.push_back(parent_seed->columns[j]);
   for(int i=0;i<m;++i){
    int image=parent_seed->images[row_order[depth+i]];
    seed_match.push_back(image<0?-1:indices[image]);
   }
  }
#endif
  int lower=assignment(cost,m,match,u,v,parent_seed?&seed_v:nullptr,parent_seed?&seed_match:nullptr);
  if(lower>=INF/2||committed+lower>target){++pruned;return;}
#ifndef SYNKIT_COLD_ASSIGNMENT
  // Below this dimension bookkeeping costs more than a fresh small LAP.
  // This is an implementation selector; both branches solve the same LAP.
  if(m>16){
   next_seed.columns.assign(n,0);next_seed.images.assign(n,-1);
   for(int j=0;j<m;++j)next_seed.columns[columns[j]]=v[j+1];
   for(int i=0;i<m;++i)next_seed.images[row_order[depth+i]]=columns[match[i]];
   child_seed=&next_seed;
  }else child_seed=nullptr;
#endif
  Perm dist,component;int dim=forced_paths(cost,m,match,u,v,dist,component,lower==remaining_budget);
  chosen=0;int best_count=INF,best_neighbors=-1;all_forced=true;
  for(int i=0;i<m;++i){
   int count=0,neighbors=0,x=row_order[depth+i];
   for(int j=0;j<m;++j){
    int edge=cost[i*m+match[j]]-u[i+1]-v[match[j]+1];
    if(committed+lower+edge+dist[component[j]*dim+component[i]]<=target){++count;forced_images[i]=columns[match[j]];}
   }
   if(count!=1)all_forced=false;
   for(int y=0;y<n;++y)if(mapping[y]>=0&&a[x*n+y]!=0)++neighbors;
   if(count<best_count||(count==best_count&&neighbors>best_neighbors)){
    chosen=i;best_count=count;best_neighbors=neighbors;
   }
  }
  // Forced-edge relaxation certificates are necessary conditions for exact
  // completion. Singleton certified domains can close the same unique leaf.
  if(all_forced&&m>1&&close_forced())return;
  for(int i=0;i<m;++i){
   int col=match[i],value=cost[chosen*m+col];
   if(value>=INF/2||dist[component[i]*dim+component[chosen]]>=INF/2)continue;
   int forced=lower+value-u[chosen+1]-v[col+1]+dist[component[i]*dim+component[chosen]];
   if(committed+forced<=target)candidates.push_back(columns[col]);
  }
  std::sort(candidates.begin(),candidates.end());
  }
  std::swap(row_order[depth],row_order[depth+chosen]);
  struct Restore {std::vector<int>&v;int a,b;~Restore(){std::swap(v[a],v[b]);}} restore{row_order,depth,depth+chosen};
  atom=row_order[depth];
  if(candidates.size()==1){
   int image=candidates[0];bool fixed=true;
   for(const auto&g:*reactant_generators)if(g[atom]!=atom){fixed=false;break;}
   if(fixed)for(const auto&g:*generators)if(g[image]!=image){fixed=false;break;}
   if(fixed){
    // Both stabilizers already fix the selected points. The row orbit is
    // just the assigned row, whose row_min/permitted entries will never be
    // read by descendants. Thus all omitted group/domain updates are inert.
    int child_request=request;
    if(batch_mode&&request>=0){
     auto found=requests[request].children.find(image);
     if(found==requests[request].children.end())return;
     child_request=found->second;
    }else if(!batch_mode&&depth<static_cast<int>(prefix.size())&&image!=prefix[depth])return;
    int delta=cross[atom*n+image];mapping[atom]=image;used[image]=1;
    update(atom,image,1,depth,columns);path.push_back(image);
    visit(depth+1,committed+delta,generators,reactant_generators,branch_level,branch_hash,child_request,child_seed);
    path.pop_back();update(atom,image,-1,depth,columns);
    used[image]=0;mapping[atom]=-1;
    return;
   }
  }
  Perm parent(n);std::iota(parent.begin(),parent.end(),0);
  auto find=[&](int x){while(parent[x]!=x){parent[x]=parent[parent[x]];x=parent[x];}return x;};
  for(const auto&g:*generators)for(int j:columns){int x=find(j),y=find(g[j]);if(x!=y)parent[y]=x;}
  Perm orbit_min(n,INF);
  for(int image:columns){int root=find(image);orbit_min[root]=std::min(orbit_min[root],image);}
  std::vector<char>seen(n,0);
  Perm row_orbit{atom};std::vector<char>row_seen(n,0);row_seen[atom]=1;
  for(size_t i=0;i<row_orbit.size();++i)for(const auto&g:*reactant_generators){
   int image=g[row_orbit[i]];if(!row_seen[image]){row_seen[image]=1;row_orbit.push_back(image);}
  }
  auto child_rows=stabilize(reactant_generators,atom,n);
  Perm choices;
  for(int image:candidates){int orbit=find(image);if(!seen[orbit]){seen[orbit]=1;choices.push_back(image);}}
  for(int image:choices){
   int child_request=request;
   if(batch_mode&&request>=0){
    auto found=requests[request].children.find(image);
    if(found==requests[request].children.end())continue;
    child_request=found->second;
   }else if(!batch_mode&&depth<static_cast<int>(prefix.size())&&image!=prefix[depth])continue;
   int delta=cross[atom*n+image];
   mapping[atom]=image;used[image]=1;
   Perm old_min;for(int row:row_orbit){old_min.push_back(row_min[row]);row_min[row]=std::max(row_min[row],image);}
   // Earlier two-sided choices exclude entire product orbits, so these
   // domains remain invariant under the current independent stabilizers.
   // A smaller orbit leader at any equivalent reactant row belongs to an
   // already enumerated branch, even when that row's numeric image is larger.
   Perm removed;
   for(int row:row_orbit)for(int col:columns)
    if(orbit_min[find(col)]<image&&permitted[row*n+col]){
     permitted[row*n+col]=0;removed.push_back(row*n+col);
    }
   update(atom,image,1,depth,columns);
   int next_level=branch_level;
   uint64_t next_hash=branch_hash;
   if(choices.size()>1&&branch_level<SHARD_BRANCHES){
    ++next_level;next_hash=shard_mix(branch_hash+static_cast<unsigned>(image+1)+0x9e3779b97f4a7c15ULL);
   }
   path.push_back(image);
   visit(depth+1,committed+delta,stabilize(generators,image,n),child_rows,next_level,next_hash,child_request,child_seed);
   path.pop_back();
   update(atom,image,-1,depth,columns);
   for(size_t i=0;i<row_orbit.size();++i)row_min[row_orbit[i]]=old_min[i];
   for(int index:removed)permitted[index]=1;
   used[image]=0;mapping[atom]=-1;
   if(status)return;
  }
 }
};
extern "C" int synkit_distance_abi(){return 3;}
extern "C" int synkit_distance_candidates(
 int n,int nt,int nl,int target,double seconds,int64_t cap,
 const int*a,const int*b,const int*ca,const int*cb,const int*order,const int*pred,
 const int*gens,int count,const int*row_gens,int row_count,int shard_index,int shards,Callback callback,void*userdata,int64_t*statistics){
 try{
  if(n<1||n>256||nt<1||nt>n||nl<0||nl>15||target<0||target>1000000||!callback||shards<1||shard_index<0||shard_index>=shards)return -2;
  Solver solver(n,nt,nl,target,seconds,cap,a,b,ca,cb,order,pred,callback,userdata);
  solver.shard_index=shard_index;solver.shards=shards;
  auto group=std::make_shared<Group>();
  for(int k=0;k<count;++k)group->emplace_back(gens+k*n,gens+(k+1)*n);
  auto rows=std::make_shared<Group>();
  for(int k=0;k<row_count;++k)rows->emplace_back(row_gens+k*n,row_gens+(k+1)*n);
  solver.visit(0,0,group,rows);
  statistics[0]=solver.nodes;statistics[1]=solver.leaves;statistics[2]=solver.accepted;statistics[3]=solver.pruned;
  return solver.status;
 }catch(...){return -1;}
}

// Additive ABI: the original entry point remains available for V8 callers.
// A normal slice emits a disjoint cover of every unvisited subtree.
using FrontierCallback=int (*)(const int*,int,void*);
extern "C" int synkit_distance_frontier(
 int n,int nt,int nl,int target,double seconds,int64_t cap,
 const int*a,const int*b,const int*ca,const int*cb,const int*order,const int*pred,
 const int*gens,int count,const int*row_gens,int row_count,int shard_index,int shards,
 Callback callback,void*userdata,int64_t*statistics,
 const int*prefix,int prefix_size,int64_t slice_nodes,FrontierCallback emit){
 try{
  if(n<1||n>256||nt<1||nt>n||nl<0||nl>15||target<0||target>1000000||
     !callback||!emit||shards!=1||shard_index!=0||prefix_size<0||prefix_size>n||slice_nodes<1)return -2;
  Solver solver(n,nt,nl,target,seconds,cap,a,b,ca,cb,order,pred,callback,userdata);
  solver.prefix.assign(prefix,prefix+prefix_size);solver.slice_nodes=slice_nodes;
  auto group=std::make_shared<Group>();
  for(int k=0;k<count;++k)group->emplace_back(gens+k*n,gens+(k+1)*n);
  auto rows=std::make_shared<Group>();
  for(int k=0;k<row_count;++k)rows->emplace_back(row_gens+k*n,row_gens+(k+1)*n);
  solver.visit(0,0,group,rows);
  statistics[0]=solver.nodes;statistics[1]=solver.leaves;
  statistics[2]=solver.accepted;statistics[3]=solver.pruned;
  for(const auto&job:solver.frontier)if(emit(job.data(),job.size(),userdata))return 3;
  return solver.status;
 }catch(...){return -1;}
}

#include <map>

// Replay the trie of disjoint requested prefixes once. Below a terminal,
// normal DFS resumes; the emitted frontier remains a disjoint exact cover.
extern "C" int synkit_distance_frontier_batch(
 int n,int nt,int nl,int target,double seconds,int64_t cap,
 const int*a,const int*b,const int*ca,const int*cb,const int*order,const int*pred,
 const int*gens,int count,const int*row_gens,int row_count,int shard_index,int shards,
 Callback callback,void*userdata,int64_t*statistics,
 const int*values,const int*offsets,int prefix_count,int64_t slice_nodes,FrontierCallback emit){
 try{
  if(n<1||n>256||nt<1||nt>n||nl<0||nl>15||target<0||target>1000000||
     !callback||!emit||shards!=1||shard_index!=0||prefix_count<1||prefix_count>16384||
     offsets[0]!=0||slice_nodes<1)return -2;
  Solver solver(n,nt,nl,target,seconds,cap,a,b,ca,cb,order,pred,callback,userdata);
  solver.batch_mode=true;solver.balance_slice=prefix_count<=4;
  solver.requests.emplace_back();solver.slice_nodes=slice_nodes;
  for(int k=0;k<prefix_count;++k){
   int begin=offsets[k],end=offsets[k+1],node=0;
   if(begin<0||end<begin||end-begin>n)return -2;
   for(int j=begin;j<end;++j){
    if(solver.requests[node].terminal||values[j]<0||values[j]>=n)return -2;
    auto found=solver.requests[node].children.find(values[j]);
    if(found==solver.requests[node].children.end()){
     int next=solver.requests.size();
     solver.requests[node].children[values[j]]=next;
     solver.requests.emplace_back();node=next;
    }else node=found->second;
   }
   if(solver.requests[node].terminal||!solver.requests[node].children.empty())return -2;
   solver.requests[node].terminal=true;
  }
  auto group=std::make_shared<Group>();
  for(int k=0;k<count;++k)group->emplace_back(gens+k*n,gens+(k+1)*n);
  auto rows=std::make_shared<Group>();
  for(int k=0;k<row_count;++k)rows->emplace_back(row_gens+k*n,row_gens+(k+1)*n);
  solver.visit(0,0,group,rows);
  statistics[0]=solver.nodes;statistics[1]=solver.leaves;
  statistics[2]=solver.accepted;statistics[3]=solver.pruned;
  statistics[4]=solver.replay_nodes;statistics[5]=solver.new_nodes;
  for(const auto&job:solver.frontier)if(emit(job.data(),job.size(),userdata))return 3;
  return solver.status;
 }catch(...){return -1;}
}

#include <limits>
struct NativeCanon {
 int n;const int*colors;const int*edges;double seconds;int64_t max_nodes;
 Clock::time_point start=Clock::now();int64_t nodes=0;bool incomplete=false,root_twins_only=false;
 Group generators;std::unordered_set<Perm,VectorHash> signatures;
 Perm best,best_key,twin_class;
 struct NeighborRows {
  Perm offsets;std::vector<std::pair<int,int>> entries;
  explicit NeighborRows(int n):offsets(n+1){entries.reserve(4*n);}
  struct Range {
   const std::pair<int,int>*first,*last;
   auto begin()const{return first;}auto end()const{return last;}
   size_t size()const{return last-first;}
  };
  Range operator[](int i)const{return {entries.data()+offsets[i],entries.data()+offsets[i+1]};}
 };
 NeighborRows neighbors;
 NativeCanon(int nn,const int*c,const int*e,double sec,int64_t budget):n(nn),colors(c),edges(e),seconds(sec),max_nodes(budget),twin_class(nn),neighbors(nn){
  std::iota(twin_class.begin(),twin_class.end(),0);
  for(int i=0;i<n;++i){
   for(int j=0;j<n;++j){
    if(edges[i*n+j]<0)throw std::runtime_error("negative edge palette ID");
    if(edges[i*n+j])neighbors.entries.push_back({j,edges[i*n+j]});
   }
   neighbors.offsets[i+1]=neighbors.entries.size();
  }
 }
 bool valid(const Perm&g){
  // A bijection preserving every nonzero colored edge also preserves absent
  // edges: its injection on the finite colored edge sets is a bijection.
  std::vector<char> seen(n,0);
  for(int i=0;i<n;++i){
   if(g[i]<0||g[i]>=n||seen[g[i]])return false;
   seen[g[i]]=1;
   if(colors[i]!=colors[g[i]])return false;
  }
  for(int i=0;i<n;++i)for(const auto&entry:neighbors[i])
   if(entry.second!=edges[g[i]*n+g[entry.first]])return false;
  return true;
 }
 void accept(const Perm&g){
  bool identity=true;for(int i=0;i<n;++i)if(g[i]!=i){identity=false;break;}
  if(!identity&&signatures.insert(g).second&&valid(g))generators.push_back(g);
 }
 using Partition=std::vector<Perm>;
 // Positive edge IDs allow 0 to terminate a cell's (color,count) list.
 // This flat encoding has exactly the lexicographic order of the legacy
 // vector<pair<-cell, vector<pair<color,count>>>> signature.
 using Signature=Perm;
 Partition refine(Partition p){
  std::vector<std::pair<int,int>> entries;entries.reserve(n);
  Signature sig;sig.reserve(4*n);
  while(true){
   Perm ci(n);for(size_t k=0;k<p.size();++k)for(int x:p[k])ci[x]=k;
   Partition next;next.reserve(n);bool split=false;
   for(auto&cell:p){
    // Singleton cells cannot split; their position and contribution to ci
    // remain unchanged, so no refinement signature is needed for them.
    if(cell.size()==1){next.push_back(std::move(cell));continue;}
    std::map<Signature,Perm> buckets;
    for(int x:cell){
     entries.clear();
     for(const auto&entry:neighbors[x])entries.push_back({ci[entry.first],entry.second});
     std::sort(entries.begin(),entries.end());
     sig.clear();
     for(size_t k=0;k<entries.size();){
      int cell_id=entries[k].first;sig.push_back(-cell_id);
      while(k<entries.size()&&entries[k].first==cell_id){
       int color=entries[k].second,count=0;
       do{++count;++k;}while(k<entries.size()&&entries[k]==std::make_pair(cell_id,color));
       sig.push_back(color);sig.push_back(count);
      }
      sig.push_back(0);
     }
     buckets.try_emplace(sig).first->second.push_back(x);
    }
    split|=buckets.size()>1;
    for(auto&entry:buckets)next.push_back(std::move(entry.second));
   }
   p=std::move(next);if(!split)return p;
  }
 }
 void visit(Partition p,bool already_refined=false){
  if(incomplete)return;
  if(nodes>=max_nodes||std::chrono::duration<double>(Clock::now()-start).count()>=seconds){incomplete=true;return;}
  ++nodes;if(!already_refined)p=refine(std::move(p));int target=-1;
  while(true){
   target=-1;
   for(size_t k=0;k<p.size();++k)if(p[k].size()>1&&(target<0||p[k].size()<p[target].size()))target=k;
   if(target<0)break;
   const auto&cell=p[target];bool twins=true;
   for(int x:cell)if(twin_class[x]!=twin_class[cell[0]]){twins=false;break;}
   if(!twins)break;
   // Exact equal-row twins are freely interchangeable (all transpositions
   // are already seeded). In an equitable partition, individualizing such
   // twins cannot split any other cell. This is the same first branch as
   // repeated legacy individualization, with identical final cell ordering.
   Partition next;
   for(size_t k=0;k<p.size();++k){
    if(static_cast<int>(k)==target){for(int x:p[k])next.push_back(Perm{x});}
    else next.push_back(std::move(p[k]));
   }
   p=std::move(next);
  }
  if(target<0){
   Perm order,key;order.reserve(n);key.reserve(n+4*n);for(const auto&cell:p)order.push_back(cell[0]);
   for(int x:order)key.push_back(colors[x]);
   // Sparse certificate has exactly dense lexicographic order: an earlier
   // nonzero entry compares larger than an implicit zero, hence -position.
   // Node colors have a fixed n-entry prefix; all present edge IDs are positive.
   // Enumerate the same upper-triangle certificate through present edges.
   // Sorting each row by canonical column preserves the exact dense order,
   // including loops, without scanning every absent edge at every leaf.
   Perm inverse(n);for(int i=0;i<n;++i)inverse[order[i]]=i;
   std::vector<std::pair<int,int>> row;row.reserve(n);
   for(int i=0;i<n;++i){
    row.clear();
    for(const auto&entry:neighbors[order[i]]){
     int j=inverse[entry.first];if(j>=i)row.push_back({j,entry.second});
    }
    std::sort(row.begin(),row.end());
    for(const auto&entry:row){key.push_back(-(i*n+entry.first));key.push_back(entry.second);}
   }
   if(best.empty()||key<best_key){best=std::move(order);best_key=std::move(key);}
   else if(key==best_key){Perm g(n);for(int i=0;i<n;++i)g[best[i]]=order[i];accept(g);}
   return;
  }
  Perm ci(n);for(size_t k=0;k<p.size();++k)for(int x:p[k])ci[x]=k;
  Perm explored,parent(n);std::iota(parent.begin(),parent.end(),0);
  auto find=[&](int x){while(parent[x]!=x){parent[x]=parent[parent[x]];x=parent[x];}return x;};
  size_t processed=0;
  for(int chosen:p[target]){
   // Generators only append during this search. Keep each node's orbit union
   // and incorporate newly discovered witnesses after returning from a child.
   if(!explored.empty())while(processed<generators.size()){
    const auto&g=generators[processed++];
    bool fixed=true;for(int i=0;i<n;++i)if(ci[i]!=ci[g[i]]){fixed=false;break;}
    if(fixed)for(int i:p[target]){int x=find(i),y=find(g[i]);if(x!=y)parent[y]=x;}
   }
   bool seen=false;for(int old:explored)if(find(old)==find(chosen)){seen=true;break;}
   if(seen)continue;
   Partition child=p;Perm rest;for(int x:p[target])if(x!=chosen)rest.push_back(x);
   child[target]=Perm{chosen};child.insert(child.begin()+target+1,rest);
   visit(std::move(child));if(incomplete)return;explored.push_back(chosen);
  }
 }
 bool group_order(uint64_t&answer){
  answer=1;
  if(root_twins_only){
   // The invariant root partition separates all non-twin vertices. Its
   // automorphism group is exactly the direct product of the twin symmetric
   // groups; no stabilizer-chain construction is needed.
   Perm counts(n,0);for(int id:twin_class)++counts[id];
   for(int count:counts)for(int k=2;k<=count;++k){
    if(answer>std::numeric_limits<uint64_t>::max()/k)return false;
    answer*=k;
   }
   return true;
  }
  // Every generator is supported in exactly one component of this union.
  // Thus the generated group is a direct product on disjoint supports.
  Perm parent(n);std::iota(parent.begin(),parent.end(),0);
  auto find=[&](int x){while(parent[x]!=x){parent[x]=parent[parent[x]];x=parent[x];}return x;};
  for(const auto&g:generators){
   int first=-1;
   for(int i=0;i<n;++i)if(g[i]!=i){
    if(first<0)first=i;else parent[find(i)]=find(first);
   }
  }
  std::map<int,Perm> components;
  for(int i=0;i<n;++i)components[find(i)].push_back(i);
  Perm index(n);
  for(const auto&entry:components){
   const auto&points=entry.second;int size=points.size();
   if(size==1)continue;
   // Adjacent transpositions of every exact equal-row twin class were
   // seeded before search. A support component contained in one such class
   // therefore has the full symmetric group: m!, with no Schreier work.
   // Components joining different twin classes retain the exact fallback.
   bool twins=true;
   for(int x:points)if(twin_class[x]!=twin_class[points[0]]){twins=false;break;}
   if(twins){
    for(int k=2;k<=size;++k){
     if(answer>std::numeric_limits<uint64_t>::max()/k)return false;
     answer*=k;
    }
    continue;
   }
   for(int i=0;i<size;++i)index[points[i]]=i;
   Group restricted;
   for(const auto&g:generators){
    bool moved=false;for(int x:points)if(g[x]!=x){moved=true;break;}
    if(!moved)continue;
    Perm small(size);for(int i=0;i<size;++i)small[i]=index[g[points[i]]];
    restricted.push_back(std::move(small));
   }
   auto group=std::make_shared<Group>(std::move(restricted));
   for(int base=0;base<size&&!group->empty();++base){
    bool moved=false;for(const auto&g:*group)if(g[base]!=base){moved=true;break;}
    if(!moved)continue;
    Perm orbit{base};std::vector<char>seen(size,0);seen[base]=1;
    for(size_t k=0;k<orbit.size();++k)for(const auto&g:*group)if(!seen[g[orbit[k]]]){
     seen[g[orbit[k]]]=1;orbit.push_back(g[orbit[k]]);
    }
    if(answer>std::numeric_limits<uint64_t>::max()/orbit.size())return false;
    answer*=orbit.size();
    bool complete=true;
    group=stabilize(group,base,size,&complete);
    if(!complete)return false;
   }
  }
  return true;
 }
};
static thread_local bool profile_canonical=false;
static thread_local int64_t canonical_times[8]={};
extern "C" void synkit_canonical_profile_enable(int enabled){
 profile_canonical=enabled!=0;std::fill(canonical_times,canonical_times+8,0);
}
extern "C" void synkit_canonical_profile_read(int64_t*out){
 std::copy(canonical_times,canonical_times+8,out);
}
static int canonical_undirected_impl(
 int n,const int*colors,const int*edges,const int*seeds,int seed_count,
 double seconds,int64_t max_nodes,int*order,uint64_t*group_order,int64_t*nodes,bool require_group){
 try{
  if(n<1||n>256||seconds<0||max_nodes<1)return -2;
  auto profile_start=profile_canonical?Clock::now():Clock::time_point{};
  NativeCanon canon(n,colors,edges,seconds,max_nodes);
  // Nonadjacent twin transpositions are exact and inexpensive initial witnesses.
  std::map<Perm,Perm> twins;
  for(int i=0;i<n;++i){
   Perm key{colors[i]};key.reserve(1+2*canon.neighbors[i].size());
   for(const auto&entry:canon.neighbors[i]){key.push_back(entry.first);key.push_back(entry.second);}
   twins.try_emplace(std::move(key)).first->second.push_back(i);
  }
  for(const auto&entry:twins)
   for(int x:entry.second)canon.twin_class[x]=entry.second[0];
  NativeCanon::Partition partition;std::map<int,Perm>buckets;
  for(int i=0;i<n;++i)buckets[colors[i]].push_back(i);
  for(const auto&entry:buckets)partition.push_back(entry.second);
  partition=canon.refine(std::move(partition));
  canon.root_twins_only=true;
  for(const auto&cell:partition)for(int x:cell)
   if(canon.twin_class[x]!=canon.twin_class[cell[0]])canon.root_twins_only=false;
  if(!canon.root_twins_only){
   for(int k=0;k<seed_count;++k)canon.accept(Perm(seeds+k*n,seeds+(k+1)*n));
   for(const auto&entry:twins)for(size_t k=1;k<entry.second.size();++k){
    Perm g(n);std::iota(g.begin(),g.end(),0);std::swap(g[entry.second[k-1]],g[entry.second[k]]);canon.accept(g);
   }
  }
  auto profile_ready=profile_canonical?Clock::now():Clock::time_point{};
  canon.visit(partition,true);*nodes=canon.nodes;
  auto profile_searched=profile_canonical?Clock::now():Clock::time_point{};
  if(canon.incomplete)return 1;
  std::copy(canon.best.begin(),canon.best.end(),order);
  *group_order=0;
  if(require_group&&!canon.group_order(*group_order))return 2;
  if(profile_canonical){
   int offset=require_group?0:4;
   canonical_times[offset]+=std::chrono::duration_cast<std::chrono::nanoseconds>(profile_ready-profile_start).count();
   canonical_times[offset+1]+=std::chrono::duration_cast<std::chrono::nanoseconds>(profile_searched-profile_ready).count();
   canonical_times[offset+2]+=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-profile_searched).count();
   ++canonical_times[offset+3];
  }
  return 0;
 }catch(...){return -1;}
}


// Additive export: old callers still receive a complete group-order proof.
extern "C" int synkit_canonical_undirected(
 int n,const int*colors,const int*edges,const int*seeds,int seed_count,
 double seconds,int64_t max_nodes,int*order,uint64_t*group_order,int64_t*nodes){
 return canonical_undirected_impl(n,colors,edges,seeds,seed_count,seconds,max_nodes,order,group_order,nodes,true);
}
extern "C" int synkit_canonical_undirected_certificate(
 int n,const int*colors,const int*edges,const int*seeds,int seed_count,
 double seconds,int64_t max_nodes,int*order,uint64_t*group_order,int64_t*nodes){
 return canonical_undirected_impl(n,colors,edges,seeds,seed_count,seconds,max_nodes,order,group_order,nodes,false);
}


// Controlled ITS matrices have zero diagonal. Emit upper-triangle edges in
// canonical order directly, avoiding per-edge Python relabeling and sorting.
extern "C" int synkit_its_sparse_edges(int n,const int*order,const int*edges,int*out){
 if(n<1||n>256)return -1;
 std::vector<char> seen(n,0);
 for(int i=0;i<n;++i){
  if(order[i]<0||order[i]>=n||seen[order[i]])return -1;
  seen[order[i]]=1;
 }
 int count=0;
 for(int i=0;i<n;++i)for(int j=i+1;j<n;++j){
  int color=edges[order[i]*n+order[j]];
  if(color){out[3*count]=i;out[3*count+1]=j;out[3*count+2]=color;++count;}
 }
 return count;
}


#include <bitset>
// A bounded cache of exact ITS patterns and explicitly witnessed R-images.
// Cache misses always fall back to complete canonicalization.
struct PatternCache {
 int n,work_limit;
 Perm base_nodes,base_edges,pending;
 Group generators;
 std::vector<std::bitset<256>> supports;
 std::unordered_set<Perm,VectorHash> known;
 size_t payload_ints=0;
 static constexpr size_t MAX_PATTERNS=1000000,MAX_INTS=16*1024*1024;
 PatternCache(int nn,const int*nodes,const int*edges,const int*gens,int count,int work):
  n(nn),work_limit(work),base_nodes(nodes,nodes+nn),base_edges(edges,edges+nn*nn){
  for(int k=0;k<count;++k){
   Perm g(gens+k*n,gens+(k+1)*n);std::vector<char>seen(n,0);bool valid=true;
   std::bitset<256> support;
   for(int i=0;i<n;++i){
    if(g[i]<0||g[i]>=n||seen[g[i]]||base_nodes[i]!=base_nodes[g[i]]){valid=false;break;}
    seen[g[i]]=1;if(g[i]!=i)support.set(i);
   }
   if(valid)for(int i=0;i<n&&valid;++i)for(int j=0;j<n;++j)
    if(base_edges[i*n+j]!=base_edges[g[i]*n+g[j]]){valid=false;break;}
   if(valid&&!support.none()){generators.push_back(std::move(g));supports.push_back(support);}
  }
 }
 bool lookup(const int*nodes,const int*edges){
  pending.clear();
  for(int i=0;i<n;++i)for(int j=i+1;j<n;++j)if(edges[i*n+j]!=base_edges[i*n+j]){
   pending.push_back(i*n+j);pending.push_back(edges[i*n+j]);
  }
  for(int i=0;i<n;++i)if(nodes[i]!=base_nodes[i]){
   pending.push_back(n*n+i);pending.push_back(nodes[i]);
  }
  return known.find(pending)!=known.end();
 }
 void remember(){
  if(pending.size()>MAX_INTS)return;
  if(known.size()+work_limit+1>MAX_PATTERNS||payload_ints+(work_limit+1)*pending.size()>MAX_INTS){
   known.clear();payload_ints=0;
  }
  std::vector<Perm> queue{pending};
  if(known.insert(pending).second)payload_ints+=pending.size();
  int work=0;
  int limit=std::min<size_t>(work_limit,(MAX_INTS-pending.size())/std::max<size_t>(1,pending.size()));
  for(size_t q=0;q<queue.size()&&work<limit;++q){
   const Perm current=queue[q];std::bitset<256> support;
   for(size_t j=0;j<current.size();j+=2){
    int pos=current[j];
    if(pos>=n*n)support.set(pos-n*n);
    else{support.set(pos/n);support.set(pos%n);}
   }
   for(size_t gi=0;gi<generators.size()&&work<limit;++gi){
    if((support&supports[gi]).none())continue;
    ++work;const auto&g=generators[gi];
    std::vector<std::pair<int,int>> entries;entries.reserve(current.size()/2);
    for(size_t j=0;j<current.size();j+=2){
     int pos=current[j],image;
     if(pos>=n*n)image=n*n+g[pos-n*n];
     else{int a=g[pos/n],b=g[pos%n];image=std::min(a,b)*n+std::max(a,b);}
     entries.push_back({image,current[j+1]});
    }
    std::sort(entries.begin(),entries.end());Perm next;next.reserve(current.size());
    for(const auto&entry:entries){next.push_back(entry.first);next.push_back(entry.second);}
    if(known.insert(next).second){payload_ints+=next.size();queue.push_back(std::move(next));}
   }
  }
 }
};
extern "C" void* synkit_pattern_cache_create(int n,const int*nodes,const int*edges,
                                            const int*gens,int count,int work){
 try{
  if(n<1||n>256||count<0||work<0||work>1024)return nullptr;
  return new PatternCache(n,nodes,edges,gens,count,work);
 }catch(...){return nullptr;}
}
extern "C" int synkit_pattern_cache_lookup(void*handle,const int*nodes,const int*edges){
 try{return static_cast<PatternCache*>(handle)->lookup(nodes,edges)?1:0;}catch(...){return -1;}
}
extern "C" int synkit_pattern_cache_remember(void*handle){
 try{static_cast<PatternCache*>(handle)->remember();return 0;}catch(...){return -1;}
}
extern "C" void synkit_pattern_cache_destroy(void*handle){delete static_cast<PatternCache*>(handle);}


// Inputs are precomputed with the public Python equality/isclose semantics.
// Output: n unary flags, n context flags, then changed upper-triangle pairs.
extern "C" int synkit_its_changes(int n,const int*mapping,const int*unary,
 const int*paired,const int*changed,int radius,int*out){
 if(n<1||n>256||radius<0)return -1;
 std::vector<int> frontier;
 for(int i=0;i<n;++i){out[i]=unary[i*n+mapping[i]];out[n+i]=out[i];}
 int count=0;
 for(int i=0;i<n;++i)for(int j=i+1;j<n;++j)if(changed[paired[i*n+j]]){
  out[2*n+2*count]=i;out[2*n+2*count+1]=j;++count;
  out[n+i]=out[n+j]=1;
 }
 for(int i=0;i<n;++i)if(out[n+i])frontier.push_back(i);
 for(int depth=0;depth<radius&&!frontier.empty();++depth){
  std::vector<int> next;
  for(int i:frontier)for(int j=0;j<n;++j)if(paired[i*n+j]&&!out[n+j]){
   out[n+j]=1;next.push_back(j);
  }
  frontier=std::move(next);
 }
 return count;
}

// Ordered resource IDs have exactly the order of (typed element, A bond, B bond).
// Each selected vertex receives its sorted external-resource multiset.
extern "C" int synkit_its_boundary(int n,const int*selected,int count,const int*paired,
 int palette_size,const int*resources,int*offsets,int*out){
 if(n<1||n>256||count<0||count>n||palette_size<1)return -1;
 std::vector<char>inside(n,0);
 for(int k=0;k<count;++k){
  int i=selected[k];if(i<0||i>=n||inside[i])return -1;inside[i]=1;
 }
 int size=0;offsets[0]=0;
 for(int k=0;k<count;++k){
  int start=size,i=selected[k];
  for(int j=0;j<n;++j)if(!inside[j]&&paired[i*n+j])
   out[size++]=resources[j*palette_size+paired[i*n+j]];
  std::sort(out+start,out+size);offsets[k+1]=size;
 }
 return size;
}


// Diagnostic oracle surface: exact assignment duals and each forced-edge optimum.
// Costs are bounded nonnegative integers; INF denotes a forbidden edge.
extern "C" int synkit_assignment_certificate(int m,const int*input,int*match_out,
 int*u_out,int*v_out,int*forced_out){
 try{
  if(m<1||m>64)return -2;
  Perm cost(input,input+m*m),match,u,v,dist,component;
  for(int value:cost)if(value<0||(value>1000000&&value!=INF))return -2;
  int lower=Solver::assignment(cost,m,match,u,v);
  if(lower>=INF)return INF;
  int dim=Solver::forced_paths(cost,m,match,u,v,dist,component);
  std::copy(match.begin(),match.end(),match_out);
  std::copy(u.begin()+1,u.end(),u_out);std::copy(v.begin()+1,v.end(),v_out);
  for(int i=0;i<m;++i)for(int j=0;j<m;++j){
   int column=match[j],path=dist[component[j]*dim+component[i]];
   forced_out[i*m+column]=(cost[i*m+column]>=INF/2||path>=INF)?INF:
    lower+cost[i*m+column]-u[i+1]-v[column+1]+path;
  }
  return lower;
 }catch(...){return -1;}
}

// Diagnostic: arbitrary stale hints must produce independently valid duals.
extern "C" int synkit_assignment_seed_certificate(int m,const int*input,const int*initial_v,
 const int*initial_match,int*match_out,
 int*u_out,int*v_out,int*forced_out){
 try{
  if(m<1||m>64||!initial_v||!initial_match)return -2;
  Perm cost(input,input+m*m),match,u,v,dist,component;
  for(int value:cost)if(value<0||(value>1000000&&value!=INF))return -2;
  Perm seed_v(initial_v,initial_v+m),seed_match(initial_match,initial_match+m);
  int lower=Solver::assignment(cost,m,match,u,v,&seed_v,&seed_match);
  if(lower>=INF)return INF;
  int dim=Solver::forced_paths(cost,m,match,u,v,dist,component);
  std::copy(match.begin(),match.end(),match_out);
  std::copy(u.begin()+1,u.end(),u_out);std::copy(v.begin()+1,v.end(),v_out);
  for(int i=0;i<m;++i)for(int j=0;j<m;++j){
   int column=match[j],path=dist[component[j]*dim+component[i]];
   forced_out[i*m+column]=(cost[i*m+column]>=INF/2||path>=INF)?INF:
    lower+cost[i*m+column]-u[i+1]-v[column+1]+path;
  }
  return lower;
 }catch(...){return -1;}
}


extern "C" int synkit_its_pair(int n,const int*mapping,const int*a,const int*b,
 int levels,const int*pair_ids,int*out){
 if(n<1||n>256||levels<1)return -1;
 for(int i=0;i<n;++i)if(mapping[i]<0||mapping[i]>=n)return -1;
 for(int i=0;i<n;++i)for(int j=0;j<n;++j)
  out[i*n+j]=(i==j)?0:pair_ids[a[i*n+j]*levels+b[mapping[i]*n+mapping[j]]];
 return 0;
}
