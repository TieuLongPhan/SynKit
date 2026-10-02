// Exact integer-distance candidate enumeration for two-sided symmetry analysis.
// ABI 1: all matrices and indices are validated by the Python wrapper.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
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
struct VectorHash {
 std::size_t operator()(const Perm& p) const noexcept {
  std::size_t h=0; for(int x:p) h=h*1315423911u+static_cast<unsigned>(x+1); return h;
 }
};
static Perm compose(const Perm&a,const Perm&b){
 Perm p(a.size());for(size_t i=0;i<a.size();++i)p[i]=b[a[i]];return p;
}
static std::shared_ptr<Group> stabilize(std::shared_ptr<Group> gs,int point,int n){
 bool fixed=true;for(const auto&g:*gs)if(g[point]!=point){fixed=false;break;}
 if(fixed)return gs;
 Perm identity(n);std::iota(identity.begin(),identity.end(),0);
 std::vector<Perm> trans(n),inv(n);std::vector<int> orbit{point};trans[point]=identity;
 int work=0;
 for(size_t k=0;k<orbit.size();++k)for(const auto&g:*gs){
  if(++work>100000)return std::make_shared<Group>();
  int x=orbit[k],y=g[x];
  if(trans[y].empty()){trans[y]=compose(trans[x],g);orbit.push_back(y);}
 }
 for(int x:orbit){inv[x].resize(n);for(int i=0;i<n;++i)inv[x][trans[x][i]]=i;}
 auto out=std::make_shared<Group>();std::unordered_set<Perm,VectorHash> seen;
 for(int x:orbit)for(const auto&g:*gs){
  Perm h=compose(compose(trans[x],g),inv[g[x]]);
  if(h!=identity&&seen.insert(h).second){
   out->push_back(std::move(h));if(out->size()>512)return std::make_shared<Group>();
  }
 }
 return out;
}
struct Solver {
 int n,nt,nl,target;double seconds;int64_t cap;
 const int *a,*b,*ca,*cb,*order,*pred;
 std::vector<int> mapping,used,cross,ha,hb,levels;
 Callback callback;void*userdata;Clock::time_point started;
 int status=0;int64_t nodes=0,leaves=0,accepted=0,pruned=0;
 Solver(int nn,int types,int lev,int tar,double sec,int64_t limit,
        const int*aa,const int*bb,const int*cca,const int*ccb,const int*ord,
        const int*pre,Callback call,void*data):
  n(nn),nt(types),nl(lev),target(tar),seconds(sec),cap(limit),
  a(aa),b(bb),ca(cca),cb(ccb),order(ord),pred(pre),
  mapping(n,-1),used(n,0),cross(n*n,0),ha(n*nt*nl),hb(n*nt*nl),
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
 void update(int atom,int image,int sign){
  for(int i=0;i<n;++i){
   for(int k=0;k<nl;++k){
    if(a[i*n+atom]>=levels[k+1])ha[(i*nt+ca[atom])*nl+k]-=sign;
    if(b[i*n+image]>=levels[k+1])hb[(i*nt+cb[image])*nl+k]-=sign;
   }
   for(int j=0;j<n;++j)cross[i*n+j]+=sign*2*std::abs(a[i*n+atom]-b[j*n+image]);
  }
 }
 // Dense Hungarian assignment with integer duals. Infeasibility fails closed.
 int assignment(const std::vector<int>&cost,int m,Perm&match,Perm&u,Perm&v){
  Perm p(m+1),way(m+1);u.assign(m+1,0);v.assign(m+1,0);
  for(int i=1;i<=m;++i){
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
 void visit(int depth,int committed,std::shared_ptr<Group> generators){
  if(status)return;
  ++nodes;
  if(expired()){status=1;return;}
  if(committed>target){++pruned;return;}
  if(depth==n){
   ++leaves;
   if(committed==target){
    ++accepted;
    if(callback(mapping.data(),userdata)){status=3;return;}
    if(cap>0&&accepted>=cap)status=2;
   }
   return;
  }
  int m=n-depth,atom=order[depth];Perm columns;for(int j=0;j<n;++j)if(!used[j])columns.push_back(j);
  std::vector<int> cost(m*m,INF);
  for(int i=0;i<m;++i){
   int x=order[depth+i],minimum=-1;
   for(int p=0;p<n;++p)if(pred[x*n+p])minimum=std::max(minimum,mapping[p]);
   for(int j=0;j<m;++j){
    int y=columns[j];if(ca[x]!=cb[y]||y<=minimum)continue;
    int value=cross[x*n+y];
    for(int t=0;t<nt;++t)for(int k=0;k<nl;++k)
     value+=(levels[k+1]-levels[k])*std::abs(ha[(x*nt+t)*nl+k]-hb[(y*nt+t)*nl+k]);
    cost[i*m+j]=value;
   }
  }
  Perm match,u,v;int lower=assignment(cost,m,match,u,v);
  if(lower>=INF/2||committed+lower>target){++pruned;return;}
  // Shortest alternating path to row 0, using nonnegative Hungarian reduced costs.
  Perm dist(m,INF);std::vector<char>done(m,0);dist[0]=0;
  for(int step=0;step<m;++step){
   int k=-1;for(int i=0;i<m;++i)if(!done[i]&&(k<0||dist[i]<dist[k]))k=i;
   if(k<0||dist[k]>=INF/2)break;done[k]=1;
   int col=match[k];
   for(int i=0;i<m;++i)if(!done[i]&&cost[i*m+col]<INF/2){
    int edge=cost[i*m+col]-u[i+1]-v[col+1];
    if(edge<0)throw std::runtime_error("negative reduced cost");
    dist[i]=std::min(dist[i],dist[k]+edge);
   }
  }
  Perm parent(n);std::iota(parent.begin(),parent.end(),0);
  auto find=[&](int x){while(parent[x]!=x){parent[x]=parent[parent[x]];x=parent[x];}return x;};
  for(const auto&g:*generators)for(int j:columns){int x=find(j),y=find(g[j]);if(x!=y)parent[y]=x;}
  Perm candidates;
  for(int i=0;i<m;++i){
   int col=match[i],value=cost[col];
   if(value>=INF/2||dist[i]>=INF/2)continue;
   int forced=lower+value-u[1]-v[col+1]+dist[i];
   if(committed+forced<=target)candidates.push_back(columns[col]);
  }
  std::sort(candidates.begin(),candidates.end());
  std::vector<char>seen(n,0);
  for(int image:candidates){
   int orbit=find(image);if(seen[orbit])continue;seen[orbit]=1;
   int delta=cross[atom*n+image];
   mapping[atom]=image;used[image]=1;
   update(atom,image,1);
   visit(depth+1,committed+delta,stabilize(generators,image,n));
   update(atom,image,-1);
   used[image]=0;mapping[atom]=-1;
   if(status)return;
  }
 }
};
extern "C" int synkit_distance_abi(){return 1;}
extern "C" int synkit_distance_candidates(
 int n,int nt,int nl,int target,double seconds,int64_t cap,
 const int*a,const int*b,const int*ca,const int*cb,const int*order,const int*pred,
 const int*gens,int count,Callback callback,void*userdata,int64_t*statistics){
 try{
  if(n<1||n>256||nt<1||nt>n||nl<0||nl>15||target<0||target>1000000||!callback)return -2;
  Solver solver(n,nt,nl,target,seconds,cap,a,b,ca,cb,order,pred,callback,userdata);
  auto group=std::make_shared<Group>();
  for(int k=0;k<count;++k)group->emplace_back(gens+k*n,gens+(k+1)*n);
  solver.visit(0,0,group);
  statistics[0]=solver.nodes;statistics[1]=solver.leaves;statistics[2]=solver.accepted;statistics[3]=solver.pruned;
  return solver.status;
 }catch(...){return -1;}
}
