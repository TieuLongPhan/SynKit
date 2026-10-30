#include "synkit_mapper/hydrogen/hydrogen.h"
#include "synkit_mapper/core/runtime.h"
#include <exception>
#include <functional>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>

namespace {
using namespace synkit_mapper;
using Values=std::vector<int32_t>;
thread_local std::string h_error;
struct BigInteger {
  std::vector<uint32_t> limbs{1};
  void multiply(uint32_t k) {uint64_t carry=0;for(auto &v:limbs) {uint64_t x=uint64_t(v)*k+carry;v=x%1000000000;carry=x/1000000000;}if(carry) limbs.push_back(carry);}
  void divide(uint32_t k) {uint64_t rest=0;for(size_t i=limbs.size();i-->0;) {uint64_t x=rest*1000000000+limbs[i];limbs[i]=x/k;rest=x%k;}if(rest) throw std::logic_error("Nonintegral H multiplicity");while(limbs.size()>1&&!limbs.back()) limbs.pop_back();}
  std::string text() const {std::ostringstream out;out<<limbs.back();for(size_t i=limbs.size()-1;i-->0;) {auto s=std::to_string(limbs[i]);out<<std::string(9-s.size(),'0')<<s;}return out.str();}
};
std::string multiplicity(const Values &rows,const Values &columns,const Values &cells,Budget &budget) {
  BigInteger value;
  for(const auto *counts:{&rows,&columns}) for(int count:*counts) for(int k=2;k<=count;++k) {if(k%64==0) budget.check();value.multiply(k);}
  size_t processed=0;
  for(int count:cells) {if(++processed%1024==0) budget.check();for(int k=2;k<=count;++k) {if(k%64==0) budget.check();value.divide(k);}}
  return value.text();
}
struct Transport {
  int n,reservoir,query;int64_t target,max_plans,max_nodes;
  Values original_rows,original_columns,rows,columns,cells,active;
  Request timing{};Budget budget;
  SynkitHydrogenTransportCallback callback;void *userdata;
  SynkitHydrogenTransportStatistics &stats;
  Transport(int size,const int32_t *r,const int32_t *c,int dummy,int q,int64_t t,
            double seconds,int64_t plans,int64_t nodes,void *cancel,
            SynkitHydrogenTransportCallback cb,void *data,SynkitHydrogenTransportStatistics &s)
      :n(size),reservoir(dummy),query(q),target(t),max_plans(plans),max_nodes(nodes),
       original_rows(r,r+size),original_columns(c,c+size),rows(original_rows),columns(original_columns),
       cells(size*size,0),timing{},budget(timing),callback(cb),userdata(data),stats(s) {
    timing.seconds=seconds;timing.cancel_token=cancel;
  }
  bool real(int i) const {return i!=reservoir;}
  int64_t edge_cost(int i,int j) const {return i==j?0:2*(int(real(i))+int(real(j)));}
  void poll() {budget.check();if(max_nodes>=0&&stats.nodes>=max_nodes) throw Stop{SYNKIT_STOP_STATES};++stats.nodes;}
  bool viable(size_t pos,int first_column,int remaining,int64_t cost) {
    int64_t out_rows=0,out_columns=0,diagonal=0;
    for(int j=0;j<n;++j) if(real(j)) out_columns+=columns[j];
    for(size_t k=pos;k<active.size();++k) {
      int i=active[k],count=k==pos?remaining:rows[i];
      if(real(i)) {out_rows+=count;if(k!=pos||i>=first_column) diagonal+=std::min(count,columns[i]);}
    }
    int64_t lower=cost+2*(out_rows+out_columns-2*diagonal);
    if(lower>target) return false;
    return query!=SYNKIT_EXACT_DISTANCE||cost+2*(out_rows+out_columns)>=target;
  }
  void emit(int64_t cost) {
    budget.check();
    if(cost>target||(query==SYNKIT_EXACT_DISTANCE&&cost!=target)) return;
    if(max_plans>=0&&stats.plans>=max_plans) throw Stop{SYNKIT_STOP_CANDIDATES};
    std::string count=multiplicity(original_rows,original_columns,cells,budget);++stats.plans;
    if(callback(cells.data(),cost,count.c_str(),userdata)) throw Stop{SYNKIT_STOP_CALLBACK};
    budget.check();
  }
  void row(size_t pos,int64_t cost) {
    poll();if(pos==active.size()) {emit(cost);return;}
    distribute(pos,0,rows[active[pos]],cost);
  }
  void distribute(size_t pos,int first_column,int remaining,int64_t cost) {
    poll();if(!remaining) {row(pos+1,cost);return;}
    if(!viable(pos,first_column,remaining,cost)) return;
    int i=active[pos];
    // Enumerate positive cells in increasing column order; omitted cells are
    // zero. Recursion depth depends on H count, not parents squared.
    for(int j=first_column;j<n;++j) if(columns[j]) {
      int maximum=std::min(remaining,columns[j]);
      for(int count=1;count<=maximum;++count) {
        int64_t next=cost+count*edge_cost(i,j);if(next>target) break;
        if(query==SYNKIT_MINIMUM_CLASSES&&i==j&&real(i)) continue;
        cells[i*n+j]+=count;columns[j]-=count;
        distribute(pos,j+1,remaining-count,next);
        columns[j]+=count;cells[i*n+j]-=count;
      }
    }
  }
  void run() {
    stats={};
    int64_t minimum=0;
    for(int i=0;i<n;++i) if(real(i)) minimum+=2*std::abs(rows[i]-columns[i]);
    stats.minimum2=minimum;
    budget.check();
    if(query==SYNKIT_MINIMUM_CLASSES) {
      target=minimum;
      for(int i=0;i<n;++i) if(real(i)) {int kept=std::min(rows[i],columns[i]);cells[i*n+i]=kept;rows[i]-=kept;columns[i]-=kept;}
    }
    for(int i=0;i<n;++i) if(rows[i]) active.push_back(i);
    if(minimum<=target) row(0,0);
    stats.complete=1;
  }
};

struct Endpoint {
  Values heavy,counts,reservoir;
  std::vector<Values> groups;
};
struct Quotient {
  Problem problem{};
  std::vector<int64_t> a,b;
  Values tr,tp,nr,np,er,ep;
  Endpoint r,p;
};
bool uniform_hydrogen_classes(const Problem &p,const SynkitHydrogenAttributes &attrs,const SynkitClassProblem &classes,Budget &budget) {
  if(!classes.node_pair_colors||!classes.edges_r||!classes.edges_p||!classes.edge_pair_colors||classes.edge_states_r<1||classes.edge_states_p<1)
    throw std::invalid_argument("Invalid exact H class palette");
  for(bool right:{false,true}) {
    auto z=right?attrs.atomic_numbers_p:attrs.atomic_numbers_r;
    auto other_z=right?attrs.atomic_numbers_r:attrs.atomic_numbers_p;
    auto dummy=right?attrs.dummies_p:attrs.dummies_r;
    auto weights=right?p.b2:p.a2;
    auto edges=right?classes.edges_p:classes.edges_r;
    int states=right?classes.edge_states_p:classes.edge_states_r;
    int other_states=right?classes.edge_states_r:classes.edge_states_p;
    if(!z||!other_z||!dummy) throw std::invalid_argument("Missing H class eligibility metadata");
    std::map<int,Values> node_signatures,edge_signatures;
    for(int i=0;i<p.n;++i) if(z[i]==1) {
      budget.check();
      Values node;
      for(int j=0;j<p.n;++j) if(other_z[j]==1) node.push_back(classes.node_pair_colors[right?j*p.n+i:i*p.n+j]);
      auto inserted=node_signatures.emplace(int(bool(dummy[i])),node);
      if(!inserted.second&&inserted.first->second!=node) return false;
      for(int j=0;j<p.n;++j) if(i!=j) {
        int state=edges[i*p.n+j];if(state<0||state>=states) throw std::invalid_argument("Invalid H edge palette state");
        Values signature;signature.reserve(other_states);
        for(int k=0;k<other_states;++k) {if(k%64==0) budget.check();signature.push_back(classes.edge_pair_colors[right?k*classes.edge_states_p+state:state*classes.edge_states_p+k]);}
        // All ordinary attachment bonds share one class state; absent states
        // also share one effect. This sufficient condition is deliberately
        // conservative for arbitrary user annotations outside chemistry.
        int kind=weights[i*p.n+j]?1:0;
        auto old=edge_signatures.emplace(kind,signature);
        if(!old.second&&old.first->second!=signature) return false;
      }
    }
  }
  return true;
}
int endpoint(const Problem &p,const SynkitHydrogenAttributes &attrs,bool right,Endpoint &out,Budget &budget) {
  const int32_t *z=right?attrs.atomic_numbers_p:attrs.atomic_numbers_r;
  const int32_t *isotope=right?attrs.isotopes_p:attrs.isotopes_r;
  const int32_t *charge=right?attrs.charges_p:attrs.charges_r;
  const int32_t *radical=right?attrs.radicals_p:attrs.radicals_r;
  const int32_t *dummy=right?attrs.dummies_p:attrs.dummies_r;
  const int32_t *stereo=right?attrs.stereo_p:attrs.stereo_r;
  const int64_t *weights=right?p.b2:p.a2;
  const int32_t *colors=right?p.node_colors_p:p.node_colors_r;
  const int32_t *edges=right?p.edge_colors_p:p.edge_colors_r;
  if(!z||!isotope||!charge||!radical||!dummy||!stereo) throw std::invalid_argument("Missing hydrogen eligibility metadata");
  bool found=false;Values positions(p.n,-1);
  for(int i=0;i<p.n;++i) {budget.check();if(stereo[i]) return SYNKIT_H_STEREO;if(z[i]==1) found=true;else {positions[i]=out.heavy.size();out.heavy.push_back(i);}}
  if(!found) return SYNKIT_H_NO_HYDROGEN;
  out.groups.resize(out.heavy.size());out.counts.assign(out.heavy.size(),0);
  bool have_real=false,have_dummy=false;int real_color=0,dummy_color=0,bond_color=0;
  for(int i=0;i<p.n;++i) if(z[i]==1) {
    budget.check();
    if(isotope[i]) return SYNKIT_H_ISOTOPE;
    if(charge[i]||radical[i]) return SYNKIT_H_ELECTRON_STATE;
    int parent=-1,degree=0;
    for(int j=0;j<p.n;++j) if(weights[i*p.n+j]) {parent=j;++degree;}
    if(dummy[i]) {
      if(degree) return SYNKIT_H_DUMMY_BOND;
      if(colors&&have_dummy&&colors[i]!=dummy_color) return SYNKIT_H_CLASS_ATTRIBUTES;
      dummy_color=colors?colors[i]:0;have_dummy=true;out.reservoir.push_back(i);continue;
    }
    if(degree!=1) return SYNKIT_H_NONTERMINAL;
    if(weights[i*p.n+parent]!=2||positions[parent]<0||z[parent]<=1) return SYNKIT_H_BOND_STATE;
    if((colors&&have_real&&colors[i]!=real_color)||(edges&&have_real&&edges[i*p.n+parent]!=bond_color)) return SYNKIT_H_CLASS_ATTRIBUTES;
    real_color=colors?colors[i]:0;bond_color=edges?edges[i*p.n+parent]:0;have_real=true;
    out.groups[positions[parent]].push_back(i);++out.counts[positions[parent]];
  }
  return SYNKIT_H_SUPPORTED;
}
int32_t fresh_type(const Problem &p) {
  std::set<int32_t> used(p.types_r,p.types_r+p.n);used.insert(p.types_p,p.types_p+p.n);
  for(int64_t value=std::numeric_limits<int32_t>::min();value<=std::numeric_limits<int32_t>::max();++value)
    if(!used.count(static_cast<int32_t>(value))) return value;
  throw std::logic_error("No available H anchor type");
}
int reduce(const Problem &p,const SynkitHydrogenAttributes &attrs,Quotient &q,Budget &budget) {
  int reason=endpoint(p,attrs,false,q.r,budget);if(reason) return reason;
  reason=endpoint(p,attrs,true,q.p,budget);if(reason) return reason;
  std::set<int32_t> hydrogen_types,heavy_types;
  for(int i=0;i<p.n;++i) {
    (attrs.atomic_numbers_r[i]==1?hydrogen_types:heavy_types).insert(p.types_r[i]);
    (attrs.atomic_numbers_p[i]==1?hydrogen_types:heavy_types).insert(p.types_p[i]);
  }
  if(hydrogen_types.size()!=1||heavy_types.count(*hydrogen_types.begin())) return SYNKIT_H_CLASS_ATTRIBUTES;
  if(q.r.heavy.size()!=q.p.heavy.size()) throw std::invalid_argument("Unequal heavy inventories");
  int m=q.r.heavy.size(),n=m+1;int32_t anchor=fresh_type(p);
  q.a.assign(n*n,0);q.b.assign(n*n,0);q.er.assign(n*n,0);q.ep.assign(n*n,0);
  for(bool right:{false,true}) {
    auto &side=right?q.p:q.r;auto &matrix=right?q.b:q.a;auto &types=right?q.tp:q.tr;
    auto &colors=right?q.np:q.nr;auto &edgecolors=right?q.ep:q.er;
    auto full=right?p.b2:p.a2;auto typed=right?p.types_p:p.types_r;
    auto nodes=right?p.node_colors_p:p.node_colors_r;auto edges=right?p.edge_colors_p:p.edge_colors_r;
    for(int i=0;i<m;++i) {
      budget.check();
      int source=side.heavy[i];types.push_back(typed[source]);colors.push_back(nodes?nodes[source]:0);
      for(int j=0;j<m;++j) {matrix[i*n+j]=full[source*p.n+side.heavy[j]];edgecolors[i*n+j]=edges?edges[source*p.n+side.heavy[j]]:0;}
      matrix[i*n+m]=matrix[m*n+i]=2*side.counts[i];
    }
    types.push_back(anchor);colors.push_back(0);
  }
  q.problem={1,n,q.a.data(),q.b.data(),q.tr.data(),q.tp.data(),q.nr.data(),q.np.data(),q.er.data(),q.ep.data()};
  return SYNKIT_H_SUPPORTED;
}
struct HydrogenSearch {
  const Problem &full;const Request &request;Quotient &q;
  int64_t max_plans;Callback callback;void *userdata;Statistics &stats;SynkitHydrogenStatistics &hs;
  Budget budget;int stop=0;bool plan_limit=false;std::exception_ptr failure;
  Values mapping;std::vector<Values> rows,columns;
  int64_t heavy_cost=0;
  HydrogenSearch(const Problem &p,const Request &r,Quotient &reduced,int64_t plans,Callback cb,void *data,Statistics &s,SynkitHydrogenStatistics &h)
      :full(p),request(r),q(reduced),max_plans(plans),callback(cb),userdata(data),stats(s),hs(h),budget(r) {}
  static int flow(const int32_t *cells,int64_t cost,const char *,void *data) {
    auto &self=*static_cast<HydrogenSearch*>(data);
    try {
      self.budget.check();
      if(self.max_plans>=0&&self.hs.transport_plans>=self.max_plans) {self.plan_limit=true;throw Stop{SYNKIT_STOP_CANDIDATES};}
      if(self.request.max_candidates>=0&&self.hs.transport_plans>=self.request.max_candidates) throw Stop{SYNKIT_STOP_CANDIDATES};
      Values lifted=self.mapping;int n=self.rows.size();std::vector<size_t> source_offset(n,0),target_offset(n,0);
      for(int i=0;i<n;++i) for(int j=0;j<n;++j) for(int count=0;count<cells[i*n+j];++count)
        lifted[self.rows[i][source_offset[i]++]]=self.columns[j][target_offset[j]++];
      int64_t total=self.heavy_cost+cost;
      if(mapping_cost2(self.full,lifted)!=total) throw std::logic_error("Hydrogen quotient full cost mismatch");
      ++self.hs.transport_plans;
      if(self.callback(lifted.data(),total,self.userdata)) throw Stop{SYNKIT_STOP_CALLBACK};
      self.budget.check();return 0;
    } catch(const Stop &s) {self.stop=s.reason;return 1;}catch(...) {self.failure=std::current_exception();return 1;}
  }
  static int heavy(const int32_t *images,int64_t,void *data) {
    auto &self=*static_cast<HydrogenSearch*>(data);
    try {
      self.budget.check();++self.hs.heavy_candidates;
      int m=self.q.r.heavy.size();self.mapping.assign(self.full.n,-1);self.rows=self.q.r.groups;self.columns.clear();
      self.heavy_cost=0;
      for(int i=0;i<m;++i) {
        if(images[i]<0||images[i]>=m) throw std::logic_error("H anchor not fixed");
        self.mapping[self.q.r.heavy[i]]=self.q.p.heavy[images[i]];
        self.columns.push_back(self.q.p.groups[images[i]]);
        for(int j=0;j<i;++j) self.heavy_cost+=std::abs(self.full.a2[self.q.r.heavy[i]*self.full.n+self.q.r.heavy[j]]-
          self.full.b2[self.q.p.heavy[images[i]]*self.full.n+self.q.p.heavy[images[j]]]);
      }
      self.rows.push_back(self.q.r.reservoir);self.columns.push_back(self.q.p.reservoir);
      Values rcounts,pcounts;for(const auto &group:self.rows) rcounts.push_back(group.size());for(const auto &group:self.columns) pcounts.push_back(group.size());
      SynkitHydrogenTransportStatistics transport{};
      int query=(self.request.query==SYNKIT_MINIMUM_CLASSES)?SYNKIT_MINIMUM_CLASSES:self.request.query;
      int64_t target=query==SYNKIT_MINIMUM_CLASSES?0:self.request.target2-self.heavy_cost;
      if(query!=SYNKIT_MINIMUM_CLASSES&&target<0) return 0;
      int result=synkit_mapper_hydrogen_transport(m+1,rcounts.data(),pcounts.data(),m,query,target,
          self.request.seconds<0?-1:self.budget.remaining(),-1,-1,self.request.cancel_token,flow,&self,&transport);
      self.hs.transport_nodes+=transport.nodes;self.hs.transport_seconds+=transport.elapsed_seconds;
      if(result) throw std::runtime_error(h_error);
      if(self.failure) return 1;
      if(self.stop) return 1;
      if(!transport.complete) {self.stop=transport.reason;return 1;}
      return 0;
    } catch(const Stop &s) {self.stop=s.reason;return 1;}catch(...) {self.failure=std::current_exception();return 1;}
  }
};
} // namespace

extern "C" int synkit_mapper_hydrogen_abi() {return 1;}
extern "C" const char *synkit_mapper_hydrogen_error() {return h_error.c_str();}
extern "C" int synkit_mapper_hydrogen_transport(int n,const int32_t *rows,const int32_t *columns,int reservoir,int query,
    int64_t target,double seconds,int64_t plans,int64_t nodes,void *cancel,SynkitHydrogenTransportCallback callback,void *data,SynkitHydrogenTransportStatistics *statistics) {
  try {
    h_error.clear();
    if(!statistics||!callback||!rows||!columns||n<1||n>1025||reservoir< -1||reservoir>=n||query<0||query>2||target<0||
       !std::isfinite(seconds)||(seconds<0&&seconds!=-1)||(plans<1&&plans!=-1)||(nodes<1&&nodes!=-1)) throw std::invalid_argument("Invalid hydrogen transport request");
    int64_t left=0,right=0;
    for(int i=0;i<n;++i) {if(rows[i]<0||columns[i]<0) throw std::invalid_argument("Negative hydrogen count");left+=rows[i];right+=columns[i];}
    if(left!=right||left>4096) throw std::invalid_argument("Unbalanced or oversized hydrogen transport");
    Transport transport(n,rows,columns,reservoir,query,target,seconds,plans,nodes,cancel,callback,data,*statistics);
    try {transport.run();} catch(const Stop &stop) {statistics->complete=0;statistics->reason=stop.reason;}
    statistics->elapsed_seconds=transport.budget.seconds();return 0;
  } catch(const std::exception &e) {h_error=e.what();return -1;}catch(...) {h_error="Native hydrogen transport failed";return -1;}
}

extern "C" int synkit_mapper_hydrogen_solve(const Problem *p,const SynkitHydrogenAttributes *attrs,const Request *request,
    int64_t max_plans,Callback callback,void *data,Statistics *statistics,SynkitHydrogenStatistics *hs) {
  const auto invocation_start=Clock::now();
  try {
    h_error.clear();if(!p||!attrs||!request||!statistics||!hs||(max_plans<1&&max_plans!=-1)) throw std::invalid_argument("Invalid hydrogen solve request");
    Budget total(*request);
    validate(*p,*request);initialize(*statistics,*request);*hs={};hs->full_nodes=p->n;
    if(request->classify||request->max_classes!=-1) return unsupported(*statistics);
    if(!callback&&request->query!=SYNKIT_MINIMUM_PROOF) throw std::invalid_argument("Hydrogen enumeration requires callback");
    auto start=Clock::now();Quotient q;
    hs->eligibility_reason=reduce(*p,*attrs,q,total);hs->reduction_seconds=elapsed(start);
    if(hs->eligibility_reason) {statistics->hydrogen_complete=0;statistics->elapsed_seconds=total.seconds();return unsupported(*statistics);}
    hs->supported=1;hs->quotient_nodes=q.problem.n;statistics->hydrogen_complete=0;
    Request raw=*request;raw.max_candidates=-1;raw.max_classes=-1;
    if(raw.query==SYNKIT_EXACT_DISTANCE) raw.query=SYNKIT_AT_MOST_DISTANCE;
    raw.seconds=request->seconds<0?-1:total.remaining();
    Request lift_request=*request;lift_request.seconds=raw.seconds;
    HydrogenSearch search(*p,lift_request,q,max_plans,callback,data,*statistics,*hs);
    start=Clock::now();int result=synkit_mapper_solve(&q.problem,&raw,HydrogenSearch::heavy,&search,statistics);
    hs->heavy_seconds=std::max(0.0,elapsed(start)-hs->transport_seconds);
    if(result) {h_error=synkit_mapper_core_error();return -1;}
    if(statistics->status==SYNKIT_UNSUPPORTED) hs->eligibility_reason=SYNKIT_H_REDUCED_BACKEND;
    if(search.failure) std::rethrow_exception(search.failure);
    if(search.stop) stopped(*statistics,search.stop);
    hs->reason=search.plan_limit?11:search.stop;
    hs->transport_complete=statistics->search_complete&&search.stop==0&&request->query!=SYNKIT_MINIMUM_PROOF;
    statistics->hydrogen_complete=hs->transport_complete;statistics->candidate_count=hs->transport_plans;
    statistics->labeled_count=-1;statistics->class_count=-1;statistics->elapsed_seconds=total.seconds();
    return 0;
  } catch(const Stop &stop) {
    if(statistics) {stopped(*statistics,stop.reason);statistics->hydrogen_complete=0;statistics->elapsed_seconds=elapsed(invocation_start);}
    if(hs) {hs->transport_complete=0;hs->reason=stop.reason;if(!hs->supported) hs->reduction_seconds=elapsed(invocation_start);}return 0;
  } catch(const std::exception &e) {h_error=e.what();return -1;}catch(...) {h_error="Native hydrogen quotient failed";return -1;}
}

static int hydrogen_solve_classified_impl(const Problem *p,const SynkitHydrogenAttributes *attrs,const Request *request,
    const SynkitClassProblem *classes,int64_t max_plans,int64_t max_nodes,SynkitClassCallback callback,void *data,
    Statistics *statistics,SynkitHydrogenStatistics *hs,SynkitClassStatistics *cs,bool updates) {
  const auto invocation_start=Clock::now();
  try {
    h_error.clear();
    if(!p||!attrs||!request||!classes||!statistics||!hs||!cs||!callback||request->query==SYNKIT_MINIMUM_PROOF||p->n!=classes->n||(max_nodes<1&&max_nodes!=-1)) throw std::invalid_argument("Invalid classified hydrogen request");
    initialize(*statistics,*request);*hs={};*cs={};hs->full_nodes=p->n;
    validate(*p,*request);
    Budget budget(*request);Request raw=*request;raw.classify=0;raw.max_classes=-1;
    auto preflight_start=Clock::now();Endpoint left,right;
    int eligibility=endpoint(*p,*attrs,false,left,budget);
    if(!eligibility) eligibility=endpoint(*p,*attrs,true,right,budget);
    if(!eligibility&&!uniform_hydrogen_classes(*p,*attrs,*classes,budget)) eligibility=SYNKIT_H_CLASS_ATTRIBUTES;
    double preflight_seconds=elapsed(preflight_start);
    if(eligibility) {
      initialize(*statistics,*request);*hs={};*cs={};hs->full_nodes=p->n;
      hs->eligibility_reason=eligibility;hs->reduction_seconds=preflight_seconds;
      statistics->classification_complete=0;statistics->hydrogen_complete=0;
      statistics->elapsed_seconds=budget.seconds();
      return unsupported(*statistics);
    }
    raw.seconds=request->seconds<0?-1:budget.remaining();
    struct Batch {int n;Values maps;std::vector<int64_t> costs;std::exception_ptr failure;};
    Batch batch{p->n,{}, {},nullptr};
    auto remember=[](const int32_t *mapping,int64_t cost,void *ptr)->int {auto &b=*static_cast<Batch*>(ptr);try {b.maps.insert(b.maps.end(),mapping,mapping+b.n);b.costs.push_back(cost);return 0;}catch(...) {b.failure=std::current_exception();return 1;}};
    *cs={};int result=synkit_mapper_hydrogen_solve(p,attrs,&raw,max_plans,remember,&batch,statistics,hs);
    hs->reduction_seconds+=preflight_seconds;
    statistics->classification_complete=0;
    if(result) return result;
    if(batch.failure) std::rethrow_exception(batch.failure);
    if(statistics->status==SYNKIT_UNSUPPORTED) return 0;
    auto classify=updates?synkit_mapper_classify_batch_updates_with_cancel:
                          synkit_mapper_classify_batch_with_cancel;
    result=classify(classes,batch.costs.size(),batch.maps.data(),batch.costs.data(),
      request->seconds<0?-1:budget.remaining(),max_nodes,request->max_classes,request->cancel_token,callback,data,cs);
    statistics->classification_complete=cs->complete;statistics->class_count=cs->classes;
    statistics->classification_seconds=cs->elapsed_seconds;statistics->elapsed_seconds=budget.seconds();
    if(result) {h_error=synkit_mapper_classification_error();return result;}
    if(!cs->complete&&statistics->status==SYNKIT_FINISHED) {const int reasons[]={0,1,2,4,5,10};statistics->status=SYNKIT_LIMIT;statistics->reason=reasons[cs->reason];}
    return 0;
  } catch(const Stop &stop) {
    if(statistics) {stopped(*statistics,stop.reason);statistics->classification_complete=statistics->hydrogen_complete=0;statistics->elapsed_seconds=elapsed(invocation_start);}
    if(hs) {hs->transport_complete=0;hs->reason=stop.reason;if(!hs->supported) hs->reduction_seconds=elapsed(invocation_start);}return 0;
  } catch(const std::exception &e) {h_error=e.what();return -1;}catch(...) {h_error="Classified hydrogen solve failed";return -1;}
}

extern "C" int synkit_mapper_hydrogen_solve_classified(const Problem *p,const SynkitHydrogenAttributes *attrs,const Request *request,
    const SynkitClassProblem *classes,int64_t max_plans,int64_t max_nodes,SynkitClassCallback callback,void *data,
    Statistics *statistics,SynkitHydrogenStatistics *hs,SynkitClassStatistics *cs) {
  return hydrogen_solve_classified_impl(p,attrs,request,classes,max_plans,max_nodes,
      callback,data,statistics,hs,cs,false);
}

extern "C" int synkit_mapper_hydrogen_solve_classified_updates(const Problem *p,const SynkitHydrogenAttributes *attrs,const Request *request,
    const SynkitClassProblem *classes,int64_t max_plans,int64_t max_nodes,SynkitClassCallback callback,void *data,
    Statistics *statistics,SynkitHydrogenStatistics *hs,SynkitClassStatistics *cs) {
  return hydrogen_solve_classified_impl(p,attrs,request,classes,max_plans,max_nodes,
      callback,data,statistics,hs,cs,true);
}
