#include "exact.h"
#include <map>
#include <numeric>
#include <functional>
using namespace crn_exact;
extern "C" {
int synkit_crn_symmetry_abi(){return 1;}
int synkit_crn_symmetry(std::size_t n,const std::int64_t* kinds,const std::int64_t* adjacency,
                       double seconds,std::size_t max_states,std::size_t max_count,char** output){
 if(!output||!std::isfinite(seconds)||seconds<=0||seconds>1e9||!max_states||!max_count)return 2;*output=nullptr;Budget budget(seconds);
 std::size_t states=0;Int count=0;std::vector<std::int64_t> best;std::vector<std::size_t> base,parent(n);std::iota(parent.begin(),parent.end(),0);
 std::vector<std::vector<std::pair<std::size_t,std::int64_t>>> incoming(n),outgoing(n);
 auto find=[&](std::size_t v){while(parent[v]!=v)v=parent[v];return v;};const char* reason="complete";
 try{
  for(std::size_t i=0;i<n;++i){budget.check();for(std::size_t j=0;j<n;++j)if(adjacency[i*n+j]){outgoing[i].push_back({j,adjacency[i*n+j]});incoming[j].push_back({i,adjacency[i*n+j]});}}
  auto refine=[&](std::vector<std::int64_t> colors){
   while(true){budget.check();std::vector<std::vector<std::int64_t>> signatures(n);std::map<std::vector<std::int64_t>,std::int64_t> ids;
    for(std::size_t v=0;v<n;++v){auto& sig=signatures[v];sig.push_back(colors[v]);
     for(auto* edges:{&incoming[v],&outgoing[v]}){std::vector<std::pair<std::int64_t,std::int64_t>> neighbors;for(auto edge:*edges)neighbors.push_back({colors[edge.first],edge.second});std::sort(neighbors.begin(),neighbors.end());sig.push_back(neighbors.size());for(auto pair:neighbors){sig.push_back(pair.first);sig.push_back(pair.second);}}
     ids.emplace(sig,0);}
    std::int64_t id=0;for(auto& entry:ids)entry.second=id++;
    std::vector<std::int64_t> next;for(const auto& sig:signatures)next.push_back(ids[sig]);if(next==colors)return colors;colors=std::move(next);
   }
  };
  std::function<void(std::vector<std::int64_t>)> visit=[&](std::vector<std::int64_t> colors){
   budget.check();if(states>=max_states)throw Limit{"state_limit"};++states;colors=refine(std::move(colors));
   std::map<std::int64_t,std::vector<std::size_t>> cells;for(std::size_t v=0;v<n;++v)cells[colors[v]].push_back(v);
   std::vector<std::size_t> selected;for(const auto& entry:cells)if(entry.second.size()>1 && (selected.empty()||entry.second.size()<selected.size()))selected=entry.second;
   if(!selected.empty()){
    auto fresh=*std::max_element(colors.begin(),colors.end())+1;for(auto v:selected){auto next=colors;next[v]=fresh;visit(std::move(next));}return;
   }
   std::vector<std::size_t> order;for(const auto& entry:cells)order.push_back(entry.second[0]);
   std::vector<std::int64_t> certificate;certificate.reserve(n+n*n);
   for(auto v:order)certificate.push_back(kinds[v]);for(auto v:order)for(auto w:order)certificate.push_back(adjacency[v*n+w]);
   if(base.empty() || certificate<best){best=std::move(certificate);base=order;count=1;std::iota(parent.begin(),parent.end(),0);}
   else if(certificate==best){++count;for(std::size_t i=0;i<n;++i){auto a=find(base[i]),b=find(order[i]);if(a!=b)parent[b]=a;}}
   if(count>=max_count)throw Limit{"count_limit"};
  };
  std::vector<std::int64_t> initial(kinds,kinds+n);visit(std::move(initial));
 }catch(const Limit& limit){reason=limit.reason;}catch(const std::bad_alloc&){reason="allocation_failure";}catch(...){return 2;}
 std::map<std::size_t,std::vector<std::size_t>> orbits;for(std::size_t v=0;v<n;++v)orbits[find(v)].push_back(v);
 std::vector<std::vector<std::size_t>> groups;for(auto entry:orbits)groups.push_back(std::move(entry.second));
 std::ostringstream out;out<<"{\"termination_reason\":\""<<reason<<"\",\"states\":"<<states<<",\"automorphism_count\":"<<count<<",\"order\":";vector_json(out,base);out<<",\"orbits\":";matrix_json(out,groups);out<<'}';return output_json(out.str(),output);
}
}
