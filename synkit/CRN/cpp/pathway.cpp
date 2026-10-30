#include "exact.h"
#include <deque>
#include <unordered_set>
using namespace crn_exact;
namespace {
struct Node { std::vector<Int> marking; std::size_t parent, transition, depth; };
std::string key(const std::vector<Int>& marking){std::ostringstream out;for(const auto& x:marking)out<<x<<';';return out.str();}
}
extern "C" {
int synkit_crn_pathway_abi(){return 1;}
int synkit_crn_pathway(std::size_t n,std::size_t m,const std::int64_t* pre,const std::int64_t* post,
 const std::int64_t* initial,const std::int64_t* exact,const std::int64_t* atleast,
 std::size_t max_states,std::size_t max_depth,double seconds,int qualitative,char** output){
 if(!output||!std::isfinite(seconds)||seconds<=0||seconds>1e9||!max_states)return 2;*output=nullptr;Budget budget(seconds);
 std::string status="inconclusive",reason="time_limit";std::size_t explored=0,deepest=0;bool exhaustive=false,limited=false;
 std::vector<std::size_t> certificate;
 try{
  auto goal=[&](const std::vector<Int>& marking){for(std::size_t i=0;i<n;++i)if((exact[i]>=0 && marking[i]!=exact[i]) || marking[i]<atleast[i])return false;return true;};
  std::vector<Node> nodes;Node first;first.parent=std::size_t(-1);first.depth=0;for(std::size_t i=0;i<n;++i)first.marking.push_back(initial[i]);
  if(goal(first.marking)){status="realizable";reason="certificate";}
  else{
   nodes.push_back(std::move(first));std::deque<std::size_t> queue{0};std::unordered_set<std::string> seen{key(nodes[0].marking)};bool found=false;
   while(!queue.empty()&&!found){
    budget.check();if(explored>=max_states)throw Limit{"state_limit"};auto index=queue.front();queue.pop_front();
    // Copy: adding successors may reallocate the node vector.
    Node current=nodes[index];++explored;deepest=std::max(deepest,current.depth);
    for(std::size_t t=0;t<m;++t){
     budget.check();bool enabled=true;for(std::size_t i=0;i<n;++i)if(current.marking[i]<(qualitative?(pre[i*m+t]>0?1:0):pre[i*m+t])){enabled=false;break;}
     if(!enabled)continue;if(current.depth>=max_depth){limited=true;continue;}
     Node child;child.parent=index;child.transition=t;child.depth=current.depth+1;child.marking=current.marking;
     for(std::size_t i=0;i<n;++i){if(qualitative){if(post[i*m+t]>0)child.marking[i]=1;}else child.marking[i]+=Int(post[i*m+t])-Int(pre[i*m+t]);}
     if(goal(child.marking)){
      certificate.push_back(t);auto p=index;while(nodes[p].parent!=std::size_t(-1)){certificate.push_back(nodes[p].transition);p=nodes[p].parent;}std::reverse(certificate.begin(),certificate.end());
      deepest=std::max(deepest,child.depth);status="realizable";reason="certificate";found=true;break;
     }
     auto signature=key(child.marking);if(seen.insert(signature).second){
      if(nodes.size()>=max_states*16)throw Limit{"frontier_limit"};queue.push_back(nodes.size());nodes.push_back(std::move(child));}
    }
   }
   if(!found){status=limited?"inconclusive":"unrealizable";reason=limited?"depth_limit":"exhausted";exhaustive=!limited;}
  }
 }catch(const Limit& limit){reason=limit.reason;status="inconclusive";certificate.clear();}catch(const std::bad_alloc&){reason="allocation_failure";}catch(...){return 2;}
 std::ostringstream out;out<<"{\"status\":\""<<status<<"\",\"termination_reason\":\""<<reason<<"\",\"explored_states\":"<<explored<<",\"max_depth_reached\":"<<deepest<<",\"exhaustive\":"<<(exhaustive?"true":"false")<<",\"certificate\":";
 if(status=="realizable")vector_json(out,certificate);else out<<"null";out<<'}';return output_json(out.str(),output);
}
}
