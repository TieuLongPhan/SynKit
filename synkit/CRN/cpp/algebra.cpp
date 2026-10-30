#include "exact.h"
#include <map>
#include <set>
#include <functional>
using namespace crn_exact;
extern "C" {
int synkit_crn_algebra_abi(){return 1;}
int synkit_crn_matrix(std::size_t n,std::size_t m,const std::int64_t* input,int kernels,double seconds,char** output){
 if(!output || !std::isfinite(seconds)||seconds<=0||seconds>1e9)return 2;*output=nullptr;Budget budget(seconds);
 try{
  Matrix s(n,std::vector<Rat>(m)),t(m,std::vector<Rat>(n));
  for(std::size_t i=0;i<n;++i){budget.check();for(std::size_t j=0;j<m;++j)s[i][j]=t[j][i]=input[i*m+j];}
  std::size_t rank=0;std::vector<std::vector<Int>> left,right;
  if(kernels&1)left=kernel(t,n,budget,rank);else{auto copy=s;rank=rref(copy,m,budget).size();}
  if(kernels&2){std::size_t ignored;right=kernel(s,m,budget,ignored);}
  std::ostringstream out;out<<"{\"termination_reason\":\"complete\",\"rank\":"<<rank<<",\"left\":";matrix_json(out,left);out<<",\"right\":";matrix_json(out,right);out<<'}';return output_json(out.str(),output);
 }catch(const Limit& l){return output_json(std::string("{\"termination_reason\":\"")+l.reason+"\"}",output);}catch(...){return 2;}
}
int synkit_crn_crnt(std::size_t n,std::size_t m,const std::int64_t* pre,const std::int64_t* post,double seconds,char** output){
 if(!output || !std::isfinite(seconds)||seconds<=0||seconds>1e9)return 2;*output=nullptr;Budget budget(seconds);
 try{
  std::map<std::vector<std::int64_t>,std::size_t> index;std::vector<std::vector<std::int64_t>> complexes;
  std::vector<std::pair<std::size_t,std::size_t>> arrows;
  auto intern=[&](const std::int64_t* data,std::size_t j){std::vector<std::int64_t> column(n);for(std::size_t i=0;i<n;++i)column[i]=data[i*m+j];auto found=index.find(column);if(found!=index.end())return found->second;std::size_t id=complexes.size();index[column]=id;complexes.push_back(std::move(column));return id;};
  Matrix vectors(m,std::vector<Rat>(n));
  for(std::size_t j=0;j<m;++j){budget.check();auto a=intern(pre,j);auto b=intern(post,j);arrows.emplace_back(a,b);for(std::size_t i=0;i<n;++i)vectors[j][i]=Int(post[i*m+j])-Int(pre[i*m+j]);}
  auto copy=vectors;std::size_t rank=rref(copy,n,budget).size(), count=complexes.size();
  std::vector<std::vector<std::size_t>> adj(count),rev(count),undirected(count);
  std::set<std::pair<std::size_t,std::size_t>> edges(arrows.begin(),arrows.end());
  for(auto edge:edges){adj[edge.first].push_back(edge.second);rev[edge.second].push_back(edge.first);undirected[edge.first].push_back(edge.second);undirected[edge.second].push_back(edge.first);}
  std::vector<int> component(count,-1),scc(count,-1);std::vector<std::vector<std::size_t>> classes;std::vector<std::size_t> order;std::vector<bool> seen(count,false);
  std::function<void(std::size_t)> visit=[&](std::size_t v){budget.check();seen[v]=true;for(auto w:adj[v])if(!seen[w])visit(w);order.push_back(v);};
  for(std::size_t v=0;v<count;++v)if(!seen[v])visit(v);
  std::function<void(std::size_t,int)> assign=[&](std::size_t v,int c){budget.check();scc[v]=c;for(auto w:rev[v])if(scc[w]<0)assign(w,c);};
  int sc=0;for(auto it=order.rbegin();it!=order.rend();++it)if(scc[*it]<0)assign(*it,sc++);
  for(std::size_t v=0;v<count;++v)if(component[v]<0){int c=classes.size();classes.push_back({});std::vector<std::size_t> stack{v};component[v]=c;while(!stack.empty()){budget.check();auto x=stack.back();stack.pop_back();classes.back().push_back(x);for(auto w:undirected[x])if(component[w]<0){component[w]=c;stack.push_back(w);}}std::sort(classes.back().begin(),classes.back().end());}
  std::vector<int> terminal(sc,1);for(auto edge:edges)if(scc[edge.first]!=scc[edge.second])terminal[scc[edge.first]]=0;
  std::vector<int> deficiencies,terminal_counts;bool weak=true,reversible=true;
  for(auto edge:edges)if(!edges.count({edge.second,edge.first}))reversible=false;
  for(std::size_t c=0;c<classes.size();++c){Matrix local;std::set<int> strong;for(auto v:classes[c])strong.insert(scc[v]);if(strong.size()!=1)weak=false;
   for(std::size_t j=0;j<m;++j)if(component[arrows[j].first]==int(c))local.push_back(vectors[j]);
   auto r=rref(local,n,budget).size();deficiencies.push_back(int(classes[c].size())-1-int(r));int terminals=0;for(auto k:strong)terminals+=terminal[k];terminal_counts.push_back(terminals);}
  std::ostringstream out;out<<"{\"termination_reason\":\"complete\",\"rank\":"<<rank<<",\"n_species\":"<<n<<",\"n_reactions\":"<<m<<",\"n_complexes\":"<<count<<",\"n_linkage_classes\":"<<classes.size()<<",\"n_terminal_strong_linkage_classes\":"<<std::count(terminal.begin(),terminal.end(),1)<<",\"deficiency\":"<<int(count)-int(classes.size())-int(rank)<<",\"is_weakly_reversible\":"<<(weak?"true":"false")<<",\"is_reversible\":"<<(reversible?"true":"false")<<",\"linkage_class_deficiencies\":";vector_json(out,deficiencies);out<<",\"terminal_counts\":";vector_json(out,terminal_counts);out<<",\"complex_coefficients\":";matrix_json(out,complexes);out<<'}';return output_json(out.str(),output);
 }catch(const Limit& l){return output_json(std::string("{\"termination_reason\":\"")+l.reason+"\"}",output);}catch(...){return 2;}
}
}
