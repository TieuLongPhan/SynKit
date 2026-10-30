#include "synkit_mapper/symmetry/classification.h"
#include "synkit_mapper/core/runtime.h"
#include "classification_stream.h"
#include <memory>
#include <exception>

namespace {
struct Candidates {
  int n;
  std::vector<int32_t> mappings;
  std::vector<int64_t> costs;
  std::exception_ptr failure;
  synkit_mapper::detail::ClassStream *stream=nullptr;
};
int remember(const int32_t *mapping,int64_t cost,void *data) {
  auto &c=*static_cast<Candidates*>(data);
  try {
    if(c.stream) {
      int reason=synkit_mapper::detail::class_stream_push(c.stream,mapping,cost);
      // A classification node/class cap preserves the independent raw
      // search status and prefix counts. Other stops end both stages.
      return reason==2||reason==3?0:reason!=0;
    }
    c.mappings.insert(c.mappings.end(),mapping,mapping+c.n);c.costs.push_back(cost);return 0;
  } catch(...) {c.failure=std::current_exception();return 1;}
}
int solve_classified_impl(
    const SynkitMapperProblem *p,const SynkitMapperRequest *request,
    const uint8_t *allowed,
    const SynkitClassProblem *classes,int64_t max_canonical_nodes,
    SynkitClassCallback callback,void *userdata,SynkitMapperStatistics *statistics,
    SynkitClassStatistics *class_statistics,bool updates,bool packed_identifier) {
  using namespace synkit_mapper;
  if(!p||!request||!classes||!callback||!statistics||!class_statistics||
     p->n!=classes->n||request->classify!=1||request->query==SYNKIT_MINIMUM_PROOF||
     (max_canonical_nodes<1&&max_canonical_nodes!=-1)||
     !std::isfinite(request->seconds)||(request->seconds<0&&request->seconds!=-1)) return -1;
  try {
    initialize(*statistics,*request);*class_statistics={};
    // The cap concerns the selected backend's accounted frontier/table.
    // Intermediate class buffers and process RSS are outside those units.
    Budget budget(*request);
    Request raw=*request;raw.classify=0;raw.max_classes=-1;
    using Stream=detail::ClassStream;
    std::unique_ptr<Stream,void(*)(Stream*)> stream(nullptr,detail::class_stream_delete);
    if(updates)stream.reset(detail::class_stream_create(*classes,
        request->seconds<0?-1:budget.remaining(),max_canonical_nodes,
        request->max_classes,request->cancel_token,callback,userdata,packed_identifier));
    Candidates candidates{p->n,{}, {},nullptr,stream.get()};
    raw.seconds=request->seconds<0?-1:budget.remaining();
    int result=allowed?synkit_mapper_solve_domains(p,&raw,allowed,remember,&candidates,statistics):
                       synkit_mapper_solve(p,&raw,remember,&candidates,statistics);
    *class_statistics={};statistics->classification_complete=0;
    if(result||candidates.failure) return -1;
    if(statistics->status==SYNKIT_UNSUPPORTED||statistics->status==SYNKIT_ERROR) return 0;
    double remaining=request->seconds<0?-1:budget.remaining();
    if(stream)detail::class_stream_finish(stream.get(),*class_statistics);
    else result=synkit_mapper_classify_batch_ex(classes,static_cast<int64_t>(candidates.costs.size()),
      candidates.mappings.data(),candidates.costs.data(),remaining,max_canonical_nodes,
      request->max_classes,request->cancel_token,callback,userdata,class_statistics,
      updates?1:0,packed_identifier?1:0);
    statistics->classification_complete=class_statistics->complete;
    statistics->class_count=class_statistics->classes;
    statistics->classification_seconds=class_statistics->elapsed_seconds;
    statistics->elapsed_seconds=budget.seconds();
    if(result) {statistics->status=SYNKIT_ERROR;return -1;}
    if(!class_statistics->complete && (statistics->status==SYNKIT_FINISHED||
       (stream&&statistics->reason==SYNKIT_STOP_CALLBACK))) {
      const int reasons[]={0,SYNKIT_STOP_TIME,SYNKIT_STOP_STATES,SYNKIT_STOP_CLASSES,
                           SYNKIT_STOP_CALLBACK,SYNKIT_STOP_CANCELLED};
      statistics->status=SYNKIT_LIMIT;statistics->reason=reasons[class_statistics->reason];
    }
    return 0;
  } catch(...) {return -1;}
}
} // namespace

extern "C" int synkit_mapper_solve_classified(
    const SynkitMapperProblem *p,const SynkitMapperRequest *request,
    const SynkitClassProblem *classes,int64_t max_canonical_nodes,
    SynkitClassCallback callback,void *userdata,SynkitMapperStatistics *statistics,
    SynkitClassStatistics *class_statistics) {
  return solve_classified_impl(p,request,nullptr,classes,max_canonical_nodes,
      callback,userdata,statistics,class_statistics,false,false);
}

extern "C" int synkit_mapper_solve_domains_classified(
    const SynkitMapperProblem *p,const SynkitMapperRequest *request,const uint8_t *allowed,
    const SynkitClassProblem *classes,int64_t max_canonical_nodes,
    SynkitClassCallback callback,void *userdata,SynkitMapperStatistics *statistics,
    SynkitClassStatistics *class_statistics) {
  if(!allowed) return -1;
  return solve_classified_impl(p,request,allowed,classes,max_canonical_nodes,
      callback,userdata,statistics,class_statistics,false,false);
}

extern "C" int synkit_mapper_solve_classified_updates(
    const SynkitMapperProblem *p,const SynkitMapperRequest *request,
    const SynkitClassProblem *classes,int64_t max_canonical_nodes,
    SynkitClassCallback callback,void *userdata,SynkitMapperStatistics *statistics,
    SynkitClassStatistics *class_statistics) {
  return solve_classified_impl(p,request,nullptr,classes,max_canonical_nodes,
      callback,userdata,statistics,class_statistics,true,false);
}

extern "C" int synkit_mapper_solve_domains_classified_updates(
    const SynkitMapperProblem *p,const SynkitMapperRequest *request,const uint8_t *allowed,
    const SynkitClassProblem *classes,int64_t max_canonical_nodes,
    SynkitClassCallback callback,void *userdata,SynkitMapperStatistics *statistics,
    SynkitClassStatistics *class_statistics) {
  if(!allowed) return -1;
  return solve_classified_impl(p,request,allowed,classes,max_canonical_nodes,
      callback,userdata,statistics,class_statistics,true,false);
}

extern "C" int synkit_mapper_solve_classified_ex(
    const SynkitMapperProblem *p,const SynkitMapperRequest *request,const uint8_t *allowed,
    const SynkitClassProblem *classes,int64_t max_canonical_nodes,
    SynkitClassCallback callback,void *userdata,SynkitMapperStatistics *statistics,
    SynkitClassStatistics *class_statistics,int32_t updates,int32_t packed_identifier) {
  if((updates!=0&&updates!=1)||(packed_identifier!=0&&packed_identifier!=1)) return -1;
  return solve_classified_impl(p,request,allowed,classes,max_canonical_nodes,
      callback,userdata,statistics,class_statistics,updates!=0,packed_identifier!=0);
}
