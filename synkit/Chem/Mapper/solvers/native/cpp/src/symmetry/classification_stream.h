
#ifndef SYNKIT_PRIVATE_CLASS_STREAM_H
#define SYNKIT_PRIVATE_CLASS_STREAM_H
#include "synkit_mapper/symmetry/classification.h"
namespace synkit_mapper::detail {
struct ClassStream;
ClassStream *class_stream_create(const SynkitClassProblem &,double,int64_t,
    int64_t,void *,SynkitClassCallback,void *,bool);
int class_stream_push(ClassStream *,const int32_t *,int64_t);
void class_stream_finish(ClassStream *,SynkitClassStatistics &);
void class_stream_delete(ClassStream *);
}
#endif
