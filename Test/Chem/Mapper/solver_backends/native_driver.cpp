// Standalone adapter qualification bridge, deliberately separate from the
// production dispatcher/build. Production entry points and validation are used.
#include "synkit_mapper/core/runtime.h"
#include "synkit_mapper/mces/mces.h"
#include "synkit_mapper/solvers/solvers.h"
#include <exception>
#include <string>
namespace {
thread_local std::string error;
}
extern "C" {
int synkit_mapper_core_abi() { return SYNKIT_MAPPER_CORE_ABI; }
uint64_t synkit_mapper_capabilities() {
  return (1ULL << SYNKIT_CP_SAT) | (1ULL << SYNKIT_SCIP) |
         (1ULL << SYNKIT_MAXSAT) | (1ULL << SYNKIT_KMILP) |
         (1ULL << SYNKIT_MCES);
}
const char *synkit_mapper_core_error() { return error.c_str(); }
const char *synkit_mapper_backend_version(int backend) {
  if (backend == SYNKIT_MCES)
    return "weighted association graph clique v1";
  return backend == SYNKIT_MAXSAT
             ? synkit_mapper::maxsat_solver_version()
             : synkit_mapper::ortools_solver_version(backend);
}
const char *synkit_mapper_backend_metadata(int backend) {
  if (backend == SYNKIT_MCES)
    return synkit_mapper::mces_run_metadata();
  return backend == SYNKIT_MAXSAT ? synkit_mapper::maxsat_run_metadata()
                                  : synkit_mapper::ortools_run_metadata();
}
void *synkit_mapper_cancel_create() { return new synkit_mapper::Cancellation; }
void synkit_mapper_cancel_request(void *token) {
  if (token)
    static_cast<synkit_mapper::Cancellation *>(token)->requested = true;
}
void synkit_mapper_cancel_free(void *token) {
  delete static_cast<synkit_mapper::Cancellation *>(token);
}
int synkit_mapper_solve(const SynkitMapperProblem *p,
                        const SynkitMapperRequest *r,
                        SynkitMapperCallback callback, void *userdata,
                        SynkitMapperStatistics *s) {
  error.clear();
  try {
    if (!p || !r || !s)
      throw std::invalid_argument("Missing test bridge arguments");
    synkit_mapper::validate(*p, *r);
    if (r->backend == SYNKIT_MCES)
      return synkit_mapper::solve_mces(*p, *r, callback, userdata, *s);
    if (r->backend == SYNKIT_MAXSAT)
      return synkit_mapper::solve_maxsat(*p, *r, callback, userdata, *s);
    return synkit_mapper::solve_ortools(*p, *r, callback, userdata, *s);
  } catch (const std::exception &e) {
    error = e.what();
    return -1;
  }
}
int synkit_mapper_solve_domains(const SynkitMapperProblem *p,
                                const SynkitMapperRequest *r,
                                const uint8_t *allowed,
                                SynkitMapperCallback callback, void *data,
                                SynkitMapperStatistics *s) {
  error.clear();
  try {
    if (!p || !r || !s || !allowed)
      throw std::invalid_argument("Missing domain test bridge arguments");
    synkit_mapper::validate(*p, *r);
    for (int i = 0; i < p->n; ++i)
      for (int j = 0; j < p->n; ++j)
        if (allowed[i * p->n + j] > 1 ||
            (allowed[i * p->n + j] && p->types_r[i] != p->types_p[j]))
          throw std::invalid_argument("Invalid kernel domain mask");
    return synkit_mapper::solve_kernel_milp(*p, *r, allowed, callback, data,
                                            *s);
  } catch (const std::exception &e) {
    error = e.what();
    return -1;
  }
}
}
