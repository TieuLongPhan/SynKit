#ifndef SYNKIT_MAPPER_SOLVERS_H
#define SYNKIT_MAPPER_SOLVERS_H
#include "synkit_mapper/mapper.h"

// Optional production adapters. Numerical SCIP proofs use the engine's
// configured tolerances (zero requested relative/absolute optimality gap);
// they are not independently checked formal certificates.
namespace synkit_mapper {
const char *ortools_solver_version(int backend);
const char *maxsat_solver_version();
const char *ortools_run_metadata();
const char *maxsat_run_metadata();
int solve_kernel_milp(const Problem &, const Request &, const uint8_t *,
                      Callback, void *, Statistics &);
} // namespace synkit_mapper
#endif
