#include "synkit_mapper/core/pmcd_witness.h"
#include "synkit_mapper/core/runtime.h"
#include <string>
#ifdef SYNKIT_MAPPER_WITH_MCES
#include "synkit_mapper/mces/mces.h"
#endif
#if defined(SYNKIT_MAPPER_WITH_ORTOOLS) || defined(SYNKIT_MAPPER_WITH_MAXSAT)
#include "synkit_mapper/solvers/solvers.h"
#endif

namespace {
thread_local std::string error;
int slap(const synkit_mapper::Problem &p, const synkit_mapper::Request &r,
         synkit_mapper::Callback callback, void *userdata,
         synkit_mapper::Statistics &s) {
  using namespace synkit_mapper;
  if (p.n > 256 || r.query != SYNKIT_MINIMUM_CLASSES || r.classify ||
      r.max_classes != -1 || r.max_frontier != -1 ||
      r.memory_limit_bytes != -1 || r.cancel_token)
    return unsupported(s);
  auto start = Clock::now();
  std::vector<int32_t> flags(p.n, 0);
  const int32_t *policy = r.slap_stop_flags ? r.slap_stop_flags : flags.data();
  SlapStatistics raw{};
  int result = synkit_slap_solve(
      p.n, p.a2, p.b2, p.types_r, p.types_p, policy, r.slap_max_fingerprints,
      r.seconds, r.max_states, r.max_candidates, callback, userdata, &raw);
  if (result)
    throw std::runtime_error(synkit_mapper_error());
  s.nodes = raw.states;
  s.leaves = raw.candidates;
  s.candidate_count = raw.candidates;
  s.labeled_count = raw.candidates;
  s.heuristic_finished = raw.completed;
  s.callback_seconds = raw.callback_seconds;
  s.enumeration_seconds = raw.elapsed_seconds;
  s.elapsed_seconds = elapsed(start);
  if (raw.best_proxy2 >= 0)
    s.incumbent2 = raw.best_proxy2 / 2;
  s.reason = SYNKIT_STOP_HEURISTIC;
  if (raw.reason) {
    const int reasons[] = {0, SYNKIT_STOP_TIME, SYNKIT_STOP_STATES,
                           SYNKIT_STOP_CANDIDATES, SYNKIT_STOP_CALLBACK};
    stopped(s, reasons[raw.reason]);
  }
  return 0;
}
} // namespace

extern "C" {
int synkit_mapper_core_abi(void) { return 1; }
const char *synkit_mapper_core_error(void) { return error.c_str(); }
const char *synkit_mapper_backend_version(int backend) {
  static thread_local std::string version;
  version = "dependency-free C++17";
  if (backend == SYNKIT_PABS)
    version = "PABS-SLAP (C++17)";
#ifdef SYNKIT_MAPPER_WITH_ORTOOLS
#if defined(__SIZEOF_INT128__)
  if (backend == SYNKIT_PABS)
    version = "PABS-SLAP (C++17; certified PDLP bounds enabled)";
#endif
  if (backend == SYNKIT_CP_SAT || backend == SYNKIT_SCIP ||
      backend == SYNKIT_KMILP)
    version = synkit_mapper::ortools_solver_version(backend);
#endif
#ifdef SYNKIT_MAPPER_WITH_MAXSAT
  if (backend == SYNKIT_MAXSAT)
    version = synkit_mapper::maxsat_solver_version();
#endif
  return version.c_str();
}
const char *synkit_mapper_backend_metadata(int backend) {
#ifdef SYNKIT_MAPPER_WITH_ORTOOLS
  if (backend == SYNKIT_CP_SAT || backend == SYNKIT_SCIP ||
      backend == SYNKIT_KMILP)
    return synkit_mapper::ortools_run_metadata();
#endif
#ifdef SYNKIT_MAPPER_WITH_MAXSAT
  if (backend == SYNKIT_MAXSAT)
    return synkit_mapper::maxsat_run_metadata();
#endif
#ifdef SYNKIT_MAPPER_WITH_MCES
  if (backend == SYNKIT_MCES)
    return synkit_mapper::mces_run_metadata();
#endif
  return "{}";
}
void *synkit_mapper_cancel_create(void) {
  return new synkit_mapper::Cancellation;
}
void synkit_mapper_cancel_request(void *token) {
  if (token)
    static_cast<synkit_mapper::Cancellation *>(token)->requested.store(true);
}
void synkit_mapper_cancel_free(void *token) {
  delete static_cast<synkit_mapper::Cancellation *>(token);
}
uint64_t synkit_mapper_capabilities(void) {
  uint64_t result = (uint64_t{1} << SYNKIT_SLAP) | (uint64_t{1} << SYNKIT_PABS);
#ifdef SYNKIT_MAPPER_WITH_ASSIGNMENT
  result |= (uint64_t{1} << SYNKIT_ABB) | (uint64_t{1} << SYNKIT_KBB);
#endif
#ifdef SYNKIT_MAPPER_WITH_ASTAR
  result |= uint64_t{1} << SYNKIT_ASTAR;
#endif
#ifdef SYNKIT_MAPPER_WITH_CPABS
  result |= uint64_t{1} << SYNKIT_CPABS;
#endif
#ifdef SYNKIT_MAPPER_WITH_SUPPORTING
  result |= (uint64_t{1} << SYNKIT_EDIT) | (uint64_t{1} << SYNKIT_DFS) |
            (uint64_t{1} << SYNKIT_SEPDP);
#endif
#ifdef SYNKIT_MAPPER_WITH_MCES
  result |= uint64_t{1} << SYNKIT_MCES;
#endif
#ifdef SYNKIT_MAPPER_WITH_ORTOOLS
  result |= (uint64_t{1} << SYNKIT_CP_SAT) | (uint64_t{1} << SYNKIT_SCIP) |
            (uint64_t{1} << SYNKIT_KMILP);
#endif
#ifdef SYNKIT_MAPPER_WITH_MAXSAT
  result |= uint64_t{1} << SYNKIT_MAXSAT;
#endif
  return result;
}
int synkit_mapper_solve(const SynkitMapperProblem *problem,
                        const SynkitMapperRequest *request,
                        SynkitMapperCallback callback, void *userdata,
                        SynkitMapperStatistics *statistics) {
  synkit_mapper_clear_pmcd_witness();
  using namespace synkit_mapper;
  error.clear();
  if (!problem || !request || !statistics) {
    error = "Missing common mapper ABI argument";
    return -1;
  }
  initialize(*statistics, *request);
  const auto start = Clock::now();
  try {
    validate(*problem, *request);
    Request adjusted = *request;
    if (adjusted.seconds >= 0)
      adjusted.seconds = std::max(0.0, adjusted.seconds - elapsed(start));
    int result = 0;
    if (cancelled(*request)) {
      stopped(*statistics, SYNKIT_STOP_CANCELLED);
      statistics->elapsed_seconds = elapsed(start);
      return 0;
    }
    switch (request->backend) {
    case SYNKIT_SLAP:
      result = slap(*problem, adjusted, callback, userdata, *statistics);
      break;
    case SYNKIT_PABS:
      result = solve_pabs(*problem, adjusted, callback, userdata, *statistics);
      break;
#ifdef SYNKIT_MAPPER_WITH_ASSIGNMENT
    case SYNKIT_ABB:
      result = solve_abb(*problem, adjusted, nullptr, callback, userdata,
                         *statistics);
      break;
#endif
#ifdef SYNKIT_MAPPER_WITH_ASTAR
    case SYNKIT_ASTAR:
      result = solve_astar(*problem, adjusted, callback, userdata, *statistics);
      break;
#endif
#ifdef SYNKIT_MAPPER_WITH_CPABS
    case SYNKIT_CPABS:
      result = solve_cpabs(*problem, adjusted, callback, userdata, *statistics);
      break;
#endif
#ifdef SYNKIT_MAPPER_WITH_SUPPORTING
    case SYNKIT_EDIT:
      result = solve_edit_support(*problem, adjusted, callback, userdata,
                                  *statistics);
      break;
    case SYNKIT_DFS:
      result = solve_dfs(*problem, adjusted, callback, userdata, *statistics);
      break;
    case SYNKIT_SEPDP:
      result = solve_sepdp(*problem, adjusted, callback, userdata, *statistics);
      break;
#endif
#ifdef SYNKIT_MAPPER_WITH_MCES
    case SYNKIT_MCES:
      result = solve_mces(*problem, adjusted, callback, userdata, *statistics);
      break;
#endif
#ifdef SYNKIT_MAPPER_WITH_ORTOOLS
    case SYNKIT_CP_SAT:
    case SYNKIT_SCIP:
      result =
          solve_ortools(*problem, adjusted, callback, userdata, *statistics);
      break;
#endif
#ifdef SYNKIT_MAPPER_WITH_MAXSAT
    case SYNKIT_MAXSAT:
      result =
          solve_maxsat(*problem, adjusted, callback, userdata, *statistics);
      break;
#endif
    default:
      result = unsupported(*statistics);
    }
    statistics->elapsed_seconds = elapsed(start);
    return result;
  } catch (const Stop &stop) {
    stopped(*statistics, stop.reason);
  } catch (const std::exception &exception) {
    statistics->status = SYNKIT_ERROR;
    error = exception.what();
    statistics->elapsed_seconds = elapsed(start);
    return -1;
  }
  statistics->elapsed_seconds = elapsed(start);
  return 0;
}
int synkit_mapper_solve_domains(const SynkitMapperProblem *problem,
                                const SynkitMapperRequest *request,
                                const uint8_t *allowed,
                                SynkitMapperCallback callback, void *userdata,
                                SynkitMapperStatistics *statistics) {
  synkit_mapper_clear_pmcd_witness();
  using namespace synkit_mapper;
  error.clear();
  if (!problem || !request || !statistics || !allowed) {
    error = "Missing kernel mapper ABI argument";
    return -1;
  }
  initialize(*statistics, *request);
  auto start = Clock::now();
  try {
    validate(*problem, *request);
    for (int i = 0; i < problem->n * problem->n; ++i)
      if (allowed[i] > 1)
        throw std::invalid_argument(
            "Kernel domains must contain only zero or one");
    Request adjusted = *request;
    if (adjusted.seconds >= 0)
      adjusted.seconds = std::max(0.0, adjusted.seconds - elapsed(start));
    int result = 0;
    switch (request->backend) {
#ifdef SYNKIT_MAPPER_WITH_ASSIGNMENT
    case SYNKIT_KBB:
      result = solve_abb(*problem, adjusted, allowed, callback, userdata,
                         *statistics);
      break;
#endif
#ifdef SYNKIT_MAPPER_WITH_ORTOOLS
    case SYNKIT_KMILP:
      result = solve_kernel_milp(*problem, adjusted, allowed, callback,
                                 userdata, *statistics);
      break;
#endif
    default:
      result = unsupported(*statistics);
    }
    statistics->elapsed_seconds = elapsed(start);
    return result;
  } catch (const Stop &stop) {
    stopped(*statistics, stop.reason);
  } catch (const std::exception &exception) {
    statistics->status = SYNKIT_ERROR;
    error = exception.what();
    statistics->elapsed_seconds = elapsed(start);
    return -1;
  }
  statistics->elapsed_seconds = elapsed(start);
  return 0;
}
}
