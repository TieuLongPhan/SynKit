#include "formulation.h"
#include "seed.h"
#include "synkit_mapper/core/runtime.h"
#include "synkit_mapper/solvers/solvers.h"
#include <sstream>

#include <cmath>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "ortools/base/version.h"
#include "ortools/linear_solver/linear_solver.h"
#include "ortools/linear_solver/linear_solver_callback.h"
#include "ortools/sat/cp_model.h"
#include "ortools/sat/cp_model_solver.h"
#include "ortools/util/time_limit.h"

namespace synkit_mapper {
namespace {
struct RunMetadata {
  int64_t assignments = 0, products = 0, constraints = 0, aggregate = 0,
          bytes = 0, seed_cost = -1;
  double seed_seconds = 0;
};
thread_local RunMetadata metadata;
using operations_research::Domain;
using operations_research::MPSolver;
using operations_research::MPVariable;
using solvers::detail::Formulation;
using solvers::detail::kInfinity;
namespace sat = operations_research::sat;
struct PhaseTimer {
  double &total;
  Clock::time_point start = Clock::now();
  ~PhaseTimer() { total += elapsed(start); }
};

bool emit(const std::vector<int32_t> &mapping, int64_t cost2,
          const Request &request, const Budget &budget, Callback callback,
          void *userdata, Statistics &stats) {
  ++stats.candidate_count;
  if (stats.incumbent2 < 0 || cost2 < stats.incumbent2)
    stats.incumbent2 = cost2;
  ++stats.leaves;
  stats.labeled_count = stats.candidate_count;
  int stop = 0;
  {
    PhaseTimer timer{stats.callback_seconds};
    if (callback)
      stop = callback(mapping.data(), cost2, userdata);
  }
  if (stop)
    stopped(stats, SYNKIT_STOP_CALLBACK);
  else if (budget.reason())
    stopped(stats, budget.reason());
  else if (request.max_candidates >= 0 &&
           stats.candidate_count >= request.max_candidates)
    stopped(stats, SYNKIT_STOP_CANDIDATES);
  return stats.reason != SYNKIT_STOP_NONE;
}

void cp_sat(const Problem &problem, const Request &request, Formulation &f,
            Budget &budget, const std::vector<int64_t> &seed, Callback callback,
            void *userdata, Statistics &stats) {
  sat::CpModelBuilder model;
  std::vector<sat::IntVar> variables;
  {
    PhaseTimer timer{stats.model_seconds};
    for (int i = 0; i < f.variables; ++i) {
      if (!(i % 1024))
        budget.check();
      variables.push_back(
          model.NewIntVar(Domain(0, i == f.cost ? f.maximum : 1)));
    }
    for (size_t i = 0; i < f.rows.size(); ++i) {
      if (!(i % 1024))
        budget.check();
      sat::LinearExpr expression;
      size_t terms = 0;
      for (auto [v, coefficient] : f.rows[i].terms) {
        if (!(++terms % 1024))
          budget.check();
        expression += coefficient * variables[v];
      }
      model.AddLinearConstraint(expression,
                                Domain(f.rows[i].lower, f.rows[i].upper));
    }
  }
  if (!seed.empty()) {
    PhaseTimer timer{stats.model_seconds};
    for (int i = 0; i < f.variables; ++i) {
      if (!(i % 1024))
        budget.check();
      model.AddHint(variables[i], seed[i]);
    }
    if (request.query == SYNKIT_MINIMUM_CLASSES ||
        request.query == SYNKIT_MINIMUM_PROOF)
      model.AddLessOrEqual(variables[f.cost], seed[f.cost]);
  }
  auto project = [&](const sat::CpSolverResponse &response) {
    std::vector<int32_t> mapping(problem.n, -1);
    for (int i = 0; i < problem.n; ++i)
      for (int j = 0; j < problem.n; ++j) {
        int x = f.x[i * problem.n + j];
        if (x >= 0 && sat::SolutionIntegerValue(response, variables[x]))
          mapping[i] = j;
      }
    int64_t cost = mapping_cost2(problem, mapping);
    if (cost != sat::SolutionIntegerValue(response, variables[f.cost]))
      throw std::runtime_error(
          "CP-SAT cost disagrees with independent rescoring");
    return std::make_pair(std::move(mapping), cost);
  };
  auto run = [&](bool enumerate) {
    budget.check();
    sat::Model solver;
    sat::SatParameters parameters;
    parameters.set_num_search_workers(1);
    parameters.set_random_seed(0);
    parameters.set_max_time_in_seconds(budget.remaining());
    parameters.set_enumerate_all_solutions(enumerate);
    solver.Add(sat::NewSatParameters(parameters));
    // Register the caller token as the secondary, read-only cancellation
    // signal. CP-SAT writes its primary external flag when stopping internal
    // subsolvers, so sharing that flag would wrongly cancel later phases.
    if (request.cancel_token)
      solver.GetOrCreate<operations_research::TimeLimit>()
          ->RegisterSecondaryExternalBooleanAsLimit(
              &static_cast<Cancellation *>(request.cancel_token)->requested);
    std::exception_ptr callback_error;
    if (enumerate)
      solver.Add(sat::NewFeasibleSolutionObserver(
          [&](const sat::CpSolverResponse &response) {
            try {
              budget.check();
              auto [mapping, cost] = project(response);
              if (emit(mapping, cost, request, budget, callback, userdata,
                       stats))
                sat::StopSearch(&solver);
            } catch (...) {
              callback_error = std::current_exception();
              sat::StopSearch(&solver);
            }
          }));
    sat::CpSolverResponse response;
    {
      PhaseTimer timer{enumerate ? stats.enumeration_seconds
                                 : stats.optimization_seconds};
      response = sat::SolveCpModel(model.Build(), &solver);
    }
    stats.nodes += response.num_branches();
    if (callback_error)
      std::rethrow_exception(callback_error);
    budget.check();
    if (response.status() == sat::CpSolverStatus::MODEL_INVALID)
      throw std::runtime_error("Invalid CP-SAT model: " +
                               response.solution_info());
    return response;
  };
  int64_t target = request.target2;
  if (request.query == SYNKIT_MINIMUM_CLASSES ||
      request.query == SYNKIT_MINIMUM_PROOF) {
    model.Minimize(variables[f.cost]);
    auto response = run(false);
    if (response.status() == sat::CpSolverStatus::OPTIMAL ||
        response.status() == sat::CpSolverStatus::FEASIBLE) {
      stats.incumbent2 = project(response).second;
      stats.lower_bound2 =
          static_cast<int64_t>(std::floor(response.best_objective_bound()));
    }
    if (response.status() == sat::CpSolverStatus::INFEASIBLE) {
      stats.search_complete = 1;
      return;
    }
    if (response.status() != sat::CpSolverStatus::OPTIMAL) {
      stopped(stats,
              cancelled(request) ? SYNKIT_STOP_CANCELLED : SYNKIT_STOP_TIME);
      return;
    }
    stats.minimum_proven = 1;
    target = stats.minimum2 = stats.lower_bound2 = stats.incumbent2;
    if (request.query == SYNKIT_MINIMUM_PROOF) {
      stats.search_complete = 1;
      return;
    }
    model.ClearObjective();
  }
  budget.check();
  if (request.query == SYNKIT_AT_MOST_DISTANCE)
    model.AddLessOrEqual(variables[f.cost], std::min(target, f.maximum));
  else
    model.AddEquality(variables[f.cost], target);
  auto response = run(true);
  if (stats.reason)
    return;
  if (response.status() == sat::CpSolverStatus::OPTIMAL ||
      response.status() == sat::CpSolverStatus::INFEASIBLE)
    stats.search_complete = 1;
  else
    stopped(stats,
            cancelled(request) ? SYNKIT_STOP_CANCELLED : SYNKIT_STOP_TIME);
}

class ScipCancellation final : public operations_research::MPCallback {
public:
  ScipCancellation(MPSolver &solver, const Budget &budget)
      : MPCallback(false, false), solver_(solver), budget_(budget) {}
  void RunCallback(operations_research::MPCallbackContext *) override {
    // This callback executes in the one solver thread. SCIP's MPSolver
    // InterruptSolve must not be called from an asynchronous watcher thread.
    if (budget_.reason())
      solver_.InterruptSolve();
  }

private:
  MPSolver &solver_;
  const Budget &budget_;
};

void scip(const Problem &problem, const Request &request, Formulation &f,
          Budget &budget, const std::vector<int64_t> &seed, Callback callback,
          void *userdata, Statistics &stats) {
  std::unique_ptr<MPSolver> solver;
  std::vector<MPVariable *> variables;
  {
    PhaseTimer timer{stats.model_seconds};
    solver.reset(MPSolver::CreateSolver("SCIP"));
    if (!solver)
      throw std::runtime_error("OR-Tools distribution has no SCIP engine");
    if (!solver->SetNumThreads(1).ok())
      throw std::runtime_error("Cannot configure one SCIP worker");
    if (!solver->SetSolverSpecificParametersAsString(
            "limits/gap = 0\nlimits/absgap = 0\nnumerics/feastol = 1e-9\n"
            "randomization/randomseedshift = 0"))
      throw std::runtime_error(
          "Cannot configure SCIP gap/tolerance parameters");
    for (int i = 0; i < f.variables; ++i) {
      if (!(i % 1024))
        budget.check();
      variables.push_back(
          solver->MakeIntVar(0, i == f.cost ? f.maximum : 1, ""));
    }
    for (size_t i = 0; i < f.rows.size(); ++i) {
      if (!(i % 1024))
        budget.check();
      const auto &r = f.rows[i];
      auto *row = solver->MakeRowConstraint(
          r.lower <= -kInfinity ? -solver->infinity() : r.lower,
          r.upper >= kInfinity ? solver->infinity() : r.upper);
      size_t terms = 0;
      for (auto [v, coefficient] : r.terms) {
        if (!(++terms % 1024))
          budget.check();
        row->SetCoefficient(variables[v], static_cast<double>(coefficient));
      }
    }
  }
  if (!seed.empty()) {
    PhaseTimer timer{stats.model_seconds};
    std::vector<std::pair<const MPVariable *, double>> hint;
    hint.reserve(variables.size());
    for (size_t i = 0; i < variables.size(); ++i) {
      if (!(i % 1024))
        budget.check();
      hint.emplace_back(variables[i], static_cast<double>(seed[i]));
    }
    solver->SetHint(hint);
    if (request.query == SYNKIT_MINIMUM_CLASSES ||
        request.query == SYNKIT_MINIMUM_PROOF)
      variables[f.cost]->SetUB(seed[f.cost]);
  }
  ScipCancellation cancellation(*solver, budget);
  if (request.cancel_token)
    solver->SetCallback(&cancellation);
  auto project = [&] {
    std::vector<int32_t> mapping(problem.n, -1);
    for (int i = 0; i < problem.n; ++i)
      for (int j = 0; j < problem.n; ++j) {
        int x = f.x[i * problem.n + j];
        if (x >= 0 && variables[x]->solution_value() > 0.5)
          mapping[i] = j;
      }
    int64_t cost = mapping_cost2(problem, mapping);
    if (std::abs(static_cast<double>(cost) -
                 variables[f.cost]->solution_value()) > 1e-5)
      throw std::runtime_error(
          "SCIP cost disagrees with independent rescoring");
    if (!solver->VerifySolution(1e-6, false))
      throw std::runtime_error(
          "SCIP returned a numerically invalid assignment");
    return std::make_pair(std::move(mapping), cost);
  };
  auto run = [&](bool enumeration) {
    budget.check();
    if (request.seconds >= 0)
      solver->set_time_limit(
          std::max(int64_t{1},
                   static_cast<int64_t>(std::ceil(1000 * budget.remaining()))));
    MPSolver::ResultStatus response;
    {
      PhaseTimer timer{enumeration ? stats.enumeration_seconds
                                   : stats.optimization_seconds};
      response = solver->Solve();
    }
    stats.nodes += std::max(int64_t{0}, solver->nodes());
    // OR-Tools maps SCIP USERINTERRUPT with no incumbent to ABNORMAL. A
    // requested cancellation/deadline is a resource stop, not engine failure.
    budget.check();
    if (response == MPSolver::ABNORMAL || response == MPSolver::MODEL_INVALID ||
        response == MPSolver::UNBOUNDED)
      throw std::runtime_error("SCIP model or solver failure");
    return response;
  };
  int64_t target = request.target2;
  if (request.query == SYNKIT_MINIMUM_CLASSES ||
      request.query == SYNKIT_MINIMUM_PROOF) {
    solver->MutableObjective()->SetCoefficient(variables[f.cost], 1);
    solver->MutableObjective()->SetMinimization();
    auto response = run(false);
    if (response == MPSolver::OPTIMAL || response == MPSolver::FEASIBLE)
      stats.incumbent2 = project().second;
    if (response == MPSolver::INFEASIBLE) {
      stats.search_complete = 1;
      return;
    }
    if (response != MPSolver::OPTIMAL) {
      stopped(stats,
              cancelled(request) ? SYNKIT_STOP_CANCELLED : SYNKIT_STOP_TIME);
      return;
    }
    stats.minimum_proven = 1;
    target = stats.minimum2 = stats.lower_bound2 = stats.incumbent2;
    if (request.query == SYNKIT_MINIMUM_PROOF) {
      stats.search_complete = 1;
      return;
    }
    solver->MutableObjective()->Clear();
  }
  if (request.query == SYNKIT_AT_MOST_DISTANCE)
    variables[f.cost]->SetUB(std::min(target, f.maximum));
  else
    variables[f.cost]->SetBounds(target, target);
  while (true) {
    auto response = run(true);
    if (response == MPSolver::INFEASIBLE) {
      stats.search_complete = 1;
      return;
    }
    if (response != MPSolver::OPTIMAL && response != MPSolver::FEASIBLE) {
      stopped(stats,
              cancelled(request) ? SYNKIT_STOP_CANCELLED : SYNKIT_STOP_TIME);
      return;
    }
    budget.check();
    auto [mapping, cost] = project();
    if (emit(mapping, cost, request, budget, callback, userdata, stats))
      return;
    // No-goods exclude the entire projected mapping, not auxiliary valuations.
    PhaseTimer timer{stats.model_seconds};
    auto *block = solver->MakeRowConstraint(-solver->infinity(), problem.n - 1);
    for (int i = 0; i < problem.n; ++i)
      block->SetCoefficient(variables[f.x[i * problem.n + mapping[i]]], 1);
  }
}
} // namespace

const char *ortools_solver_version(int backend) {
  static const std::string cp =
      "OR-Tools " + operations_research::OrToolsVersionString();
  static const std::string scip_version = [] {
    auto solver = std::unique_ptr<MPSolver>(MPSolver::CreateSolver("SCIP"));
    return solver ? solver->SolverVersion() : std::string("SCIP unavailable");
  }();
  return backend == SYNKIT_SCIP ? scip_version.c_str() : cp.c_str();
}

const char *ortools_run_metadata() {
  static thread_local std::string value;
  std::ostringstream out;
  out << "{\"formulation\":\"sparse_assignment_rlt_seed_v2\",\"assignment_"
         "variables\":"
      << metadata.assignments << ",\"product_variables\":" << metadata.products
      << ",\"model_constraints\":" << metadata.constraints
      << ",\"aggregate_constraints\":" << metadata.aggregate
      << ",\"accounted_formulation_bytes\":" << metadata.bytes
      << ",\"seed_cost2\":" << metadata.seed_cost
      << ",\"seed_seconds\":" << metadata.seed_seconds
      << ",\"solver_peak_memory_bytes\":-1,\"seed_scope\":\"input_only_"
         "feasible_incumbent\"}";
  value = out.str();
  return value.c_str();
}

int solve_ortools_domain(const Problem &problem, const Request &request,
                         const uint8_t *allowed, Callback callback,
                         void *userdata, Statistics &stats) {
  initialize(stats, request);
  metadata = {};
  Budget budget(request);
  struct Finish {
    Statistics &s;
    Budget &b;
    ~Finish() { s.elapsed_seconds = b.seconds(); }
  } finish{stats, budget};
  if (request.max_states != -1 || request.max_classes != -1 ||
      request.max_frontier != -1 || request.memory_limit_bytes != -1 ||
      request.symmetry || request.classify)
    return unsupported(stats);
  stats.labeled_count = 0;
  stats.peak_frontier = stats.peak_frontier_bytes = -1;
  try {
    budget.check();
    Formulation formulation;
    {
      PhaseTimer timer{stats.model_seconds};
      formulation = solvers::detail::formulate(
          problem.n, problem.a2, problem.b2, problem.types_r, problem.types_p,
          [&] { budget.check(); }, allowed);
    }
    metadata.products = formulation.products.size();
    metadata.assignments = formulation.variables - 1 - metadata.products;
    metadata.constraints = formulation.rows.size();
    metadata.aggregate = formulation.aggregate_constraints;
    metadata.bytes = formulation.accounted_bytes();
    std::vector<int64_t> seed;
    {
      PhaseTimer timer{stats.model_seconds};
      PhaseTimer seed_timer{metadata.seed_seconds};
      auto mapping = solvers::detail::feasible_seed(problem, budget, allowed);
      if (!mapping.empty()) {
        metadata.seed_cost = mapping_cost2(problem, mapping);
        stats.incumbent2 = metadata.seed_cost;
        seed = solvers::detail::seed_values(formulation, mapping,
                                            metadata.seed_cost);
      }
    }
    if (request.query == SYNKIT_EXACT_DISTANCE &&
        request.target2 > formulation.maximum) {
      stats.search_complete = 1;
      return 0;
    }
    if (request.backend == SYNKIT_CP_SAT)
      cp_sat(problem, request, formulation, budget, seed, callback, userdata,
             stats);
    else if (request.backend == SYNKIT_SCIP || request.backend == 9)
      scip(problem, request, formulation, budget, seed, callback, userdata,
           stats);
    else
      return unsupported(stats);
  } catch (const Stop &stop) {
    stopped(stats, stop.reason);
  }
  return 0;
}
int solve_ortools(const Problem &p, const Request &r, Callback cb, void *data,
                  Statistics &s) {
  if (r.backend == 9) {
    initialize(s, r);
    return unsupported(s);
  }
  return solve_ortools_domain(p, r, nullptr, cb, data, s);
}
int solve_kernel_milp(const Problem &p, const Request &r,
                      const uint8_t *allowed, Callback cb, void *data,
                      Statistics &s) {
  if (!allowed || r.backend != 9) {
    initialize(s, r);
    return unsupported(s);
  }
  return solve_ortools_domain(p, r, allowed, cb, data, s);
}
} // namespace synkit_mapper
