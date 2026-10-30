#include "formulation.h"
#include "seed.h"
#include "synkit_mapper/core/runtime.h"
#include "synkit_mapper/solvers/solvers.h"
#include "weighted_cnf.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#ifndef SYNKIT_MAPPER_OPENWBO_EXECUTABLE
#define SYNKIT_MAPPER_OPENWBO_EXECUTABLE ""
#endif

#ifndef SYNKIT_MAPPER_EVALMAXSAT_EXECUTABLE
#define SYNKIT_MAPPER_EVALMAXSAT_EXECUTABLE ""
#endif

namespace synkit_mapper {
namespace {
struct RunMetadata {
  int64_t products = 0, constraints = 0, formulation_bytes = 0, cnf_bytes = 0,
          cnf_variables = 0, cnf_clauses = 0, seed_cost = -1,
          engine_peak_rss = -1;
  double seed_seconds = 0;
};
thread_local RunMetadata metadata;
bool modern_engine() {
  return std::strlen(SYNKIT_MAPPER_EVALMAXSAT_EXECUTABLE) != 0;
}
const char *engine_path() {
  return modern_engine() ? SYNKIT_MAPPER_EVALMAXSAT_EXECUTABLE
                         : SYNKIT_MAPPER_OPENWBO_EXECUTABLE;
}
using solvers::detail::Formulation;
using solvers::detail::kInfinity;
using solvers::detail::Terms;
struct PhaseTimer {
  double &total;
  Clock::time_point start = Clock::now();
  ~PhaseTimer() { total += elapsed(start); }
};
#if defined(__unix__) || defined(__APPLE__)
struct TemporaryFiles {
  std::string directory, input, output;
  TemporaryFiles() {
    auto pattern =
        (std::filesystem::temp_directory_path() / "synkit-maxsat-XXXXXX")
            .string();
    std::vector<char> name(pattern.begin(), pattern.end());
    name.push_back('\0');
    char *created = ::mkdtemp(name.data());
    if (!created)
      throw std::runtime_error("Cannot create MaxSAT temporary directory");
    directory = created;
    input = directory + (modern_engine() ? "/mapping.wcnf" : "/mapping.opb");
    output = directory + "/solver.log";
  }
  ~TemporaryFiles() {
    ::unlink(input.c_str());
    ::unlink(output.c_str());
    ::rmdir(directory.c_str());
  }
};
pid_t wait_child(pid_t pid, int *status, int options) {
#if defined(__linux__) || defined(__APPLE__)
  struct rusage usage{};
  pid_t observed = ::wait4(pid, status, options, &usage);
  if (observed == pid) {
#if defined(__APPLE__)
    int64_t bytes = usage.ru_maxrss;
#else
    int64_t bytes = static_cast<int64_t>(usage.ru_maxrss) * 1024;
#endif
    metadata.engine_peak_rss = std::max(metadata.engine_peak_rss, bytes);
  }
  return observed;
#else
  return ::waitpid(pid, status, options);
#endif
}
struct Child {
  pid_t pid = -1;
  ~Child() {
    if (pid > 0) {
      ::kill(pid, SIGKILL);
      int status;
      while (wait_child(pid, &status, 0) < 0 && errno == EINTR) {
      }
    }
  }
};
struct Response {
  enum State { Unknown, Feasible, Optimum, Infeasible } state = Unknown;
  std::vector<int8_t> values;
};

Response run_openwbo(const TemporaryFiles &files, int variables,
                     Budget &budget) {
  budget.check();
  int output = ::open(files.output.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
  if (output < 0)
    throw std::runtime_error("Cannot open MaxSAT output file");
  const char *executable = engine_path();
  const bool modern = modern_engine();
  Child child;
  child.pid = ::fork();
  if (child.pid == 0) {
    // Only async-signal-safe calls between fork and exec (Python may have
    // other threads). No shell, interpolation, or inherited solver state.
    if (::dup2(output, STDOUT_FILENO) < 0 || ::dup2(output, STDERR_FILENO) < 0)
      ::_exit(126);
    ::close(output);
    if (modern)
      ::execl(executable, executable, "--old", files.input.c_str(),
              static_cast<char *>(nullptr));
    else
      ::execl(executable, executable, "-formula=1", "-algorithm=4",
              "-verbosity=0", "-print-model", files.input.c_str(),
              static_cast<char *>(nullptr));
    ::_exit(127);
  }
  ::close(output);
  if (child.pid < 0)
    throw std::runtime_error("Cannot start native Open-WBO engine");
  int status = 0;
  while (true) {
    budget.check();
    pid_t observed = wait_child(child.pid, &status, WNOHANG);
    if (observed == child.pid) {
      child.pid = -1;
      break;
    }
    if (observed < 0 && errno != EINTR)
      throw std::runtime_error("Cannot wait for native Open-WBO engine");
    ::poll(nullptr, 0,
           5); // Cancellation/deadline polling; not a solver worker.
  }
  if (!WIFEXITED(status) ||
      (WEXITSTATUS(status) != 0 && WEXITSTATUS(status) != 10 &&
       WEXITSTATUS(status) != 20 && WEXITSTATUS(status) != 30)) {
    std::ifstream failed(files.output);
    std::string diagnostic((std::istreambuf_iterator<char>(failed)), {});
    if (diagnostic.size() > 2000)
      diagnostic = diagnostic.substr(diagnostic.size() - 2000);
    throw std::runtime_error("Native Open-WBO failed (wait status " +
                             std::to_string(status) + "): " + diagnostic);
  }
  Response response;
  response.values.resize(variables, -1);
  std::ifstream stream(files.output);
  std::string line;
  while (std::getline(stream, line)) {
    if (line == "s OPTIMUM FOUND")
      response.state = Response::Optimum;
    else if (line == "s UNSATISFIABLE")
      response.state = Response::Infeasible;
    else if (line == "s SATISFIABLE")
      response.state = Response::Feasible;
    else if (line.rfind("v ", 0) == 0) {
      std::istringstream literals(line.substr(2));
      std::string literal;
      size_t parsed = 0;
      while (literals >> literal) {
        if (!(++parsed % 1024))
          budget.check();
        bool positive = literal[0] != '-';
        size_t offset = positive ? 0 : 1;
        if (offset >= literal.size() || (!modern && literal[offset] != 'x'))
          throw std::runtime_error("Unexpected Open-WBO PB model literal");
        size_t prefix = offset + (modern ? 0 : 1), used = 0;
        long long variable = std::stoll(literal.substr(prefix), &used);
        if (used != literal.size() - prefix || variable < 1 ||
            variable > variables)
          throw std::runtime_error("Invalid Open-WBO PB variable identifier");
        response.values[variable - 1] = positive;
      }
    }
  }
  if (!stream.eof())
    throw std::runtime_error("Cannot read Open-WBO output");
  if (response.state == Response::Unknown)
    throw std::runtime_error(
        "Open-WBO terminated without a certified solver status");
  budget.check();
  return response;
}

void expression(std::ostream &out, const Terms &terms, Budget &budget) {
  size_t written = 0;
  for (auto [variable, coefficient] : terms) {
    if (!(++written % 1024))
      budget.check();
    out << (coefficient >= 0 ? " +" : " ") << coefficient << " x"
        << variable + 1;
  }
}
void row(std::ostream &out, const Terms &terms, int64_t lower, int64_t upper,
         Budget &budget) {
  if (lower == upper) {
    expression(out, terms, budget);
    out << " = " << lower << " ;\n";
  } else {
    if (lower > -kInfinity) {
      expression(out, terms, budget);
      out << " >= " << lower << " ;\n";
    }
    if (upper < kInfinity) {
      expression(out, terms, budget);
      out << " <= " << upper << " ;\n";
    }
  }
}

// The shared sparse product model is reused, except the integer cost variable
// becomes one forced Boolean z. The chemical objective is C*z - sum(w*y)
// with z=1. In the satisfaction phase z=0 and actual cost is constrained by
// sum(w*y)+z == C-target (or >= for a ball). Each y is an exact conjunction.
void write_problem(const TemporaryFiles &files, const Formulation &f,
                   bool optimize, int query, int64_t target,
                   const std::vector<std::vector<int32_t>> &blocked,
                   int64_t seed_cost, Budget &budget) {
  std::ofstream stream(files.input, std::ios::binary | std::ios::trunc);
  if (!stream)
    throw std::runtime_error("Cannot write native MaxSAT input");
  stream << "* Native sparse full-space atom assignment, doubled CD\nmin:";
  const auto &cost_terms = f.rows.back().terms;
  if (optimize && f.maximum) {
    stream << " +" << f.maximum << " x" << f.cost + 1;
    size_t written = 0;
    for (auto [variable, coefficient] : cost_terms) {
      if (!(++written % 1024))
        budget.check();
      if (variable != f.cost)
        stream << " -" << coefficient << " x" << variable + 1;
    }
  } else
    stream << " +1 x" << f.cost + 1;
  stream << " ;\n";
  row(stream, {{f.cost, 1}}, optimize && f.maximum ? 1 : 0,
      optimize && f.maximum ? 1 : 0, budget);
  for (size_t i = 0; i + 1 < f.rows.size(); ++i) {
    if (!(i % 1024))
      budget.check();
    row(stream, f.rows[i].terms, f.rows[i].lower, f.rows[i].upper, budget);
  }
  if (optimize && seed_cost >= 0 && f.maximum > seed_cost) {
    Terms overlap;
    for (auto term : cost_terms)
      if (term.first != f.cost)
        overlap.push_back(term);
    row(stream, overlap, f.maximum - seed_cost, kInfinity, budget);
  }
  if (!optimize) {
    int64_t bound = f.maximum - std::min(target, f.maximum);
    row(stream, cost_terms, bound,
        query == SYNKIT_AT_MOST_DISTANCE ? kInfinity : bound, budget);
  }
  for (size_t b = 0; b < blocked.size(); ++b) {
    if (!(b % 256))
      budget.check();
    Terms nogood;
    for (int i = 0; i < f.n; ++i)
      nogood.emplace_back(f.x[i * f.n + blocked[b][i]], 1);
    row(stream, nogood, -kInfinity, f.n - 1, budget);
  }
  // Upstream maps the file read-only, uses strlen(), and incorrectly attempts
  // to write if its length differs from st_size. Keep no interior NUL and
  // ensure kernel zero-padding exists in the final mmap page.
  const auto page_size = ::sysconf(_SC_PAGESIZE);
  if (page_size > 0 && static_cast<int64_t>(stream.tellp()) % page_size == 0)
    stream.put('\n');
  stream.close();
  if (!stream)
    throw std::runtime_error("Failed to complete MaxSAT input file");
}

std::pair<std::vector<int32_t>, int64_t> project(const Problem &problem,
                                                 const Formulation &f,
                                                 const Response &response,
                                                 Budget &budget) {
  std::vector<int32_t> mapping(problem.n, -1);
  for (int i = 0; i < problem.n; ++i)
    for (int j = 0; j < problem.n; ++j) {
      int x = f.x[i * problem.n + j];
      if (x >= 0) {
        if (response.values[x] < 0)
          throw std::runtime_error("Open-WBO omitted an atom variable");
        if (response.values[x])
          mapping[i] = j;
      }
    }
  int64_t cost = mapping_cost2(problem, mapping);
  int64_t actual = f.maximum;
  size_t checked = 0;
  for (auto [variable, coefficient] : f.rows.back().terms)
    if (variable != f.cost) {
      if (!(++checked % 1024))
        budget.check();
      if (response.values[variable] < 0)
        throw std::runtime_error("Open-WBO omitted a product variable");
      actual -= coefficient * response.values[variable];
    }
  if (cost != actual)
    throw std::runtime_error(
        "Open-WBO cost disagrees with independent rescoring");
  return {std::move(mapping), cost};
}
#endif
} // namespace

const char *maxsat_solver_version() {
  return modern_engine()
             ? "EvalMaxSAT CLI, OLL/CaDiCaL, external native process"
             : "Open-WBO 2.1, OLL/Glucose4.1, external native process";
}
const char *maxsat_run_metadata() {
  static thread_local std::string value;
  std::ostringstream out;
  out << "{\"formulation\":\""
      << (modern_engine() ? "exact_weighted_cnf_seed_v2"
                          : "sparse_pb_rlt_seed_v2")
      << "\",\"engine\":\"" << (modern_engine() ? "EvalMaxSAT" : "Open-WBO")
      << "\",\"product_variables\":" << metadata.products
      << ",\"model_constraints\":" << metadata.constraints
      << ",\"accounted_formulation_bytes\":" << metadata.formulation_bytes
      << ",\"accounted_cnf_bytes\":" << metadata.cnf_bytes
      << ",\"cnf_variables\":" << metadata.cnf_variables
      << ",\"cnf_clauses\":" << metadata.cnf_clauses
      << ",\"seed_cost2\":" << metadata.seed_cost
      << ",\"seed_seconds\":" << metadata.seed_seconds
      << ",\"engine_peak_rss_bytes\":" << metadata.engine_peak_rss
      << ",\"engine_peak_rss_scope\":\"child_process_lifetime_including_fork_"
         "exec\""
      << ",\"cnf_reused_for_enumeration\":"
      << (modern_engine() ? "true" : "false")
      << ",\"seed_scope\":\"input_only_feasible_incumbent_bound\"}";
  value = out.str();
  return value.c_str();
}

int solve_maxsat(const Problem &problem, const Request &request,
                 Callback callback, void *userdata, Statistics &stats) {
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
#if defined(__unix__) || defined(__APPLE__)
  if (!std::strlen(engine_path()))
    return unsupported(stats);
  if (::access(engine_path(), X_OK))
    throw std::runtime_error("Configured Open-WBO executable is unavailable");
  stats.labeled_count = 0;
  stats.peak_frontier = stats.peak_frontier_bytes = -1;
  try {
    budget.check();
    Formulation f;
    {
      PhaseTimer timer{stats.model_seconds};
      f = solvers::detail::formulate(problem.n, problem.a2, problem.b2,
                                     problem.types_r, problem.types_p,
                                     [&] { budget.check(); });
    }
    if (request.query == SYNKIT_EXACT_DISTANCE && request.target2 > f.maximum) {
      stats.search_complete = 1;
      return 0;
    }
    metadata.products = f.products.size();
    metadata.constraints = f.rows.size();
    metadata.formulation_bytes = f.accounted_bytes();
    {
      PhaseTimer timer{stats.model_seconds};
      PhaseTimer seed_timer{metadata.seed_seconds};
      auto seed = solvers::detail::feasible_seed(problem, budget);
      if (!seed.empty())
        metadata.seed_cost = stats.incumbent2 = mapping_cost2(problem, seed);
    }
    TemporaryFiles files;
    std::vector<std::vector<int32_t>> blocked;
    std::unique_ptr<solvers::detail::WeightedCnf> satisfaction_cnf;
    size_t encoded_blocks = 0;
    auto run = [&](bool optimize, int64_t target) {
      int variables = f.variables;
      {
        PhaseTimer timer{stats.model_seconds};
        if (modern_engine()) {
          std::unique_ptr<solvers::detail::WeightedCnf> optimization_cnf;
          if (optimize)
            optimization_cnf =
                std::make_unique<solvers::detail::WeightedCnf>(f, budget);
          bool fresh = !satisfaction_cnf;
          if (!optimize && fresh)
            satisfaction_cnf =
                std::make_unique<solvers::detail::WeightedCnf>(f, budget);
          auto &cnf = optimize ? *optimization_cnf : *satisfaction_cnf;
          if (optimize) {
            for (const auto &product : f.products)
              cnf.soft.emplace_back(product.benefit, product.variable + 1);
            if (metadata.seed_cost >= 0 && f.maximum > metadata.seed_cost) {
              cnf.build_sum(f);
              cnf.bound(f.maximum - metadata.seed_cost, true);
            }
          } else if (fresh) {
            cnf.build_sum(f);
            cnf.bound(f.maximum - std::min(target, f.maximum), true);
            if (request.query != SYNKIT_AT_MOST_DISTANCE)
              cnf.bound(f.maximum - target, false);
          }
          while (!optimize && encoded_blocks < blocked.size()) {
            budget.check();
            const auto &mapping = blocked[encoded_blocks++];
            std::vector<int> nogood;
            for (int i = 0; i < f.n; ++i)
              nogood.push_back(-f.x[i * f.n + mapping[i]] - 1);
            cnf.add(std::move(nogood));
          }
          variables = cnf.variables;
          metadata.cnf_variables = variables;
          metadata.cnf_clauses = cnf.hard.size() + cnf.soft.size();
          metadata.cnf_bytes =
              std::max(metadata.cnf_bytes, cnf.accounted_bytes());
          cnf.write(files.input);
        } else
          write_problem(files, f, optimize, request.query, target, blocked,
                        metadata.seed_cost, budget);
      }
      PhaseTimer timer{optimize ? stats.optimization_seconds
                                : stats.enumeration_seconds};
      return run_openwbo(files, variables, budget);
    };
    int64_t target = request.target2;
    if (request.query == SYNKIT_MINIMUM_CLASSES ||
        request.query == SYNKIT_MINIMUM_PROOF) {
      auto response = run(true, 0);
      if (response.state == Response::Infeasible) {
        stats.search_complete = 1;
        return 0;
      }
      auto [mapping, cost] = project(problem, f, response, budget);
      stats.incumbent2 = cost;
      if (response.state != Response::Optimum) {
        stopped(stats,
                cancelled(request) ? SYNKIT_STOP_CANCELLED : SYNKIT_STOP_TIME);
        return 0;
      }
      stats.minimum_proven = 1;
      target = stats.minimum2 = stats.lower_bound2 = cost;
      if (request.query == SYNKIT_MINIMUM_PROOF) {
        stats.search_complete = 1;
        return 0;
      }
    }
    while (true) {
      auto response = run(false, target);
      if (response.state == Response::Infeasible) {
        stats.search_complete = 1;
        return 0;
      }
      auto [mapping, cost] = project(problem, f, response, budget);
      if ((request.query == SYNKIT_AT_MOST_DISTANCE && cost > target) ||
          (request.query != SYNKIT_AT_MOST_DISTANCE && cost != target))
        throw std::runtime_error(
            "Open-WBO emitted a mapping outside the requested shell");
      budget.check();
      ++stats.candidate_count;
      if (stats.incumbent2 < 0 || cost < stats.incumbent2)
        stats.incumbent2 = cost;
      ++stats.leaves;
      stats.labeled_count = stats.candidate_count;
      int stop = 0;
      {
        PhaseTimer callback_timer{stats.callback_seconds};
        PhaseTimer enumeration_timer{stats.enumeration_seconds};
        if (callback)
          stop = callback(mapping.data(), cost, userdata);
      }
      if (stop) {
        stopped(stats, SYNKIT_STOP_CALLBACK);
        return 0;
      }
      budget.check();
      if (request.max_candidates >= 0 &&
          stats.candidate_count >= request.max_candidates) {
        stopped(stats, SYNKIT_STOP_CANDIDATES);
        return 0;
      }
      blocked.push_back(std::move(mapping));
    }
  } catch (const Stop &stop) {
    stopped(stats, stop.reason);
  }
#else
  (void)problem;
  (void)callback;
  (void)userdata;
  return unsupported(stats);
#endif
  return 0;
}
} // namespace synkit_mapper
