// Included inside sparse_root_lp only with OR-Tools and exact wide arithmetic.
// This numerical module returns untrusted ORIGINAL-index vectors. It has no
// authority to declare a lower bound, optimal map, or completed enumeration.
template <class Check>
bool make_initial_dual(const PreparedModel &model, std::vector<double> &proposal,
                       Check check) {
  const int n = model.n, m = model.variable_rows.size();
  proposal.clear();
  if (m != int(model.columns.size()) || model.rows.size() < size_t(2 * m))
    return false;
  std::vector<int64_t> twice_dual(model.rows.size(), 0);
  // Each overlap variable appears in four marginal inequalities. A multiplier
  // -max(overlap)/2 on each makes its original reduced cost nonnegative.
  for (size_t r = 0; r < model.rows.size(); ++r) {
    if (!(r % 128))
      check();
    const auto &row = model.rows[r];
    if (row.equality || row.rhs || row.terms.size() < 2)
      continue;
    const auto &last = row.terms.back();
    if (last.coefficient != -1 || last.column < 0 ||
        last.column >= model.assignment_variables)
      continue;
    int64_t largest = 0;
    bool marginal = true;
    for (size_t k = 0; k + 1 < row.terms.size(); ++k) {
      const auto &term = row.terms[k];
      if (term.coefficient != 1 || term.column < model.assignment_variables ||
          size_t(term.column) >= model.costs.size() ||
          model.costs[term.column] > 0 || model.costs[term.column] % 2) {
        marginal = false;
        break;
      }
      largest = std::max(largest, -model.costs[term.column] / 2);
    }
    if (marginal)
      twice_dual[r] = -largest;
  }
  auto reduced_costs = [&]() {
    std::vector<Wide> reduced;
    reduced.reserve(model.costs.size());
    for (int64_t value : model.costs)
      reduced.push_back(Wide(2) * value);
    size_t visited = 0;
    for (size_t r = 0; r < model.rows.size(); ++r)
      if (twice_dual[r])
        for (const auto &term : model.rows[r].terms) {
          if (!(visited++ % 1024))
            check();
          reduced[term.column] -= Wide(term.coefficient) * twice_dual[r];
        }
    return reduced;
  };
  auto reduced = reduced_costs();
  if (m) {
    const int safe_entry = INF / (8 * m) - 1;
    if (safe_entry < 1)
      return false;
    std::vector<Wide> shifts(m);
    Wide largest = 0;
    std::vector<int> position(n, -1);
    for (int j = 0; j < m; ++j)
      position[model.columns[j]] = j;
    for (int i = 0; i < m; ++i) {
      check();
      const int row = model.variable_rows[i];
      if (model.domains[row].empty())
        return false;
      shifts[i] = reduced[model.assignment[row * n + model.domains[row][0]]];
      for (int image : model.domains[row])
        shifts[i] = std::min(shifts[i], reduced[model.assignment[row * n + image]]);
      for (int image : model.domains[row])
        largest = std::max(largest, reduced[model.assignment[row * n + image]] - shifts[i]);
    }
    const Wide quantum = std::max(Wide(1), (largest + safe_entry - 1) / safe_entry);
    std::vector<int> cost(m * m, INF);
    for (int i = 0; i < m; ++i) {
      check();
      const int row = model.variable_rows[i];
      for (int image : model.domains[row]) {
        const Wide value = (reduced[model.assignment[row * n + image]] - shifts[i]) / quantum;
        if (value < 0 || value > safe_entry || position[image] < 0)
          return false;
        cost[i * m + position[image]] = int(value);
      }
    }
    Perm mapping, u, v;
    check();
    const int optimum = Solver::assignment(cost, m, mapping, u, v);
    check();
    if (optimum >= INF / 2 || mapping.size() != size_t(m) ||
        u.size() != size_t(m + 1) || v.size() != size_t(m + 1))
      return false;
    std::vector<bool> used(m, false);
    int64_t primal = 0, dual = 0;
    for (int i = 0; i < m; ++i) {
      const int j = mapping[i];
      if (j < 0 || j >= m || used[j] || cost[i * m + j] >= INF / 2)
        return false;
      used[j] = true;
      primal += cost[i * m + j];
      dual += int64_t(u[i + 1]) + v[i + 1];
      for (int k = 0; k < m; ++k)
        if (cost[i * m + k] < INF / 2 &&
            int64_t(cost[i * m + k]) - u[i + 1] - v[k + 1] < 0)
          return false;
      const Wide row_value = shifts[i] + quantum * u[i + 1];
      const Wide column_value = quantum * v[i + 1];
      // Stay inside the existing numerical-proposal clipping range.
      if (row_value < -2000000 || row_value > 2000000 ||
          column_value < -2000000 || column_value > 2000000)
        return false;
      twice_dual[i] = int64_t(row_value);
      twice_dual[m + i] = int64_t(column_value);
    }
    if (primal != optimum || dual != optimum)
      return false;
  }
  reduced = reduced_costs();
  for (Wide value : reduced)
    if (value < 0)
      return false;
  proposal.resize(twice_dual.size());
  for (size_t r = 0; r < twice_dual.size(); ++r) {
    if (!(r % 128))
      check();
    if ((!model.rows[r].equality && twice_dual[r] > 0) ||
        twice_dual[r] < -2000000 || twice_dual[r] > 2000000)
      return false;
    proposal[r] = 0.5 * twice_dual[r];
  }
  check();
  return true;
}

class PDLPProvider {
  std::unique_ptr<operations_research::pdlp::QuadraticProgram> qp_;
  std::vector<double> primal_, dual_;
  std::optional<operations_research::pdlp::PrimalAndDualSolution> initial_;

public:
  int reason = 0;
  int64_t iterations = 0;
  template <class Check>
  PDLPProvider(const std::vector<int64_t> &costs, const std::vector<Row> &rows,
               int64_t constant, Check check) {

    qp_ = std::make_unique<operations_research::pdlp::QuadraticProgram>(
        costs.size(), rows.size());
    qp_->objective_offset = constant;
    qp_->objective_scaling_factor = 1;
    qp_->variable_lower_bounds.setZero();
    qp_->variable_upper_bounds.setOnes();
    for (size_t j = 0; j < costs.size(); ++j) {
      if (!(j % 128))
        check();
      qp_->objective_vector[j] = costs[j];
    }
    std::vector<Eigen::Triplet<double, int64_t>> coefficients;
    size_t nonzeros = 0;
    for (const auto &row : rows)
      nonzeros += row.terms.size();
    coefficients.reserve(nonzeros);
    for (size_t r = 0; r < rows.size(); ++r) {
      if (!(r % 128))
        check();
      const auto &row = rows[r];
      qp_->constraint_lower_bounds[r] =
          row.equality ? row.rhs : -std::numeric_limits<double>::infinity();
      qp_->constraint_upper_bounds[r] = row.rhs;
      for (const auto &term : row.terms)
        coefficients.emplace_back(r, term.column, term.coefficient);
    }
    qp_->constraint_matrix.setFromTriplets(coefficients.begin(),
                                           coefficients.end());
    check();
  }

  template <class Check>
  void set_initial_primal(const std::vector<int64_t> &point, Check check,
                          const std::vector<double> *dual = nullptr) {
    if (!qp_ || point.size() != size_t(qp_->objective_vector.size()))
      return;
    operations_research::pdlp::PrimalAndDualSolution candidate;
    candidate.primal_solution.resize(point.size());
    candidate.dual_solution =
        Eigen::VectorXd::Zero(qp_->constraint_lower_bounds.size());
    for (size_t j = 0; j < point.size(); ++j) {
      if (!(j % 128))
        check();
      if (point[j] != 0 && point[j] != 1)
        return;
      candidate.primal_solution[j] = point[j];
    }
    if (dual && dual->size() == size_t(candidate.dual_solution.size()))
      for (size_t r = 0; r < dual->size(); ++r) {
        if (!(r % 128))
          check();
        if (!std::isfinite((*dual)[r]))
          return;
        candidate.dual_solution[r] = (*dual)[r];
      }
    check();
    initial_ = std::move(candidate);
  }

  void solve(double seconds,
             int64_t stop_target = std::numeric_limits<int64_t>::max(),
             int64_t cost_step = 0) {
    if (!qp_)
      return;
    operations_research::pdlp::PrimalDualHybridGradientParams parameters;
    parameters.set_num_threads(1);
    parameters.set_verbosity_level(0);
    parameters.set_record_iteration_stats(false);
    parameters.mutable_termination_criteria()->set_time_sec_limit(
        std::max(0.0, seconds));
    auto *optimality = parameters.mutable_termination_criteria()
                           ->mutable_simple_optimality_criteria();
    optimality->set_eps_optimal_absolute(1e-6);
    optimality->set_eps_optimal_relative(1e-6);
    // This scalar controls WHEN to request a vector, never WHAT is proved.
    // The returned original-index vector still passes certify_proposal.
    const bool target_available =
        cost_step > 0 && stop_target >= 0 &&
        stop_target != std::numeric_limits<int64_t>::max();
    const double trigger = double(stop_target) - double(cost_step) * 0.5;
    std::atomic<bool> interrupt{false};
    std::function<void(
        const operations_research::pdlp::IterationCallbackInfo &)>
        callback;
    if (target_available) {
      callback =
          [&](const operations_research::pdlp::IterationCallbackInfo &info) {
#ifdef SYNKIT_MAPPER_TEST_HOOKS
            ++harvest_trace[1];
#endif // SYNKIT_MAPPER_TEST_HOOKS
            if (info.iteration_type !=
                operations_research::pdlp::IterationType::kNormal)
              return;
            bool current_seen = false, average_seen = false, invalid = false;
            double current = std::numeric_limits<double>::infinity();
            double average = std::numeric_limits<double>::infinity();
            for (const auto &point :
                 info.iteration_stats.convergence_information()) {
              if (point.candidate_type() !=
                      operations_research::pdlp::POINT_TYPE_CURRENT_ITERATE &&
                  point.candidate_type() !=
                      operations_research::pdlp::POINT_TYPE_AVERAGE_ITERATE)
                continue;
              const bool is_current =
                  point.candidate_type() ==
                  operations_research::pdlp::POINT_TYPE_CURRENT_ITERATE;
              if (is_current)
                current_seen = true;
              else
                average_seen = true;
              if (!point.has_corrected_dual_objective()) {
                invalid = true;
                continue;
              }
              const double value = point.corrected_dual_objective();
              if (!std::isfinite(value)) {
                invalid = true;
                continue;
              }
              if (is_current)
                current = std::min(current, value);
              else
                average = std::min(average, value);
#ifdef SYNKIT_MAPPER_TEST_HOOKS
              const int64_t diagnostic = int64_t(
                  std::llround(std::max(-1e12, std::min(1e12, value)) * 1e6));
              harvest_trace[14] = diagnostic;
              harvest_trace[15] = std::max(harvest_trace[15], diagnostic);
#endif // SYNKIT_MAPPER_TEST_HOOKS
            }
#ifdef SYNKIT_MAPPER_TEST_HOOKS
            harvest_trace[10] =
                int(current_seen) + 2 * int(average_seen) + 4 * int(invalid);
#endif // SYNKIT_MAPPER_TEST_HOOKS
       // SAME callback iteration, both possible returned points.
       // Historical scalar maxima are diagnostics only and never trigger
       // interruption.
            if (current_seen && average_seen && !invalid &&
                current >= trigger && average >= trigger) {
#ifdef SYNKIT_MAPPER_TEST_HOOKS
              harvest_trace[2] = 1;
#endif // SYNKIT_MAPPER_TEST_HOOKS
              interrupt.store(true, std::memory_order_relaxed);
            }
          };
    }
#ifdef SYNKIT_MAPPER_TEST_HOOKS
    const auto started = Clock::now();
#endif // SYNKIT_MAPPER_TEST_HOOKS
    auto result = operations_research::pdlp::PrimalDualHybridGradient(
        std::move(*qp_), parameters, std::move(initial_),
        target_available ? &interrupt : nullptr, nullptr, callback);
#ifdef SYNKIT_MAPPER_TEST_HOOKS
    harvest_trace[5] = int64_t(elapsed(started) * 1e6);
#endif // SYNKIT_MAPPER_TEST_HOOKS
    qp_.reset();
    reason = int(result.solve_log.termination_reason());
    iterations = result.solve_log.iteration_count();
#ifdef SYNKIT_MAPPER_TEST_HOOKS
    harvest_trace[3] = reason;
    harvest_trace[4] = iterations;
#endif // SYNKIT_MAPPER_TEST_HOOKS
    primal_.resize(result.primal_solution.size());
    for (int64_t i = 0; i < result.primal_solution.size(); ++i)
      primal_[i] = result.primal_solution[i];
    dual_.resize(result.dual_solution.size());
    for (int64_t i = 0; i < result.dual_solution.size(); ++i)
      dual_[i] = result.dual_solution[i];
  }
  const std::vector<double> &primal() const { return primal_; }
  const std::vector<double> &dual() const { return dual_; }
};
