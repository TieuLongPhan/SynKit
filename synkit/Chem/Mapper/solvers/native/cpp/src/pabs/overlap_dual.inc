// Private to bridge.cpp; not a standalone translation unit.
namespace {

static thread_local RootDualBranchCertificate *conditioned_dual_capture =
    nullptr;
static thread_local int64_t conditioned_dual_statistics[6] = {};
struct CaptureConditionedDual {
  RootDualBranchCertificate *previous;
  explicit CaptureConditionedDual(RootDualBranchCertificate *value)
      : previous(conditioned_dual_capture) {
    conditioned_dual_capture = value;
  }
  ~CaptureConditionedDual() { conditioned_dual_capture = previous; }
};

struct OverlapDualFactor {
  int i, k;
  int64_t weight;
  std::vector<int64_t> first, second;
};
int64_t overlap_dual_floor_base(const Problem &p, Budget &budget,
                                double allowance,
                                int64_t *statistics = nullptr) {
  // Gauge offsets can grow during diffusion without improving its bound.
  // Limit every unary/message: with m <= 128, <= 400 factors and the PABS
  // input mass <= 500000, all subsequent sums/differences fit int64_t.
  // Aborting an update retains the last fully evaluated bound and snapshot.
  constexpr int64_t potential_limit = 1000000000;
  const auto begun = Clock::now();
  int64_t local[8] = {};
  if (!statistics)
    statistics = local;
  std::fill(statistics, statistics + 8, 0);
  int64_t best = 0;
  auto expired = [&]() {
    budget.check();
    return elapsed(begun) >= allowance;
  };
  const int n = p.n;
  std::map<int32_t, int> count, position;
  for (int i = 0; i < n; ++i) {
    ++count[p.types_r[i]];
    position[p.types_p[i]] = i;
  }
  Perm rows, columns, variable(n, -1), fixed(n, -1);
  for (int i = 0; i < n; ++i) {
    if (count[p.types_r[i]] == 1)
      fixed[i] = position.at(p.types_r[i]);
    else {
      variable[i] = rows.size();
      rows.push_back(i);
    }
    if (count[p.types_p[i]] != 1)
      columns.push_back(i);
  }
  const int m = rows.size();
  statistics[0] = m;
  if (m > 128 || rows.size() != columns.size() || allowance <= 0)
    return 0;
  statistics[7] = 1;
  std::vector<std::vector<int>> domains(m);
  std::vector<int64_t> unary(m * m, 0);
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < m; ++j)
      if (p.types_r[rows[i]] == p.types_p[columns[j]])
        domains[i].push_back(j);
  std::vector<OverlapDualFactor> factors;
  int64_t constant = 0;
  for (int i = 0; i < n; ++i) {
    if (expired())
      return best;
    for (int k = i + 1; k < n; ++k) {
      const int64_t a = p.a2[i * n + k];
      constant += a + p.b2[i * n + k];
      if (!a)
        continue;
      if (fixed[i] >= 0 && fixed[k] >= 0)
        constant -= 2 * std::min(a, p.b2[fixed[i] * n + fixed[k]]);
      else if (fixed[i] >= 0 || fixed[k] >= 0) {
        const int v = fixed[i] >= 0 ? k : i,
                  f = fixed[i] >= 0 ? fixed[i] : fixed[k];
        for (int j : domains[variable[v]])
          unary[variable[v] * m + j] -=
              2 * std::min(a, p.b2[columns[j] * n + f]);
      } else {
        factors.push_back({variable[i], variable[k], a,
                           std::vector<int64_t>(m, 0),
                           std::vector<int64_t>(m, 0)});
        if (factors.size() > 400) {
          statistics[7] = 0;
          return best;
        }
      }
    }
  }
  statistics[1] = factors.size();

  std::vector<std::vector<std::pair<int, int64_t>>> positive(m);
  for (int j = 0; j < m; ++j)
    for (int l = 0; l < m; ++l) {
      const auto weight = p.b2[columns[j] * n + columns[l]];
      if (weight > 0)
        positive[j].push_back({l, weight});
    }
  std::vector<int64_t> factor_minima(factors.size(), 0);
  int64_t factor_sum = 0;
  for (size_t e = 0; e < factors.size(); ++e) {
    if (expired())
      return best;
    const auto &factor = factors[e];
    int64_t value = INT64_MAX;
    std::vector<int64_t> marginal;
    sparse_pair_marginals(m, factor.weight, factor.first, factor.second,
                          domains[factor.i], domains[factor.k], positive,
                          marginal);
    for (int j : domains[factor.i])
      value = std::min(value, marginal[j]);
    if (value == INT64_MAX)
      return best;
    factor_minima[e] = value;
    factor_sum += value;
  }
  auto evaluate = [&](int forced_row, std::vector<int64_t> *forced) {
    if (expired())
      return false;
    Perm costs(m * m, INF), matching, u, v;
    int64_t shift = 0;
    std::vector<int64_t> row_shift(m, 0);
    for (int i = 0; i < m; ++i) {
      int64_t minimum = INT64_MAX;
      for (int j : domains[i])
        minimum = std::min(minimum, unary[i * m + j]);
      if (minimum == INT64_MAX)
        return false;
      shift += minimum;
      row_shift[i] = minimum;
      for (int j : domains[i]) {
        const int64_t value = unary[i * m + j] - minimum;
        // Bound the entire LAP sum and every reduced-path calculation.
        if (value >= INF / (8 * std::max(1, m)))
          return false;
        costs[i * m + j] = static_cast<int>(value);
      }
    }
    int assigned = 0;
    if (m) {
      assigned = Solver::assignment(costs, m, matching, u, v);
      if (assigned >= INF / 2)
        return false;
    }
    const int64_t bound = constant + shift + assigned + factor_sum;
    ++statistics[4];
    best = std::max(best, bound);
    statistics[5] = best;

    if (conditioned_dual_capture &&
        bound > conditioned_dual_capture->root_value) {
      if (expired())
        return false;
      RootDualBranchCertificate snapshot;
      snapshot.n = n;
      snapshot.m = m;
      snapshot.constant = constant + shift + assigned;
      snapshot.root_value = bound;
      snapshot.rows = rows;
      snapshot.columns = columns;
      snapshot.row_columns = domains;
      snapshot.column_index.assign(n, -1);
      snapshot.types_r.assign(p.types_r, p.types_r + n);
      snapshot.types_p.assign(p.types_p, p.types_p + n);
      for (int j = 0; j < m; ++j)
        snapshot.column_index[columns[j]] = j;
      snapshot.reduced_unary.assign(m * m,
                                    RootDualBranchCertificate::forbidden);
      snapshot.target.resize(m * m);
      for (int i = 0; i < m; ++i) {
        for (int j : domains[i]) {
          const int64_t reduced =
              unary[i * m + j] - row_shift[i] - u[i + 1] - v[j + 1];
          if (reduced < 0)
            throw std::runtime_error("Infeasible conditioned LAP dual");
          snapshot.reduced_unary[i * m + j] = reduced;
        }
        for (int j = 0; j < m; ++j)
          snapshot.target[i * m + j] = p.b2[columns[i] * n + columns[j]];
      }
      for (size_t edge = 0; edge < factors.size(); ++edge) {
        const auto &e = factors[edge];
        snapshot.factors.push_back({e.i,
                                    e.k,
                                    e.weight,
                                    factor_minima[edge],
                                    e.first,
                                    e.second,
                                    {},
                                    {}});
      }
      // The snapshot owns every coefficient; later message updates cannot
      // mutate it. Conditional tables are prepared once after diffusion.
      if (expired())
        return false;
      *conditioned_dual_capture = std::move(snapshot);
    }

    if (forced) {
      if (expired())
        return false;
      forced->assign(m, INT64_MAX);
      Perm distance;
      Solver::forced_to_row(costs, m, matching, u, v, forced_row, distance);
      for (int matched_row = 0; matched_row < m; ++matched_row) {
        const int column = matching[matched_row];
        const int path = distance[matched_row];
        if (costs[forced_row * m + column] >= INF / 2 || path >= INF)
          continue;
        (*forced)[column] = shift + assigned + costs[forced_row * m + column] -
                            u[forced_row + 1] - v[column + 1] + path;
      }
    }
    return true;
  };
  if (!evaluate(-1, nullptr))
    return best;
  for (int pass = 0; pass < 8; ++pass) {
    for (size_t edge = 0; edge < factors.size(); ++edge)
      for (int side = 0; side < 2; ++side) {
        if (expired())
          return best;
        auto &e = factors[edge];
        const int row = side ? e.k : e.i, other = side ? e.i : e.k;
        auto &message = side ? e.second : e.first;
        std::vector<int64_t> marginal(m, INT64_MAX), assignment_marginal;
        if (m <= 64 && !evaluate(row, &assignment_marginal))
          return best;
        sparse_pair_marginals(m, e.weight, side ? e.second : e.first,
                              side ? e.first : e.second, domains[row],
                              domains[other], positive, marginal);
        int64_t new_minimum = INT64_MAX;
        for (int j : domains[row]) {
          if (marginal[j] == INT64_MAX)
            return best;
          const int64_t node =
              m <= 64 ? assignment_marginal[j] : unary[row * m + j];
          if (node == INT64_MAX)
            return best;
          int64_t difference = marginal[j] - node, denominator = 2;
          const int64_t delta =
              difference >= 0
                  ? difference / denominator
                  : -((-difference + denominator - 1) / denominator);
          const int64_t next_unary = unary[row * m + j] + delta;
          const int64_t next_message = message[j] + delta;
          if (next_unary < -potential_limit || next_unary > potential_limit ||
              next_message < -potential_limit || next_message > potential_limit)
            return best;
          unary[row * m + j] = next_unary;
          message[j] = next_message;
          new_minimum = std::min(new_minimum, marginal[j] - delta);
        }
        factor_sum += new_minimum - factor_minima[edge];
        factor_minima[edge] = new_minimum;
        ++statistics[3];
      }
    ++statistics[2];
    if (!evaluate(-1, nullptr))
      return best;
    if (factors.empty())
      break;
  }
  statistics[6] =
      std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - begun)
          .count();
  return best;
}
} // namespace

namespace {
int64_t overlap_dual_floor(const Problem &p, Budget &budget, double allowance,
                           int64_t *statistics = nullptr) {
  int64_t local[8] = {};
  if (!statistics)
    statistics = local;
  int64_t bound = overlap_dual_floor_base(p, budget, allowance, statistics);
  int64_t gcd = 0, mass = 0;
  for (int i = 0; i < p.n; ++i) {
    budget.check();
    for (int j = 0; j < i; ++j) {
      const auto a = p.a2[i * p.n + j], b = p.b2[i * p.n + j];
      gcd = std::gcd(gcd, a);
      gcd = std::gcd(gcd, b);
      mass += a + b;
    }
  }
  // Since |a-b| == a+b (mod 2g), every feasible score has this residue.
  // Round a valid integer lower bound upward to the next feasible residue.
  if (gcd) {
    const int64_t step = 2 * gcd, residue = mass % step;
    bound += (residue - bound % step + step) % step;
  }
  statistics[5] = bound;
  return bound;
}
} // namespace

namespace {
static thread_local int64_t conditioned_triplet_statistics[8] = {};
int64_t conditioned_overlap_dual_floor(const Problem &p, Budget &budget,
                                       double allowance,
                                       RootDualBranchCertificate &certificate,
                                       int64_t *stats = nullptr) {
  std::fill(conditioned_triplet_statistics, conditioned_triplet_statistics + 8,
            0);
  const auto started = Clock::now();
  const auto deadline = started + std::chrono::duration_cast<Clock::duration>(
                                      std::chrono::duration<double>(allowance));
  CaptureConditionedDual capture(&certificate);
  const double reserve = std::min(0.005, allowance * 0.2);
  int64_t scalar =
      overlap_dual_floor(p, budget, std::max(0.0, allowance - reserve), stats);
  budget.check();
  if (certificate.n && Clock::now() < deadline)
    certificate.finalize(deadline);
  if (certificate.ready && Clock::now() < deadline) {
    const int64_t grouped =
        certificate.triplet_floor(deadline, conditioned_triplet_statistics);
    budget.check();
    if (grouped > scalar) {
      int64_t gcd = 0, mass = 0;
      for (int i = 0; i < p.n; ++i)
        for (int j = 0; j < i; ++j) {
          gcd = std::gcd(gcd, p.a2[i * p.n + j]);
          gcd = std::gcd(gcd, p.b2[i * p.n + j]);
          mass += p.a2[i * p.n + j] + p.b2[i * p.n + j];
        }
      scalar = grouped;
      if (gcd) {
        const int64_t step = 2 * gcd, residue = mass % step;
        scalar += (residue - scalar % step + step) % step;
      }
    }
  }
  conditioned_dual_statistics[2] = certificate.ready;
  conditioned_dual_statistics[3] = certificate.m;
  conditioned_dual_statistics[4] = certificate.factors.size();
  conditioned_dual_statistics[5] =
      std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() -
                                                           started)
          .count();
  return scalar;
}
} // namespace
