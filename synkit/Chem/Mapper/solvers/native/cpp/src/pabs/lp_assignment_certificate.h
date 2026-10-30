#pragma once
// A call-local, exact assignment relaxation of the checked original LP dual.
// No primal anchors, guessed domains or previous solves enter this certificate.
#include "lp_certificate.h"
namespace synkit_mapper {
namespace sparse_root_lp {
struct AssignmentCertificate {
  bool ready = false;
  int n = 0, m = 0;
  int64_t root_bound2 = 0;
#ifdef SYNKIT_MAPPER_TEST_HOOKS
  int64_t calls = 0, prunes = 0;
#endif // SYNKIT_MAPPER_TEST_HOOKS
#if defined(__SIZEOF_INT128__)
  Wide base = 0, quantum = 1;
  int64_t gcd = 0, mass = 0;
  std::vector<int> rows, columns, row_position, column_position;
  std::vector<int> penalties;

  int64_t round_bound(Wide numerator) const {
    Wide value = numerator >= 0
                     ? (numerator + certificate_scale - 1) / certificate_scale
                     : numerator / certificate_scale;
    if (gcd) {
      const Wide step = 2 * Wide(gcd);
      Wide correction = (Wide(mass) - value) % step;
      if (correction < 0)
        correction += step;
      value += correction;
    }
    // The complete typed space has a mapping of cost at most mass. Clipping
    // only weakens an optional bound and keeps the legacy callback arithmetic
    // within the production graph mass guard.
    return int64_t(std::max(Wide(0), std::min(Wide(mass), value)));
  }

  template <class Check>
  static AssignmentCertificate
  build(const PreparedModel &model, const IntegerCertificate &certificate,
        const std::vector<Wide> &reduced, Check check) {
    AssignmentCertificate result;
    result.n = model.n;
    result.m = int(model.variable_rows.size());
    result.mass = model.mass;
    result.gcd = model.gcd;
    if (!result.m || result.n > 256 || result.m != int(model.columns.size()) ||
        reduced.size() != size_t(model.assignment_variables))
      return result;
    check();
    const int n = result.n, m = result.m;
    result.rows = model.variable_rows;
    result.columns = model.columns;
    result.row_position.assign(n, -1);
    result.column_position.assign(n, -1);
    for (int i = 0; i < m; ++i) {
      result.row_position[result.rows[i]] = i;
      result.column_position[result.columns[i]] = i;
    }
    result.base = certificate.numerator;
    for (Wide value : reduced)
      result.base -= std::min(Wide(0), value);
    std::vector<Wide> shifts(m);
    Wide largest = 0;
    for (int i = 0; i < m; ++i) {
      check();
      int row = result.rows[i];
      const auto &domain = model.domains[row];
      if (domain.empty())
        return AssignmentCertificate{};
      Wide minimum = reduced[model.assignment[row * n + domain[0]]];
      for (int image : domain)
        minimum = std::min(minimum, reduced[model.assignment[row * n + image]]);
      shifts[i] = minimum;
      result.base += minimum;
      for (int image : domain)
        largest = std::max(largest, reduced[model.assignment[row * n + image]] -
                                        minimum);
    }
    const int safe_entry = INF / (8 * m) - 1;
    if (safe_entry < 1)
      return AssignmentCertificate{};
    result.quantum = std::max(Wide(1), (largest + safe_entry - 1) / safe_entry);
    std::vector<int> cost(m * m, INF);
    for (int i = 0; i < m; ++i) {
      check();
      int row = result.rows[i];
      for (int image : model.domains[row]) {
        int col = result.column_position[image];
        if (col < 0)
          return AssignmentCertificate{};
        Wide value = (reduced[model.assignment[row * n + image]] - shifts[i]) /
                     result.quantum;
        if (value < 0 || value > safe_entry)
          return AssignmentCertificate{};
        cost[i * m + col] = int(value);
      }
    }
    check();
    Perm match, u, v;
    int optimum = Solver::assignment(cost, m, match, u, v);
    check();
    if (optimum >= INF / 2 || int(match.size()) != m ||
        int(u.size()) != m + 1 || int(v.size()) != m + 1)
      return AssignmentCertificate{};
    int64_t dual = 0, primal = 0;
    std::vector<char> used(m, false);
    result.penalties.assign(m * m, INF);
    for (int i = 0; i < m; ++i) {
      check();
      int image = match[i];
      if (image < 0 || image >= m || used[image]++ ||
          cost[i * m + image] >= INF / 2)
        return AssignmentCertificate{};
      primal += cost[i * m + image];
      dual += int64_t(u[i + 1]) + v[i + 1];
      for (int j = 0; j < m; ++j)
        if (cost[i * m + j] < INF / 2) {
          int64_t value = int64_t(cost[i * m + j]) - u[i + 1] - v[j + 1];
          if (value < 0 || value >= INF)
            return AssignmentCertificate{};
          result.penalties[i * m + j] = int(value);
        }
    }
    if (primal != optimum || dual != optimum)
      return AssignmentCertificate{};
    result.base += result.quantum * optimum;
    result.root_bound2 = result.round_bound(result.base);
    check();
    result.ready = true;
    return result;
  }

  int64_t bound(const int *mapping, const int *used) const {
    if (!ready)
      return 0;
    Wide extra = 0;
    for (int i = 0; i < m; ++i) {
      int image = mapping[rows[i]];
      if (image >= 0) {
        if (image >= n || column_position[image] < 0)
          return mass;
        int penalty = penalties[i * m + column_position[image]];
        if (penalty >= INF / 2)
          return mass;
        extra += penalty;
      } else if (m <= 64) {
        // Dropping nonnegative future-row minima for larger problems gives
        // a weaker valid certificate with O(m) prefix evaluation.
        int least = INF;
        for (int j = 0; j < m; ++j)
          if (!used[columns[j]])
            least = std::min(least, penalties[i * m + j]);
        if (least >= INF / 2)
          return mass;
        extra += least;
      }
    }
    return round_bound(base + quantum * extra);
  }
#else
  int64_t bound(const int *, const int *) const { return 0; }
#endif
};
} // namespace sparse_root_lp
} // namespace synkit_mapper
