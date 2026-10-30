#pragma once

// Shared unrestricted assignment model for the optional OR-Tools engines.
// A2/B2 and costs are doubled chemical distance. If y = x_ij AND x_kl,
// D2 = sum(A2) + sum(B2) - 2 sum(min(A2_ik,B2_jl) y).
// All three AND inequalities are mandatory, including for numeric shells.
#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace synkit_mapper::solvers::detail {
constexpr int64_t kInfinity = std::numeric_limits<int64_t>::max() / 4;
using Terms = std::vector<std::pair<int, int64_t>>;
struct Row {
  Terms terms;
  int64_t lower, upper;
};
struct Product {
  int variable, left, right;
  int64_t benefit;
};
struct Formulation {
  int n = 0, cost = -1, variables = 0;
  int64_t maximum = 0;
  std::vector<int> x;
  std::vector<Row> rows;
  std::vector<Product> products;
  int64_t aggregate_constraints = 0;
  int64_t accounted_bytes() const {
    int64_t size = sizeof(*this) + x.capacity() * sizeof(int) +
                   rows.capacity() * sizeof(Row) +
                   products.capacity() * sizeof(Product);
    for (const auto &row : rows)
      size += row.terms.capacity() * sizeof(Terms::value_type);
    return size;
  }
  int variable() {
    if (variables == std::numeric_limits<int>::max())
      throw std::overflow_error(
          "Assignment model exceeds int variable indices");
    return variables++;
  }
  void add(Terms terms, int64_t lower, int64_t upper) {
    rows.push_back({std::move(terms), lower, upper});
  }
};

inline Formulation formulate(int n, const int64_t *a, const int64_t *b,
                             const int32_t *left, const int32_t *right,
                             const std::function<void()> &check,
                             const uint8_t *allowed = nullptr) {
  Formulation f;
  f.n = n;
  f.x.resize(static_cast<size_t>(n) * n, -1);
  for (int i = 0; i < n; ++i) {
    check();
    Terms row;
    for (int j = 0; j < n; ++j)
      if (left[i] == right[j] && (!allowed || allowed[i * n + j])) {
        int x = f.variable();
        f.x[i * n + j] = x;
        row.emplace_back(x, 1);
      }
    f.add(std::move(row), 1, 1);
  }
  for (int j = 0; j < n; ++j) {
    check();
    Terms column;
    for (int i = 0; i < n; ++i)
      if (f.x[i * n + j] >= 0)
        column.emplace_back(f.x[i * n + j], 1);
    f.add(std::move(column), 1, 1);
  }
  struct Edge {
    int i, k;
    int64_t weight;
  };
  std::vector<Edge> left_edges, right_edges;
  for (int i = 0; i < n; ++i) {
    check();
    for (int k = i + 1; k < n; ++k) {
      if (a[i * n + k])
        left_edges.push_back({i, k, a[i * n + k]});
      if (b[i * n + k])
        right_edges.push_back({i, k, b[i * n + k]});
      int64_t increment = a[i * n + k] + b[i * n + k];
      if (increment > kInfinity - f.maximum)
        throw std::overflow_error(
            "Assignment cost exceeds supported integer range");
      f.maximum += increment;
    }
  }
  Terms cost;
  size_t checks = 0;
  for (const auto &e : left_edges) {
    check();
    std::map<int, Terms> aggregate;
    for (const auto &g : right_edges)
      for (int direction = 0; direction != 2; ++direction) {
        if (!(++checks % 1024))
          check();
        int j = direction ? g.k : g.i, l = direction ? g.i : g.k;
        int u = f.x[e.i * n + j], v = f.x[e.k * n + l];
        if (u < 0 || v < 0)
          continue;
        int y = f.variable();
        f.add({{y, 1}, {u, -1}}, -kInfinity, 0);
        f.add({{y, 1}, {v, -1}}, -kInfinity, 0);
        f.add({{y, 1}, {u, -1}, {v, -1}}, -1, kInfinity);
        int64_t benefit = 2 * std::min(e.weight, g.weight);
        cost.emplace_back(y, benefit);
        f.products.push_back({y, u, v, benefit});
        aggregate[u].emplace_back(y, 1);
        aggregate[v].emplace_back(y, 1);
      }
    // RLT strengthening: for a fixed source edge and one endpoint image,
    // at most one product can be selected. In integral models this follows
    // from bijectivity; the aggregate inequality strengthens the MILP LP.
    for (auto &[assignment, terms] : aggregate) {
      check();
      if (terms.size() < 2)
        continue;
      terms.emplace_back(assignment, -1);
      f.add(std::move(terms), -kInfinity, 0);
      ++f.aggregate_constraints;
    }
  }
  f.cost = f.variable();
  cost.emplace_back(f.cost, 1);
  f.add(std::move(cost), f.maximum, f.maximum);
  return f;
}
} // namespace synkit_mapper::solvers::detail
