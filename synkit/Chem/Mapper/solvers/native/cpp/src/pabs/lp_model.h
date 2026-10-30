#pragma once
// Exact sparse typed-assignment model in doubled chemical-distance units.
// Every true typed bijection embeds into these x/y/total variables. The
// inequalities are a relaxation; numerical feasibility never proves PMCD.
#include "synkit_mapper/core/runtime.h"
#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>
namespace synkit_mapper {
namespace sparse_root_lp {
constexpr int max_variables = 100000, max_constraints = 300000,
              max_nonzeros = 2000000;
struct Term {
  int column;
  int coefficient;
};
struct Row {
  std::vector<Term> terms;
  int rhs;
  bool equality;
};
struct ModelEdge {
  int i, k;
  int64_t weight;
};
struct ModelProduct {
  int index, e, f, left, right;
};
struct PreparedModel {
  int n = 0;
  int64_t constant = 0, mass = 0, gcd = 0, assignment_variables = 0,
          auxiliary_variables = 0, nonzeros = 0;
  std::vector<int64_t> costs;
  std::vector<Row> rows;
  std::vector<int> fixed, variable_rows, columns, assignment, source_totals,
      target_totals;
  std::vector<std::vector<int>> domains;
  std::vector<ModelEdge> source_edges, target_edges;
  std::vector<ModelProduct> products;
};
struct ModelLimit {};
template <class Check>
std::unique_ptr<PreparedModel> build_model(const Problem &p, Check check) {
  if (p.n < 1 || p.n > 256 || !p.a2 || !p.b2 || !p.types_r || !p.types_p)
    return nullptr;
  auto model = std::make_unique<PreparedModel>();
  model->n = p.n;
  check();
  const int n = p.n;
  std::map<int32_t, int> counts, other, target_position;
  for (int i = 0; i < n; ++i) {
    ++counts[p.types_r[i]];
    ++other[p.types_p[i]];
    target_position[p.types_p[i]] = i;
  }
  if (counts != other)
    return nullptr;
  std::vector<int> fixed(n, -1), rows, cols;
  for (int i = 0; i < n; ++i)
    if (counts[p.types_r[i]] == 1)
      fixed[i] = target_position[p.types_r[i]];
    else
      rows.push_back(i);
  for (int j = 0; j < n; ++j)
    if (counts[p.types_p[j]] != 1)
      cols.push_back(j);
  int64_t mass = 0, gcd = 0;
  for (int i = 0; i < n; ++i) {
    check();
    if (p.a2[i * n + i] || p.b2[i * n + i])
      return nullptr;
    for (int k = 0; k < n; ++k) {
      const int64_t a = p.a2[i * n + k], b = p.b2[i * n + k];
      if (a < 0 || b < 0 || a > 1000000 || b > 1000000 ||
          a != p.a2[k * n + i] || b != p.b2[k * n + i])
        return nullptr;
      if (k < i) {
        mass += a + b;
        gcd = std::gcd(gcd, a);
        gcd = std::gcd(gcd, b);
      }
    }
  }
  model->constant = mass;
  for (int i = 0; i < n; ++i)
    if (fixed[i] >= 0)
      for (int k = 0; k < i; ++k)
        if (fixed[k] >= 0)
          model->constant -=
              2 * std::min(p.a2[i * n + k], p.b2[fixed[i] * n + fixed[k]]);
  std::vector<int> x(n * n, -1);
  std::vector<std::vector<int>> domains(n);
  auto variable = [&](int64_t cost) {
    if (model->costs.size() >= max_variables)
      throw ModelLimit{};
    int index = model->costs.size();
    model->costs.push_back(cost);
    return index;
  };
  int nonzeros = 0;
  auto constraint = [&](std::vector<Term> terms, int rhs,
                        bool equality = false) {
    if (model->rows.size() >= max_constraints ||
        nonzeros + terms.size() > max_nonzeros)
      throw ModelLimit{};
    nonzeros += terms.size();
    model->rows.push_back({std::move(terms), rhs, equality});
  };
  for (int i : rows) {
    check();
    for (int j : cols)
      if (p.types_r[i] == p.types_p[j]) {
        int64_t unary = 0;
        for (int k = 0; k < n; ++k)
          if (fixed[k] >= 0)
            unary -= 2 * std::min(p.a2[i * n + k], p.b2[j * n + fixed[k]]);
        x[i * n + j] = variable(unary);
        domains[i].push_back(j);
      }
  }
  model->assignment_variables = model->costs.size();
  for (int i : rows) {
    std::vector<Term> terms;
    for (int j : domains[i])
      terms.push_back({x[i * n + j], 1});
    constraint(std::move(terms), 1, true);
  }
  for (int j : cols) {
    std::vector<Term> terms;
    for (int i : rows)
      if (x[i * n + j] >= 0)
        terms.push_back({x[i * n + j], 1});
    constraint(std::move(terms), 1, true);
  }
  using Edge = ModelEdge;
  std::vector<Edge> source, target;
  for (size_t r = 0; r < rows.size(); ++r)
    for (size_t s = r + 1; s < rows.size(); ++s) {
      const int i = rows[r], k = rows[s];
      if (p.a2[i * n + k])
        source.push_back({i, k, p.a2[i * n + k]});
    }
  for (size_t r = 0; r < cols.size(); ++r)
    for (size_t s = r + 1; s < cols.size(); ++s) {
      const int j = cols[r], l = cols[s];
      if (p.b2[j * n + l])
        target.push_back({j, l, p.b2[j * n + l]});
    }
  using Product = ModelProduct;
  std::vector<Product> products;
  std::vector<std::vector<int>> sy(source.size()), ty(target.size());
  std::vector<int> source_totals(source.size(), -1),
      target_totals(target.size(), -1);
  std::vector<std::vector<int>> so(source.size() * n), si(source.size() * n),
      to(target.size() * n), ti(target.size() * n);
  for (size_t e = 0; e < source.size(); ++e) {
    check();
    const auto &s = source[e];
    for (size_t f = 0; f < target.size(); ++f) {
      if (!(f % 64))
        check();
      const auto &t = target[f];
      for (int direction = 0; direction < 2; ++direction) {
        const int j = direction ? t.k : t.i, l = direction ? t.i : t.k;
        if (x[s.i * n + j] < 0 || x[s.k * n + l] < 0)
          continue;
        const int index = variable(-2 * std::min(s.weight, t.weight));
        const int product = products.size();
        products.push_back({index, int(e), int(f), j, l});
        sy[e].push_back(product);
        ty[f].push_back(product);
        so[e * n + j].push_back(index);
        si[e * n + l].push_back(index);
        to[f * n + (direction ? s.k : s.i)].push_back(index);
        ti[f * n + (direction ? s.i : s.k)].push_back(index);
      }
    }
  }
  auto marginal = [&](const std::vector<int> &indices, int assignment) {
    if (indices.empty())
      return;
    std::vector<Term> terms;
    terms.reserve(indices.size() + 1);
    for (int index : indices)
      terms.push_back({index, 1});
    terms.push_back({assignment, -1});
    constraint(std::move(terms), 0);
  };
  for (size_t e = 0; e < source.size(); ++e) {
    check();
    const auto &s = source[e];
    // Existing marginals and sum_j x[i,j]=1 imply 0<=sum_e y<=1.
    // A zero-cost total in [0,1] therefore preserves the exact LP projection.
    int total = -1;
    if (p.types_r[s.i] == p.types_r[s.k]) {
      total = variable(0);
      source_totals[e] = total;
      ++model->auxiliary_variables;
      std::vector<Term> definition;
      definition.reserve(sy[e].size() + 1);
      definition.push_back({total, 1});
      for (int h : sy[e])
        definition.push_back({products[h].index, -1});
      constraint(std::move(definition), 0, true);
    }
    for (int j : domains[s.i])
      marginal(so[e * n + j], x[s.i * n + j]);
    for (int j : domains[s.k])
      marginal(si[e * n + j], x[s.k * n + j]);
    for (int j : domains[s.i])
      if (x[s.k * n + j] >= 0) {
        std::vector<Term> terms;
        // sum(nonincident y)=total-sum(incident y); the positive
        // target edges are loop-free, so these incident lists are disjoint.
        terms.push_back({total, 1});
        for (int index : so[e * n + j])
          terms.push_back({index, -1});
        for (int index : si[e * n + j])
          terms.push_back({index, -1});
        terms.push_back({x[s.i * n + j], 1});
        terms.push_back({x[s.k * n + j], 1});
        constraint(std::move(terms), 1);
      }
  }
  for (size_t f = 0; f < target.size(); ++f) {
    check();
    const auto &t = target[f];
    int total = -1;
    if (p.types_p[t.i] == p.types_p[t.k]) {
      total = variable(0);
      target_totals[f] = total;
      ++model->auxiliary_variables;
      std::vector<Term> definition;
      definition.reserve(ty[f].size() + 1);
      definition.push_back({total, 1});
      for (int h : ty[f])
        definition.push_back({products[h].index, -1});
      constraint(std::move(definition), 0, true);
    }
    for (int i : rows) {
      if (x[i * n + t.i] >= 0)
        marginal(to[f * n + i], x[i * n + t.i]);
      if (x[i * n + t.k] >= 0)
        marginal(ti[f * n + i], x[i * n + t.k]);
      if (x[i * n + t.i] >= 0 && x[i * n + t.k] >= 0) {
        std::vector<Term> terms;
        terms.push_back({total, 1});
        for (int index : to[f * n + i])
          terms.push_back({index, -1});
        for (int index : ti[f * n + i])
          terms.push_back({index, -1});
        terms.push_back({x[i * n + t.i], 1});
        terms.push_back({x[i * n + t.k], 1});
        constraint(std::move(terms), 1);
      }
    }
  }
  model->mass = mass;
  model->gcd = gcd;
  model->nonzeros = nonzeros;
  model->fixed = std::move(fixed);
  model->variable_rows = std::move(rows);
  model->columns = std::move(cols);
  model->assignment = std::move(x);
  model->domains = std::move(domains);
  model->source_edges = std::move(source);
  model->target_edges = std::move(target);
  model->products = std::move(products);
  model->source_totals = std::move(source_totals);
  model->target_totals = std::move(target_totals);
  return model;
}

#if defined(__SIZEOF_INT128__)
// A freshly computed typed map supplies a feasible numerical starting point
// only. Check its complete original-index embedding independently before PDLP
// sees it; the exact dual checker remains the sole source of LP lower bounds.
template <class Check>
bool embed_incumbent(const Problem &p, const PreparedModel &model,
                     const std::vector<int32_t> &mapping, int64_t expected_cost,
                     std::vector<int64_t> &point, Check check) {
  __extension__ typedef __int128 SeedWide;
  const int n = model.n;
  point.clear();
  if (p.n != n || n < 1 || mapping.size() != size_t(n) ||
      expected_cost < 0)
    return false;
  std::vector<bool> used(n, false);
  SeedWide original_cost = 0;
  for (int i = 0; i < n; ++i) {
    check();
    const int j = mapping[i];
    if (j < 0 || j >= n || used[j] ||
        p.types_r[i] != p.types_p[j] ||
        (model.fixed[i] >= 0 && model.fixed[i] != j))
      return false;
    used[j] = true;
  }
  for (int i = 0; i < n; ++i) {
    check();
    for (int k = i + 1; k < n; ++k)
      original_cost +=
          std::abs(p.a2[i * n + k] -
                   p.b2[mapping[i] * n + mapping[k]]);
  }
  if (original_cost != expected_cost)
    return false;
  point.assign(model.costs.size(), 0);
  for (int i : model.variable_rows) {
    const int x = model.assignment[i * n + mapping[i]];
    if (x < 0 || size_t(x) >= point.size())
      return false;
    point[x] = 1;
  }
  size_t visited = 0;
  for (const auto &product : model.products) {
    if (!(visited++ % 128))
      check();
    const auto &edge = model.source_edges[product.e];
    if (mapping[edge.i] != product.left ||
        mapping[edge.k] != product.right)
      continue;
    point[product.index] = 1;
    for (int total : {model.source_totals[product.e],
                      model.target_totals[product.f]}) {
      if (total < 0)
        continue;
      if (point[total] != 0)
        return false;
      point[total] = 1;
    }
  }
  SeedWide objective = model.constant;
  for (size_t j = 0; j < point.size(); ++j) {
    if (!(j % 128))
      check();
    if (point[j] != 0 && point[j] != 1)
      return false;
    objective += SeedWide(model.costs[j]) * point[j];
  }
  if (objective != original_cost)
    return false;
  visited = 0;
  for (const auto &row : model.rows) {
    SeedWide sum = 0;
    for (const auto &term : row.terms) {
      if (!(visited++ % 1024))
        check();
      if (term.column < 0 || size_t(term.column) >= point.size())
        return false;
      sum += SeedWide(term.coefficient) * point[term.column];
    }
    if (row.equality ? sum != row.rhs : sum > row.rhs)
      return false;
  }
  check();
  return true;
}
#endif
} // namespace sparse_root_lp
} // namespace synkit_mapper
