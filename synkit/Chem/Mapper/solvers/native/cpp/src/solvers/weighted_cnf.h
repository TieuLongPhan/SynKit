#pragma once
#include "formulation.h"
#include "synkit_mapper/core/runtime.h"
#include <fstream>

namespace synkit_mapper::solvers::detail {
// Exact Boolean arithmetic encoding only. Optimization remains in a maintained
// native MaxSAT engine. Products and every adder/comparator gate are
// equivalences.
struct WeightedCnf {
  int variables, zero;
  int64_t sum_weights = 0;
  Budget &budget;
  std::vector<std::vector<int>> hard;
  std::vector<std::pair<int64_t, int>> soft;
  std::vector<int> bits;
  explicit WeightedCnf(const Formulation &f, Budget &b)
      : variables(f.variables), zero(f.cost + 1), budget(b) {
    hard.push_back({-zero});
    for (int i = 0; i < f.n; ++i) {
      budget.check();
      std::vector<int> row, column;
      for (int j = 0; j < f.n; ++j) {
        if (f.x[i * f.n + j] >= 0)
          row.push_back(f.x[i * f.n + j] + 1);
        if (f.x[j * f.n + i] >= 0)
          column.push_back(f.x[j * f.n + i] + 1);
      }
      exactly_one(row);
      exactly_one(column);
    }
    size_t checked = 0;
    for (const auto &p : f.products) {
      if (!(++checked % 1024))
        budget.check();
      int y = p.variable + 1, u = p.left + 1, v = p.right + 1;
      add({-y, u});
      add({-y, v});
      add({y, -u, -v});
      if (p.benefit > std::numeric_limits<int64_t>::max() - sum_weights)
        throw std::overflow_error("MaxSAT weight sum overflow");
      sum_weights += p.benefit;
    }
  }
  int variable() {
    if (variables == std::numeric_limits<int>::max())
      throw std::overflow_error("MaxSAT Boolean index overflow");
    return ++variables;
  }
  void add(std::vector<int> clause) {
    if (std::find(clause.begin(), clause.end(), -zero) != clause.end())
      return;
    clause.erase(std::remove(clause.begin(), clause.end(), zero), clause.end());
    std::sort(clause.begin(), clause.end());
    clause.erase(std::unique(clause.begin(), clause.end()), clause.end());
    for (int literal : clause)
      if (std::binary_search(clause.begin(), clause.end(), -literal))
        return;
    hard.push_back(std::move(clause));
  }
  void exactly_one(const std::vector<int> &xs) {
    add(xs);
    if (xs.size() < 2)
      return;
    int previous = variable();
    add({-xs[0], previous});
    for (size_t i = 1; i + 1 < xs.size(); ++i) {
      int next = variable();
      add({-xs[i], next});
      add({-previous, next});
      add({-xs[i], -previous});
      previous = next;
    }
    add({-xs.back(), -previous});
  }
  int conjunction(int a, int b) {
    if (a == zero || b == zero || a == -b)
      return zero;
    if (a == -zero || a == b)
      return b;
    if (b == -zero)
      return a;
    int z = variable();
    add({-z, a});
    add({-z, b});
    add({z, -a, -b});
    return z;
  }
  int disjunction(int a, int b) {
    if (a == -zero || b == -zero || a == -b)
      return -zero;
    if (a == zero || a == b)
      return b;
    if (b == zero)
      return a;
    int z = variable();
    add({z, -a});
    add({z, -b});
    add({-z, a, b});
    return z;
  }
  int parity(int a, int b) {
    if (a == zero)
      return b;
    if (b == zero)
      return a;
    if (a == -zero)
      return -b;
    if (b == -zero)
      return -a;
    if (a == b)
      return zero;
    if (a == -b)
      return -zero;
    int z = variable();
    add({-a, -b, -z});
    add({a, b, -z});
    add({a, -b, z});
    add({-a, b, z});
    return z;
  }
  int majority(int a, int b, int c) {
    if (a == b)
      return a;
    if (a == -b)
      return c;
    if (a == c)
      return a;
    if (a == -c)
      return b;
    if (b == c)
      return b;
    if (b == -c)
      return a;
    if (a == zero)
      return conjunction(b, c);
    if (b == zero)
      return conjunction(a, c);
    if (c == zero)
      return conjunction(a, b);
    if (a == -zero)
      return disjunction(b, c);
    if (b == -zero)
      return disjunction(a, c);
    if (c == -zero)
      return disjunction(a, b);
    int z = variable();
    add({-a, -b, z});
    add({-a, -c, z});
    add({-b, -c, z});
    add({a, b, -z});
    add({a, c, -z});
    add({b, c, -z});
    return z;
  }
  void build_sum(const Formulation &f) {
    if (!bits.empty())
      return;
    std::vector<std::vector<int>> buckets(64);
    size_t checked = 0;
    for (const auto &p : f.products) {
      if (!(++checked % 512))
        budget.check();
      uint64_t weight = p.benefit;
      for (int bit = 0; weight; ++bit, weight >>= 1)
        if (weight & 1)
          buckets[bit].push_back(p.variable + 1);
    }
    for (size_t bit = 0; bit + 1 < buckets.size(); ++bit) {
      budget.check();
      auto &bucket = buckets[bit];
      while (bucket.size() >= 3) {
        if (!(++checked % 512))
          budget.check();
        int a = bucket.back();
        bucket.pop_back();
        int b = bucket.back();
        bucket.pop_back();
        int c = bucket.back();
        bucket.pop_back();
        bucket.push_back(parity(parity(a, b), c));
        buckets[bit + 1].push_back(majority(a, b, c));
      }
    }
    int carry = zero;
    int width = 1;
    for (uint64_t value = sum_weights; value >>= 1;)
      ++width;
    for (int bit = 0; bit < width; ++bit) {
      int a = buckets[bit].empty() ? zero : buckets[bit][0];
      int b = buckets[bit].size() < 2 ? zero : buckets[bit][1];
      bits.push_back(parity(parity(a, b), carry));
      carry = majority(a, b, carry);
    }
    if (carry != zero)
      add({-carry});
  }
  void bound(int64_t value, bool lower) {
    if (lower ? value <= 0 : value >= sum_weights)
      return;
    if (lower ? value > sum_weights : value < 0) {
      add({});
      return;
    }
    int equal = -zero;
    for (int bit = static_cast<int>(bits.size()) - 1; bit >= 0; --bit) {
      bool one = (static_cast<uint64_t>(value) >> bit) & 1;
      if (lower && one)
        add({-equal, bits[bit]});
      if (!lower && !one)
        add({-equal, -bits[bit]});
      equal = conjunction(equal, one ? bits[bit] : -bits[bit]);
    }
  }
  int64_t accounted_bytes() const {
    int64_t bytes = sizeof(*this) + hard.capacity() * sizeof(std::vector<int>) +
                    soft.capacity() * sizeof(soft[0]) +
                    bits.capacity() * sizeof(int);
    for (const auto &clause : hard)
      bytes += clause.capacity() * sizeof(int);
    return bytes;
  }
  void write(const std::string &path) {
    std::ofstream out(path);
    if (!out)
      throw std::runtime_error("Cannot write weighted CNF");
    int64_t total = 0;
    for (auto [weight, literal] : soft) {
      (void)literal;
      total += weight;
    }
    if (total == std::numeric_limits<int64_t>::max())
      throw std::overflow_error("WCNF top weight overflow");
    int64_t top = total + 1;
    out << "p wcnf " << variables << ' ' << hard.size() + soft.size() << ' '
        << top << '\n';
    size_t checked = 0;
    for (const auto &clause : hard) {
      if (!(++checked % 512))
        budget.check();
      out << top;
      for (int v : clause)
        out << ' ' << v;
      out << " 0\n";
    }
    for (auto [weight, literal] : soft) {
      if (!(++checked % 512))
        budget.check();
      out << weight << ' ' << literal << " 0\n";
    }
    out.close();
    if (!out)
      throw std::runtime_error("Failed to finish weighted CNF");
  }
};
} // namespace synkit_mapper::solvers::detail
