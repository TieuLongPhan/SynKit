#pragma once
#include "formulation.h"
#include "synkit_mapper/core/runtime.h"
#include <functional>

namespace synkit_mapper::solvers::detail {
// Input-only feasible incumbent. Neither the matching nor local improvement
// fixes any search domain; only a validated objective bound/hint is exported.
inline std::vector<int32_t> feasible_seed(const Problem &p,
                                          const Budget &budget,
                                          const uint8_t *allowed = nullptr) {
  std::vector<int32_t> inverse(p.n, -1), mapping(p.n, -1);
  std::function<bool(int, std::vector<uint8_t> &)> augment =
      [&](int row, std::vector<uint8_t> &seen) {
        budget.check();
        for (int j = 0; j < p.n; ++j)
          if (!seen[j] && p.types_r[row] == p.types_p[j] &&
              (!allowed || allowed[row * p.n + j])) {
            seen[j] = 1;
            if (inverse[j] < 0 || augment(inverse[j], seen)) {
              inverse[j] = row;
              return true;
            }
          }
        return false;
      };
  for (int i = 0; i < p.n; ++i) {
    std::vector<uint8_t> seen(p.n, 0);
    if (!augment(i, seen))
      return {};
  }
  for (int j = 0; j < p.n; ++j)
    mapping[inverse[j]] = j;
  for (int pass = 0; pass < 2; ++pass) {
    bool changed = false;
    for (int i = 0; i < p.n; ++i) {
      budget.check();
      for (int k = i + 1; k < p.n; ++k) {
        if (p.types_r[i] != p.types_r[k] ||
            (allowed && (!allowed[i * p.n + mapping[k]] ||
                         !allowed[k * p.n + mapping[i]])))
          continue;
        int64_t delta = 0;
        for (int v = 0; v < p.n; ++v)
          if (v != i && v != k) {
            delta += std::abs(p.a2[i * p.n + v] -
                              p.b2[mapping[k] * p.n + mapping[v]]) +
                     std::abs(p.a2[k * p.n + v] -
                              p.b2[mapping[i] * p.n + mapping[v]]) -
                     std::abs(p.a2[i * p.n + v] -
                              p.b2[mapping[i] * p.n + mapping[v]]) -
                     std::abs(p.a2[k * p.n + v] -
                              p.b2[mapping[k] * p.n + mapping[v]]);
          }
        if (delta < 0) {
          std::swap(mapping[i], mapping[k]);
          changed = true;
        }
      }
    }
    if (!changed)
      break;
  }
  (void)mapping_cost2(p, mapping);
  return mapping;
}
inline std::vector<int64_t> seed_values(const Formulation &f,
                                        const std::vector<int32_t> &mapping,
                                        int64_t cost) {
  if (mapping.empty())
    return {};
  std::vector<int64_t> values(f.variables, 0);
  for (int i = 0; i < f.n; ++i)
    values[f.x[i * f.n + mapping[i]]] = 1;
  for (const auto &product : f.products)
    values[product.variable] = values[product.left] && values[product.right];
  values[f.cost] = cost;
  return values;
}
} // namespace synkit_mapper::solvers::detail
