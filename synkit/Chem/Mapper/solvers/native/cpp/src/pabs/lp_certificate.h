#pragma once

// Mathematical authority for the optional LP bound. No numerical solver or
// solver status is trusted: any sign-valid integer multiplier vector gives a
// box-Lagrangian lower bound on the original exact integer formulation.
#include "lp_model.h"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace synkit_mapper {
namespace sparse_root_lp {
constexpr int64_t certificate_scale = 1000000;

#if defined(__SIZEOF_INT128__)
__extension__ typedef __int128 Wide;
struct IntegerCertificate {
  Wide numerator = 0;
  int64_t bound2 = 0;
  std::vector<int64_t> dual;
};

inline std::string decimal(Wide value) {
  if (!value)
    return "0";
  const bool negative = value < 0;
  if (negative)
    value = -value;
  std::string result;
  while (value) {
    result.push_back(char('0' + value % 10));
    value /= 10;
  }
  if (negative)
    result.push_back('-');
  std::reverse(result.begin(), result.end());
  return result;
}

// Equality multipliers are free. Upper-inequality multipliers must be <= 0.
// For 0<=z<=1 and Az<=b this is C+b*d+sum(min(0,c-A^T*d)).
template <class Check>
IntegerCertificate evaluate_integer_dual(const PreparedModel &model,
                                         const std::vector<int64_t> &dual,
                                         Check check,
                                         std::vector<Wide> *assignment_reduced = nullptr) {
  if (dual.size() != model.rows.size() || model.costs.size() > max_variables ||
      model.rows.size() > max_constraints)
    throw std::invalid_argument("Invalid LP certificate dimensions");
  std::vector<Wide> reduced;
  reduced.reserve(model.costs.size());
  for (int64_t cost : model.costs)
    reduced.push_back(Wide(cost) * certificate_scale);
  Wide lower = Wide(model.constant) * certificate_scale;
  size_t nonzeros = 0;
  for (size_t i = 0; i < model.rows.size(); ++i) {
    if (!(i % 128))
      check();
    const auto &row = model.rows[i];
    const int64_t multiplier = dual[i];
    if ((!row.equality && multiplier > 0) || multiplier < -1000000000000LL ||
        multiplier > 1000000000000LL)
      throw std::invalid_argument("Invalid LP certificate multiplier");
    lower += Wide(row.rhs) * multiplier;
    nonzeros += row.terms.size();
    if (nonzeros > max_nonzeros)
      throw std::invalid_argument("LP certificate exceeds nonzero cap");
    for (const auto &term : row.terms) {
      if (term.column < 0 || size_t(term.column) >= reduced.size())
        throw std::invalid_argument("Invalid LP certificate column");
      reduced[term.column] -= Wide(term.coefficient) * multiplier;
    }
  }
  for (Wide value : reduced)
    if (value < 0)
      lower += value;
  Wide rounded = lower >= 0
                     ? (lower + certificate_scale - 1) / certificate_scale
                     : lower / certificate_scale;
  // Every true mapping has cost mass-2*overlap, hence the ORIGINAL matrix
  // lattice mass modulo 2*gcd. No floating rounding enters this step.
  if (model.gcd) {
    const Wide step = 2 * Wide(model.gcd);
    Wide residue = (Wide(model.mass) - rounded) % step;
    if (residue < 0)
      residue += step;
    rounded += residue;
  }
  check();
  if (rounded < std::numeric_limits<int64_t>::min() ||
      rounded > std::numeric_limits<int64_t>::max())
    throw std::overflow_error("LP certificate bound exceeds int64");
  if (assignment_reduced)
    assignment_reduced->assign(reduced.begin(),
                               reduced.begin() + model.assignment_variables);
  return {lower, int64_t(rounded), dual};
}

// Floating vectors are only proposals; clipping/projection cannot invalidate
// the subsequent exact bound. Even a numerical failure may yield a usable
// vector, provided its original dimensions are correct.
template <class Check>
IntegerCertificate certify_proposal(const PreparedModel &model,
                                    const std::vector<double> &proposal,
                                    Check check,
                                    std::vector<Wide> *assignment_reduced = nullptr) {
  if (proposal.size() != model.rows.size())
    throw std::invalid_argument("Invalid LP proposal dimensions");
  std::vector<int64_t> dual(proposal.size());
  for (size_t i = 0; i < proposal.size(); ++i) {
    if (!(i % 128))
      check();
    double value = std::isfinite(proposal[i]) ? proposal[i] : 0;
    value = std::max(-1000000.0, std::min(1000000.0, value));
    int64_t encoded = std::llround(value * certificate_scale);
    if (!model.rows[i].equality)
      encoded = std::min(int64_t{0}, encoded);
    dual[i] = encoded;
  }
  return evaluate_integer_dual(model, dual, check, assignment_reduced);
}
#endif // exact wide arithmetic; unsupported compilers use the PABS fallback.
} // namespace sparse_root_lp
} // namespace synkit_mapper
