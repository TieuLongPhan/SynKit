#pragma once
// Optional numerical proposals, checked by exact integer certificates before
// they can improve the PABS lower bound. The incumbent is changed ONLY after
// full typed-bijection scoring. No LP status or anchor restricts exact search.
#include "lp_assignment_certificate.h"
#include "lp_certificate.h"
#include "lp_model.h"
#include <atomic>
#include <functional>
#include <tuple>
#if defined(SYNKIT_MAPPER_WITH_ORTOOLS) && defined(__SIZEOF_INT128__)
#include "ortools/pdlp/primal_dual_hybrid_gradient.h"
#endif
namespace synkit_mapper {
namespace sparse_root_lp {
struct LocalStop {};
// Candidate-private diagnostics, never read by any proof decision.
#ifdef SYNKIT_MAPPER_TEST_HOOKS
inline thread_local int64_t harvest_trace[20] = {};
#endif // SYNKIT_MAPPER_TEST_HOOKS
#if defined(SYNKIT_MAPPER_WITH_ORTOOLS) && defined(__SIZEOF_INT128__)
#include "lp_provider.inc"
#endif
// All state is call-local in the production implementation.
inline void clear() {
#ifdef SYNKIT_MAPPER_TEST_HOOKS
  std::fill(harvest_trace, harvest_trace + 20, 0);
  harvest_trace[8] = harvest_trace[11] = -1;
  harvest_trace[14] = harvest_trace[15] = std::numeric_limits<int64_t>::min();
#endif // SYNKIT_MAPPER_TEST_HOOKS
}
inline int64_t run(const Problem &p, const Budget &budget, double allowance,
                   int64_t stop_target = std::numeric_limits<int64_t>::max(),
                   std::vector<int32_t> *incumbent_mapping = nullptr,
                   int64_t *incumbent_cost = nullptr,
                   AssignmentCertificate *assignment_certificate = nullptr) {
#if !defined(SYNKIT_MAPPER_WITH_ORTOOLS) || !defined(__SIZEOF_INT128__)
  (void)p;
  (void)budget;
  (void)allowance;
  (void)stop_target;
  (void)incumbent_mapping;
  (void)incumbent_cost;
  (void)assignment_certificate;
  // Unavailable optional arithmetic/solver never substitutes a floating bound.
  // Existing exact PABS search remains responsible for the proof.
  return 0;
#else
  const auto started = Clock::now();
  int64_t best = 0;
  if (p.n < 1 || p.n > 256 || allowance <= 0)
    return 0;
#ifdef SYNKIT_MAPPER_TEST_HOOKS
  harvest_trace[0] = 1;
  harvest_trace[9] = stop_target;
  harvest_trace[12] = int64_t(allowance * 1e6);
#endif // SYNKIT_MAPPER_TEST_HOOKS
  auto check = [&]() {
    budget.check();
    if (elapsed(started) >= allowance)
      throw LocalStop{};
  };
  try {
    auto model = build_model(p, check);
    if (!model)
      return 0;
#ifdef SYNKIT_MAPPER_TEST_HOOKS
    harvest_trace[18] = model->costs.size();
    harvest_trace[19] = model->rows.size();
#endif // SYNKIT_MAPPER_TEST_HOOKS
    PDLPProvider provider(model->costs, model->rows, model->constant, check);
    if (incumbent_mapping && incumbent_cost) {
      std::vector<int64_t> point;
      if (embed_incumbent(p, *model, *incumbent_mapping, *incumbent_cost,
                          point, check)) {
        std::vector<double> dual;
        const bool dual_ready = make_initial_dual(*model, dual, check);
        provider.set_initial_primal(point, check, dual_ready ? &dual : nullptr);
      }
    }
    check();
    const double remaining =
        std::min(allowance - elapsed(started), budget.remaining());
    if (remaining < 0.003)
      return 0;
    const double reserve = std::min(0.1, remaining * 0.2);
    provider.solve(remaining - reserve, stop_target, 2 * model->gcd);
#ifdef SYNKIT_MAPPER_TEST_HOOKS
    harvest_trace[17] = provider.dual().size();
#endif // SYNKIT_MAPPER_TEST_HOOKS
    if (provider.dual().size() == model->rows.size()) {
#ifdef SYNKIT_MAPPER_TEST_HOOKS
      ++harvest_trace[6];
#endif // SYNKIT_MAPPER_TEST_HOOKS
      std::vector<Wide> assignment_reduced;
      auto certificate = certify_proposal(
          *model, provider.dual(), check,
          assignment_certificate ? &assignment_reduced : nullptr);
#ifdef SYNKIT_MAPPER_TEST_HOOKS
      ++harvest_trace[7];
      harvest_trace[8] = certificate.bound2;
#endif // SYNKIT_MAPPER_TEST_HOOKS
      if (certificate.bound2 > best)
        best = certificate.bound2;
      if (assignment_certificate) {
        auto next = AssignmentCertificate::build(*model, certificate,
                                                 assignment_reduced, check);
        if (next.ready) {
          best = std::max(best, next.root_bound2);
          *assignment_certificate = std::move(next);
        }
      }
    }
    if (best < stop_target && incumbent_mapping && incumbent_cost &&
        provider.primal().size() == model->costs.size()) {
      const int n = p.n;
      const auto &rows = model->variable_rows;
      const auto &cols = model->columns;
      const auto &x = model->assignment;
      const auto &domains = model->domains;
      const auto &fixed = model->fixed;
      const auto &source = model->source_edges;
      const auto &products = model->products;
      auto primal_value = [&](int index) { return provider.primal()[index]; };
      // The selected decoder retains the bridge-private LAP implementation and
      // tie order. This header is included after the legacy kernel in
      // bridge.cpp.
#include "lp_rounding.inc"
      round_primal(
          true); // Partial PDLP iterates are untrusted but usable proposals.
    }
#ifdef SYNKIT_MAPPER_TEST_HOOKS
    if (incumbent_cost)
      harvest_trace[11] = *incumbent_cost;
#endif // SYNKIT_MAPPER_TEST_HOOKS
  } catch (const LocalStop &) {
#ifdef SYNKIT_MAPPER_TEST_HOOKS
    harvest_trace[16] = 1;
#endif // SYNKIT_MAPPER_TEST_HOOKS
  } catch (const Stop &) {
#ifdef SYNKIT_MAPPER_TEST_HOOKS
    harvest_trace[16] = 2;
#endif // SYNKIT_MAPPER_TEST_HOOKS
  } catch (const ModelLimit &) {
#ifdef SYNKIT_MAPPER_TEST_HOOKS
    harvest_trace[16] = 3;
#endif // SYNKIT_MAPPER_TEST_HOOKS
  } catch (const std::exception &) {
#ifdef SYNKIT_MAPPER_TEST_HOOKS
    harvest_trace[16] = 4;
#endif // SYNKIT_MAPPER_TEST_HOOKS
  }
  return best;
#endif
}
} // namespace sparse_root_lp
} // namespace synkit_mapper
#ifdef SYNKIT_MAPPER_TEST_HOOKS
#include "lp_test_hooks.h"
#endif

#ifdef SYNKIT_MAPPER_TEST_HOOKS
// Read-only, candidate-private post-clock attribution. No previous result is
// fed into a later solve; solve_pabs_original clears this slot at entry.
extern "C" int synkit_private_lp_harvest_statistics(int64_t *out,
                                                    int capacity) {
  if (!out || capacity < 20)
    return -1;
  std::copy(synkit_mapper::sparse_root_lp::harvest_trace,
            synkit_mapper::sparse_root_lp::harvest_trace + 20, out);
  return 20;
}
extern "C" void synkit_private_lp_harvest_clear() {
  synkit_mapper::sparse_root_lp::clear();
}
#endif // SYNKIT_MAPPER_TEST_HOOKS
