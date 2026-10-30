#pragma once

// Explicit test builds only. No production snapshots, serializers or runtime
// provider selectors are installed by these mathematical-oracle entry points.
#ifdef SYNKIT_MAPPER_TEST_HOOKS
#include <iomanip>
#include <set>
#include <sstream>

namespace synkit_mapper {
namespace sparse_root_lp {
inline Problem test_problem(int n, const int64_t *a, const int64_t *b,
                            const int32_t *tr, const int32_t *tp) {
  Problem p{};
  p.abi_version = 1;
  p.n = n;
  p.a2 = a;
  p.b2 = b;
  p.types_r = tr;
  p.types_p = tp;
  return p;
}

template <class Values>
void test_array(std::ostream &out, const Values &values) {
  out << '[';
  for (size_t i = 0; i < values.size(); ++i) {
    if (i)
      out << ',';
    out << values[i];
  }
  out << ']';
}

inline void test_model_json(std::ostream &out, const PreparedModel &m) {
  out << "{\"valid\":true,\"n\":" << m.n << ",\"constant\":" << m.constant
      << ",\"mass\":" << m.mass << ",\"gcd\":" << m.gcd
      << ",\"scale\":" << certificate_scale << ",\"objective_coefficients\":";
  test_array(out, m.costs);
  out << ",\"constraints\":[";
  for (size_t r = 0; r < m.rows.size(); ++r) {
    if (r)
      out << ',';
    const auto &row = m.rows[r];
    out << "{\"rhs\":" << row.rhs
        << ",\"equal\":" << (row.equality ? "true" : "false") << ",\"coeff\":[";
    for (size_t k = 0; k < row.terms.size(); ++k) {
      if (k)
        out << ',';
      out << '[' << row.terms[k].column << ',' << row.terms[k].coefficient
          << ']';
    }
    out << "]}";
  }
  std::vector<std::string> keys(m.costs.size());
  for (int i : m.variable_rows)
    for (int j : m.domains[i])
      keys[m.assignment[i * m.n + j]] =
          "[\"x\"," + std::to_string(i) + ',' + std::to_string(j) + ']';
  for (const auto &z : m.products) {
    const auto &e = m.source_edges[z.e];
    keys[z.index] = "[\"y\"," + std::to_string(e.i) + ',' +
                    std::to_string(e.k) + ',' + std::to_string(z.left) + ',' +
                    std::to_string(z.right) + ']';
  }
  for (size_t e = 0; e < m.source_totals.size(); ++e)
    if (m.source_totals[e] >= 0)
      keys[m.source_totals[e]] = "[\"ts\"," +
                                 std::to_string(m.source_edges[e].i) + ',' +
                                 std::to_string(m.source_edges[e].k) + ']';
  for (size_t e = 0; e < m.target_totals.size(); ++e)
    if (m.target_totals[e] >= 0)
      keys[m.target_totals[e]] = "[\"tt\"," +
                                 std::to_string(m.target_edges[e].i) + ',' +
                                 std::to_string(m.target_edges[e].k) + ']';
  out << "],\"keys\":[";
  for (size_t i = 0; i < keys.size(); ++i) {
    if (i)
      out << ',';
    out << keys[i];
  }
  out << "]}";
}

#if defined(__SIZEOF_INT128__)
inline void test_certificate_json(std::ostream &out,
                                  const IntegerCertificate &certificate) {
  out << "{\"valid\":true,\"scale\":" << certificate_scale
      << ",\"integer_box_bound_numerator\":" << decimal(certificate.numerator)
      << ",\"rounded_original_lattice_bound2\":" << certificate.bound2
      << ",\"bound2\":" << certificate.bound2 << ",\"dual\":";
  test_array(out, certificate.dual);
  out << '}';
}
#endif
} // namespace sparse_root_lp
} // namespace synkit_mapper

extern "C" const char *synkit_test_lp_model_json(int n, const int64_t *a,
                                                 const int64_t *b,
                                                 const int32_t *tr,
                                                 const int32_t *tp) {
  using namespace synkit_mapper::sparse_root_lp;
  static thread_local std::string text;
  try {
    auto model = build_model(test_problem(n, a, b, tr, tp), [] {});
    if (!model)
      throw std::invalid_argument("Invalid LP model");
    std::ostringstream out;
    test_model_json(out, *model);
    text = out.str();
  } catch (...) {
    text = "{\"valid\":false}";
  }
  return text.c_str();
}

extern "C" const char *synkit_test_lp_incumbent_point_json(
    int n, const int64_t *a, const int64_t *b, const int32_t *tr,
    const int32_t *tp, const int32_t *mapping, int64_t cost,
    double seconds) {
  using namespace synkit_mapper;
  using namespace synkit_mapper::sparse_root_lp;
  static thread_local std::string text;
#if defined(__SIZEOF_INT128__)
  try {
    if (n < 1 || !mapping || !std::isfinite(seconds) || seconds < 0)
      throw std::invalid_argument("Invalid seed point arguments");
    Request request{};
    request.seconds = seconds > 0 ? seconds : -1;
    Budget budget(request);
    auto check = [&] { budget.check(); };
    auto problem = test_problem(n, a, b, tr, tp);
    auto model = build_model(problem, check);
    std::vector<int64_t> point;
    if (!model || !embed_incumbent(problem, *model,
                                   std::vector<int32_t>(mapping, mapping + n),
                                   cost, point, check))
      throw std::invalid_argument("Invalid incumbent point");
    std::ostringstream out;
    out << "{\"valid\":true,\"point\":";
    test_array(out, point);
    out << ",\"model\":";
    test_model_json(out, *model);
#if defined(SYNKIT_MAPPER_WITH_ORTOOLS)
    std::vector<double> initial_dual;
    const bool dual_ready = make_initial_dual(*model, initial_dual, check);
    out << std::setprecision(17);
    out << ",\"initial_dual_ready\":" << (dual_ready ? "true" : "false");
    if (dual_ready) {
      out << ",\"initial_dual\":";
      test_array(out, initial_dual);
      out << ",\"initial_certificate\":";
      test_certificate_json(out, certify_proposal(*model, initial_dual, check));
    }
    if (seconds > 0) {
      PDLPProvider provider(model->costs, model->rows, model->constant, check);
      provider.set_initial_primal(point, check, dual_ready ? &initial_dual : nullptr);
      provider.solve(budget.remaining() * 0.8, cost, 2 * model->gcd);
      auto certificate = certify_proposal(*model, provider.dual(), check);
      out << ",\"certificate\":";
      test_certificate_json(out, certificate);
      out << ",\"provider_reason\":" << provider.reason;
    }
#endif
    out << '}';
    text = out.str();
  } catch (...) {
    text = "{\"valid\":false}";
  }
#else
  (void)n; (void)a; (void)b; (void)tr; (void)tp; (void)mapping;
  (void)cost; (void)seconds;
  text = "{\"valid\":false,\"wide_arithmetic_available\":false}";
#endif
  return text.c_str();
}

extern "C" const char *synkit_test_lp_integer_certificate_json(
    int n, const int64_t *a, const int64_t *b, const int32_t *tr,
    const int32_t *tp, const int64_t *dual, int count) {
  using namespace synkit_mapper::sparse_root_lp;
  static thread_local std::string text;
#if defined(__SIZEOF_INT128__)
  try {
    auto model = build_model(test_problem(n, a, b, tr, tp), [] {});
    if (!model || count < 0 || size_t(count) != model->rows.size() ||
        (count && !dual))
      throw std::invalid_argument("Invalid LP certificate");
    std::vector<int64_t> proposed;
    if (count)
      proposed.assign(dual, dual + count);
    auto checked = evaluate_integer_dual(*model, proposed, [] {});
    std::ostringstream out;
    test_certificate_json(out, checked);
    text = out.str();
  } catch (...) {
    text = "{\"valid\":false}";
  }
#else
  (void)n;
  (void)a;
  (void)b;
  (void)tr;
  (void)tp;
  (void)dual;
  (void)count;
  text = "{\"valid\":false,\"wide_arithmetic_available\":false}";
#endif
  return text.c_str();
}

extern "C" const char *
synkit_test_lp_default_certificate_json(int n, const int64_t *a,
                                        const int64_t *b, const int32_t *tr,
                                        const int32_t *tp, double seconds) {
  using namespace synkit_mapper;
  using namespace synkit_mapper::sparse_root_lp;
  static thread_local std::string text;
#if defined(SYNKIT_MAPPER_WITH_ORTOOLS) && defined(__SIZEOF_INT128__)
  try {
    if (!std::isfinite(seconds) || seconds <= 0)
      throw std::invalid_argument("Invalid LP budget");
    Request request{};
    request.seconds = seconds;
    Budget budget(request);
    auto check = [&] { budget.check(); };
    auto model = build_model(test_problem(n, a, b, tr, tp), check);
    if (!model)
      throw std::invalid_argument("Invalid LP model");
    PDLPProvider provider(model->costs, model->rows, model->constant, check);
    provider.solve(budget.remaining() * 0.8);
    auto checked = certify_proposal(*model, provider.dual(), check);
    std::ostringstream out;
    test_certificate_json(out, checked);
    text = out.str();
    text.pop_back();
    text += ",\"available\":true,\"threads\":1,\"provider_reason\":" +
            std::to_string(provider.reason) + ",\"primal_dimension\":" +
            std::to_string(provider.primal().size()) +
            ",\"dual_dimension\":" + std::to_string(provider.dual().size()) +
            ",\"solver_status_is_not_proof\":true}";
  } catch (...) {
    text = "{\"available\":true,\"valid\":false}";
  }
#else
  (void)n;
  (void)a;
  (void)b;
  (void)tr;
  (void)tp;
  (void)seconds;
  text = "{\"available\":false,\"valid\":false}";
#endif
  return text.c_str();
}
#endif // SYNKIT_MAPPER_TEST_HOOKS

#if defined(SYNKIT_MAPPER_TEST_HOOKS) && defined(__SIZEOF_INT128__)
extern "C" const char *synkit_test_lp_assignment_certificate_json(
    int n, const int64_t *a, const int64_t *b, const int32_t *tr,
    const int32_t *tp, const int64_t *dual, int count,
    const int32_t *partial, const int32_t *seed) {
  using namespace synkit_mapper;
  using namespace synkit_mapper::sparse_root_lp;
  static thread_local std::string text;
  try {
    auto model = build_model(test_problem(n, a, b, tr, tp), [] {});
    if (!model || !partial || count < 0 || size_t(count) != model->rows.size() ||
        (count && !dual)) throw std::invalid_argument("Invalid assignment oracle");
    std::vector<int64_t> proposed;
    if (count) proposed.assign(dual, dual + count);
    std::vector<Wide> reduced;
    auto box = evaluate_integer_dual(*model, proposed, [] {}, &reduced);
    auto certificate = AssignmentCertificate::build(*model, box, reduced, [] {});
    if (!certificate.ready) throw std::invalid_argument("Unavailable assignment oracle");
    Perm mapping(partial, partial + n), used(n, 0);
    for (int i = 0; i < n; ++i) {
      int j = mapping[i];
      if (j < -1 || (j >= 0 && (j >= n || used[j]++ || tr[i] != tp[j])))
        throw std::invalid_argument("Invalid partial injection");
    }
    std::ostringstream out;
    out << "{\"valid\":true,\"n\":" << n << ",\"m\":" << certificate.m
        << ",\"box_numerator\":" << decimal(box.numerator)
        << ",\"base_numerator\":" << decimal(certificate.base)
        << ",\"quantum\":" << decimal(certificate.quantum)
        << ",\"root_bound2\":" << certificate.root_bound2
        << ",\"conditional_bound2\":" << certificate.bound(mapping.data(), used.data())
        << ",\"rows\":";
    test_array(out, certificate.rows);
    out << ",\"columns\":"; test_array(out, certificate.columns);
    out << ",\"penalties\":"; test_array(out, certificate.penalties);
    out << ",\"assignment_reduced\":[";
    for (size_t i = 0; i < reduced.size(); ++i) {
      if (i) out << ',';
      out << decimal(reduced[i]);
    }
    out << ']';
    if (seed) {
      std::vector<int> aa(a, a + n*n), bb(b, b + n*n), cr(n), cp(n), order(n),
          pred(n*n, 0), best(n), rg, pg;
      std::map<int32_t, int> types;
      std::set<int> levels(aa.begin(), aa.end());
      levels.insert(bb.begin(), bb.end());
      for (int i=0;i<n;++i) {
        cr[i]=types.emplace(tr[i],types.size()).first->second;
        cp[i]=types.emplace(tp[i],types.size()).first->second;
        order[i]=i;
      }
      auto twins = [&](const std::vector<int> &matrix, const int32_t *labels,
                       std::vector<int> &generators) {
        for (int i=0;i<n;++i) for(int j=i+1;j<n;++j) {
          if(labels[i]!=labels[j]) continue;
          bool equal=true;
          for(int k=0;k<n;++k)
            if(k!=i && k!=j && matrix[i*n+k]!=matrix[j*n+k]) equal=false;
          if(!equal) continue;
          Perm g(n); std::iota(g.begin(),g.end(),0); std::swap(g[i],g[j]);
          generators.insert(generators.end(),g.begin(),g.end());
        }
      };
      twins(aa,tr,rg);twins(bb,tp,pg);
      int best4=0;int64_t statistics[6]={};
      int status=native_minimum_impl(n,types.size(),levels.size()-1,2.0,
          aa.data(),bb.data(),cr.data(),cp.data(),order.data(),pred.data(),
          pg.data(),pg.size()/n,rg.data(),rg.size()/n,seed,best.data(),&best4,
          statistics,0,0,false,int(2*std::max(int64_t(0),certificate.root_bound2)),
          nullptr,nullptr,nullptr,
          +[](const int *mapping,const int *used,int,void *pointer) {
            return static_cast<AssignmentCertificate *>(pointer)->bound(mapping,used);
          },&certificate);
      out << ",\"search_status\":" << status << ",\"minimum2\":" << best4/2
          << ",\"mapping\":";test_array(out,best);
    }
    out << '}';text=out.str();
  } catch (const std::exception &e) {
    text="{\"valid\":false}";
  }
  return text.c_str();
}
#endif
