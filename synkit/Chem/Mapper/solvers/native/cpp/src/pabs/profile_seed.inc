// Private to bridge.cpp; assignment profile bound and feasible incumbents.
namespace pabs {
bool prepare_seed(const Problem &p, Budget &budget, bool polish,
                  std::vector<int32_t> &mapping, int64_t &incumbent2,
                  int64_t &lower_bound2, bool *original_profile_zero) {
  std::fill(coherent_seed_statistics, coherent_seed_statistics + 20, 0);
  coherent_seed_mapping.clear();
  if (original_profile_zero)
    *original_profile_zero = false;
  budget.check();
  if (p.n < 1 || !p.a2 || !p.b2 || !p.types_r || !p.types_p)
    throw std::invalid_argument("Invalid shared PABS seed problem");
  if (p.n > SYNKIT_PABS_MAX_VERTICES)
    return false;
  const int n = p.n;
  std::map<int32_t, int> types, inventory;
  std::vector<int> cr(n), cp(n);
  std::set<int64_t> levels;
  int64_t maximum2 = 0;
  for (int i = 0; i < n; ++i) {
    cr[i] = types.emplace(p.types_r[i], types.size()).first->second;
    cp[i] = types.emplace(p.types_p[i], types.size()).first->second;
    ++inventory[p.types_r[i]];
    --inventory[p.types_p[i]];
    for (int j = 0; j < n; ++j) {
      const auto a = p.a2[i * n + j], b = p.b2[i * n + j];
      if (a < 0 || b < 0)
        throw std::invalid_argument("Negative shared PABS seed weight");
      if (a > 1024 || b > 1024)
        return false;
      levels.insert(a);
      levels.insert(b);
      if (j > i)
        maximum2 += a + b;
    }
  }
  if (levels.size() > 16 || maximum2 > 500000)
    return false;
  for (const auto &entry : inventory)
    if (entry.second)
      throw std::invalid_argument("Incompatible shared PABS seed inventories");
  std::vector<std::vector<std::vector<int>>> pa(
      n, std::vector<std::vector<int>>(types.size())),
      pb = pa;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      if (i != j) {
        if (p.a2[i * n + j])
          pa[i][cr[j]].push_back(static_cast<int>(p.a2[i * n + j]));
        if (p.b2[i * n + j])
          pb[i][cp[j]].push_back(static_cast<int>(p.b2[i * n + j]));
      }
  for (auto *profiles : {&pa, &pb})
    for (auto &row : *profiles)
      for (auto &values : row)
        std::sort(values.begin(), values.end(), std::greater<int>());
  Perm costs(n * n, INF), candidate, u, v;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      if (cr[i] == cp[j]) {
        int value = 0;
        for (size_t t = 0; t < types.size(); ++t) {
          const auto &left = pa[i][t], &right = pb[j][t];
          const size_t common = std::min(left.size(), right.size());
          for (size_t k = 0; k < common; ++k)
            value += std::abs(left[k] - right[k]);
          for (size_t k = common; k < left.size(); ++k)
            value += left[k];
          for (size_t k = common; k < right.size(); ++k)
            value += right[k];
        }
        costs[i * n + j] = value;
      }

  // Unique hard types fix their images in every feasible permutation. Their
  // edges to unresolved atoms are unary costs, so count them twice before
  // dividing the incident-profile relaxation by two. A fixed row contains
  // only fixed-to-fixed costs, so every contribution is counted at most twice.
  std::map<int, std::pair<int, int>> occurrences;
  for (int i = 0; i < n; ++i)
    ++occurrences[cr[i]].first;
  for (int j = 0; j < n; ++j)
    occurrences[cp[j]].second = j;
  Perm fixed_rows, fixed_targets;
  for (int i = 0; i < n; ++i)
    if (occurrences[cr[i]].first == 1) {
      fixed_rows.push_back(i);
      fixed_targets.push_back(occurrences[cr[i]].second);
    }
  if (!fixed_rows.empty())
    for (int i = 0; i < n; ++i) {
      budget.check();
      const bool fixed = occurrences[cr[i]].first == 1;
      for (int j = 0; j < n; ++j)
        if (cr[i] == cp[j]) {
          int discrepancy = 0;
          for (size_t k = 0; k < fixed_rows.size(); ++k)
            discrepancy += std::abs(p.a2[i * n + fixed_rows[k]] -
                                    p.b2[j * n + fixed_targets[k]]);
          if (fixed)
            costs[i * n + j] = discrepancy;
          else
            costs[i * n + j] += discrepancy;
        }
    }
  const int profile_minimum = Solver::assignment(costs, n, candidate, u, v);
  if (profile_minimum >= INF / 2)
    throw std::runtime_error("No compatible PABS seed");
  budget.check();
  const int64_t score = mapping_cost2(
      p, std::vector<int32_t>(candidate.begin(), candidate.end()));
  mapping.assign(candidate.begin(), candidate.end());
  incumbent2 = score;
  // Every endpoint pair contributes at most twice to the profile relaxation.
  lower_bound2 = (profile_minimum + 1) / 2;
  if (original_profile_zero)
    *original_profile_zero = profile_minimum == 0;
  // A bounded small-support embedding only proposes a feasible typed map.
  // No failure, cutoff, domain deletion or embedding status supplies a bound.
  if (polish && incumbent2 > lower_bound2 && n <= 512) {
    int64_t ma = 0, mb = 0;
    int ea = 0, eb = 0;
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < i; ++j) {
        ma += p.a2[i * n + j];
        mb += p.b2[i * n + j];
        ea += p.a2[i * n + j] > 0;
        eb += p.b2[i * n + j] > 0;
      }
    const int64_t gap = std::abs(ma - mb), excess = lower_bound2 - gap;
    if (gap > 0 && excess >= 0 && excess <= 8 && excess % 2 == 0 &&
        (ma < mb ? ea : eb) <= 64) {
      const double remaining = budget.remaining();
      const double allowance =
          remaining < 0 ? 0.05 : std::min(0.05, remaining * 0.10);
      if (allowance > 0) {
        Perm proposed(n, -1);
        int64_t score = -1;
        const int result = pabs::sparse_embedding_seed(
            n, p.a2, p.b2, p.types_r, p.types_p, lower_bound2, allowance,
            100000, proposed.data(), &score, bridge_embedding_statistics);
        bridge_embedding_statistics[10] = result;
        bridge_embedding_statistics[12] = lower_bound2;
        budget.check();
        if (result == 1) {
          std::vector<char> seen(n, 0);
          for (int i = 0; i < n; ++i) {
            const int j = proposed[i];
            if (j < 0 || j >= n || seen[j]++ || p.types_r[i] != p.types_p[j])
              throw std::runtime_error(
                  "Invalid embedding seed typed bijection");
          }
          const int64_t verified = mapping_cost2(
              p, std::vector<int32_t>(proposed.begin(), proposed.end()));
          if (verified != score || verified != lower_bound2)
            throw std::runtime_error(
                "Embedding seed does not meet the fresh certified bound");
          mapping = std::move(proposed);
          incumbent2 = verified;
          bridge_embedding_statistics[11] = 1;
          bridge_embedding_statistics[13] = verified;
        }
      }
    }
  }
  double coherent_debit = 0;
  if (polish && incumbent2 > lower_bound2)
    coherent_debit = propose_coherent_seed(p, costs, cr, cp, mapping,
                                           incumbent2, lower_bound2, budget);
  if (polish && lower_bound2 < 8 && incumbent2 > lower_bound2) {
    const int floor = chemical_positive_floor(p, cr, cp, lower_bound2, budget);
    if (floor > incumbent2)
      throw std::runtime_error("Chemical floor exceeds feasible incumbent");
    lower_bound2 = std::max<int64_t>(lower_bound2, floor);
  }
  if (polish) {
    SparseSeedSwap swap_delta(p);
    Perm inverse(n);
    for (int k = 0; k < n; ++k)
      inverse[mapping[k]] = k;
    // Input-only incumbent improvement. Equal-type swaps preserve the full
    // feasible mapping space; no reference correspondence or hard domain is
    // introduced. Skip polishing when the starting map already meets the LB.
    for (int pass = 0; pass < 8 && incumbent2 > lower_bound2; ++pass) {
      bool improved = false;
      for (int i = 0; i < n && incumbent2 > lower_bound2; ++i)
        for (int j = i + 1; j < n && incumbent2 > lower_bound2; ++j) {
          budget.check();
          if (cr[i] != cr[j])
            continue;
          const int64_t delta = swap_delta.delta(mapping, inverse, i, j);
          if (delta < 0) {
            SparseSeedSwap::accept(mapping, inverse, i, j);
            incumbent2 += delta;
            improved = true;
          }
        }
      if (!improved)
        break;
    }
    if (mapping_cost2(p, std::vector<int32_t>(mapping.begin(),
                                              mapping.end())) != incumbent2)
      throw std::runtime_error("PABS polished seed does not match rescoring");
    if (incumbent2 > lower_bound2)
      propose_context_seed(p, costs, cr, cp, mapping, incumbent2, lower_bound2,
                           budget);
    budget.check();
    improve_incumbent(p, costs, cr, cp, mapping, lower_bound2, budget,
                      incumbent2, coherent_debit);
    budget.check();
  }
  return true;
}
} // namespace pabs
