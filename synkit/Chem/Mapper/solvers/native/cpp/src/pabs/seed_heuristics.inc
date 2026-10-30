// Private to bridge.cpp; not a standalone translation unit.
// Only neighbors of the two source or target vertices can contribute to a
// two-swap delta. All four weights vanish outside this union. The inverse
// permutation is maintained on acceptance; no feasible mapping is restricted.
class SparseSeedSwap {
  const Problem &p;
  std::vector<Perm> left, right;
  std::vector<uint32_t> marked;
  uint32_t epoch = 0;
  bool ready = false;
  void prepare() {
    if (ready)
      return;
    left.resize(p.n);
    right.resize(p.n);
    marked.assign(p.n, 0);
    for (int i = 0; i < p.n; ++i)
      for (int j = 0; j < p.n; ++j)
        if (i != j) {
          if (p.a2[i * p.n + j])
            left[i].push_back(j);
          if (p.b2[i * p.n + j])
            right[i].push_back(j);
        }
    ready = true;
  }

public:
  explicit SparseSeedSwap(const Problem &problem) : p(problem) {}
  int64_t delta(const Perm &mapping, const Perm &inverse, int i, int j) {
    prepare();
    if (++epoch == 0) {
      std::fill(marked.begin(), marked.end(), 0);
      epoch = 1;
    }
    const int a = mapping[i], b = mapping[j];
    int64_t change = 0;
    auto visit = [&](int k) {
      if (k == i || k == j || marked[k] == epoch)
        return;
      marked[k] = epoch;
      const int c = mapping[k];
      change += std::abs(p.a2[i * p.n + k] - p.b2[b * p.n + c]) +
                std::abs(p.a2[j * p.n + k] - p.b2[a * p.n + c]) -
                std::abs(p.a2[i * p.n + k] - p.b2[a * p.n + c]) -
                std::abs(p.a2[j * p.n + k] - p.b2[b * p.n + c]);
    };
    for (int k : left[i])
      visit(k);
    for (int k : left[j])
      visit(k);
    for (int k : right[a])
      visit(inverse[k]);
    for (int k : right[b])
      visit(inverse[k]);
    return change;
  }
  int64_t overlap(const Perm &mapping, int i, int j) {
    prepare();
    int64_t reward = 0;
    for (int k : left[i])
      reward += std::min(p.a2[i * p.n + k], p.b2[j * p.n + mapping[k]]);
    return reward;
  }
  static void accept(Perm &mapping, Perm &inverse, int i, int j) {
    std::swap(mapping[i], mapping[j]);
    inverse[mapping[i]] = i;
    inverse[mapping[j]] = j;
  }
};

// Soft structural context only proposes a full typed permutation. In
// particular equal WL colors never remove a domain value or certify symmetry.
// Unique-type vertices retain fixed colors so score-anchor degree changes do
// not contaminate every heavy-atom color in subsequent refinement rounds.
void propose_context_seed(const Problem &p, const Perm &profile,
                          const std::vector<int> &tr,
                          const std::vector<int> &tp, Perm &best,
                          int64_t &best2, int64_t lower2, Budget &budget) {
  const int n = p.n;
  if (n > 512 || budget.remaining() < 0.15)
    return;
  const auto begun = Clock::now();
  const double allowance = std::min(0.02, budget.remaining() * 0.05);
  auto expired = [&]() {
    budget.check();
    return elapsed(begun) >= allowance;
  };
  std::map<int, int> counts;
  for (int t : tr)
    ++counts[t];
  std::vector<Perm> ar(n), ap(n);
  for (int i = 0; i < n; ++i) {
    if (expired())
      return;
    for (int j = 0; j < n; ++j) {
      if (p.a2[i * n + j])
        ar[i].push_back(j);
      if (p.b2[i * n + j])
        ap[i].push_back(j);
    }
  }
  Perm cr = tr, cp = tp, penalty(n * n, 0);
  for (int round = 0; round < 3; ++round) {
    std::map<std::vector<int64_t>, int> intern;
    auto refine = [&](const Perm &colors, const Perm &types,
                      const std::vector<Perm> &adj, const int64_t *matrix,
                      Perm &next) {
      next.resize(n);
      for (int i = 0; i < n; ++i) {
        if (expired())
          return false;
        std::vector<int64_t> key;
        if (counts.at(types[i]) == 1)
          key = {-1, types[i]};
        else {
          std::vector<std::pair<int, int64_t>> incident;
          for (int j : adj[i])
            incident.emplace_back(colors[j], matrix[i * n + j]);
          std::sort(incident.begin(), incident.end());
          key = {0, colors[i], static_cast<int64_t>(incident.size())};
          for (auto entry : incident) {
            key.push_back(entry.first);
            key.push_back(entry.second);
          }
        }
        next[i] = intern.emplace(std::move(key), intern.size()).first->second;
      }
      return true;
    };
    Perm nr, np;
    if (!refine(cr, tr, ar, p.a2, nr) || !refine(cp, tp, ap, p.b2, np))
      return;
    cr = std::move(nr);
    cp = std::move(np);
    for (int i = 0; i < n; ++i) {
      if (expired())
        return;
      for (int j = 0; j < n; ++j)
        if (tr[i] == tp[j] && cr[i] != cp[j])
          penalty[i * n + j] += round + 1;
    }
  }
  Perm costs(n * n, INF), candidate, u, v;
  for (int i = 0; i < n; ++i) {
    if (expired())
      return;
    for (int j = 0; j < n; ++j)
      if (tr[i] == tp[j]) {
        // Keep all arithmetic strictly within the existing LAP integer range.
        if (profile[i * n + j] > (INF - 7) / 8)
          return;
        costs[i * n + j] = profile[i * n + j] * 8 + penalty[i * n + j];
      }
  }
  if (Solver::assignment(costs, n, candidate, u, v) >= INF / 2)
    return;
  budget.check();
  std::vector<bool> seen(n, false);
  for (int i = 0; i < n; ++i) {
    const int j = candidate[i];
    if (j < 0 || j >= n || seen[j] || tr[i] != tp[j])
      throw std::runtime_error("Invalid context-LAP seed");
    seen[j] = true;
  }
  int64_t score = mapping_cost2(
      p, std::vector<int32_t>(candidate.begin(), candidate.end()));
  budget.check();

  // Polish this alternative without discarding the already polished profile
  // incumbent. Every swap is a full typed permutation; its exact sparse delta
  // is independently checked again with the whole score before acceptance.
  Perm inverse(n), marked(n, -1);
  for (int i = 0; i < n; ++i)
    inverse[candidate[i]] = i;
  int epoch = 0;
  for (int pass = 0; pass < 8 && score > lower2 && !expired(); ++pass) {
    bool improved = false;
    for (int i = 0; i < n && score > lower2 && !expired(); ++i)
      for (int j = i + 1; j < n && score > lower2; ++j) {
        if (tr[i] != tr[j])
          continue;
        ++epoch;
        const int a = candidate[i], b = candidate[j];
        int64_t delta = 0;
        auto visit = [&](int k) {
          if (k == i || k == j || marked[k] == epoch)
            return;
          marked[k] = epoch;
          const int c = candidate[k];
          delta += std::abs(p.a2[i * n + k] - p.b2[b * n + c]) +
                   std::abs(p.a2[j * n + k] - p.b2[a * n + c]) -
                   std::abs(p.a2[i * n + k] - p.b2[a * n + c]) -
                   std::abs(p.a2[j * n + k] - p.b2[b * n + c]);
        };
        for (int k : ar[i])
          visit(k);
        for (int k : ar[j])
          visit(k);
        for (int k : ap[a])
          visit(inverse[k]);
        for (int k : ap[b])
          visit(inverse[k]);
        if (delta < 0) {
          std::swap(candidate[i], candidate[j]);
          inverse[candidate[i]] = i;
          inverse[candidate[j]] = j;
          score += delta;
          improved = true;
        }
      }
    if (!improved)
      break;
  }
  if (score != mapping_cost2(
                   p, std::vector<int32_t>(candidate.begin(), candidate.end())))
    throw std::runtime_error("Context seed polish disagrees with full score");
  budget.check();
  if (score < best2) {
    best = std::move(candidate);
    best2 = score;
  }
}
// This entire routine supplies upper candidates only. Balanced WL cells are
// temporary labels for one restricted SLAP proposal; they never enter exact
// domains, lower bounds, endpoint groups, or enumeration/classification.
static thread_local int64_t coherent_seed_statistics[20] = {};
static thread_local Perm coherent_seed_mapping;

double propose_coherent_seed(const Problem &p, const Perm &profile,
                             const Perm &tr, const Perm &tp, Perm &best,
                             int64_t &best2, int64_t lower2, Budget &budget) {
  const int n = p.n;
  const double remaining = budget.remaining();
  if (n < 256 || n > 512 || best2 - lower2 < 24 ||
      (remaining >= 0 && remaining < 0.25))
    return 0;
  const auto begun = Clock::now();
  // Borrow from, rather than supplement, the existing large-problem SLAP
  // allowance. Its later unrestricted call is debited by all elapsed work.
  const double nominal = remaining < 0 ? 1.0 : std::min(1.0, remaining * 0.20);
  const double allowance = std::min(0.15, nominal * 0.90);
  coherent_seed_statistics[0] = 1;
  coherent_seed_statistics[2] = best2;
  coherent_seed_statistics[15] = static_cast<int64_t>(allowance * 1e9);
  auto expired = [&]() {
    budget.check();
    return elapsed(begun) >= allowance;
  };
  auto finish = [&]() {
    const double seconds = elapsed(begun);
    coherent_seed_statistics[11] = best2;
    coherent_seed_statistics[14] = static_cast<int64_t>(seconds * 1e9);
    coherent_seed_statistics[16] = coherent_seed_statistics[14];
    return seconds;
  };
  std::map<int, int> counts;
  for (int t : tr)
    ++counts[t];
  std::vector<Perm> ar(n), ap(n);
  for (int i = 0; i < n; ++i) {
    if (expired())
      return finish();
    for (int j = 0; j < n; ++j) {
      if (p.a2[i * n + j])
        ar[i].push_back(j);
      if (p.b2[i * n + j])
        ap[i].push_back(j);
    }
  }
  Perm cr = tr, cp = tp, penalty(n * n, 0), chosen_r, chosen_p;
  int smallest_maximum = n + 1, selected_depth = 0;
  double smallest_ambiguity = std::numeric_limits<double>::infinity();
  std::vector<double> log_factorial(n + 1, 0.0);
  for (int i = 2; i <= n; ++i)
    log_factorial[i] = log_factorial[i - 1] + std::log(static_cast<double>(i));
  for (int round = 0; round < 32; ++round) {
    std::map<std::vector<int64_t>, int> intern;
    auto refine = [&](const Perm &colors, const Perm &types,
                      const std::vector<Perm> &adj, const int64_t *matrix,
                      Perm &next) {
      next.resize(n);
      for (int i = 0; i < n; ++i) {
        if (expired())
          return false;
        std::vector<int64_t> key;
        if (counts.at(types[i]) == 1)
          key = {-1, types[i]};
        else {
          std::vector<std::pair<int, int64_t>> incident;
          for (int j : adj[i])
            incident.emplace_back(colors[j], matrix[i * n + j]);
          std::sort(incident.begin(), incident.end());
          key = {0, colors[i], static_cast<int64_t>(incident.size())};
          for (auto edge : incident) {
            key.push_back(edge.first);
            key.push_back(edge.second);
          }
        }
        next[i] = intern.emplace(std::move(key), intern.size()).first->second;
      }
      return true;
    };
    Perm nr, np;
    if (!refine(cr, tr, ar, p.a2, nr) || !refine(cp, tp, ap, p.b2, np))
      return finish();
    cr = std::move(nr);
    cp = std::move(np);
    for (int i = 0; i < n; ++i) {
      if (expired())
        return finish();
      for (int j = 0; j < n; ++j)
        if (tr[i] == tp[j] && cr[i] != cp[j])
          penalty[i * n + j] += round + 1;
    }
    const int depth = round + 1;
    if (depth != 8 && depth != 16 && depth != 32)
      continue;
    std::map<int, int> nr_counts, np_counts;
    for (int c : cr)
      ++nr_counts[c];
    for (int c : cp)
      ++np_counts[c];
    std::map<std::pair<int, int>, int> registry;
    Perm lr(n), lp(n);
    for (int i = 0; i < n; ++i) {
      const int rc = nr_counts[cr[i]] == np_counts[cr[i]] ? cr[i] : -1;
      const int pc = nr_counts[cp[i]] == np_counts[cp[i]] ? cp[i] : -1;
      lr[i] = registry.emplace(std::make_pair(tr[i], rc), registry.size())
                  .first->second;
      lp[i] = registry.emplace(std::make_pair(tp[i], pc), registry.size())
                  .first->second;
    }
    Perm balance(registry.size(), 0), sizes(registry.size(), 0);
    for (int c : lr) {
      ++balance[c];
      ++sizes[c];
    }
    for (int c : lp)
      --balance[c];
    if (std::any_of(balance.begin(), balance.end(),
                    [](int value) { return value != 0; }))
      throw std::runtime_error("Unbalanced coherent heuristic labels");
    const int maximum = *std::max_element(sizes.begin(), sizes.end());
    // Estimate actual SLAP neighborhood-class work. A large fallback cell
    // can already split almost completely against its fixed neighbor labels.
    // These exact signatures only rank candidate starts; they prove nothing.
    double ambiguity = 0;
    int maximum_class = 0;
    for (int side = 0; side < 2; ++side) {
      const auto &labels = side ? lp : lr;
      const auto &adj = side ? ap : ar;
      const auto *matrix = side ? p.b2 : p.a2;
      std::map<std::vector<int64_t>, int> classes;
      for (int i = 0; i < n; ++i) {
        if (expired())
          return finish();
        std::vector<std::pair<int, int64_t>> incident;
        for (int j : adj[i])
          incident.emplace_back(labels[j], matrix[i * n + j]);
        std::sort(incident.begin(), incident.end());
        std::vector<int64_t> key{labels[i],
                                 static_cast<int64_t>(incident.size())};
        for (auto edge : incident) {
          key.push_back(edge.first);
          key.push_back(edge.second);
        }
        ++classes[std::move(key)];
      }
      for (const auto &entry : classes) {
        ambiguity += log_factorial[entry.second];
        maximum_class = std::max(maximum_class, entry.second);
      }
    }
    if (ambiguity < smallest_ambiguity ||
        (ambiguity == smallest_ambiguity && maximum_class < smallest_maximum)) {
      smallest_maximum = maximum_class;
      smallest_ambiguity = ambiguity;
      chosen_r = std::move(lr);
      chosen_p = std::move(lp);
      selected_depth = depth;
      coherent_seed_statistics[6] = registry.size();
      coherent_seed_statistics[5] = maximum;
      coherent_seed_statistics[18] = maximum_class;
      coherent_seed_statistics[19] = static_cast<int64_t>(ambiguity * 1e6);
    }
  }
  coherent_seed_statistics[4] = selected_depth;

  Perm candidate(n, -1), u, v;
  for (const auto &entry : counts) {
    if (expired())
      return finish();
    Perm rows, columns;
    for (int i = 0; i < n; ++i)
      if (tr[i] == entry.first)
        rows.push_back(i);
    for (int j = 0; j < n; ++j)
      if (tp[j] == entry.first)
        columns.push_back(j);
    if (rows.size() != columns.size())
      throw std::runtime_error("Coherent proposal hard-type inventory");
    const int size = rows.size();
    Perm costs(size * size), assigned;
    for (int i = 0; i < size; ++i)
      for (int j = 0; j < size; ++j) {
        const int base = profile[rows[i] * n + columns[j]];
        if (base > (INF - 529) / 1024)
          return finish();
        costs[i * size + j] = base * 1024 + penalty[rows[i] * n + columns[j]];
      }
    if (Solver::assignment(costs, size, assigned, u, v) >= INF / 2)
      return finish();
    for (int i = 0; i < size; ++i)
      candidate[rows[i]] = columns[assigned[i]];
  }
  auto retain = [&](const int32_t *mapping, int64_t supplied) {
    std::vector<int32_t> value(mapping, mapping + n);
    std::vector<bool> seen(n, false);
    for (int i = 0; i < n; ++i) {
      const int j = value[i];
      if (j < 0 || j >= n || seen[j] || p.types_r[i] != p.types_p[j])
        throw std::runtime_error("Invalid coherent proposal typed bijection");
      seen[j] = true;
    }
    const int64_t actual = mapping_cost2(p, value);
    if (actual != supplied)
      throw std::runtime_error("Coherent proposal full score mismatch");
    if (actual < best2) {
      best.assign(value.begin(), value.end());
      best2 = actual;
      coherent_seed_mapping = best;
      coherent_seed_statistics[10] = 1;
    }
  };
  const int64_t raw = mapping_cost2(
      p, std::vector<int32_t>(candidate.begin(), candidate.end()));
  retain(candidate.data(), raw);
  coherent_seed_statistics[3] = raw;
  coherent_seed_statistics[1] = 1;
  coherent_seed_statistics[12] = static_cast<int64_t>(elapsed(begun) * 1e9);
  if (!expired() && best2 > lower2) {
    struct Receiver {
      decltype(retain) &accept;
      bool invalid = false;
      static int receive(const int32_t *map, int64_t score, void *pointer) {
        auto &self = *static_cast<Receiver *>(pointer);
        try {
          self.accept(map, score);
        } catch (...) {
          self.invalid = true;
        }
        return 1;
      }
    } receiver{retain};
    Perm flags(n, 0);
    SlapStatistics statistics{};
    const auto slap_begun = Clock::now();
    synkit_slap_solve(n, p.a2, p.b2, chosen_r.data(), chosen_p.data(),
                      flags.data(), 10000, allowance - elapsed(begun), -1, 1,
                      Receiver::receive, &receiver, &statistics);
    coherent_seed_statistics[7] = statistics.candidates;
    coherent_seed_statistics[8] = statistics.states;
    coherent_seed_statistics[9] = statistics.reason;
    coherent_seed_statistics[13] =
        static_cast<int64_t>(elapsed(slap_begun) * 1e9);
    if (receiver.invalid)
      throw std::runtime_error("Invalid balanced coherent SLAP candidate");
    budget.check();
  }
  return finish();
}

// This is an incumbent heuristic only: all candidates are full typed
// permutations, independently rescored before being retained. Neither the
// profile lower bound nor the exact PABS traversal depends on these starts.
void improve_incumbent(const Problem &p, const Perm &profile_costs,
                       const std::vector<int> &types_r,
                       const std::vector<int> &types_p, Perm &best,
                       int64_t lower2, Budget &budget, int64_t &best2,
                       double slap_debit = 0) {
  const int n = p.n;
  const double remaining = budget.remaining();
  const double restart_cap =
      remaining < 0 ? 0.25 : std::min(0.25, remaining * 0.03);
  const double existing_slap_cap =
      (n <= 512 && remaining >= 0)
          ? std::max(0.0, (n <= 256 ? std::min(0.1, remaining * 0.12)
                                    : std::min(1.0, remaining * 0.20)) -
                              slap_debit)
          : 0.0;
  // Same pre-SLAP clock and worst stage envelope; reuse time left by an
  // early candidate instead of testing an already expired 3% timer.
  const double seconds = std::max(restart_cap, existing_slap_cap);
  if (seconds <= 0 || best2 <= lower2)
    return;
  const auto started = Clock::now();
  auto expired = [&]() {
    budget.check();
    return elapsed(started) >= seconds;
  };
  if (n <= 512) {
    struct SeedReceiver {
      const Problem &problem;
      Perm &best;
      int64_t &best2;
      bool invalid = false;
      static int receive(const int32_t *mapping, int64_t cost2, void *pointer) {
        auto &self = *static_cast<SeedReceiver *>(pointer);
        std::vector<int32_t> candidate(mapping, mapping + self.problem.n);
        std::vector<bool> seen(self.problem.n, false);
        for (int i = 0; i < self.problem.n; ++i) {
          const int j = candidate[i];
          if (j < 0 || j >= self.problem.n || seen[j] ||
              self.problem.types_r[i] != self.problem.types_p[j]) {
            self.invalid = true;
            return 1;
          }
          seen[j] = true;
        }
        const int64_t score = mapping_cost2(self.problem, candidate);
        if (score != cost2) {
          self.invalid = true;
          return 1;
        }
        if (score < self.best2) {
          self.best.assign(candidate.begin(), candidate.end());
          self.best2 = score;
        }
        return 1;
      }
    } receiver{p, best, best2};
    // Native SLAP supplies one heuristic candidate only. Its status, proxy
    // costs and WL partitions never provide a bound or restrict PABS domains.
    // Retain the existing small-problem allowance. Larger problems need more
    // time to finish one SLAP candidate; an unfinished heuristic supplies no
    // UB.
    const double slap_seconds =
        std::max(0.0, (n <= 256 ? std::min(0.1, remaining * 0.12)
                                : std::min(1.0, remaining * 0.20)) -
                          slap_debit);
    coherent_seed_statistics[17] = static_cast<int64_t>(slap_seconds * 1e9);
    std::vector<int32_t> flags(n, 0);
    SlapStatistics statistics{};
    if (slap_seconds > 0)
      synkit_slap_solve(n, p.a2, p.b2, p.types_r, p.types_p, flags.data(),
                        10000, slap_seconds, -1, 1, SeedReceiver::receive,
                        &receiver, &statistics);
    if (receiver.invalid)
      throw std::runtime_error("Invalid native SLAP incumbent candidate");
    budget.check();
  }
  uint64_t random = 0x8cc8cebc9273a21bULL;
  auto next = [&]() {
    random ^= random << 13;
    random ^= random >> 7;
    random ^= random << 17;
    return random;
  };
  auto score = [&](const Perm &mapping) {
    return mapping_cost2(p,
                         std::vector<int32_t>(mapping.begin(), mapping.end()));
  };
  SparseSeedSwap swap_delta(p);
  auto polish = [&](Perm &mapping) {
    Perm inverse(n);
    for (int k = 0; k < n; ++k)
      inverse[mapping[k]] = k;
    int64_t value = score(mapping);
    for (int pass = 0; pass < 8 && value > lower2 && !expired(); ++pass) {
      bool improved = false;
      for (int i = 0; i < n && !expired(); ++i)
        for (int j = i + 1; j < n; ++j) {
          if (types_r[i] != types_r[j])
            continue;
          const int64_t delta = swap_delta.delta(mapping, inverse, i, j);
          if (delta < 0) {
            SparseSeedSwap::accept(mapping, inverse, i, j);
            value += delta;
            improved = true;
          }
        }
      if (!improved)
        break;
    }
    if (score(mapping) != value)
      throw std::runtime_error("PABS restart seed does not match rescoring");
    return value;
  };
  Perm costs(n * n), candidate, u, v;
  for (int attempt = 0; attempt < 128 && best2 > lower2 && !expired();
       ++attempt) {
    candidate = best;
    if (attempt % 4 == 3) {
      // A deterministic perturbation can leave a two-swap local minimum.
      for (int k = 0; k < 2 + attempt % 7; ++k) {
        const int i = next() % n, j = next() % n;
        if (types_r[i] == types_r[j])
          std::swap(candidate[i], candidate[j]);
      }
    } else if (attempt % 4 == 0) {
      // Perturb only seed costs. The unmodified profile-LAP bound is retained.
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
          costs[i * n + j] = types_r[i] == types_p[j]
                                 ? profile_costs[i * n + j] * 8 + next() % 8
                                 : INF;
      if (Solver::assignment(costs, n, candidate, u, v) >= INF / 2)
        throw std::runtime_error("No compatible PABS restart assignment");
    } else {
      // Since |a-b| = a+b-2min(a,b), overlap-LAP is a useful local proxy.
      // It is used solely to propose a feasible map, never as a lower bound.
      Perm maximum(n, 0);
      for (int i = 0; i < n; ++i) {
        if (expired())
          break;
        for (int j = 0; j < n; ++j)
          if (types_r[i] == types_p[j]) {
            const int reward = swap_delta.overlap(candidate, i, j);
            costs[i * n + j] = reward;
            maximum[i] = std::max(maximum[i], reward);
          }
      }
      if (expired())
        break;
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
          costs[i * n + j] = types_r[i] == types_p[j]
                                 ? maximum[i] - costs[i * n + j] + next() % 4
                                 : INF;
      if (Solver::assignment(costs, n, candidate, u, v) >= INF / 2)
        throw std::runtime_error("No compatible PABS overlap assignment");
    }
    const int64_t value = polish(candidate);
    if (value < best2) {
      best = std::move(candidate);
      best2 = value;
    }
  }
}
