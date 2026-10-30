// Private to bridge.cpp; exact algebraic lower bound.
// Integer normalized objective variables obey 2*e_i+h_i+sum_j b_ij=c_type.
// Conserved total h then makes total CD even. CD2 is impossible: a single
// bond edit has odd endpoint degree, while two bond units cannot have zero
// signed degree everywhere. Pure unary changes cost at least3. Therefore a
// nonzero distance is >=4 (raw doubled cost>=8). Guards are checked exactly;
// component/WL disagreement is only a negative isomorphism certificate.
int chemical_positive_floor(const Problem &p, const Perm &tr, const Perm &tp,
                            int64_t known_lower, Budget &budget,
                            int *reason = nullptr) {
  const int n = p.n;
  if (reason)
    *reason = 0;
  if (n < 3 || n > 256 || budget.remaining() < 0.05)
    return 0;
  const auto started = Clock::now();
  const double allowance = std::min(0.005, budget.remaining() * 0.01);
  auto expired = [&]() {
    budget.check();
    return elapsed(started) >= allowance;
  };
  std::map<int, int> counts, other_counts, positions;
  for (int i = 0; i < n; ++i) {
    ++counts[tr[i]];
    ++other_counts[tp[i]];
    positions[tp[i]] = i;
  }
  if (counts != other_counts)
    return 0;
  Perm dr(n, 0), dp(n, 0);
  std::vector<Perm> ar(n), ap(n);
  for (int i = 0; i < n; ++i) {
    if (expired())
      return 0;
    if (p.a2[i * n + i] || p.b2[i * n + i])
      return 0;
    for (int j = 0; j < n; ++j) {
      const int64_t a = p.a2[i * n + j], b = p.b2[i * n + j];
      if (a < 0 || b < 0 || a % 2 || b % 2 || a != p.a2[j * n + i] ||
          b != p.b2[j * n + i] || a > 1024 || b > 1024)
        return 0;
      dr[i] += a / 2;
      dp[i] += b / 2;
      if (a)
        ar[i].push_back(j);
      if (b)
        ap[i].push_back(j);
    }
  }
  int er = -1, hr = -1, ep = -1, hp = -1;
  for (int orientation = 0; orientation < 2 && er < 0; ++orientation) {
    const int e = n - 2 + orientation, h = n - 1 - orientation;
    if (counts[tr[e]] != 1 || counts[tr[h]] != 1)
      continue;
    const int pe = positions[tr[e]], ph = positions[tr[h]];
    if (p.a2[e * n + h] || p.b2[pe * n + ph] || dr[h] != dp[ph])
      continue;
    std::map<int, int64_t> constants;
    bool valid = true;
    for (int side = 0; side < 2 && valid; ++side) {
      const auto &types = side ? tp : tr;
      const auto &degree = side ? dp : dr;
      const auto *matrix = side ? p.b2 : p.a2;
      const int electron = side ? pe : e, hydrogen = side ? ph : h;
      for (int i = 0; i < n; ++i)
        if (i != electron && i != hydrogen) {
          const int64_t value = degree[i] + matrix[i * n + electron] / 2;
          auto entry = constants.emplace(types[i], value);
          if (entry.first->second != value) {
            valid = false;
            break;
          }
        }
    }
    if (valid) {
      er = e;
      hr = h;
      ep = pe;
      hp = ph;
    }
  }
  if (er < 0 || expired())
    return 0;
  if (known_lower > 0) {
    if (reason)
      *reason = 1;
    return 8;
  }
  auto components = [&](const Perm &types, const std::vector<Perm> &adj,
                        const int64_t *matrix, int e, int h) {
    std::vector<std::vector<int64_t>> signatures;
    Perm seen(n, 0);
    seen[e] = seen[h] = 1;
    for (int root = 0; root < n; ++root)
      if (!seen[root]) {
        std::vector<std::vector<int64_t>> vertices;
        Perm pending{root};
        seen[root] = 1;
        while (!pending.empty()) {
          int i = pending.back();
          pending.pop_back();
          int degree = 0, weight = 0;
          for (int j : adj[i])
            if (j != e && j != h) {
              ++degree;
              weight += matrix[i * n + j];
              if (!seen[j]) {
                seen[j] = 1;
                pending.push_back(j);
              }
            }
          vertices.push_back(
              {types[i], matrix[i * n + e], matrix[i * n + h], degree, weight});
        }
        std::sort(vertices.begin(), vertices.end());
        std::vector<int64_t> key{static_cast<int64_t>(vertices.size())};
        for (const auto &vertex : vertices)
          key.insert(key.end(), vertex.begin(), vertex.end());
        signatures.push_back(std::move(key));
      }
    std::sort(signatures.begin(), signatures.end());
    return signatures;
  };
  auto left = components(tr, ar, p.a2, er, hr),
       right = components(tp, ap, p.b2, ep, hp);
  if (expired())
    return 0;
  if (left != right) {
    if (reason)
      *reason = 2;
    return 8;
  }
  Perm cr = tr, cp = tp;
  std::set<int> previous(tr.begin(), tr.end());
  size_t previous_count = previous.size();
  for (int round = 0; round < 6; ++round) {
    std::map<std::vector<int64_t>, int> intern;
    auto refine = [&](const Perm &colors, const std::vector<Perm> &adj,
                      const int64_t *matrix, Perm &next) {
      next.resize(n);
      for (int i = 0; i < n; ++i) {
        if (expired())
          return false;
        std::vector<std::pair<int, int64_t>> neighbors;
        for (int j : adj[i])
          neighbors.emplace_back(colors[j], matrix[i * n + j]);
        std::sort(neighbors.begin(), neighbors.end());
        std::vector<int64_t> key{colors[i],
                                 static_cast<int64_t>(neighbors.size())};
        for (auto edge : neighbors) {
          key.push_back(edge.first);
          key.push_back(edge.second);
        }
        next[i] = intern.emplace(std::move(key), intern.size()).first->second;
      }
      return true;
    };
    Perm nr, np;
    if (!refine(cr, ar, p.a2, nr) || !refine(cp, ap, p.b2, np))
      return 0;
    Perm balance(intern.size(), 0);
    for (int color : nr)
      ++balance[color];
    for (int color : np)
      --balance[color];
    if (std::any_of(balance.begin(), balance.end(),
                    [](int value) { return value != 0; })) {
      if (expired())
        return 0;
      if (reason)
        *reason = 3;
      return 8;
    }
    if (intern.size() == previous_count)
      break;
    previous_count = intern.size();
    cr = std::move(nr);
    cp = std::move(np);
  }
  return 0;
}
