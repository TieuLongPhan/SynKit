// Private to bridge.cpp; not a standalone translation unit.
namespace synkit_mapper {
namespace {
struct ReducedBlocks {
  std::vector<int64_t> a, b;
  int64_t offset = 0;
};
ReducedBlocks reduce_constant_blocks(const Problem &p, Budget &budget) {
  const int n = p.n;
  struct Block {
    int64_t amin = INT64_MAX, amax = -1, bmin = INT64_MAX, bmax = -1;
  };
  std::map<int32_t, int> registry;
  Perm tr(n), tp(n);
  for (int i = 0; i < n; ++i) {
    tr[i] = registry.emplace(p.types_r[i], registry.size()).first->second;
    tp[i] = registry.emplace(p.types_p[i], registry.size()).first->second;
  }
  const int nt = registry.size();
  Perm balance(nt, 0);
  for (int i = 0; i < n; ++i) {
    ++balance[tr[i]];
    --balance[tp[i]];
  }
  if (std::any_of(balance.begin(), balance.end(), [](int x) { return x != 0; }))
    throw std::invalid_argument("Unbalanced reduction types");
  std::vector<Block> blocks(nt * nt);
  auto index = [&](int a, int b) {
    if (a > b)
      std::swap(a, b);
    return a * nt + b;
  };
  for (int i = 0; i < n; ++i) {
    budget.check();
    for (int j = i + 1; j < n; ++j) {
      auto &left = blocks[index(tr[i], tr[j])];
      auto &right = blocks[index(tp[i], tp[j])];
      left.amin = std::min(left.amin, p.a2[i * n + j]);
      left.amax = std::max(left.amax, p.a2[i * n + j]);
      right.bmin = std::min(right.bmin, p.b2[i * n + j]);
      right.bmax = std::max(right.bmax, p.b2[i * n + j]);
    }
  }
  ReducedBlocks result;
  result.a.assign(n * n, 0);
  result.b.assign(n * n, 0);
  for (int i = 0; i < n; ++i) {
    budget.check();
    for (int j = i + 1; j < n; ++j) {
      const auto &left = blocks[index(tr[i], tr[j])];
      const auto &right = blocks[index(tp[i], tp[j])];
      int64_t a = 0, b = 0;
      if (left.bmin == left.bmax)
        result.offset += std::abs(p.a2[i * n + j] - left.bmin);
      else if (left.amin != left.amax)
        a = p.a2[i * n + j] - std::min(left.amin, left.bmin);
      if (right.bmin != right.bmax && right.amin == right.amax)
        result.offset += std::abs(p.b2[i * n + j] - right.amin);
      else if (right.bmin != right.bmax && right.amin != right.amax)
        b = p.b2[i * n + j] - std::min(right.amin, right.bmin);
      result.a[i * n + j] = result.a[j * n + i] = a;
      result.b[i * n + j] = result.b[j * n + i] = b;
    }
  }
  std::set<int64_t> levels(result.a.begin(), result.a.end());
  levels.insert(result.b.begin(), result.b.end());
  if (levels.size() > 16) {
    // Removing constant-opposite blocks cannot enlarge the old alphabet.
    // If centering would exceed the kernel's alphabet, omit centering only.
    for (int i = 0; i < n; ++i) {
      budget.check();
      for (int j = i + 1; j < n; ++j) {
        const auto &left = blocks[index(tr[i], tr[j])];
        const auto &right = blocks[index(tp[i], tp[j])];
        result.a[i * n + j] = result.a[j * n + i] =
            (left.amin != left.amax && left.bmin != left.bmax) ? p.a2[i * n + j]
                                                               : 0;
        result.b[i * n + j] = result.b[j * n + i] =
            (right.amin != right.amax && right.bmin != right.bmax)
                ? p.b2[i * n + j]
                : 0;
      }
    }
  }
  return result;
}
struct RestoreOriginalScore {
  Statistics &s;
  int64_t offset;
  double preparation;
  ~RestoreOriginalScore() {
    s.model_seconds += preparation;
    if (s.minimum2 >= 0)
      s.minimum2 += offset;
    if (s.incumbent2 >= 0)
      s.incumbent2 += offset;
    if (s.lower_bound2 >= 0)
      s.lower_bound2 += offset;
    if (offset) {
      std::array<int32_t, 1024> mapping;
      int64_t cost = -1;
      int proven = 0;
      int n =
          synkit_mapper_last_pmcd_witness(mapping.data(), 1024, &cost, &proven);
      if (n > 0 && proven)
        synkit_mapper_store_pmcd_witness(n, mapping.data(), cost + offset,
                                         proven);
    }
  }
};
} // namespace
int solve_pabs(const Problem &p, const Request &r, Callback callback,
               void *userdata, Statistics &s) {
  // Only numeric proof uses the larger objective-equivalence group. Exact
  // enumeration keeps the original chemical endpoint groups and class keys.
  if (r.query != SYNKIT_MINIMUM_PROOF || p.n > 512)
    return solve_pabs_original(p, r, callback, userdata, s);
  synkit_mapper_clear_pmcd_witness();
  const auto started = Clock::now();
  Budget budget(r);
  budget.check();
  auto reduced = reduce_constant_blocks(p, budget);
  budget.check();
  Problem q = p;
  q.a2 = reduced.a.data();
  q.b2 = reduced.b.data();
  // These are objective symmetries only: the raw chemical classes are never
  // quotiented by a graph with removed cost-constant edges.
  q.node_colors_r = q.node_colors_p = nullptr;
  q.edge_colors_r = q.edge_colors_p = nullptr;
  Request remaining = r;
  if (r.seconds >= 0)
    remaining.seconds = std::max(0.0, budget.remaining());
  RestoreOriginalScore restore{s, reduced.offset, elapsed(started)};
  return solve_pabs_original(q, remaining, callback, userdata, s);
}
} // namespace synkit_mapper
