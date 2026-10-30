// Bridge-private proposal decoder. Requires legacy Solver::assignment, Perm
// and INF from the existing kernel; it never changes exact-search domains.
auto round_primal = [&](bool refine) {
  if (provider.primal().size() != model->costs.size())
    return;
  if (incumbent_mapping && incumbent_cost &&
      incumbent_mapping->size() == size_t(n)) {
    if (mapping_cost2(p, *incumbent_mapping) != *incumbent_cost)
      throw std::runtime_error("Invalid LP-rounding incoming seed");
  }
  const int m = rows.size();
  std::vector<double> prior(n * n, 0);
  for (int i : rows)
    for (int j : domains[i]) {
      double value = primal_value(x[i * n + j]);
      prior[i * n + j] =
          std::isfinite(value) ? std::max(0.0, std::min(1.0, value)) : 0.0;
    }
  auto accept = [&](const std::vector<int32_t> &candidate) {
    check();
    const int64_t cost = mapping_cost2(p, candidate);
    check();
    if (incumbent_mapping && incumbent_cost && cost < *incumbent_cost) {
      *incumbent_mapping = candidate;
      *incumbent_cost = cost;
      stop_target = cost;
    }
  };
  int anchor_i = -1, anchor_j = -1, anchor_k = -1, anchor_l = -1;
  auto lap = [&](const std::vector<double> &weights,
                 std::vector<int32_t> &candidate) {
    check();
    Perm costs(m * m, INF), matching, u, v;
    for (int r = 0; r < m; ++r) {
      double maximum = 0;
      for (int j : domains[rows[r]])
        maximum = std::max(maximum, weights[rows[r] * n + j]);
      for (int c = 0; c < m; ++c)
        if (x[rows[r] * n + cols[c]] >= 0) {
          const int source = rows[r], target = cols[c];
          // Temporary proposal-only anchors. Original domains, groups and
          // exact search are never modified by these local LAP exclusions.
          if (anchor_i >= 0 && ((source == anchor_i && target != anchor_j) ||
                                (source == anchor_k && target != anchor_l) ||
                                (source != anchor_i && target == anchor_j) ||
                                (source != anchor_k && target == anchor_l)))
            continue;
          double reward =
              maximum > 0 ? weights[rows[r] * n + cols[c]] / maximum : 0;
          if (!std::isfinite(reward))
            reward = 0;
          reward = std::max(0.0, std::min(1.0, reward));
          costs[r * m + c] = int(std::llround(10000 * (1 - reward)));
        }
    }
    if (m && Solver::assignment(costs, m, matching, u, v) >= INF / 2)
      return false;
    candidate.assign(fixed.begin(), fixed.end());
    for (int r = 0; r < m; ++r)
      candidate[rows[r]] = cols[matching[r]];
    if (anchor_i >= 0 &&
        (candidate[anchor_i] != anchor_j || candidate[anchor_k] != anchor_l))
      throw std::runtime_error("LP proposal LAP violated a temporary anchor");
    check();
    accept(candidate);
    return true;
  };
  std::vector<int32_t> candidate;
  if (!lap(prior, candidate))
    return;
  if (!refine || best >= stop_target)
    return;
  // Deterministic pair-agreement refinement is heuristic only. Products
  // incident to the current image reward coherent edge correspondences.
  // A fresh input-only seed is a second starting point when supplied.
  std::vector<std::vector<int32_t>> starts{candidate};
  if (incumbent_mapping && *incumbent_mapping != candidate)
    starts.push_back(*incumbent_mapping);
  for (auto current : starts)
    for (int pass = 0; pass < 4 && best < stop_target; ++pass) {
      check();
      std::vector<double> agreement = prior;
      for (const auto &z : products) {
        const auto &edge = source[z.e];
        double value = primal_value(z.index);
        if (!std::isfinite(value))
          continue;
        value = std::max(0.0, std::min(1.0, value));
        if (current[edge.k] == z.right)
          agreement[edge.i * n + z.left] += value;
        if (current[edge.i] == z.left)
          agreement[edge.k * n + z.right] += value;
      }
      std::vector<int32_t> next;
      if (!lap(agreement, next) || next == current)
        break;
      current = std::move(next);
    }
  if (best >= stop_target)
    return;
  // Break fractional-face tie inconsistencies by holding one positive
  // LP edge correspondence during a few otherwise unchanged LAP rounds.
  // The two typed images are distinct, so residual type counts balance.
  struct Anchor {
    int i, k, j, l;
    double support, first, second, rank;
  };
  std::vector<Anchor> anchors;
  anchors.reserve(std::min(size_t(16), products.size()));
  auto preferred = [](const Anchor &a, const Anchor &b) {
    if (a.rank != b.rank)
      return a.rank > b.rank;
    return std::tie(a.i, a.k, a.j, a.l) < std::tie(b.i, b.k, b.j, b.l);
  };
  for (size_t h = 0; h < products.size(); ++h) {
    if (!(h % 128))
      check();
    const auto &z = products[h];
    const auto &e = source[z.e];
    double y = primal_value(z.index);
    if (!std::isfinite(y))
      continue;
    y = std::max(0.0, std::min(1.0, y));
    const double first = prior[e.i * n + z.left],
                 second = prior[e.k * n + z.right];
    const double uncertainty = 2 - first - second;
    if (y <= 1e-9 || uncertainty <= 1e-12)
      continue;
    Anchor a{e.i, e.k, z.left, z.right, y, first, second, y * uncertainty};
    if (anchors.size() < 16) {
      anchors.push_back(a);
      std::sort(anchors.begin(), anchors.end(), preferred);
    } else if (preferred(a, anchors.back())) {
      anchors.back() = a;
      std::sort(anchors.begin(), anchors.end(), preferred);
    }
  }
  for (const auto &a : anchors) {
    if (best >= stop_target)
      break;
    check();
    anchor_i = a.i;
    anchor_j = a.j;
    anchor_k = a.k;
    anchor_l = a.l;
    std::vector<int32_t> current;
    if (!lap(prior, current))
      continue;
    for (int pass = 0; pass < 4 && best < stop_target; ++pass) {
      check();
      std::vector<double> agreement = prior;
      for (const auto &z : products) {
        const auto &e = source[z.e];
        double value = primal_value(z.index);
        if (!std::isfinite(value))
          continue;
        value = std::max(0.0, std::min(1.0, value));
        if (current[e.k] == z.right)
          agreement[e.i * n + z.left] += value;
        if (current[e.i] == z.left)
          agreement[e.k * n + z.right] += value;
      }
      std::vector<int32_t> next;
      if (!lap(agreement, next) || next == current)
        break;
      current = std::move(next);
    }
  }
  anchor_i = anchor_j = anchor_k = anchor_l = -1;
};
