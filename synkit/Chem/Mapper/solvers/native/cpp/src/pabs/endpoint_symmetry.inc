// Private to bridge.cpp; verified endpoint automorphism subgroups.
std::vector<int> twin_generators(const Problem &p, bool product) {
  const int n = p.n;
  const auto *matrix = product ? p.b2 : p.a2;
  const auto *types = product ? p.types_p : p.types_r;
  const auto *nodes = product ? p.node_colors_p : p.node_colors_r;
  const auto *edges = product ? p.edge_colors_p : p.edge_colors_r;
  std::vector<int> generators;
  for (int i = 0; i < n; ++i)
    for (int k = i + 1; k < n; ++k) {
      if (types[i] != types[k] || (nodes && nodes[i] != nodes[k]) ||
          (edges && edges[i * n + i] != edges[k * n + k]))
        continue;
      bool equal = true;
      for (int j = 0; j < n; ++j)
        if (j != i && j != k &&
            (matrix[i * n + j] != matrix[k * n + j] ||
             (edges && edges[i * n + j] != edges[k * n + j]))) {
          equal = false;
          break;
        }
      if (!equal)
        continue;
      for (int j = 0; j < n; ++j)
        generators.push_back(j == i ? k : j == k ? i : j);
      break;
    }
  return generators;
}

// Individualization/refinement only discovers candidate symmetries. Every
// retained permutation is verified against all typed, attributed, weighted
// endpoint edges by NativeCanon::accept. A discovery cutoff retains a subgroup
// and can only weaken pruning; it cannot remove a feasible mapping class.
std::vector<int> endpoint_generators(const Problem &p, bool product,
                                     Budget &budget) {
  const int n = p.n;
  auto result = twin_generators(p, product);
  const auto *weights = product ? p.b2 : p.a2;
  const auto *types = product ? p.types_p : p.types_r;
  const auto *nodes = product ? p.node_colors_p : p.node_colors_r;
  const auto *edges = product ? p.edge_colors_p : p.edge_colors_r;
  std::map<std::pair<int32_t, int32_t>, int> node_ids;
  std::map<std::pair<int64_t, int32_t>, int> edge_ids;
  Perm colors(n), palette(n * n, 0), parent(n);
  std::iota(parent.begin(), parent.end(), 0);
  auto find = [&](int x) {
    while (parent[x] != x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  };
  for (int i = 0; i < n; ++i) {
    colors[i] = node_ids
                    .emplace(std::make_pair(types[i], nodes ? nodes[i] : 0),
                             node_ids.size())
                    .first->second;
    for (int j = 0; j < n; ++j) {
      const int64_t weight = weights[i * n + j];
      const int32_t edge = edges ? edges[i * n + j] : 0;
      if (weight || edge)
        palette[i * n + j] =
            edge_ids.emplace(std::make_pair(weight, edge), edge_ids.size() + 1)
                .first->second;
    }
  }
  const double seconds = std::min(0.05, budget.remaining() * 0.02);
  NativeCanon canon(n, colors.data(), palette.data(), seconds, 10000);
  for (size_t offset = 0; offset < result.size(); offset += n) {
    Perm generator(result.begin() + offset, result.begin() + offset + n);
    canon.accept(generator);
    // The existing exact transpositions connect freely interchangeable twins,
    // including adjacent true twins. They may be individualized together.
    for (int i = 0; i < n; ++i)
      if (generator[i] != i) {
        const int x = find(i), y = find(generator[i]);
        parent[y] = x;
      }
  }
  for (int i = 0; i < n; ++i)
    canon.twin_class[i] = find(i);
  const size_t initial = canon.generators.size();
  std::map<int, Perm> buckets;
  for (int i = 0; i < n; ++i)
    buckets[colors[i]].push_back(i);
  NativeCanon::Partition partition;
  for (auto &entry : buckets)
    partition.push_back(std::move(entry.second));
  partition = canon.refine(std::move(partition));
  bool twins_only = true;
  for (const auto &cell : partition)
    for (int x : cell)
      if (canon.twin_class[x] != canon.twin_class[cell.front()])
        twins_only = false;
  if (!twins_only && seconds > 0)
    canon.visit(std::move(partition), true);
  const size_t stop = std::min(canon.generators.size(), initial + 32);
  for (size_t k = initial; k < stop; ++k)
    result.insert(result.end(), canon.generators[k].begin(),
                  canon.generators[k].end());
  budget.check();
  return result;
}
