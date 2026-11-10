#ifndef SYNKIT_STEREO_HPP
#define SYNKIT_STEREO_HPP

#include "stereo.h"
#include <algorithm>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace synkit { namespace stereo {
struct Edge { int32_t source, target, color = 0; };
struct Graph {
  std::vector<int32_t> colors;
  std::vector<Edge> edges; /* each undirected edge once */
  std::vector<SynkitStereoFrame> frames;
};
struct Options {
  double seconds = -1;
  int64_t max_nodes = -1, workspace_bytes = -1, output_bytes = -1;
  int32_t max_depth = 1024;
  bool defer_stereo = false;
  void *cancel_token = nullptr;
};
struct Result {
  SynkitStereoStatistics statistics{};
  std::vector<int32_t> mapping;
  bool complete() const { return statistics.complete != 0; }
  bool isomorphic() const {
    if (!complete()) throw std::logic_error("Stereo comparison is incomplete");
    return statistics.isomorphic == 1;
  }
};

namespace detail {
// Convenience preparation is outside the core's tracked workspace/deadline.
struct Packed {
  std::vector<int64_t> offsets;
  std::vector<int32_t> neighbors, edge_colors;
  SynkitStereoGraph view{};
  explicit Packed(const Graph &g) {
    if (g.colors.size() > static_cast<size_t>(INT32_MAX))
      throw std::invalid_argument("Too many vertices");
    std::vector<std::vector<std::pair<int32_t, int32_t>>> rows(g.colors.size());
    for (const auto &e : g.edges) {
      if (e.source < 0 || e.target < 0 || static_cast<size_t>(e.source) >= rows.size() ||
          static_cast<size_t>(e.target) >= rows.size() || e.source == e.target)
        throw std::invalid_argument("Invalid edge endpoints");
      rows[e.source].emplace_back(e.target, e.color);
      rows[e.target].emplace_back(e.source, e.color);
    }
    offsets.push_back(0);
    for (auto &row : rows) {
      std::sort(row.begin(), row.end());
      for (const auto &e : row) { neighbors.push_back(e.first); edge_colors.push_back(e.second); }
      offsets.push_back(static_cast<int64_t>(neighbors.size()));
    }
    view = {static_cast<int64_t>(g.colors.size()), static_cast<int64_t>(neighbors.size()),
            offsets.data(), neighbors.data(), g.colors.data(), edge_colors.data(),
            static_cast<int64_t>(g.frames.size()), g.frames.data()};
  }
};
} // namespace detail

inline Result isomorphism(const Graph &source, const Graph &target, const Options &options = {}) {
  detail::Packed a(source), b(target);
  SynkitStereoRequest request{1, sizeof(SynkitStereoRequest), a.view, b.view,
    options.seconds, options.max_nodes, options.workspace_bytes, options.output_bytes,
    options.max_depth, options.defer_stereo ? SYNKIT_STEREO_DEFER_CHECKS : 0, options.cancel_token};
  SynkitStereoResult *raw = nullptr;
  if (synkit_stereo_match(&request, &raw) != 0)
    throw std::invalid_argument(synkit_stereo_error());
  std::unique_ptr<SynkitStereoResult, decltype(&synkit_stereo_free)> owned(raw, synkit_stereo_free);
  Result result;
  result.statistics = *synkit_stereo_statistics(raw);
  int64_t size = 0;
  const auto *mapping = synkit_stereo_mapping(raw, &size);
  if (size) result.mapping.assign(mapping, mapping + size);
  return result;
}
}} // namespace synkit::stereo
#endif
