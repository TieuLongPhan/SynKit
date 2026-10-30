// Standalone SLAP on a permuted three-node path, with no Python runtime.
#include "synkit_mapper/slap.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>

namespace {
constexpr int N = 3;
constexpr std::array<int64_t, N * N> A2{0, 2, 0, 2, 0, 2, 0, 2, 0};
constexpr std::array<int64_t, N * N> B2{0, 2, 2, 2, 0, 0, 2, 0, 0};

struct Output {
  int count = 0;
  bool valid = true;
};

int receive(const int32_t *mapping, int64_t cost2, void *userdata) {
  auto &output = *static_cast<Output *>(userdata);
  std::array<bool, N> seen{};
  for (int i = 0; i < N; ++i) {
    if (mapping[i] < 0 || mapping[i] >= N || seen[mapping[i]]) {
      output.valid = false;
      return 1;
    }
    seen[mapping[i]] = true;
  }
  int64_t score2 = 0;
  for (int i = 0; i < N; ++i)
    for (int j = i + 1; j < N; ++j)
      score2 += std::abs(A2[i * N + j] - B2[mapping[i] * N + mapping[j]]);
  if (score2 != cost2 || score2 != 0) {
    output.valid = false;
    return 1;
  }
  ++output.count;
  std::cout << "candidate " << output.count << ": CD=" << cost2 / 2.0
            << ", mapping=";
  for (int i = 0; i < N; ++i)
    std::cout << (i ? "," : "") << mapping[i];
  std::cout << '\n';
  return 0;
}
} // namespace

int main() {
  const std::array<int32_t, N> labels{6, 6, 6};
  const std::array<int32_t, N> stop_flags{0, 0, 0};
  Output output;
  SlapStatistics statistics{};
  const int status = synkit_slap_solve(
      N, A2.data(), B2.data(), labels.data(), labels.data(), stop_flags.data(),
      10000, -1, -1, -1, receive, &output, &statistics);
  if (status || !output.valid || !statistics.completed || !output.count) {
    std::cerr << "SLAP example failed: " << synkit_mapper_error() << '\n';
    return 1;
  }
  std::cout << "Finished heuristic exploration: " << statistics.states
            << " states, " << statistics.lap_calls << " LAP calls.\n"
            << "SLAP provides no general optimality or all-class guarantee.\n";
  return 0;
}
