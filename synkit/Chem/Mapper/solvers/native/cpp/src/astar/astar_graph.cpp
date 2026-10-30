#include "synkit_mapper/mapper.h"

#include <iostream>

int main() {
  const int64_t a2[] = {0, 1, 0, 1, 0, 2, 0, 2, 0};
  const int64_t b2[] = {0, 2, 0, 2, 0, 1, 0, 1, 0};
  const int32_t types[] = {6, 6, 6};
  SynkitMapperProblem problem{};
  problem.abi_version = 1;
  problem.n = 3;
  problem.a2 = a2;
  problem.b2 = b2;
  problem.types_r = problem.types_p = types;
  SynkitMapperRequest request{};
  request.abi_version = 1;
  request.backend = SYNKIT_ASTAR;
  request.workers = 1;
  request.seconds = -1;
  request.max_states = request.max_candidates = request.max_classes = -1;
  request.max_frontier = request.memory_limit_bytes = -1;
  SynkitMapperStatistics statistics{};
  const int status = synkit_mapper::solve_astar(
      problem, request,
      [](const int32_t *mapping, int64_t cost2, void *) {
        std::cout << "mapping " << mapping[0] << ' ' << mapping[1] << ' '
                  << mapping[2] << "; doubled CD " << cost2 << '\n';
        return 0;
      },
      nullptr, statistics);
  std::cout << "minimum proved " << statistics.minimum_proven
            << "; search complete " << statistics.search_complete
            << "; peak frontier " << statistics.peak_frontier
            << "; accounted frontier bytes " << statistics.peak_frontier_bytes
            << '\n';
  return status || !statistics.minimum_proven || !statistics.search_complete;
}
