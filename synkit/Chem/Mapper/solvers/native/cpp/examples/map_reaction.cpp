#include "synkit_mapper/reaction/reaction.h"
#include <cstdlib>
#include <iostream>

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "usage: mapper_reaction 'reactants>>products' [backend_id] [seconds]\n";
    return 2;
  }
  SynkitReactionOptions options{1, 1, 1, 0, 0, 0, 0, 100000, 10000};
  SynkitMapperRequest request{};
  request.abi_version = 1;
  request.backend = argc > 2 ? std::atoi(argv[2]) : SYNKIT_PABS;
  request.query = SYNKIT_MINIMUM_CLASSES;
  request.workers = 1;
  request.seconds = argc > 3 ? std::atof(argv[3]) : 10;
  request.max_states = request.max_classes = request.max_frontier =
      request.memory_limit_bytes = -1;
  request.max_candidates = 100000;
  request.slap_max_fingerprints = 10000;
  char *json = nullptr;
  if (synkit_mapper_reaction_solve(argv[1], &options, &request, &json)) {
    std::cerr << synkit_mapper_reaction_error() << '\n';
    return 1;
  }
  std::cout << json << '\n';
  synkit_mapper_reaction_free(json);
  return 0;
}
