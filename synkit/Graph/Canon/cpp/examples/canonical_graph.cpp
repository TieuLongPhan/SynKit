#include "synkit_canon/canon.h"
#include <iostream>
int main() {
  const int64_t offsets[] = {0, 1, 3, 4};
  const int32_t neighbors[] = {1, 0, 2, 1}, colors[] = {0, 0, 0};
  SynkitCanonRequest request{1, sizeof(SynkitCanonRequest), 3, 4, offsets,
    neighbors, colors, 2.0, 10000, 1048576, 1048576, 100, nullptr};
  SynkitCanonResult *result = nullptr;
  if (synkit_canon_search(&request, &result) != 0 || !result ||
      !synkit_canon_statistics(result)->complete) return 1;
  int64_t size = 0;
  const auto *bytes = synkit_canon_bytes(result, &size);
  if (!bytes || size != 40) { synkit_canon_free(result); return 2; }
  std::cout << "Complete canonical path certificate: " << size << " bytes\n";
  synkit_canon_free(result);
}
