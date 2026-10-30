#include "synkit_mapper/core/pmcd_witness.h"
#include <algorithm>
#include <vector>
namespace {
thread_local std::vector<int32_t> last_mapping;
thread_local int64_t last_cost2 = -1;
thread_local int last_proven = 0;
} // namespace
extern "C" void synkit_mapper_clear_pmcd_witness(void) {
  last_mapping.clear();
  last_cost2 = -1;
  last_proven = 0;
}
extern "C" void synkit_mapper_store_pmcd_witness(int n, const int32_t *mapping,
                                                 int64_t cost2, int proven) {
  synkit_mapper_clear_pmcd_witness();
  if (n < 1 || n > 1024 || !mapping || cost2 < 0)
    return;
  last_mapping.assign(mapping, mapping + n);
  last_cost2 = cost2;
  last_proven = proven != 0;
}
extern "C" int synkit_mapper_last_pmcd_witness(int32_t *mapping, int capacity,
                                               int64_t *cost2, int *proven) {
  if (!cost2 || !proven || capacity < 0 || (!mapping && capacity))
    return -1;
  *cost2 = last_cost2;
  *proven = last_proven;
  int n = last_mapping.size();
  if (!mapping && capacity == 0)
    return n;
  if (capacity < n)
    return -1;
  std::copy(last_mapping.begin(), last_mapping.end(), mapping);
  return n;
}
