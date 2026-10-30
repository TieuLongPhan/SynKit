#pragma once
#include "synkit_mapper/mapper.h"
namespace synkit_mapper {
// Full-bijection weighted common-edge overlap, via size-n cliques of the
// explicit compatible-correspondence association graph. Not binary-only MCES.
int solve_mces(const Problem &, const Request &, Callback, void *,
               Statistics &);
const char *mces_run_metadata();
} // namespace synkit_mapper
