#pragma once
#include <cstdint>

namespace synkit_mapper::pabs {
// Exhaustively compare small weighted deletion decks. A timeout preserves only
// levels already excluded; hashes are filters, exact canonical forms decide.
int edge_deck(int n, const int64_t *a, const int64_t *b, const int32_t *types_r,
              const int32_t *types_p, int units, double seconds,
              int64_t node_limit, int64_t upper2, int32_t *mapping,
              int64_t *lower2, int64_t *witness2, int64_t *statistics);
int edge_deck_sizes(int n, const int64_t *a, const int64_t *b,
                    const int32_t *types_r, const int32_t *types_p, int units,
                    int64_t *sizes);
// Proposal only. Failure or truncation never supplies an optimality bound.
int sparse_embedding_seed(int n, const int64_t *a, const int64_t *b,
                          const int32_t *types_r, const int32_t *types_p,
                          int64_t target2, double seconds, int64_t node_limit,
                          int32_t *mapping, int64_t *cost2,
                          int64_t *statistics);
} // namespace synkit_mapper::pabs
