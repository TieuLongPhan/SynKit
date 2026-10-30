// Private mathematical oracles, enabled only with SYNKIT_MAPPER_TEST_HOOKS.
namespace {
synkit_mapper::Request pabs_oracle_request(double seconds) {
  synkit_mapper::Request request{};
  request.abi_version = 1;
  request.backend = SYNKIT_PABS;
  request.query = SYNKIT_MINIMUM_PROOF;
  request.workers = 1;
  request.seconds = seconds;
  request.max_states = request.max_candidates = request.max_classes =
      request.max_frontier = request.memory_limit_bytes = -1;
  return request;
}

synkit_mapper::Problem pabs_oracle_problem(
    int n, const int64_t *a, const int64_t *b, const int32_t *tr,
    const int32_t *tp, const synkit_mapper::Request &request,
    bool bounded_dual = false) {
  // Enforce dimensions before validate() can access the caller's arrays.
  if (n < 1 || n > (bounded_dual ? 512 : 1024))
    throw std::invalid_argument("Invalid mathematical oracle dimensions");
  synkit_mapper::Problem problem{};
  problem.abi_version = 1;
  problem.n = n;
  problem.a2 = a;
  problem.b2 = b;
  problem.types_r = tr;
  problem.types_p = tp;
  synkit_mapper::validate(problem, request);
  if (bounded_dual) {
    if (request.seconds < 0)
      throw std::invalid_argument("Dual oracle requires a finite allowance");
    int64_t mass = 0;
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < i; ++j) {
        const int64_t av = a[i * n + j], bv = b[i * n + j];
        if (av > 1024 || bv > 1024)
          throw std::invalid_argument("Dual oracle weight limit exceeded");
        mass += av + bv;
        if (mass > 500000)
          throw std::invalid_argument("Dual oracle mass limit exceeded");
      }
  }
  return problem;
}
} // namespace

extern "C" int synkit_private_seed_probe(int n, const int64_t *a,
                                         const int64_t *b, const int32_t *r,
                                         const int32_t *p, int polish,
                                         double seconds, int32_t *out,
                                         int64_t *upper, int64_t *lower,
                                         int *reason) {
  if (n < 1 || n > 1024 || !a || !b || !r || !p || !out || !upper || !lower ||
      !reason)
    return -1;
  synkit_mapper::Problem problem{};
  problem.abi_version = 1;
  problem.n = n;
  problem.a2 = a;
  problem.b2 = b;
  problem.types_r = r;
  problem.types_p = p;
  synkit_mapper::Request request{};
  request.abi_version = 1;
  request.backend = SYNKIT_PABS;
  request.query = SYNKIT_MINIMUM_PROOF;
  request.workers = 1;
  request.seconds = seconds;
  request.max_states = request.max_candidates = request.max_classes =
      request.max_frontier = request.memory_limit_bytes = -1;
  std::vector<int32_t> mapping(n, -7);
  *upper = -11;
  *lower = -13;
  *reason = 0;
  try {
    synkit_mapper::validate(problem, request);
    synkit_mapper::Budget budget(request);
    bool supported = synkit_mapper::pabs::prepare_seed(problem, budget, polish,
                                                       mapping, *upper, *lower);
    std::copy(mapping.begin(), mapping.end(), out);
    return supported ? 1 : 0;
  } catch (const synkit_mapper::Stop &stop) {
    std::copy(mapping.begin(), mapping.end(), out);
    *reason = stop.reason;
    return 2;
  } catch (...) {
    return -1;
  }
}

extern "C" int64_t synkit_private_sparse_seed_delta(int n, const int64_t *a,
                                                    const int64_t *b,
                                                    const int32_t *m, int i,
                                                    int j) {
  synkit_mapper::Problem p{};
  p.n = n;
  p.a2 = a;
  p.b2 = b;
  synkit_mapper::SparseSeedSwap scorer(p);
  std::vector<int32_t> mapping(m, m + n), inverse(n);
  for (int k = 0; k < n; ++k)
    inverse[mapping[k]] = k;
  return scorer.delta(mapping, inverse, i, j);
}

extern "C" int64_t synkit_private_sparse_seed_overlap(int n, const int64_t *a,
                                                      const int64_t *b,
                                                      const int32_t *m, int i,
                                                      int j) {
  synkit_mapper::Problem p{};
  p.n = n;
  p.a2 = a;
  p.b2 = b;
  synkit_mapper::SparseSeedSwap scorer(p);
  return scorer.overlap(std::vector<int32_t>(m, m + n), i, j);
}

extern "C" int synkit_private_chemical_floor(int n, const int64_t *a,
                                             const int64_t *b,
                                             const int32_t *tr,
                                             const int32_t *tp, int64_t lower,
                                             int32_t *reason) {
  try {
    const auto request = pabs_oracle_request(-1);
    const auto p = pabs_oracle_problem(n, a, b, tr, tp, request);
    synkit_mapper::Budget budget(request);
    return synkit_mapper::chemical_positive_floor(
        p, std::vector<int32_t>(tr, tr + n), std::vector<int32_t>(tp, tp + n),
        lower, budget, reason);
  } catch (...) {
    return -1;
  }
}

extern "C" int synkit_private_constant_block_reduction(
    int n, const int64_t *a, const int64_t *b, const int32_t *tr,
    const int32_t *tp, int64_t *out_a, int64_t *out_b, int64_t *offset) {
  if (!out_a || !out_b || !offset)
    return -1;
  try {
    const auto r = pabs_oracle_request(-1);
    const auto p = pabs_oracle_problem(n, a, b, tr, tp, r);
    synkit_mapper::Budget budget(r);
    auto reduced = synkit_mapper::reduce_constant_blocks(p, budget);
    std::copy(reduced.a.begin(), reduced.a.end(), out_a);
    std::copy(reduced.b.begin(), reduced.b.end(), out_b);
    *offset = reduced.offset;
    return 0;
  } catch (...) {
    return -1;
  }
}

extern "C" int synkit_private_overlap_dual_floor(
    int n, const int64_t *a, const int64_t *b, const int32_t *tr,
    const int32_t *tp, double seconds, int64_t *bound, int64_t *stats) {
  if (!bound || !stats)
    return -1;
  *bound = 0;
  std::fill(stats, stats + 8, 0);
  try {
    const auto r = pabs_oracle_request(seconds);
    const auto p = pabs_oracle_problem(n, a, b, tr, tp, r, true);
    synkit_mapper::Budget budget(r);
    *bound = synkit_mapper::overlap_dual_floor(p, budget, seconds, stats);
    return stats[7] ? 0 : 1;
  } catch (const synkit_mapper::Stop &) {
    return 2;
  } catch (...) {
    return -1;
  }
}

extern "C" int synkit_private_conditioned_dual_prepare(
    int n, const int64_t *a, const int64_t *b, const int32_t *tr,
    const int32_t *tp, double seconds, void **handle, int64_t *root,
    int64_t *stats) {
  if (!handle || !root || !stats)
    return -1;
  *handle = nullptr;
  *root = 0;
  std::fill(stats, stats + 8, 0);
  auto certificate = std::make_unique<RootDualBranchCertificate>();
  try {
    const auto r = pabs_oracle_request(seconds);
    const auto p = pabs_oracle_problem(n, a, b, tr, tp, r, true);
    synkit_mapper::Budget budget(r);
    *root = synkit_mapper::conditioned_overlap_dual_floor(p, budget, seconds,
                                                          *certificate, stats);
    if (!certificate->ready)
      return 1;
    *handle = certificate.release();
    return 0;
  } catch (const synkit_mapper::Stop &) {
    return 2;
  } catch (...) {
    return -1;
  }
}

extern "C" int synkit_private_conditioned_dual_query(void *handle, int n,
                                                     const int32_t *partial,
                                                     int64_t *bound) {
  if (!handle || !partial || !bound)
    return -1;
  auto &certificate = *static_cast<RootDualBranchCertificate *>(handle);
  if (!certificate.ready || n != certificate.n)
    return -1;
  Perm seen(n, 0);
  for (int i = 0; i < n; ++i)
    if (partial[i] >= 0) {
      int j = partial[i];
      if (j >= n || seen[j]++ ||
          certificate.types_r[i] != certificate.types_p[j])
        return -1;
    }
  *bound = certificate.lower_bound(partial, seen.data());
  return 0;
}

extern "C" void synkit_private_conditioned_dual_release(void *handle) {
  delete static_cast<RootDualBranchCertificate *>(handle);
}

extern "C" int synkit_private_conditioned_dual_query_threshold(
    void *handle, int n, const int32_t *partial, int quarter_target,
    int64_t *bound) {
  if (!handle || !partial || !bound)
    return -1;
  auto &certificate = *static_cast<RootDualBranchCertificate *>(handle);
  if (!certificate.ready || n != certificate.n)
    return -1;
  Perm seen(n, 0);
  for (int i = 0; i < n; ++i)
    if (partial[i] >= 0) {
      int j = partial[i];
      if (j >= n || seen[j]++ ||
          certificate.types_r[i] != certificate.types_p[j])
        return -1;
    }
  *bound = certificate.lower_bound(partial, seen.data(), quarter_target);
  return 0;
}

extern "C" int synkit_private_conditioned_triplet_query(void *handle,
                                                        double seconds,
                                                        int64_t *bound,
                                                        int64_t *stats) {
  if (!handle || !bound || !stats || seconds < 0 || !std::isfinite(seconds))
    return -1;
  try {
    const auto &c = *static_cast<RootDualBranchCertificate *>(handle);
    *bound = c.triplet_floor(Clock::now() +
                                 std::chrono::duration_cast<Clock::duration>(
                                     std::chrono::duration<double>(seconds)),
                             stats);
    return 0;
  } catch (...) {
    return -1;
  }
}
