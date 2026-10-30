// Exact nonnegative cone generators by Farkas elimination.
// Boost.Multiprecision is header-only; coefficients have arbitrary precision.
#include <boost/multiprecision/cpp_int.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Integer = boost::multiprecision::cpp_int;
using Clock = std::chrono::steady_clock;
struct Limit { const char* reason; };
struct Row {
    std::vector<Integer> values;
    std::vector<std::uint64_t> support;
    std::size_t support_size = 0;
};
struct Search {
    std::size_t n, columns, max_rows, max_results, max_bits;
    std::uint64_t max_candidates, candidates = 0;
    std::size_t peak_rows = 0, eliminated = 0;
    Clock::time_point deadline;
    void check() const {
        if (Clock::now() >= deadline) throw Limit{"time_limit"};
    }
    bool full_rank(const std::int64_t* matrix) const {
        if (n > columns) return false;
        constexpr std::int64_t prime = 1000003;
        std::vector<std::vector<std::int64_t>> work(n, std::vector<std::int64_t>(columns));
        for (std::size_t i=0; i<n; ++i) {
            check();
            for (std::size_t j=0; j<columns; ++j) {
                auto v = matrix[i*columns+j] % prime;
                work[i][j] = v < 0 ? v+prime : v;
            }
        }
        std::size_t rank=0;
        for (std::size_t col=0; col<columns && rank<n; ++col) {
            check();
            std::size_t pivot=rank;
            while (pivot<n && work[pivot][col]==0) ++pivot;
            if (pivot==n) continue;
            std::swap(work[pivot], work[rank]);
            std::int64_t inverse=1, power=prime-2, base=work[rank][col];
            while (power) { if (power&1) inverse=(inverse*base)%prime; base=(base*base)%prime; power>>=1; }
            for (std::size_t j=col; j<columns; ++j) work[rank][j]=(work[rank][j]*inverse)%prime;
            for (std::size_t i=rank+1; i<n; ++i) {
                check();
                auto factor=work[i][col];
                if (!factor) continue;
                for (std::size_t j=col; j<columns; ++j)
                    work[i][j]=(work[i][j] - factor*work[rank][j]%prime + prime)%prime;
            }
            ++rank;
        }
        return rank==n;
    }
    void normalize(Row& row) const {
        Integer g = 0;
        for (std::size_t i=0; i<n; ++i) {
            check();
            if (row.values[i] != 0) {
                if (max_bits && boost::multiprecision::msb(row.values[i]) >= max_bits)
                    throw Limit{"coefficient_limit"};
                Integer b = row.values[i];
                while (b != 0) { Integer r = g % b; g = std::move(b); b = std::move(r); }
                if (g == 1) break;
            }
        }
        if (g > 1) for (auto& v: row.values) { check(); v /= g; }
        row.support.assign((n+63)/64, 0);
        row.support_size = 0;
        for (std::size_t i=0; i<n; ++i) if (row.values[i] != 0) {
            if (max_bits && boost::multiprecision::msb(row.values[i]) >= max_bits)
                throw Limit{"coefficient_limit"};
            row.support[i/64] |= std::uint64_t(1) << (i%64);
            ++row.support_size;
        }
    }
    static bool subset(const Row& a, const Row& b) {
        if (a.support_size > b.support_size) return false;
        for (std::size_t w=0; w<a.support.size(); ++w)
            if (a.support[w] & ~b.support[w]) return false;
        return true;
    }
    void insert(std::vector<Row>& kept, Row row) {
        check();
        // Incremental antichain filtering avoids materializing all pair rows.
        for (const auto& k: kept) {
            check();
            if (subset(k, row)) return;
        }
        kept.erase(std::remove_if(kept.begin(), kept.end(), [&](const Row& k) {
            check(); return subset(row, k);
        }), kept.end());
        if (kept.size() >= max_rows) throw Limit{"row_limit"};
        kept.push_back(std::move(row));
        peak_rows = std::max(peak_rows, kept.size());
    }
};
char* copy_text(const std::string& text) {
    auto* buffer = static_cast<char*>(std::malloc(text.size()+1));
    if (!buffer) return nullptr;
    std::memcpy(buffer, text.c_str(), text.size()+1);
    return buffer;
}
}
extern "C" {
int synkit_crn_semiflows_abi() { return 1; }
// Returns JSON in *output, owned by caller via synkit_crn_free_text.
// On any resource limit no intermediate (non-kernel) tableau rows are exposed.
int synkit_crn_semiflows(std::size_t n, std::size_t columns, const std::int64_t* matrix,
                        double seconds, std::uint64_t max_candidates,
                        std::size_t max_rows, std::size_t max_results, std::size_t max_bits,
                        char** output) {
    if (!output || !std::isfinite(seconds) || seconds <= 0 || seconds > 1e9) return 2;
    *output = nullptr;
    Search search{n, columns, max_rows, max_results, max_bits, max_candidates};
    search.deadline = Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
    const char* reason = "complete";
    std::vector<Row> rows;
    try {
        const bool empty_kernel = search.full_rank(matrix);
        for (std::size_t i=0; !empty_kernel && i<n; ++i) {
            search.check();
            Row row;
            row.values.resize(n+columns);
            row.values[i] = 1;
            for (std::size_t j=0; j<columns; ++j) row.values[n+j] = matrix[i*columns+j];
            search.normalize(row);
            if (rows.size() >= max_rows) throw Limit{"row_limit"};
            rows.push_back(std::move(row));
        }
        search.peak_rows = rows.size();
        for (std::size_t col=0; col<columns && !rows.empty(); ++col) {
            search.check();
            std::vector<std::size_t> pos, neg;
            std::vector<Row> kept;
            for (std::size_t i=0; i<rows.size(); ++i) {
                const auto& v = rows[i].values[n+col];
                if (v > 0) pos.push_back(i);
                else if (v < 0) neg.push_back(i);
                else kept.push_back(rows[i]); // Existing rows already form an antichain.
            }
            for (auto pi: pos) for (auto ni: neg) {
                search.check();
                if (search.candidates >= max_candidates) throw Limit{"candidate_limit"};
                ++search.candidates;
                const Row& p = rows[pi]; const Row& m = rows[ni];
                // Positive combinations have exactly the union support. Reject
                // dominated supports before arbitrary-precision arithmetic.
                Row support_union;
                support_union.support.resize(p.support.size());
                for (std::size_t w=0; w<p.support.size(); ++w) {
                    auto word=p.support[w] | m.support[w];
                    support_union.support[w]=word;
                    while (word) { word &= word-1; ++support_union.support_size; }
                }
                bool dominated=false;
                for (const auto& k: kept) {
                    search.check();
                    if (Search::subset(k, support_union)) { dominated=true; break; }
                }
                if (dominated) continue;
                Integer a = p.values[n+col], b = -m.values[n+col];
                Row combined;
                combined.values.resize(n+columns);
                for (std::size_t j=0; j<n+columns; ++j) {
                    search.check();
                    combined.values[j] = b*p.values[j] + a*m.values[j];
                }
                search.normalize(combined);
                search.insert(kept, std::move(combined));
            }
            rows = std::move(kept);
            ++search.eliminated;
        }
        if (rows.size() > max_results) throw Limit{"result_limit"};
    } catch (const Limit& limit) {
        reason = limit.reason;
        rows.clear();
    } catch (const std::bad_alloc&) {
        reason = "allocation_failure";
        rows.clear();
    } catch (...) { return 2; }
    try {
        std::ostringstream out;
        out << "{\"termination_reason\":\"" << reason << "\",\"candidates\":" << search.candidates
            << ",\"peak_rows\":" << search.peak_rows << ",\"eliminated_columns\":" << search.eliminated
            << ",\"flows\":[";
        for (std::size_t r=0; r<rows.size(); ++r) {
            if (r) out << ',';
            out << '[';
            for (std::size_t i=0; i<n; ++i) { if (i) out << ','; out << rows[r].values[i]; }
            out << ']';
        }
        out << "]}";
        *output = copy_text(out.str());
        return *output ? 0 : 2;
    } catch (...) { return 2; }
}
}
