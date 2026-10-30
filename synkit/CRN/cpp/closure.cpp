// Exact Boolean incidence closure. Coefficients remain in the Python model.
#include <cstdint>
#include <cstddef>
#include <cstdlib>
extern "C" {
int synkit_crn_abi() { return 1; }
void synkit_crn_free_text(void* pointer) { std::free(pointer); }
int synkit_crn_closure(std::size_t words, std::size_t transitions,
                      const std::uint64_t* trigger, const std::uint64_t* required,
                      std::uint64_t* current) {
    bool changed = true;
    while (changed) {
        changed = false;
        for (std::size_t t=0; t<transitions; ++t) {
            bool hit=false, satisfied=false;
            for (std::size_t w=0; w<words; ++w) {
                hit |= bool(trigger[t*words+w] & current[w]);
                satisfied |= bool(required[t*words+w] & current[w]);
            }
            if (hit && !satisfied) {
                for (std::size_t w=0; w<words; ++w)
                    current[w] &= ~trigger[t*words+w];
                changed=true;
            }
        }
    }
    return 0;
}
}
