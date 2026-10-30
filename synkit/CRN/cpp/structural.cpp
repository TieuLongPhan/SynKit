// Exact minimal siphon/trap models: implication propagation and closure minimization.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>
namespace {
using Mask = std::vector<std::uint64_t>;
using Clock = std::chrono::steady_clock;
struct Limit { const char* reason; };
struct Hash {
    std::size_t operator()(const Mask& mask) const {
        std::size_t h=0;
        for (auto w: mask) h ^= std::hash<std::uint64_t>{}(w)+0x9e3779b9+(h<<6)+(h>>2);
        return h;
    }
};
struct Search {
    std::size_t n, words, transitions;
    const std::uint64_t* trigger;
    const std::uint64_t* required;
    Clock::time_point deadline;
    void check() const { if (deadline != Clock::time_point::max() && Clock::now() >= deadline) throw Limit{"time_limit"}; }
    static bool nonempty(const Mask& mask) {
        return std::any_of(mask.begin(), mask.end(), [](std::uint64_t w){return w!=0;});
    }
    Mask closure(Mask current) const {
        bool changed=true;
        while (changed) {
            check(); changed=false;
            for (std::size_t t=0; t<transitions; ++t) {
                if ((t & 127) == 0) check();
                bool hit=false, satisfied=false;
                for (std::size_t w=0; w<words; ++w) {
                    hit |= bool(trigger[t*words+w] & current[w]);
                    satisfied |= bool(required[t*words+w] & current[w]);
                }
                if (hit && !satisfied) {
                    for (std::size_t w=0; w<words; ++w) current[w] &= ~trigger[t*words+w];
                    changed=true;
                }
            }
        }
        return current;
    }
};
}
extern "C" {
int synkit_crn_structural_abi() { return 2; }
int synkit_crn_structural_sets(std::size_t n, std::size_t transitions,
                              const std::uint64_t* trigger, const std::uint64_t* required,
                              double seconds, std::uint64_t max_nodes, std::size_t max_results,
                              char** output) {
    if (!output || !std::isfinite(seconds) || seconds < 0 || seconds > 1e9) return 2;
    *output=nullptr;
    Search search{n, (n+63)/64, transitions, trigger, required,
        seconds == 0 ? Clock::time_point::max() :
        Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds))};
    struct State { Mask selected, excluded; };
    std::vector<Mask> results;
    std::unordered_set<Mask,Hash> emitted;
    std::uint64_t evaluations=0, states=0, closures=0;
    std::size_t peak=0;
    const char* reason="complete";
    auto tick = [&]() {
        search.check();
        if (max_nodes && evaluations>=max_nodes) throw Limit{"node_limit"};
        ++evaluations;
    };
    auto count = [](const Mask& mask) {
        std::size_t total=0;
        for (auto word: mask) {
#if defined(__GNUC__) || defined(__clang__)
            total += __builtin_popcountll(word);
#else
            while (word) { word &= word - 1; ++total; }
#endif
        }
        return total;
    };
    try {
        // Each minimal model has one least selected species. These roots and
        // the ordered positive/negative branches partition the model space.
        for (std::size_t seed=0; seed<n; ++seed) {
            State root{Mask(search.words,0),Mask(search.words,0)};
            root.selected[seed/64] |= std::uint64_t(1)<<(seed%64);
            for (std::size_t p=0; p<seed; ++p)
                root.excluded[p/64] |= std::uint64_t(1)<<(p%64);
            std::vector<State> stack;
            stack.push_back(std::move(root));
            while (!stack.empty()) {
                tick(); ++states;
                State current=std::move(stack.back()); stack.pop_back();
                bool conflict=false, changed=true;
                while (changed && !conflict) {
                    search.check(); changed=false;
                    // An emitted minimal model blocks every strict superset.
                    for (const auto& known: results) {
                        bool blocked=false;
                        Mask missing(search.words,0);
                        for (std::size_t w=0; w<search.words; ++w) {
                            blocked |= bool(known[w] & current.excluded[w]);
                            missing[w] = known[w] & ~current.selected[w];
                        }
                        if (blocked) continue;
                        auto remaining=count(missing);
                        if (!remaining) { conflict=true; break; }
                        if (remaining==1) {
                            for (std::size_t w=0; w<search.words; ++w) {
                                auto next=current.excluded[w] | missing[w];
                                changed |= next != current.excluded[w];
                                current.excluded[w]=next;
                            }
                        }
                    }
                    if (conflict) break;
                    for (std::size_t t=0; t<transitions; ++t) {
                        if ((t & 127)==0) search.check();
                        bool hit=false, satisfied=false;
                        Mask choices(search.words,0);
                        for (std::size_t w=0; w<search.words; ++w) {
                            hit |= bool(trigger[t*search.words+w] & current.selected[w]);
                            satisfied |= bool(required[t*search.words+w] & current.selected[w]);
                            choices[w]=required[t*search.words+w] & ~current.excluded[w];
                        }
                        if (satisfied) continue;
                        auto available=count(choices);
                        if (!available) {
                            for (std::size_t w=0; w<search.words; ++w) {
                                auto next=current.excluded[w] | trigger[t*search.words+w];
                                changed |= next != current.excluded[w];
                                current.excluded[w]=next;
                            }
                        } else if (hit && available==1) {
                            for (std::size_t w=0; w<search.words; ++w) {
                                auto next=current.selected[w] | choices[w];
                                changed |= next != current.selected[w];
                                current.selected[w]=next;
                            }
                        }
                    }
                    for (std::size_t w=0; w<search.words; ++w)
                        conflict |= bool(current.selected[w] & current.excluded[w]);
                }
                if (conflict) continue;
                Mask choices(search.words,0);
                bool needs_branch=false;
                std::size_t best=n+1;
                for (std::size_t t=0; t<transitions; ++t) {
                    bool hit=false, satisfied=false;
                    Mask available(search.words,0);
                    for (std::size_t w=0; w<search.words; ++w) {
                        hit |= bool(trigger[t*search.words+w] & current.selected[w]);
                        satisfied |= bool(required[t*search.words+w] & current.selected[w]);
                        available[w]=required[t*search.words+w] & ~current.excluded[w];
                    }
                    if (hit && !satisfied && count(available)<best) {
                        choices=std::move(available); best=count(choices); needs_branch=true;
                    }
                }
                if (!needs_branch) {
                    Mask minimal=current.selected;
                    for (std::size_t p=0; p<n; ++p) {
                        if (!(minimal[p/64] & (std::uint64_t(1)<<(p%64)))) continue;
                        tick(); ++closures;
                        Mask child=minimal;
                        child[p/64] &= ~(std::uint64_t(1)<<(p%64));
                        child=search.closure(std::move(child));
                        if (Search::nonempty(child)) minimal=std::move(child);
                    }
                    if (emitted.insert(minimal).second) results.push_back(std::move(minimal));
                    if (max_results && results.size()>=max_results &&
                        (!stack.empty() || seed+1<n)) throw Limit{"result_limit"};
                } else {
                    // Choose one required member; earlier choices are false
                    // in later alternatives, avoiding redundant unions.
                    for (std::size_t p=0; p<n; ++p) {
                        auto bit=std::uint64_t(1)<<(p%64);
                        if (!(choices[p/64] & bit)) continue;
                        State child=current;
                        child.selected[p/64] |= bit;
                        stack.push_back(std::move(child));
                        current.excluded[p/64] |= bit;
                    }
                    peak=std::max(peak,stack.size());
                }
            }
        }
    } catch (const Limit& limit) { reason=limit.reason; }
      catch (const std::bad_alloc&) { reason="allocation_failure"; }
      catch (...) { return 2; }
    try {
        std::ostringstream out;
        out << "{\"termination_reason\":\"" << reason
            << "\",\"algorithm\":\"implication propagation and closure minimization\""
            << ",\"search_evaluations\":" << evaluations
            << ",\"states_explored\":" << states
            << ",\"closure_evaluations\":" << closures
            << ",\"peak_frontier\":" << peak << ",\"sets\":[";
        for (std::size_t r=0; r<results.size(); ++r) {
            if (r) out << ',';
            out << '['; bool first=true;
            for (std::size_t p=0; p<n; ++p) if (results[r][p/64] & (std::uint64_t(1)<<(p%64))) {
                if (!first) out << ',';
                first=false; out << p;
            }
            out << ']';
        }
        out << "]}";
        auto text=out.str();
        auto* buffer=static_cast<char*>(std::malloc(text.size()+1));
        if (!buffer) return 2;
        std::memcpy(buffer,text.c_str(),text.size()+1); *output=buffer;
        return 0;
    } catch (...) { return 2; }
}
}

extern "C" int synkit_crn_exhaustive_siphons(std::size_t n,std::size_t transitions,
 const std::uint64_t* pre,const std::uint64_t* post,double seconds,char** output){
 if(!output||n>24||!std::isfinite(seconds)||seconds<=0||seconds>1e9)return 2;
 *output=nullptr;auto deadline=Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
 std::vector<std::uint64_t> minimal;const char* reason="complete";std::uint64_t inspected=0;
 try{
  for(std::uint64_t mask=1;mask<(std::uint64_t(1)<<n);++mask){
   if(Clock::now()>=deadline)throw Limit{"time_limit"};++inspected;bool valid=true;
   for(std::size_t t=0;t<transitions;++t)if((post[t]&mask) && !(pre[t]&mask)){valid=false;break;}
   if(!valid)continue;bool dominated=false;for(auto m:minimal)if((m&mask)==m){dominated=true;break;}
   if(!dominated)minimal.push_back(mask);
  }
 }catch(const Limit& l){reason=l.reason;}catch(...){return 2;}
 std::ostringstream out;out<<"{\"termination_reason\":\""<<reason<<"\",\"subsets\":"<<inspected<<",\"sets\":[";
 for(std::size_t r=0;r<minimal.size();++r){if(r)out<<',';out<<'[';bool first=true;for(std::size_t i=0;i<n;++i)if(minimal[r]&(std::uint64_t(1)<<i)){if(!first)out<<',';first=false;out<<i;}out<<']';}out<<"]}";
 auto text=out.str();auto* buffer=static_cast<char*>(std::malloc(text.size()+1));if(!buffer)return 2;std::memcpy(buffer,text.c_str(),text.size()+1);*output=buffer;return 0;
}
