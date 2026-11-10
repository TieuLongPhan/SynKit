#include "synkit_canon/configured.hpp"
#include <map>
#include <numeric>

namespace synkit { namespace stereo {
std::vector<Configuration> local_arrangements(const Configuration &source) {
  const int family=static_cast<int>(source.geometry);
  std::vector<Configuration> result;
  auto add=[&](Configuration d) { d.fixed=true; result.push_back(std::move(d)); };
  if (family>=1 && family<=4) {
    const size_t arity=family==3?6:(family==4?7:5);
    if (source.frame.size()!=arity) throw std::invalid_argument("Invalid atom arrangement arity");
    std::vector<int> p(arity); std::iota(p.begin(),p.end(),0);
    do { auto d=source; d.parity=family==2?0:1; for (size_t i=0;i<arity;++i) d.frame[i]=source.frame[p[i]]; add(d); }
    while (std::next_permutation(p.begin()+1,p.end()));
  } else if (family>=5 && family<=8) {
    if (source.frame.size()!=6) throw std::invalid_argument("Invalid terminal arrangement arity");
    for (int end=0;end<2;++end) for (int left=0;left<2;++left) for (int right=0;right<2;++right) {
      auto d=source; d.parity=family==5 || family==8?0:1;
      const int l=end?4:0,r=end?0:4;
      d.frame={source.frame[l+left],source.frame[l+1-left],source.frame[2+end],source.frame[3-end],source.frame[r+right],source.frame[r+1-right]};
      if (end && (family==7 || family==8)) std::reverse(d.path.begin(),d.path.end()); add(d);
    }
  } else if (family==9 || family==10) {
    if (source.path.size()<3) throw std::invalid_argument("Missing path/plane arrangement support");
    bool cyclic=family==10 || source.cyclic;
    for (int reverse=0;reverse<2;++reverse) for (size_t offset=0;offset<(cyclic?source.path.size():1);++offset) for (int sign:{-1,1}) {
      auto d=source; if (reverse) std::reverse(d.path.begin(),d.path.end());
      std::rotate(d.path.begin(),d.path.begin()+offset,d.path.end()); d.parity=sign; add(d);
    }
  } else if (family==11) {
    auto d=source; d.parity=1; add(d); d.parity=-1; add(d);
  } else throw std::invalid_argument("Unsupported local arrangement geometry");
  // Reject repeated ports rather than silently shrinking the declared raw
  // domain; valid fixed configurations already require distinct references.
  for (const auto &d:result) {
    auto refs=d.frame; std::sort(refs.begin(),refs.end());
    if (std::adjacent_find(refs.begin(),refs.end())!=refs.end()) throw std::invalid_argument("Repeated arrangement port");
  }
  return result;
}
}} // namespace
