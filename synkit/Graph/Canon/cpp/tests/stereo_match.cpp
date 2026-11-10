#include "../examples/three_center.h"
#include "../src/stereo_groups.h"
#include <future>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <tuple>

namespace {
using namespace synkit::stereo;
using synkit_stereo_detail::Image;
int checks = 0;
void require(bool condition, const char *message) {
  ++checks;
  if (!condition) throw std::runtime_error(message);
}
template<class F> void invalid(F run) {
  bool rejected = false;
  try { run(); } catch (const std::invalid_argument &) { rejected = true; }
  require(rejected, "Malformed/unsupported input was accepted");
}
int sign(const Image &p, int begin, int end) {
  int odd = 0;
  for (int i = begin; i < end; ++i) for (int j = i + 1; j < end; ++j) odd ^= p[i] > p[j];
  return odd ? -1 : 1;
}
// Independent geometric predicates, rather than the production group table.
bool allowed(int geometry, const Image &p) {
  if (geometry == SYNKIT_TETRAHEDRAL) return p[0] == 0 && sign(p, 1, 5) == 1;
  if (geometry == SYNKIT_SQUARE_PLANAR) {
    if (p[0] != 0) return false;
    auto opposite = [](int a, int b) { return std::abs(a - b) == 2; };
    return opposite(p[1], p[3]) && opposite(p[2], p[4]);
  }
  if (geometry == SYNKIT_TRIGONAL_BIPYRAMIDAL)
    return p[0] == 0 && p[1] <= 2 && p[2] <= 2 &&
           p[3] >= 3 && p[4] >= 3 && p[5] >= 3 &&
           (p[1] == 1 ? 1 : -1) == sign(p, 3, 6);
  if (geometry == SYNKIT_OCTAHEDRAL) {
    if (p[0] != 0) return false;
    const int xyz[7][3] = {{0,0,0}, {0,0,1}, {0,0,-1}, {1,0,0}, {0,1,0}, {-1,0,0}, {0,-1,0}};
    auto dot = [&](int a, int b) {
      return xyz[a][0]*xyz[b][0] + xyz[a][1]*xyz[b][1] + xyz[a][2]*xyz[b][2];
    };
    for (int i = 1; i <= 6; ++i) for (int j = 1; j <= 6; ++j)
      if (dot(i,j) != dot(p[i],p[j])) return false;
    const auto &x = xyz[p[3]], &y = xyz[p[4]], &z = xyz[p[1]];
    const int determinant = x[0]*(y[1]*z[2]-y[2]*z[1]) -
                            x[1]*(y[0]*z[2]-y[2]*z[0]) + x[2]*(y[0]*z[1]-y[1]*z[0]);
    return determinant == 1;
  }
  const bool reversed = p[2] == 3 && p[3] == 2;
  if (!reversed && !(p[2] == 2 && p[3] == 3)) return false;
  const int left = reversed ? 4 : 0, right = reversed ? 0 : 4;
  if (!((p[0] == left && p[1] == left+1) || (p[0] == left+1 && p[1] == left)) ||
      !((p[4] == right && p[5] == right+1) || (p[4] == right+1 && p[5] == right))) return false;
  if (geometry==SYNKIT_PLANAR_BOND) return (p[0]!=left)==(p[4]!=right);
  // Independent coordinate witness: reversing ends changes both t and the
  // order of the terminal vectors, so it cannot reverse axial chirality.
  const int xyz[6][3]={{1,0,-1},{-1,0,-1},{0,0,-1},{0,0,1},{0,1,1},{0,-1,1}};
  int t[3],l[3],r[3];
  for (int i=0;i<3;++i) { t[i]=xyz[p[3]][i]-xyz[p[2]][i]; l[i]=xyz[p[0]][i]-xyz[p[2]][i]; r[i]=xyz[p[4]][i]-xyz[p[3]][i]; }
  return t[0]*(l[1]*r[2]-l[2]*r[1])-t[1]*(l[0]*r[2]-l[2]*r[0])+t[2]*(l[0]*r[1]-l[1]*r[0])>0;
}
Graph local(int geometry, bool unique = true) {
  Graph g;
  const int arity = synkit_stereo_detail::group(geometry).arity;
  for (int i = 0; i < arity; ++i) g.colors.push_back(unique ? i : 0);
  if (geometry >= SYNKIT_PLANAR_BOND) {
    g.edges = {{0,2}, {1,2}, {2,3}, {3,4}, {3,5}};
  } else for (int i = 1; i < arity; ++i) g.edges.push_back({0,i});
  SynkitStereoFrame f{geometry,
    geometry == SYNKIT_SQUARE_PLANAR || geometry == SYNKIT_PLANAR_BOND ? 0 : 1, {}};
  std::iota(f.vertices, f.vertices + arity, 0);
  g.frames.push_back(f);
  return g;
}
Graph relabel(const Graph &g, const std::vector<int32_t> &p) {
  Graph h;
  h.colors.resize(g.colors.size());
  for (size_t i = 0; i < p.size(); ++i) h.colors[p[i]] = g.colors[i];
  for (const auto &e : g.edges) h.edges.push_back({p[e.source], p[e.target], e.color});
  for (auto f : g.frames) {
    for (int i = 0; i < synkit_stereo_detail::group(f.geometry).arity; ++i) f.vertices[i] = p[f.vertices[i]];
    h.frames.push_back(f);
  }
  std::reverse(h.edges.begin(), h.edges.end());
  std::reverse(h.frames.begin(), h.frames.end());
  return h;
}
bool witness(const Graph &a, const Graph &b, const std::vector<int32_t> &p, bool stereo = true) {
  if (p.size() != a.colors.size() || b.colors.size() != p.size()) return false;
  std::set<int32_t> unique(p.begin(), p.end());
  if (unique.size() != p.size()) return false;
  std::set<std::tuple<int,int,int>> edges;
  for (const auto &e : b.edges) edges.emplace(std::min(e.source,e.target), std::max(e.source,e.target), e.color);
  for (size_t i = 0; i < p.size(); ++i)
    if (p[i] < 0 || static_cast<size_t>(p[i]) >= p.size() || a.colors[i] != b.colors[p[i]]) return false;
  if (a.edges.size() != b.edges.size()) return false;
  for (const auto &e : a.edges)
    if (!edges.count({std::min(p[e.source],p[e.target]), std::max(p[e.source],p[e.target]),e.color})) return false;
  if (!stereo) return true;
  if (a.frames.size() != b.frames.size()) return false;
  // Independent inverse-position permutation test on the actual global map.
  for (const auto &f : a.frames) {
    bool found = false;
    for (const auto &h : b.frames) if (f.geometry == h.geometry) {
      Image permutation{};
      const int arity = synkit_stereo_detail::group(f.geometry).arity;
      bool all = true;
      // Tests using this oracle have positive/zero parity.
      for (int i = 0; i < arity; ++i) {
        auto at = std::find(h.vertices, h.vertices + arity, p[f.vertices[i]]);
        if (at == h.vertices + arity) { all = false; break; }
        permutation[i] = static_cast<int32_t>(at - h.vertices);
      }
      if (all && allowed(f.geometry, permutation)) { found = true; break; }
    }
    if (!found) return false;
  }
  return true;
}
bool brute(const Graph &a, const Graph &b) {
  if (a.colors.size() != b.colors.size()) return false;
  std::vector<int32_t> p(a.colors.size()); std::iota(p.begin(), p.end(), 0);
  do { if (witness(a,b,p)) return true; } while (std::next_permutation(p.begin(),p.end()));
  return false;
}
void groups() {
  const int sizes[] = {0,12,8,6,24,4,4};
  for (int geometry = 1; geometry <= 6; ++geometry) {
    const auto derived = synkit_stereo_detail::group(geometry);
    std::set<Image> images(derived.images,derived.images+derived.size);
    require(images.size() == static_cast<size_t>(sizes[geometry]),"Derived group has duplicates or wrong size");
    for (const auto &left : images) for (const auto &right : images) {
      Image composition{};
      for (int i=0;i<derived.arity;++i) composition[i]=left[right[i]];
      require(images.count(composition) == 1,"Derived group is not closed under composition");
    }
    const auto base = local(geometry);
    const int arity = static_cast<int>(base.colors.size());
    Image p{}; std::iota(p.begin(), p.begin()+arity, 0);
    int accepted = 0;
    do {
      // Keep owners in place/up to endpoint exchange, since other permutations
      // would describe an invalid local incidence rather than a stereo state.
      if (geometry < 5 && p[0] != 0) continue;
      if (geometry >= 5 && !((p[2]==2 && p[3]==3) || (p[2]==3 && p[3]==2))) continue;
      if (geometry >= 5) {
        const bool reversed = p[2] == 3;
        if ((p[0] >= 4) != reversed || (p[1] >= 4) != reversed ||
            (p[4] >= 4) == reversed || (p[5] >= 4) == reversed) continue;
      }
      Graph other = base;
      std::copy_n(p.begin(), arity, other.frames[0].vertices);
      const auto result = isomorphism(base, other);
      require(result.complete(), "Local permutation search stopped");
      const bool expected = allowed(geometry, p);
      require(result.isomorphic() == expected, "Local group differs from geometric oracle");
      accepted += expected;
      if (expected) require(witness(base, other, result.mapping), "Invalid group witness");
      if (base.frames[0].parity == 1) {
        auto negative = other;
        negative.frames[0].parity = -1;
        // Independently absorb the negative orientation into the target frame.
        Image q = p;
        if (geometry == 3) std::swap(q[4],q[5]);
        else if (geometry == 6) std::swap(q[0],q[1]);
        else std::swap(q[1],q[2]);
        require(isomorphism(base, negative).isomorphic() == allowed(geometry,q), "Negative parity mismatch");
        auto both = base; both.frames[0].parity = -1;
        require(isomorphism(both, negative).isomorphic() == expected, "Double parity mismatch");
      }
    } while (std::next_permutation(p.begin(), p.begin()+arity));
    require(accepted == sizes[geometry], "Wrong preserving group order");
  }
}
void three_sites() {
  std::mt19937 rng(20261102);
  for (bool aldehyde : {false,true}) for (bool middle_only : {false,true}) {
    for (const auto &a : three_center::words) for (const auto &b : three_center::words) {
      const auto ga = three_center::molecule(a,aldehyde,middle_only);
      const auto gb = three_center::molecule(b,aldehyde,middle_only);
      const bool expected = middle_only ? (!aldehyde || a[1] == b[1]) :
        (a == b || (!aldehyde && three_center::rotated(a) == b));
      auto result = isomorphism(ga,gb);
      require(result.complete() && result.isomorphic() == expected, "Three-center partition mismatch");
      if (expected) require(witness(ga,gb,result.mapping), "Three-center witness invalid");
      Options deferred; deferred.defer_stereo = true;
      require(isomorphism(ga,gb,deferred).isomorphic() == expected, "Deferred search disagrees");
      std::vector<int32_t> p(gb.colors.size()); std::iota(p.begin(),p.end(),0);
      std::shuffle(p.begin(),p.end(),rng);
      const auto renamed = relabel(gb,p);
      result = isomorphism(ga,renamed);
      require(result.isomorphic() == expected, "Relabeling changed identity");
      if (expected) require(witness(ga,renamed,result.mapping), "Relabeled witness invalid");
    }
  }
  const auto flip = isomorphism(three_center::molecule("LRL"),three_center::molecule("RLR"));
  require(flip.mapping[1] == 3 && flip.mapping[2] == 2 && flip.mapping[3] == 1,
          "Symmetric case must reverse the whole chain");
}
void random_oracle() {
  std::mt19937 rng(17);
  for (int sample = 0; sample < 100; ++sample) {
    auto a = local(SYNKIT_TETRAHEDRAL,false);
    a.colors.push_back(rng()%2); a.colors.push_back(rng()%2);
    for (int i = 0; i < 7; ++i) for (int j = std::max(5,i+1); j < 7; ++j)
      if (rng()%3 == 0) a.edges.push_back({i,j,static_cast<int32_t>(rng()%2)});
    for (auto &color : a.colors) color = rng()%2;
    auto b = a;
    if (rng()%2) std::swap(b.frames[0].vertices[1],b.frames[0].vertices[2]);
    if (rng()%3 == 0) b.colors[rng()%7] ^= 1;
    const bool expected = brute(a,b);
    auto result = isomorphism(a,b);
    require(result.isomorphic() == expected, "Matcher disagrees with independent exhaustive oracle");
    if (expected) require(witness(a,b,result.mapping), "Random witness invalid");
  }
  // WL cannot distinguish these 3-regular graphs. Refinement equality must
  // not be treated as an isomorphism certificate.
  Graph prism, bipartite; prism.colors.resize(6); bipartite.colors.resize(6);
  prism.edges = {{0,1},{1,2},{2,0},{3,4},{4,5},{5,3},{0,3},{1,4},{2,5}};
  for (int i=0;i<3;++i) for(int j=3;j<6;++j) bipartite.edges.push_back({i,j});
  require(!isomorphism(prism,bipartite).isomorphic(), "WL collision accepted as identity");
}
void boundaries() {
  auto a = local(SYNKIT_TETRAHEDRAL);
  Graph empty;
  require(isomorphism(empty,empty).isomorphic(), "Empty graphs differ");
  require(!isomorphism(a,empty).isomorphic(), "Different sizes match");
  auto b = a; b.colors[1] = 99;
  require(!isomorphism(a,b).isomorphic(), "Shared palette values ignored");
  b = a; b.edges[0].color = 1;
  require(!isomorphism(a,b).isomorphic(), "Edge color ignored");
  b = a; b.frames.clear();
  require(!isomorphism(a,b).isomorphic(), "Missing frame inventory ignored");
  b = a; b.frames.push_back(b.frames.front()); invalid([&]{isomorphism(a,b);});
  b = a; b.frames[0].parity=0; invalid([&]{isomorphism(a,b);});
  b = a; b.frames[0].geometry=99; invalid([&]{isomorphism(a,b);});
  b = a; b.frames[0].vertices[1]=100; invalid([&]{isomorphism(a,b);});
  b = a; b.frames[0].vertices[1]=b.frames[0].vertices[2]; invalid([&]{isomorphism(a,b);});
  b = a; b.edges.pop_back(); invalid([&]{isomorphism(a,b);});
  b = a; b.edges.push_back(b.edges[0]); invalid([&]{isomorphism(a,b);});
  b = a; b.edges[0].color=-1; invalid([&]{isomorphism(a,b);});
  for (int reason : {1,2,3,4,5,6}) {
    Options opts;
    void *token = nullptr;
    if (reason == 1) opts.seconds = 0;
    if (reason == 2) opts.max_nodes = 1;
    if (reason == 3) opts.workspace_bytes = 1;
    if (reason == 4) opts.output_bytes = 1;
    if (reason == 5) {
      token = synkit_canon_cancel_create(); require(token != nullptr,"Token allocation failed");
      synkit_canon_cancel_request(token); opts.cancel_token = token;
    }
    if (reason == 6) opts.max_depth = 0;
    auto result = isomorphism(a,a,opts);
    synkit_canon_cancel_free(token);
    require(!result.complete() && result.statistics.isomorphic == -1 && result.mapping.empty() &&
            result.statistics.reason == reason, "Stopped search presented a verdict or witness");
    bool rejected = false;
    try { result.isomorphic(); } catch (const std::logic_error &) { rejected = true; }
    require(rejected,"Incomplete comparison allowed as bool");
  }
  auto options = Options{}; options.max_nodes=0; invalid([&]{isomorphism(a,a,options);});
  options = Options{}; options.max_depth=1025; invalid([&]{isomorphism(a,a,options);});
  // Malformed CSR must not be searched, even when the other graph differs.
  detail::Packed packed(a);
  SynkitStereoRequest r{1,sizeof(SynkitStereoRequest),packed.view,packed.view,-1,-1,-1,-1,1024,0,nullptr};
  packed.neighbors[0]=999;
  SynkitStereoResult *raw=nullptr;
  require(synkit_stereo_match(&r,&raw)==-1 && !raw,"Invalid raw CSR accepted");
  require(synkit_stereo_match(nullptr,&raw)==-1 && !raw,"Null request accepted");
  int64_t size=99;
  require(!synkit_stereo_mapping(nullptr,&size) && size==0,"Null result mapping invalid");
  // Calls share no search state.
  std::vector<std::future<bool>> jobs;
  for (int i=0;i<4;++i) jobs.push_back(std::async(std::launch::async,[a]{return isomorphism(a,a).isomorphic();}));
  for (auto &job : jobs) require(job.get(),"Concurrent comparison disagrees");
}
void pruning() {
  const auto symmetric_a = three_center::molecule("LLL");
  const auto symmetric_b = three_center::molecule("LRL");
  auto incremental = isomorphism(symmetric_a,symmetric_b);
  Options deferred; deferred.defer_stereo=true;
  auto completed_only = isomorphism(symmetric_a,symmetric_b,deferred);
  require(!incremental.isomorphic() && !completed_only.isomorphic(),"Coupled sites incorrectly match");
  require(incremental.statistics.nodes > 0 && incremental.statistics.stereo_pruned > 0 &&
          incremental.statistics.leaves < completed_only.statistics.leaves,
          "Joint stereo constraint was not pruned during partial mapping");
  std::cout << "Three coupled sites: incremental " << incremental.statistics.nodes << " nodes, "
            << incremental.statistics.leaves << " leaves; deferred " << completed_only.statistics.nodes
            << " nodes, " << completed_only.statistics.leaves << " leaves\n";
  auto a = local(SYNKIT_TETRAHEDRAL);
  for (int i=0;i<7;++i) a.colors.push_back(99); // independent structural permutations
  auto b = a; std::swap(b.frames[0].vertices[1],b.frames[0].vertices[2]);
  const auto fast = isomorphism(a,b);
  Options opts; opts.defer_stereo=true;
  const auto late = isomorphism(a,b,opts);
  require(!fast.isomorphic() && !late.isomorphic(),"Pruning changed the answer");
  require(fast.statistics.nodes < late.statistics.nodes && fast.statistics.leaves < late.statistics.leaves,
          "Incremental stereo pruning did not reduce search");
  std::cout << "Stereo contradiction with 7 interchangeable spectators: incremental "
            << fast.statistics.nodes << " nodes, " << fast.statistics.leaves << " leaves; deferred "
            << late.statistics.nodes << " nodes, " << late.statistics.leaves << " leaves\n";
  // Repeated-color ligands cannot be rejected merely because a displayed
  // frame has opposite parity: a global atom permutation may repair it.
  a = local(SYNKIT_TETRAHEDRAL,false); b = a;
  std::swap(b.frames[0].vertices[1],b.frames[0].vertices[2]);
  require(isomorphism(a,b).isomorphic(),"Equivalent ligands falsely treated as distinct");
}
} // namespace
int main() {
  try {
    groups(); three_sites(); random_oracle(); boundaries(); pruning();
    std::cout << checks << " native stereo assertions passed\n";
    return 0;
  } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
