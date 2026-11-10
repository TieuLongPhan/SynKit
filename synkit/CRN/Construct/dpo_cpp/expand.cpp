// CRN expansion with the typed implicit-H DPO kernel.
// RDKit provides molecular validation, construction and canonical I/O.
#include <GraphMol/GraphMol.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/PeriodicTable.h>
#include <GraphMol/SmilesParse/SmilesWrite.h>
#include <RDGeneral/RDLog.h>
#include <RDGeneral/versions.h>
#include <algorithm>
#include <boost/json.hpp>
#include <boost/json/src.hpp>
#include <chrono>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <synkit_partial/partial.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace json = boost::json;
using I = int64_t;
using Clock = std::chrono::steady_clock;
namespace {
struct Limit {
  std::string reason;
};
struct Unsupported : std::runtime_error {
  using std::runtime_error::runtime_error;
};
I number(const json::value &v) {
  if (v.is_int64())
    return v.as_int64();
  if (v.is_uint64() && v.as_uint64() <= I(std::numeric_limits<I>::max()))
    return I(v.as_uint64());
  throw Unsupported("Wire integer outside the native representation");
}
I integer(const json::object &o, const char *k, I fallback = 0) {
  const auto *v = o.if_contains(k);
  return v ? number(*v) : fallback;
}
bool flag(const json::object &o, const char *k, bool fallback) {
  const auto *v = o.if_contains(k);
  return v ? v->as_bool() : fallback;
}
SkpBond bond(const json::value &v) {
  const auto &o = v.as_object();
  return {integer(o, "order2"), integer(o, "sigma2"), integer(o, "pi2")};
}
SkpDomain domain(const json::object &o, const char *key) {
  const auto &v = o.at(key).as_object();
  return {integer(v, "offset"), integer(v, "count")};
}
std::vector<I> element_codes(const json::object &wire) {
  std::vector<I> out;
  for (const auto &v : wire.at("symbols").as_array()) {
    const auto symbol = std::string(v.as_string());
    out.push_back(symbol == "*" ? 0 : RDKit::PeriodicTable::getTable()->getAtomicNumber(symbol));
  }
  return out;
}
SkpState state(const json::value &v, const std::vector<I> &codes) {
  const auto &o = v.as_object();
  I code = integer(o, "element");
  if (code < 0 || size_t(code) >= codes.size())
    throw Unsupported("Invalid element code");
  SkpState s{codes[code],      integer(o, "charge"), integer(o, "hcount"), integer(o, "radical"),
             integer(o, "lp"), integer(o, "ve"),     integer(o, "flags")};
  if (s.element <= 1 || (s.flags & 2) || (s.flags & 5) != 5)
    throw Unsupported("Native CRN DPO currently requires persistent, concrete, non-aromatic heavy atoms");
  return s;
}
struct Graph {
  std::vector<SkpAtom> atoms;
  std::vector<SkpEdge> edges;
  std::string key;
  // Eligibility is prepared only when a graph becomes a reactant.
  std::vector<std::vector<signed char>> eligibility;
};
struct Rule {
  std::vector<SkpRuleAtom> atoms;
  std::vector<SkpRuleEdge> edges;
  std::vector<I> domains;
  std::vector<std::vector<I>> components;
  I candidate_limit = -1, radical_policy = 0;
  SkpRequest request(const Graph &g) const {
    return {g.atoms.data(),
            I(g.atoms.size()),
            g.edges.data(),
            I(g.edges.size()),
            atoms.data(),
            I(atoms.size()),
            edges.data(),
            I(edges.size()),
            domains.data(),
            I(domains.size()),
            0,
            candidate_limit,
            radical_policy};
  }
};
std::vector<std::vector<I>> components(I n, const std::vector<std::pair<I, I>> &edges) {
  std::vector<std::vector<I>> adjacent(n), out;
  std::vector<bool> seen(n, false);
  for (auto [a, b] : edges) {
    if (a < 0 || b < 0 || a >= n || b >= n || a == b)
      throw Unsupported("Invalid native graph edge");
    adjacent[a].push_back(b);
    adjacent[b].push_back(a);
  }
  for (I start = 0; start < n; ++start)
    if (!seen[start]) {
      std::vector<I> c{start};
      seen[start] = true;
      for (size_t j = 0; j < c.size(); ++j)
        for (I next : adjacent[c[j]])
          if (!seen[next]) {
            c.push_back(next);
            seen[next] = true;
          }
      std::sort(c.begin(), c.end());
      out.push_back(std::move(c));
    }
  return out;
}
Rule read_rule(const json::object &wire) {
  if (flag(wire, "partial", false))
    throw Unsupported("Native CRN DPO requires full rule application");
  const auto codes = element_codes(wire);
  Rule r;
  r.candidate_limit = integer(wire, "candidate_limit", -1);
  r.radical_policy = integer(wire, "radical_policy");
  for (const auto &v : wire.at("domains").as_array())
    r.domains.push_back(number(v));
  for (const auto &v : wire.at("rule_atoms").as_array()) {
    const auto &a = v.as_object();
    SkpRuleAtom x{state(a.at("left"), codes), state(a.at("right"), codes),   integer(a, "atom_map"),
                  integer(a, "baseline_lp"),  integer(a, "match_mask"),      integer(a, "role", -1),
                  integer(a, "owner", -1),    integer(a, "capacity", 1),     integer(a, "budget", -1),
                  integer(a, "side"),         integer(a, "identity", -1),    integer(a, "materialization"),
                  integer(a, "virtual_kind"), integer(a, "stereo_slot", -1), integer(a, "fixed_charge"),
                  domain(a, "elements"),      domain(a, "charges"),          domain(a, "radicals"),
                  domain(a, "bond_orders")};
    if (x.role != -1 || x.owner != -1 || x.virtual_kind || x.stereo_slot != -1 || x.fixed_charge)
      throw Unsupported("Symbolic ports, wildcards and unsupported charge models are outside the native CRN profile");
    if (x.left.element != x.right.element || x.left.ve != x.right.ve)
      throw Unsupported("Native CRN rules must preserve their atomic carrier");
    r.atoms.push_back(x);
  }
  if (r.atoms.empty())
    throw Unsupported("Empty DPO rule");
  std::vector<std::pair<I, I>> left_edges;
  for (const auto &v : wire.at("rule_edges").as_array()) {
    const auto &a = v.as_object();
    SkpRuleEdge x{integer(a, "u"), integer(a, "v"), bond(a.at("left")), bond(a.at("right")), integer(a, "min_pi2", -1)};
    r.edges.push_back(x);
    if (x.left.order2)
      left_edges.emplace_back(x.u, x.v);
  }
  r.components = components(I(r.atoms.size()), left_edges);
  return r;
}
Graph read_graph(const json::object &wire) {
  auto codes = element_codes(wire);
  Graph g;
  for (const auto &v : wire.at("atoms").as_array())
    g.atoms.push_back({state(v.as_object().at("state"), codes), I(g.atoms.size() + 1)});
  for (const auto &v : wire.at("edges").as_array()) {
    const auto &a = v.as_object();
    g.edges.push_back({integer(a, "u"), integer(a, "v"), bond(a.at("state"))});
  }
  return g;
}
std::vector<std::pair<I, I>> graph_pairs(const Graph &g) {
  std::vector<std::pair<I, I>> e;
  for (const auto &b : g.edges)
    e.emplace_back(b.u, b.v);
  return e;
}
std::string canonical(const Graph &g) {
  if (g.atoms.empty())
    throw Unsupported("Empty material product");
  RDKit::RWMol mol;
  for (const auto &a : g.atoms) {
    auto *atom = new RDKit::Atom(unsigned(a.state.element));
    atom->setFormalCharge(int(a.state.charge));
    atom->setNumExplicitHs(unsigned(a.state.hcount));
    atom->setNumRadicalElectrons(unsigned(a.state.radical));
    atom->setNoImplicit(true);
    mol.addAtom(atom, false, true);
  }
  std::vector<I> occupancy(g.atoms.size());
  for (const auto &e : g.edges) {
    if (e.u < 0 || e.v < 0 || size_t(e.u) >= g.atoms.size() || size_t(e.v) >= g.atoms.size())
      throw Unsupported("Invalid product edge");
    if (e.state.sigma2 != 2 || e.state.pi2 != e.state.order2 - 2)
      throw Unsupported("Native CRN molecular IO requires standard sigma/pi bond partitions");
    RDKit::Bond::BondType kind;
    if (e.state.order2 == 2)
      kind = RDKit::Bond::SINGLE;
    else if (e.state.order2 == 4)
      kind = RDKit::Bond::DOUBLE;
    else if (e.state.order2 == 6)
      kind = RDKit::Bond::TRIPLE;
    else
      throw Unsupported("Native CRN IO requires non-aromatic integral bonds");
    mol.addBond(unsigned(e.u), unsigned(e.v), kind);
    occupancy[e.u] += e.state.order2;
    occupancy[e.v] += e.state.order2;
  }
  for (size_t i = 0; i < g.atoms.size(); ++i) {
    const auto &s = g.atoms[i].state;
    if (s.hcount < 0 || s.lp < 0 || s.radical < 0 ||
        s.ve != RDKit::PeriodicTable::getTable()->getNouterElecs(unsigned(s.element)) ||
        2 * s.charge != 2 * s.ve - 4 * s.lp - 2 * s.radical - 2 * s.hcount - occupancy[i])
      throw Unsupported("Native product violates its supplied local Lewis state");
  }
  RDKit::MolOps::sanitizeMol(mol);
  for (size_t i = 0; i < g.atoms.size(); ++i) {
    const auto &s = g.atoms[i].state;
    const auto *a = mol.getAtomWithIdx(unsigned(i));
    if (a->getIsAromatic())
      throw Unsupported("Aromatic products require an aromatic native CRN IO profile");
    if (a->getTotalNumHs() != s.hcount || a->getNumRadicalElectrons() != s.radical || a->getFormalCharge() != s.charge)
      throw Unsupported("Molecular IO changed the native DPO product state");
  }
  return RDKit::MolToSmiles(mol);
}
std::string raw_key(const Graph &g) {
  std::ostringstream s;
  for (const auto &a : g.atoms)
    s << a.state.element << ',' << a.state.charge << ',' << a.state.hcount << ',' << a.state.radical << ','
      << a.state.lp << ',' << a.state.ve << ';';
  s << '|';
  for (const auto &e : g.edges)
    s << e.u << ',' << e.v << ',' << e.state.order2 << ',' << e.state.sigma2 << ',' << e.state.pi2 << ';';
  return s.str();
}
using Result = std::unique_ptr<SkpResult, decltype(&skp_free)>;
Result execute(const Rule &r, const Graph &g) {
  auto request = r.request(g);
  Result out(skp_execute(&request), &skp_free);
  if (!out)
    throw std::runtime_error("SynKit DPO kernel allocation failed");
  const int status = skp_status(out.get());
  if (status == 2)
    throw Limit{std::string("kernel_limit: ") + skp_error(out.get())};
  if (status)
    throw Unsupported(std::string("SynKit DPO kernel: ") + skp_error(out.get()));
  if (!skp_search_complete(out.get()))
    throw Limit{"incomplete_kernel_search"};
  return out;
}
Rule identity_component(const Rule &r, size_t component) {
  Rule q;
  q.radical_policy = r.radical_policy;
  q.candidate_limit = r.candidate_limit;
  std::vector<I> remap(r.atoms.size(), -1);
  for (I n : r.components[component]) {
    remap[n] = I(q.atoms.size());
    auto a = r.atoms[n];
    a.right = a.left;
    a.atom_map = I(q.atoms.size() + 1);
    q.atoms.push_back(a);
  }
  for (const auto &e : r.edges)
    if (e.left.order2 && remap[e.u] >= 0 && remap[e.v] >= 0)
      q.edges.push_back({remap[e.u], remap[e.v], e.left, e.left, e.min_pi2});
  q.domains = r.domains;
  return q;
}
struct Options {
  int repeats = 50, max_components = 3;
  bool frontier = true, self = false, skip_no_change = true, allow_empty = false, strip = false, dedup = true,
       across = false;
  I mixtures = 50000, tasks = 200000, bindings = 50000000;
  std::string component_policy = "union";
  std::optional<Clock::time_point> deadline;
};
Options options(const json::object &o) {
  Options x;
  x.repeats = int(integer(o, "repeats", 50));
  x.max_components = int(integer(o, "max_components", 3));
  x.frontier = flag(o, "use_frontier", true);
  x.self = flag(o, "allow_self_mixtures", false);
  x.skip_no_change = flag(o, "skip_no_change", true);
  x.allow_empty = flag(o, "allow_empty_side", false);
  x.strip = flag(o, "strip_spectators", false);
  x.dedup = flag(o, "dedup_delta", true);
  x.across = flag(o, "dedup_across_rules", false);
  x.mixtures = integer(o, "max_mixtures_per_rule_step", 50000);
  x.tasks = integer(o, "max_tasks_per_step", 200000);
  x.bindings = integer(o, "max_bindings", 50000000);
  if (auto *v = o.if_contains("component_policy"))
    x.component_policy = std::string(v->as_string());
  if (x.component_policy != "union" && x.component_policy != "distinct")
    throw Unsupported("component_policy must be union or distinct");
  if (integer(o, "threads", 1) != 1 || flag(o, "explicit_h", false))
    throw Unsupported("Native DPO CRN currently requires one thread and implicit hydrogen");
  if (x.repeats < 0 || x.max_components < 1 || x.mixtures < 1 || x.tasks < 1 || x.bindings < 1)
    throw Unsupported("Invalid native DPO expansion bounds");
  if (auto *v = o.if_contains("time_limit"); v && !v->is_null()) {
    const double t = v->is_double() ? v->as_double() : double(number(*v));
    if (t < 0)
      throw Unsupported("Negative expansion time limit");
    x.deadline = Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(t));
  }
  return x;
}
struct Stats {
  int step = 0;
  I pool = 0, frontier = 0, bindings = 0, tasks = 0, applications = 0, outcomes = 0, events = 0;
};
struct Event {
  int step, rule;
  std::vector<int> left, right, all_right;
};
class Expansion {
  Options o;
  std::vector<Rule> rules;
  std::vector<std::vector<Rule>> queries;
  std::vector<Graph> graphs;
  std::unordered_map<std::string, int> ids;
  std::unordered_map<std::string, std::string> raw_keys;
  std::set<std::string> pool, frontier;
  std::unordered_set<std::string> attempts, deltas;
  std::vector<Event> events;
  std::vector<Stats> steps;
  I binding_calls = 0, binding_ns = 0, kernel_calls = 0, kernel_ns = 0;
  int steps_run = 0;
  std::string status = "complete", stop = "repeats";
  void check() const {
    if (o.deadline && Clock::now() >= *o.deadline)
      throw Limit{"time_limit"};
  }
  void prepare(Graph &g) {
    auto raw = raw_key(g);
    auto cached = raw_keys.find(raw);
    if (cached != raw_keys.end())
      g.key = cached->second;
    else {
      g.key = canonical(g);
      raw_keys.emplace(std::move(raw), g.key);
    }
  }
  int insert(Graph g) {
    if (g.key.empty())
      prepare(g);
    auto found = ids.find(g.key);
    if (found != ids.end())
      return found->second;
    const int id = int(graphs.size());
    ids.emplace(g.key, id);
    graphs.push_back(std::move(g));
    return id;
  }
  bool compatible(int id, int r, int c) {
    auto &g = graphs[id];
    if (g.eligibility.empty()) {
      g.eligibility.resize(rules.size());
      for (size_t j = 0; j < rules.size(); ++j)
        g.eligibility[j].assign(rules[j].components.size(), -1);
    }
    auto &answer = g.eligibility[r][c];
    if (answer < 0) {
      auto result = execute(queries[r][c], g);
      ++binding_calls;
      binding_ns += skp_stats(result.get())->elapsed_ns;
      answer = skp_application_count(result.get()) > 0;
    }
    return answer;
  }
  std::vector<std::vector<int>> candidates(int r, const std::vector<int> &ranked, Stats &s) {
    const int n = int(rules[r].components.size());
    if (n > o.max_components)
      throw Unsupported("DPO rule exceeds max_components");
    std::vector<std::vector<int>> slots(n);
    for (int c = 0; c < n; ++c)
      for (int id : ranked)
        if (compatible(id, r, c))
          slots[c].push_back(id);
    if (o.component_policy == "union") {
      // SynReactor matches the disconnected rule inside the selected union.
      // Several source components may bind one molecule; others are spectators.
      std::vector<std::vector<int>> out;
      std::vector<int> selected;
      std::function<void(size_t)> choose = [&](size_t start) {
        if (int(selected.size()) == n) {
          if (o.frontier &&
              std::none_of(selected.begin(), selected.end(), [&](int id) { return frontier.count(graphs[id].key); }))
            return;
          for (int c = 0; c < n; ++c)
            if (std::none_of(selected.begin(), selected.end(), [&](int id) { return compatible(id, r, c); }))
              return;
          out.push_back(selected);
          if (I(out.size()) > o.mixtures)
            throw Limit{"mixture_limit"};
          return;
        }
        for (size_t j = start; j < ranked.size(); ++j) {
          ++s.bindings;
          if ((s.bindings & 4095) == 0)
            check();
          if (s.bindings > o.bindings)
            throw Limit{"binding_limit"};
          selected.push_back(ranked[j]);
          choose(o.self ? j : j + 1);
          selected.pop_back();
        }
      };
      choose(0);
      return out;
    }
    std::set<std::vector<int>> unique;
    std::vector<int> assignment(n);
    std::function<void(int)> bind = [&](int c) {
      if (c == n) {
        if (o.frontier &&
            std::none_of(assignment.begin(), assignment.end(), [&](int i) { return frontier.count(graphs[i].key); }))
          return;
        auto sorted = assignment;
        std::sort(sorted.begin(), sorted.end(), [&](int a, int b) { return graphs[a].key < graphs[b].key; });
        unique.insert(std::move(sorted));
        if (I(unique.size()) > o.mixtures)
          throw Limit{"mixture_limit"};
        return;
      }
      for (int id : slots[c]) {
        ++s.bindings;
        if ((s.bindings & 4095) == 0)
          check();
        if (s.bindings > o.bindings)
          throw Limit{"binding_limit"};
        if (!o.self && std::find(assignment.begin(), assignment.begin() + c, id) != assignment.begin() + c)
          continue;
        assignment[c] = id;
        bind(c + 1);
      }
    };
    bind(0);
    std::vector<std::vector<int>> out(unique.begin(), unique.end());
    auto order = [&](const std::vector<int> &mix) {
      std::vector<std::string> best;
      for (size_t i = 0; i < mix.size(); ++i)
        if (!o.frontier || frontier.count(graphs[mix[i]].key)) {
          std::vector<std::string> value{graphs[mix[i]].key};
          for (size_t j = 0; j < mix.size(); ++j)
            if (i != j)
              value.push_back(graphs[mix[j]].key);
          if (best.empty() || value < best)
            best = std::move(value);
        }
      return best;
    };
    std::sort(out.begin(), out.end(), [&](const auto &a, const auto &b) { return order(a) < order(b); });
    return out;
  }
  std::vector<Graph> products(const SkpResult *result, I app) {
    I na = 0, ne = 0;
    auto *atoms = skp_atoms(result, app, &na);
    auto *edges = skp_edges(result, app, &ne);
    Graph whole;
    for (I i = 0; i < na; ++i) {
      if (atoms[i].introduced || atoms[i].port || (atoms[i].right.flags & 1) == 0)
        throw Unsupported("Native CRN DPO requires concrete carrier-preserving products");
      whole.atoms.push_back({atoms[i].right, i + 1});
    }
    for (I i = 0; i < ne; ++i)
      if (edges[i].right.order2)
        whole.edges.push_back({edges[i].u, edges[i].v, edges[i].right});
    auto cs = components(na, graph_pairs(whole));
    std::vector<Graph> out;
    for (const auto &c : cs) {
      Graph g;
      std::vector<I> remap(na, -1);
      for (I node : c) {
        remap[node] = I(g.atoms.size());
        g.atoms.push_back({whole.atoms[node].state, I(g.atoms.size() + 1)});
      }
      for (const auto &e : whole.edges)
        if (remap[e.u] >= 0 && remap[e.v] >= 0)
          g.edges.push_back({remap[e.u], remap[e.v], e.state});
      prepare(g);
      out.push_back(std::move(g));
    }
    std::sort(out.begin(), out.end(), [](const Graph &a, const Graph &b) { return a.key < b.key; });
    return out;
  }
  void apply(int rule, const std::vector<int> &mix, int step, Stats &stats, std::set<std::string> &next) {
    Graph host;
    std::vector<int> origin;
    for (size_t c = 0; c < mix.size(); ++c) {
      const auto &g = graphs[mix[c]];
      I offset = I(host.atoms.size());
      for (const auto &a : g.atoms) {
        host.atoms.push_back({a.state, I(host.atoms.size() + 1)});
        origin.push_back(int(c));
      }
      for (const auto &e : g.edges)
        host.edges.push_back({e.u + offset, e.v + offset, e.state});
    }
    auto result = execute(rules[rule], host);
    ++kernel_calls;
    ++stats.applications;
    kernel_ns += skp_stats(result.get())->elapsed_ns;
    std::set<std::vector<std::string>> seen;
    for (I app = 0; app < skp_application_count(result.get()); ++app) {
      const I *mapping = skp_mapping(result.get(), app);
      std::set<int> used;
      bool valid = true;
      for (size_t p = 0; p < rules[rule].atoms.size(); ++p)
        if (mapping[p] < 0 || size_t(mapping[p]) >= origin.size())
          throw Unsupported("Incomplete native full-rule mapping");
      // Each disconnected left component must bind one distinct molecule copy.
      for (const auto &c : rules[rule].components) {
        int component = origin.at(mapping[c[0]]);
        for (I p : c)
          if (mapping[p] < 0 || origin.at(mapping[p]) != component)
            valid = false;
        if (!used.insert(component).second)
          valid = false;
      }
      if (o.component_policy == "distinct" && (!valid || used.size() != mix.size()))
        continue;
      ++stats.outcomes;
      auto staged = products(result.get(), app);
      std::vector<std::string> keys;
      for (const auto &g : staged)
        keys.push_back(g.key);
      if (!seen.insert(keys).second)
        continue;
      std::map<std::string, int> left_counts, right_counts;
      for (int id : mix)
        ++left_counts[graphs[id].key];
      for (const auto &k : keys)
        ++right_counts[k];
      if (o.skip_no_change && left_counts == right_counts)
        continue;
      std::string delta = o.across ? "*" : std::to_string(rule);
      delta += '\x1e';
      for (const auto &[k, n] : left_counts)
        for (int j = right_counts[k]; j < n; ++j)
          delta += k + '\x1f';
      delta += '\x1e';
      for (const auto &[k, n] : right_counts)
        for (int j = left_counts[k]; j < n; ++j)
          delta += k + '\x1f';
      if (o.dedup && deltas.count(delta))
        continue;
      auto left = mix;
      std::vector<bool> keep(keys.size(), true);
      if (o.strip) {
        left.erase(std::remove_if(left.begin(), left.end(),
                                  [&](int id) {
                                    return right_counts.count(graphs[id].key) && right_counts.at(graphs[id].key) > 0;
                                  }),
                   left.end());
        for (size_t j = 0; j < keys.size(); ++j)
          if (left_counts.count(keys[j]) && left_counts.at(keys[j]) > 0)
            keep[j] = false;
      }
      if (!o.allow_empty && (left.empty() || std::none_of(keep.begin(), keep.end(), [](bool v) { return v; })))
        continue;
      if (o.dedup)
        deltas.insert(std::move(delta));
      // Publish projected graphs only after every event acceptance check.
      std::vector<int> right, kept_right;
      for (size_t j = 0; j < staged.size(); ++j) {
        int id = insert(std::move(staged[j]));
        right.push_back(id);
        if (keep[j])
          kept_right.push_back(id);
      }
      events.push_back({step, rule, std::move(left), std::move(kept_right), right});
      ++stats.events;
      for (int id : right)
        if (pool.insert(graphs[id].key).second)
          next.insert(graphs[id].key);
    }
  }

public:
  explicit Expansion(const json::object &input) : o(options(input.at("options").as_object())) {
    for (const auto &wire : input.at("rules").as_array())
      rules.push_back(read_rule(wire.as_object()));
    queries.resize(rules.size());
    for (size_t r = 0; r < rules.size(); ++r)
      for (size_t c = 0; c < rules[r].components.size(); ++c)
        queries[r].push_back(identity_component(rules[r], c));
  }
  json::object run(const json::object &input) {
    json::array seed_keys;
    for (const auto &wire : input.at("seeds").as_array()) {
      auto g = read_graph(wire.as_object());
      if (components(I(g.atoms.size()), graph_pairs(g)).size() != 1)
        throw Unsupported("Each seed must be one connected molecule");
      int id = insert(std::move(g));
      seed_keys.emplace_back(graphs[id].key);
      pool.insert(graphs[id].key);
      frontier.insert(graphs[id].key);
    }
    try {
      for (int step = 1; step <= o.repeats && !pool.empty(); ++step) {
        check();
        if (o.frontier && frontier.empty()) {
          stop = "fixed_point";
          break;
        }
        Stats stats;
        stats.step = step;
        stats.pool = I(pool.size());
        stats.frontier = I(frontier.size());
        std::vector<int> ranked;
        for (const auto &s : pool)
          ranked.push_back(ids.at(s));
        std::vector<std::pair<int, std::vector<int>>> tasks;
        for (size_t r = 0; r < rules.size(); ++r)
          for (auto mix : candidates(int(r), ranked, stats)) {
            std::string attempt = std::to_string(r);
            for (int id : mix)
              attempt += '\x1f' + graphs[id].key;
            if (!attempts.insert(attempt).second)
              continue;
            if (I(tasks.size()) >= o.tasks)
              throw Limit{"task_limit"};
            tasks.emplace_back(int(r), std::move(mix));
          }
        stats.tasks = I(tasks.size());
        if (tasks.empty()) {
          steps.push_back(stats);
          stop = "no_tasks";
          break;
        }
        std::set<std::string> next;
        for (const auto &[r, mix] : tasks) {
          check();
          apply(r, mix, step, stats, next);
        }
        frontier = std::move(next);
        steps.push_back(stats);
        steps_run = step;
        if (o.frontier && frontier.empty()) {
          stop = "fixed_point";
          break;
        }
      }
    } catch (const Limit &e) {
      status = "limit";
      stop = e.reason;
    }
    json::object out{{"protocol", 1},
                     {"engine", "synkit_dpo_cpp"},
                     {"controller", "cpp"},
                     {"kernel", "synkit_dpo_cpp"},
                     {"kernel_abi", skp_abi()},
                     {"rdkit_version", RDKit::rdkitVersion},
                     {"status", status},
                     {"stop_reason", stop},
                     {"steps_run", steps_run},
                     {"hydrogen_model", "implicit"},
                     {"component_policy", o.component_policy},
                     {"seed_keys", std::move(seed_keys)},
                     {"binding_tests", binding_calls},
                     {"binding_elapsed_ns", binding_ns},
                     {"kernel_calls", kernel_calls},
                     {"kernel_elapsed_ns", kernel_ns}};
    json::array species;
    for (const auto &g : graphs)
      species.emplace_back(g.key);
    out["species"] = std::move(species);
    json::array es;
    for (const auto &e : events)
      es.push_back(json::object{{"step", e.step},
                                {"rule_index", e.rule},
                                {"reactants", json::value_from(e.left)},
                                {"products", json::value_from(e.right)},
                                {"all_products", json::value_from(e.all_right)}});
    out["events"] = std::move(es);
    json::array ss;
    for (const auto &s : steps)
      ss.push_back(json::object{{"step", s.step},
                                {"pool", s.pool},
                                {"frontier", s.frontier},
                                {"bindings", s.bindings},
                                {"tasks", s.tasks},
                                {"candidate_mixtures", s.tasks},
                                {"applications", s.applications},
                                {"outcomes", s.outcomes},
                                {"events", s.events},
                                {"capped_rules", json::array{}},
                                {"task_budget_exhausted", false}});
    out["steps"] = std::move(ss);
    out["pool"] = json::value_from(pool);
    out["frontier"] = json::value_from(frontier);
    out["rules"] = json::array{};
    return out;
  }
};
json::object handle(const std::string &line) {
  try {
    const auto input = json::parse(line).as_object();
    if (integer(input, "protocol") != 1)
      throw Unsupported("Unsupported native DPO protocol");
    if (flag(input, "probe", false))
      return {{"protocol", 1},
              {"status", "complete"},
              {"engine", "synkit_dpo_cpp"},
              {"controller", "cpp"},
              {"kernel", "synkit_dpo_cpp"},
              {"kernel_abi", skp_abi()},
              {"rdkit_version", RDKit::rdkitVersion}};
    return Expansion(input).run(input);
  } catch (const Limit &e) {
    return {{"protocol", 1}, {"status", "limit"}, {"stop_reason", e.reason}};
  } catch (const Unsupported &e) {
    return {{"protocol", 1}, {"status", "unsupported"}, {"error", e.what()}};
  } catch (const std::exception &e) {
    return {{"protocol", 1}, {"status", "error"}, {"error", e.what()}};
  }
}
} // namespace
int main(int argc, char **argv) {
  RDLog::InitLogs();
  boost::logging::disable_logs("rdApp.warning");
  const bool serve = argc == 2 && std::string(argv[1]) == "--serve";
  std::string line;
  if (serve) {
    while (std::getline(std::cin, line))
      std::cout << json::serialize(handle(line)) << std::endl;
  } else {
    std::ostringstream input;
    input << std::cin.rdbuf();
    std::cout << json::serialize(handle(input.str())) << std::endl;
  }
  return 0;
}
