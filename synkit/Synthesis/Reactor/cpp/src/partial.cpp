#include "synkit_partial/partial.h"

#include <algorithm>
#include <chrono>
#include <functional>
#include <limits>
#include <map>
#include <new>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using I = int64_t;
using Key = std::pair<I, I>;
Key key(I u, I v) { return u < v ? Key{u, v} : Key{v, u}; }
struct Failure : std::runtime_error {
  int status;
  Failure(int s, const std::string &message) : std::runtime_error(message), status(s) {}
};
struct Found {};
struct Application {
  std::vector<I> mapping;
  std::vector<SkpProductAtom> atoms;
  std::vector<SkpProductEdge> edges;
};
bool equal(SkpBond a, SkpBond b) {
  return a.order2 == b.order2 && a.sigma2 == b.sigma2 && a.pi2 == b.pi2;
}
bool equal(SkpState a, SkpState b) {
  return a.element == b.element && a.charge == b.charge
      && a.hcount == b.hcount && a.radical == b.radical
      && a.lp == b.lp && a.ve == b.ve && a.flags == b.flags;
}
void count(I x, const char *name) {
  if (x < 0 || x > 1000000000)
    throw Failure(1, std::string(name) + " must be a non-negative bounded integer");
}
void state_valid(SkpState s, bool wildcard) {
  count(s.hcount, "hydrogen count"); count(s.radical, "radical count");
  count(s.lp, "lone-pair count"); count(s.ve, "valence electron count");
  if (s.charge < -1000000000 || s.charge > 1000000000 || s.element < 0
      || s.flags < 0 || (s.flags & ~I(7)))
    throw Failure(1, "invalid atom label");
  if (!wildcard && !(s.flags & 4)) throw Failure(1, "concrete atoms require valence electrons");
}
void bond_valid(SkpBond b) {
  count(b.order2, "bond order"); count(b.sigma2, "sigma order"); count(b.pi2, "pi order");
  if (b.sigma2 % 2 || b.pi2 % 2 || b.sigma2 > 2)
    throw Failure(1, "sigma and pi occupancies must be integral");
  if (b.order2 != 3 && b.order2 != b.sigma2 + b.pi2)
    throw Failure(1, "non-aromatic bond order must equal sigma plus pi");
  if (b.order2 == 3 && (b.sigma2 != 2 || (b.pi2 != 0 && b.pi2 != 2)))
    throw Failure(1, "aromatic bonds require one sigma and zero or one pi pair");
}
}

struct SkpResult {
  int status = 0;
  std::string error;
  SkpStats stats{};
  std::vector<Application> applications;
  I rejected_completions = 0;
  bool search_complete = true;
};

namespace {
class Kernel {
  const SkpRequest &r;
  SkpResult &out;
  bool observation;
  bool first_only;
  std::vector<std::map<I, SkpBond>> host;
  std::vector<std::map<I, const SkpRuleEdge *>> source;
  std::vector<std::vector<I>> components, domains;
  std::vector<std::vector<std::vector<I>>> embeddings;
  std::vector<I> mapping;
  std::vector<bool> used;
  size_t embedding_states = 0;
  std::vector<I> host_component, host_sizes;
  std::vector<bool> host_consistent;
  std::vector<bool> disappearing, represented;
  std::vector<std::vector<std::vector<I>>> component_images;
  std::vector<std::vector<I>> chosen_images;
  std::vector<size_t> rule_component;
  size_t embedding_component = 0;

  bool admits(SkpDomain domain, I value) const {
    if (domain.count == -1) return true;
    for (I i = 0; i < domain.count; ++i)
      if (r.domains[domain.offset + i] == value) return true;
    return false;
  }
  bool node_matches(I p, I h) const {
    const auto &a = r.rule_atoms[p];
    if (!(a.left.flags & 1)) return false;
    auto s = r.atoms[h].state, q = a.left;
    if (a.role >= 0)
      return s.element != 0 && admits(a.elements, s.element)
          && admits(a.charges, s.charge) && admits(a.radicals, s.radical);
    I m = a.match_mask;
    // Concrete atomic types are immutable, independently of optional guards.
    if (s.element != q.element || s.ve != q.ve) return false;
    if ((m & 2) && s.charge != q.charge) return false;
    if ((m & 4) && s.hcount < q.hcount) return false;
    if ((m & 16) && s.lp < q.lp) return false;
    if ((m & 32) && bool(s.flags & 2) != bool(q.flags & 2)) return false;
    if (m & 8) {
      if (r.radical_policy == 1 ? s.radical < q.radical : s.radical != q.radical)
        return false;
    }
    return true;
  }
  bool edge_matches(const SkpRuleEdge &p, SkpBond h) const {
    if (p.min_pi2 >= 0) {
      return h.order2 != 3 && h.pi2 >= p.min_pi2 && h.sigma2 == p.left.sigma2;
    }
    if (p.left.order2 == 3 && h.order2 == 3) return true;
    return equal(p.left, h);
  }
  bool wildcard_bindings() const {
    for (I p = 0; p < r.n_rule_atoms; ++p) {
      const auto &a = r.rule_atoms[p];
      if (mapping[p] >= 0 && !(observation && !(a.right.flags & 1)) && r.atoms[mapping[p]].state.radical
          + a.right.radical - a.left.radical < 0) return false;
      if (mapping[p] < 0 || a.role < 0) continue;
      if (a.capacity < 1 || a.budget == 0 || a.side == 2) return false;
      I h = mapping[p];
      if (a.identity >= 0 && r.atoms[h].atom_map != a.identity) return false;
      if (a.materialization == 2) return false;
      if (a.owner >= 0 && mapping[a.owner] < 0) return false;
      if (a.bond_orders.count >= 0 && source[p].empty()) return false;
      for (const auto &entry : source[p]) {
        I neighbor = mapping[entry.first];
        if (neighbor < 0) return false;
        if (!admits(a.bond_orders, host[h].at(neighbor).order2)) return false;
      }
    }
    return true;
  }
  bool collision_free() const {
    for (I e = 0; e < r.n_rule_edges; ++e) {
      const auto &b = r.rule_edges[e];
      if (b.left.order2 == 0 && b.right.order2 > 0 && mapping[b.u] >= 0
          && mapping[b.v] >= 0 && host[mapping[b.u]].count(mapping[b.v]))
        return false;
    }
    return true;
  }
  void embed_component(const std::vector<I> &nodes, size_t level,
                       std::vector<std::vector<I>> &result) {
    if (++embedding_states > 1000000)
      throw Failure(2, "structural embedding search exceeded one million states");
    if (level == nodes.size()) {
      if (first_only) { accept(); return; }
      if (r.candidate_limit >= 0 && result.size() >= static_cast<size_t>(r.candidate_limit))
        throw Failure(2, "structural component embedding limit exceeded");
      result.push_back(mapping);
      if (disappearing[embedding_component]) represented[host_component[mapping[nodes[0]]]] = true;
      return;
    }
    I p = -1;
    size_t best_bound = 0, best_domain = std::numeric_limits<size_t>::max();
    for (I node : nodes) {
      if (mapping[node] >= 0) continue;
      size_t bound = 0;
      for (const auto &edge : source[node]) bound += mapping[edge.first] >= 0;
      if (p < 0 || bound > best_bound
          || (bound == best_bound && domains[node].size() < best_domain)) {
        p = node; best_bound = bound; best_domain = domains[node].size();
      }
    }
    // Grow from an already bound neighbor. This avoids independently placing
    // disconnected pattern vertices and only later discovering incompatible
    // bonds. Domains remain sorted dense host indices.
    I anchor = -1;
    for (const auto &edge : source[p]) if (mapping[edge.first] >= 0
        && (anchor < 0 || host[mapping[edge.first]].size() < host[anchor].size()))
      anchor = mapping[edge.first];
    std::vector<I> candidates;
    if (anchor < 0) candidates = domains[p];
    else for (const auto &edge : host[anchor])
      if (std::binary_search(domains[p].begin(), domains[p].end(), edge.first))
        candidates.push_back(edge.first);
    for (I h : candidates) {
      if (used[h]) continue;
      size_t component = first_only ? rule_component[p] : embedding_component;
      if (disappearing[component]
          && first_only && (host_sizes[host_component[h]] != static_cast<I>(components[component].size())
                            || host[h].size() != source[p].size())) continue;
      if (!first_only && disappearing[embedding_component]
          && (represented[host_component[h]] || host_sizes[host_component[h]] != static_cast<I>(nodes.size())
              || host[h].size() != source[p].size())) continue;
      bool valid = true;
      for (const auto &edge : source[p]) {
        I neighbor = mapping[edge.first];
        if (neighbor < 0) continue;
        auto found = host[h].find(neighbor);
        if (found == host[h].end() || !edge_matches(*edge.second, found->second)) {
          valid = false; break;
        }
      }
      if (!valid) continue;
      mapping[p] = h; used[h] = true;
      embed_component(nodes, level + 1, result);
      mapping[p] = -1; used[h] = false;
    }
  }
  I budget(I p, int side) const {
    const auto &a = r.rule_atoms[p];
    auto s = side == 0 ? a.left : a.right;
    I bonds2 = 0;
    for (I e = 0; e < r.n_rule_edges; ++e) {
      const auto &b = r.rule_edges[e];
      if (b.u != p && b.v != p) continue;
      auto value = side == 0 ? b.left : b.right;
      bonds2 += value.sigma2 + value.pi2;
    }
    I missing2 = 2 * s.ve - 4 * (s.lp + a.baseline_lp)
               - 2 * s.radical - 2 * s.hcount - 2 * s.charge - bonds2;
    if (missing2 < 0 || missing2 % 2)
      throw Failure(3, "external single-bond budget must be a non-negative integer");
    return missing2 / 2;
  }
  void assign_aromatic_phase(Application &app,
                             std::map<Key, SkpProductEdge> &edges) const {
    // A normal aromatic bond contains one sigma pair and zero or one pi pair.
    // Solve the product's local Lewis equations simultaneously, including
    // unchanged aromatic context. Charges/resources are inputs to this solve,
    // never adjusted to make an inconsistent Kekule assignment pass.
    std::vector<SkpProductEdge *> aromatic;
    std::vector<std::vector<size_t>> incident(app.atoms.size());
    std::vector<I> required2(app.atoms.size());
    for (size_t i = 0; i < app.atoms.size(); ++i) {
      const auto &s = app.atoms[i].right;
      required2[i] = 2 * s.ve - 4 * s.lp - 2 * s.radical
                   - 2 * s.hcount - 2 * s.charge;
    }
    for (auto &entry : edges) {
      auto &edge = entry.second;
      auto &b = edge.right;
      if (b.order2 != 3) {
        required2[edge.u] -= b.sigma2 + b.pi2;
        required2[edge.v] -= b.sigma2 + b.pi2;
        continue;
      }
      if (b.sigma2 != 2 || (b.pi2 != 0 && b.pi2 != 2))
        throw Failure(3, "aromatic bonds require one sigma and zero or one pi pair");
      for (I node : {edge.u, edge.v}) {
        if ((app.atoms[node].right.flags & 6) != 6)
          throw Failure(3, "aromatic phase assignment requires material aromatic endpoints");
        required2[node] -= 2;
        incident[node].push_back(aromatic.size());
      }
      aromatic.push_back(&edge);
    }
    if (aromatic.empty()) return;
    std::vector<I> demand(app.atoms.size(), 0);
    std::vector<bool> current_valid(incident.size(), true);
    for (size_t node = 0; node < incident.size(); ++node) {
      if (incident[node].empty()) continue;
      if (required2[node] < 0 || required2[node] % 2
          || required2[node] / 2 > static_cast<I>(incident[node].size()))
        throw Failure(3, "product aromatic pi demand has no Kekule realization");
      demand[node] = required2[node] / 2;
      I current = 0;
      for (size_t edge : incident[node]) current += aromatic[edge]->right.pi2 / 2;
      current_valid[node] = current == demand[node];
    }
    // Propagation fixes degree-zero and fully saturated residual demands.
    // Branching is deterministic and prefers the supplied phase. The finite
    // bound is a reported search refusal, not a truncated outcome population.
    size_t states = 0;
    std::vector<bool> visited(incident.size(), false);
    for (size_t start = 0; start < incident.size(); ++start) {
      if (visited[start] || incident[start].empty()) continue;
      std::vector<size_t> nodes{start}, component;
      visited[start] = true;
      bool valid = true;
      for (size_t index = 0; index < nodes.size(); ++index) {
        size_t node = nodes[index];
        valid = valid && current_valid[node];
        for (size_t edge : incident[node]) {
          const auto *b = aromatic[edge];
          if (static_cast<size_t>(b->u) == node) component.push_back(edge);
          size_t neighbor = static_cast<size_t>(b->u) == node ? b->v : b->u;
          if (!visited[neighbor]) { visited[neighbor] = true; nodes.push_back(neighbor); }
        }
      }
      // Already valid components are exactly preserved, including spectators.
      if (valid) continue;
      std::function<bool(std::vector<int> &)> solve = [&](std::vector<int> &phase) {
        if (++states > 1000000)
          throw Failure(2, "aromatic Kekule phase search exceeded one million states");
        bool changed;
        do {
          changed = false;
          for (size_t node : nodes) {
            I needed = demand[node], unknown = 0;
            for (size_t edge : incident[node]) {
              if (phase[edge] < 0) ++unknown;
              else needed -= phase[edge];
            }
            if (needed < 0 || needed > unknown) return false;
            if (!unknown || (needed != 0 && needed != unknown)) continue;
            int value = needed ? 1 : 0;
            for (size_t edge : incident[node]) if (phase[edge] < 0) {
              phase[edge] = value; changed = true;
            }
          }
        } while (changed);
        auto next_edge = std::find_if(component.begin(), component.end(),
                                      [&](size_t edge) { return phase[edge] < 0; });
        if (next_edge == component.end()) return true;
        size_t next = *next_edge;
        int preferred = static_cast<int>(aromatic[next]->right.pi2 / 2);
        for (int value : {preferred, 1 - preferred}) {
          auto branch = phase; branch[next] = value;
          if (solve(branch)) { phase = std::move(branch); return true; }
        }
        return false;
      };
      std::vector<int> phase(aromatic.size(), -1);
      for (size_t edge : component) {
        for (I node : {aromatic[edge]->u, aromatic[edge]->v}) {
          I origin = app.atoms[node].origin;
          if (origin >= 0 && r.rule_atoms[origin].role >= 0)
            phase[edge] = static_cast<int>(aromatic[edge]->right.pi2 / 2);
        }
      }
      if (!solve(phase))
        throw Failure(3, "product aromatic pi demands admit no common Kekule assignment");
      for (size_t edge : component)
        aromatic[edge]->right.pi2 = 2 * phase[edge];
    }
  }
  Application rewrite() const {
    Application app;
    app.mapping = mapping;
    std::vector<I> total = mapping;
    std::vector<bool> fixed;
    std::map<Key, SkpProductEdge> edges;
    I next_map = 0;
    for (I i = 0; i < r.n_atoms; ++i) {
      auto a = r.atoms[i];
      next_map = std::max(next_map, a.atom_map);
      app.atoms.push_back({a.state, a.state, a.atom_map, -1, 0, -1, 0});
      fixed.push_back(true);
    }
    auto fresh_map = [&]() {
      if (next_map == std::numeric_limits<I>::max()) throw Failure(3, "atom map overflow");
      return ++next_map;
    };
    for (I p = 0; p < r.n_rule_atoms; ++p) if (total[p] < 0) {
      const auto &a = r.rule_atoms[p];
      auto s = a.left; s.lp += a.baseline_lp;
      if (a.role >= 0) {
        s.flags &= ~I(4);
        if (a.capacity < 1 || a.budget == 0 || a.side == 2)
          throw Failure(3, "missing wildcard requires one input binding resource");
        for (const auto &entry : source[p])
          if (!admits(a.bond_orders, entry.second->left.order2))
            throw Failure(3, "missing wildcard has an incompatible bond domain");
      }
      total[p] = static_cast<I>(app.atoms.size());
      app.atoms.push_back({s, s, fresh_map(), p, 1, -1, 0});
      fixed.push_back(a.fixed_charge != 0);
    }
    for (I e = 0; e < r.n_edges; ++e) {
      const auto &b = r.edges[e];
      edges[key(b.u, b.v)] = {b.u, b.v, b.state, b.state};
    }
    for (I e = 0; e < r.n_rule_edges; ++e) {
      const auto &b = r.rule_edges[e];
      if ((mapping[b.u] < 0 || mapping[b.v] < 0) && b.left.order2 > 0)
        edges[key(total[b.u], total[b.v])] = {total[b.u], total[b.v], b.left, b.left};
    }
    for (I p = 0; p < r.n_rule_atoms; ++p) if (mapping[p] < 0) {
      const auto &a = r.rule_atoms[p];
      auto &atom = app.atoms[total[p]];
      if (a.owner >= 0) atom.owner = total[a.owner];
      if (a.role >= 0) continue;
      if (observation && (!(a.left.flags & 1) || !(a.right.flags & 1))) {
        int known = (a.left.flags & 1) ? 0 : 1;
        if (!(a.fixed_charge & 2) && budget(p, known) != 0)
          throw Failure(3, "one-sided observation has unspecified external context");
        continue;
      }
      I ports = budget(p, 0);
      if (ports != budget(p, 1))
        throw Failure(3, "external attachment demand differs between rule endpoints");
      if (ports > 1000000) throw Failure(3, "external attachment count exceeds representation limit");
      for (I j = 0; j < ports; ++j) {
        I index = static_cast<I>(app.atoms.size());
        SkpState symbol{}; symbol.flags = 1;
        app.atoms.push_back({symbol, symbol, fresh_map(), -1, 1, total[p], 1});
        fixed.push_back(true);
        SkpBond unit{2, 2, 0};
        edges[key(total[p], index)] = {total[p], index, unit, unit};
      }
    }
    for (I p = 0; p < r.n_rule_atoms; ++p) {
      const auto &a = r.rule_atoms[p];
      auto &atom = app.atoms[total[p]];
      atom.origin = p;
      auto s = atom.left;
      if (observation && (!(a.left.flags & 1) || !(a.right.flags & 1))) {
        s = a.right;
        if (s.flags & 1) s.lp += a.baseline_lp;
        atom.right = s;
        fixed[total[p]] = true;
        continue;
      }
      s.hcount += a.right.hcount - a.left.hcount;
      s.lp += a.right.lp - a.left.lp;
      if (a.role < 0) {
        s.radical += a.right.radical - a.left.radical;
        s.charge = a.right.charge;
        s.flags = (s.flags & ~I(2)) | (a.right.flags & 2);
      }
      if (s.hcount < 0 || s.lp < 0 || s.radical < 0)
        throw Failure(3, "rewrite exhausts an electron or hydrogen resource");
      atom.right = s;
      fixed[total[p]] = a.role >= 0 || a.fixed_charge != 0;
    }
    for (I e = 0; e < r.n_rule_edges; ++e) {
      const auto &b = r.rule_edges[e];
      if (b.left.order2 == 0 && b.right.order2 == 0) continue;
      auto k = key(total[b.u], total[b.v]);
      auto found = edges.find(k);
      if (found == edges.end()) {
        edges[k] = {k.first, k.second, b.left, b.right};
      } else if (b.min_pi2 >= 0) {
        auto original = found->second.left;
        found->second.right = {
          original.order2 + b.right.order2 - b.left.order2,
          original.sigma2 + b.right.sigma2 - b.left.sigma2,
          original.pi2 + b.right.pi2 - b.left.pi2};
      } else if (b.left.order2 == 3 && b.right.order2 == 3) {
        // Preserve the matched Kekule phase for unchanged aromatic context.
      } else {
        found->second.right = b.right;
      }
    }
    if (observation) for (auto &entry : edges) {
      auto &edge = entry.second;
      if (!(app.atoms[edge.u].right.flags & 1) || !(app.atoms[edge.v].right.flags & 1))
        edge.right = {};
    }
    assign_aromatic_phase(app, edges);
    std::vector<I> sums2(app.atoms.size(), 0);
    for (const auto &entry : edges) {
      auto b = entry.second;
      if (b.right.order2 < 0 || b.right.sigma2 < 0 || b.right.pi2 < 0)
        throw Failure(3, "rewrite exhausts a bond resource");
      sums2[b.u] += b.right.sigma2 + b.right.pi2;
      sums2[b.v] += b.right.sigma2 + b.right.pi2;
      app.edges.push_back(b);
    }
    for (size_t i = 0; i < app.atoms.size(); ++i) {
      auto &s = app.atoms[i].right;
      if (!(s.flags & 4) || fixed[i] || (s.flags & 2)) continue;
      I charge2 = 2 * s.ve - 4 * s.lp - 2 * s.radical - 2 * s.hcount - sums2[i];
      if (charge2 % 2) throw Failure(3, "product charge is fractional");
      s.charge = charge2 / 2;
    }
    for (size_t i = 0; i < app.atoms.size(); ++i) {
      const auto &s = app.atoms[i].right;
      if (observation && (!(s.flags & 1)
          || (app.atoms[i].origin < 0 && i < host_consistent.size() && !host_consistent[i])
          || (app.atoms[i].origin >= 0 && (r.rule_atoms[app.atoms[i].origin].fixed_charge & 1)))) continue;
      if ((s.flags & 4) && 2 * s.charge !=
          2 * s.ve - 4 * s.lp - 2 * s.radical - 2 * s.hcount - sums2[i])
        throw Failure(3, "product lies outside the supported local Lewis model");
    }
    I actual_delta = 0, declared_delta = 0;
    for (const auto &atom : app.atoms) actual_delta +=
      ((atom.right.flags & 1) ? atom.right.charge : 0)
      - ((atom.left.flags & 1) ? atom.left.charge : 0);
    for (I i = 0; i < r.n_rule_atoms; ++i)
      declared_delta += ((r.rule_atoms[i].right.flags & 1) ? r.rule_atoms[i].right.charge : 0)
                      - ((r.rule_atoms[i].left.flags & 1) ? r.rule_atoms[i].left.charge : 0);
    if (actual_delta != declared_delta)
      throw Failure(3, "application changes the declared net charge exchange");
    return app;
  }
  void accept() {
    ++out.stats.candidates;
    if (r.candidate_limit >= 0 && out.stats.candidates > r.candidate_limit)
      throw Failure(2, "structural candidate limit exceeded");
    if (!wildcard_bindings() || !collision_free()) return;
    ++out.stats.mappings;
    try {
      out.applications.push_back(rewrite());
    } catch (const Failure &error) {
      bool proper = false;
      for (I p = 0; p < r.n_rule_atoms; ++p)
        proper = proper || ((r.rule_atoms[p].left.flags & 1) && mapping[p] < 0);
      if (!observation || error.status != 3 || (!proper && !first_only)) throw;
      ++out.rejected_completions;
    }
    if (first_only && !out.applications.empty()) throw Found{};
  }
  void join(const std::vector<I> &chosen, size_t level) {
    if (level == chosen.size()) { accept(); return; }
    I c = chosen[level];
    for (size_t index = 0; index < embeddings[c].size(); ++index) {
      const auto &embedding = embeddings[c][index];
      const auto &image = component_images[c][index];
      bool valid = true;
      if (disappearing[c]) for (size_t earlier = 0; earlier < level; ++earlier) {
        I other = chosen[earlier];
        if (disappearing[other] && component_images[c] == component_images[other]
            && chosen_images[other] >= image) valid = false;
      }
      for (I p : components[c]) if (used[embedding[p]]) { valid = false; break; }
      if (!valid) continue;
      for (I p : components[c]) { mapping[p] = embedding[p]; used[embedding[p]] = true; }
      chosen_images[c] = image;
      join(chosen, level + 1);
      for (I p : components[c]) { used[embedding[p]] = false; mapping[p] = -1; }
    }
  }
  void subsets(I next, I remaining, std::vector<I> &chosen) {
    if (!remaining) { join(chosen, 0); return; }
    I n = static_cast<I>(components.size());
    for (I i = next; i <= n - remaining; ++i) {
      bool prefix = true;
      if (disappearing[i]) for (I earlier = 0; earlier < i; ++earlier)
        if (disappearing[earlier] && component_images[i] == component_images[earlier]
            && std::find(chosen.begin(), chosen.end(), earlier) == chosen.end()) prefix = false;
      if (!prefix) continue;
      chosen.push_back(i); subsets(i + 1, remaining - 1, chosen); chosen.pop_back();
    }
  }
public:
  Kernel(const SkpRequest &request, SkpResult &result, bool observed = false, bool first = false)
    : r(request), out(result), observation(observed), first_only(first) {}
  void run() {
    if (r.n_atoms < 0 || r.n_atoms > 1000000 || r.n_rule_atoms < (observation ? 0 : 1)
        || r.n_rule_atoms > 10000 || r.n_edges < 0 || r.n_rule_edges < 0
        || r.n_domains < 0 || r.candidate_limit < -1 || (r.partial != 0 && r.partial != 1)
        || r.radical_policy < 0
        || r.radical_policy > 2 || (r.n_atoms && !r.atoms) || (r.n_rule_atoms && !r.rule_atoms)
        || (r.n_edges && !r.edges) || (r.n_rule_edges && !r.rule_edges)
        || (r.n_domains && !r.domains)) throw Failure(1, "invalid native request");
    if (first_only && r.partial) throw Failure(1, "first witness requires full source binding");
    host.resize(r.n_atoms); source.resize(r.n_rule_atoms);
    domains.resize(r.n_rule_atoms); mapping.assign(r.n_rule_atoms, -1); used.assign(r.n_atoms, false);
    std::map<I, bool> maps;
    for (I i = 0; i < r.n_atoms; ++i) {
      state_valid(r.atoms[i].state, false);
      if (r.atoms[i].state.element == 0 || !(r.atoms[i].state.flags & 1))
        throw Failure(1, "host must be concrete and present");
      if (r.atoms[i].atom_map <= 0 || maps.count(r.atoms[i].atom_map))
        throw Failure(1, "host atom maps must be positive and unique");
      maps[r.atoms[i].atom_map] = true;
    }
    for (I i = 0; i < r.n_edges; ++i) {
      auto e = r.edges[i];
      if (e.u < 0 || e.v < 0 || e.u >= r.n_atoms || e.v >= r.n_atoms
          || e.u == e.v || host[e.u].count(e.v)) throw Failure(1, "host must be simple");
      bond_valid(e.state);
      if (!e.state.order2) throw Failure(1, "host bonds require positive order");
      host[e.u][e.v] = e.state; host[e.v][e.u] = e.state;
    }
    for (I i = 0; i < r.n_atoms; ++i) {
      auto s = r.atoms[i].state;
      I bonds2 = 0;
      for (const auto &entry : host[i]) bonds2 += entry.second.sigma2 + entry.second.pi2;
      bool consistent = 2 * s.charge == 2 * s.ve - 4 * s.lp - 2 * s.radical - 2 * s.hcount - bonds2;
      host_consistent.push_back(consistent);
      if (!observation && !consistent)
        throw Failure(1, "host lies outside the supported local Lewis model");
    }
    host_component.assign(r.n_atoms, -1);
    for (I start = 0; start < r.n_atoms; ++start) if (host_component[start] < 0) {
      I component = static_cast<I>(host_sizes.size());
      std::vector<I> nodes{start}; host_component[start] = component;
      for (size_t i = 0; i < nodes.size(); ++i)
        for (const auto &edge : host[nodes[i]]) if (host_component[edge.first] < 0) {
          host_component[edge.first] = component; nodes.push_back(edge.first);
        }
      host_sizes.push_back(static_cast<I>(nodes.size()));
    }
    for (I i = 0; i < r.n_rule_atoms; ++i) {
      const auto &a = r.rule_atoms[i];
      state_valid(a.left, a.role >= 0 || (observation && !(a.left.flags & 1)));
      state_valid(a.right, a.role >= 0 || (observation && !(a.right.flags & 1)));
      bool persistent = (a.left.flags & 1) && (a.right.flags & 1);
      if ((!observation && !persistent) || (!(a.left.flags & 1) && !(a.right.flags & 1))
          || (persistent && (a.left.element != a.right.element || a.left.ve != a.right.ve)))
        throw Failure(1, "native rules must preserve atoms and atomic valence labels");
      if (a.role >= 0 && !equal(a.left, a.right))
        throw Failure(1, "wildcard boundary atom states must be read-only");
      if (a.role < -1 || a.role > 4 || a.owner < -1 || a.owner >= r.n_rule_atoms
          || a.capacity < 0 || a.capacity > 1 || a.budget < -1
          || a.side < 0 || a.side > 2 || a.materialization < 0 || a.materialization > 2
          || a.virtual_kind != 0 || a.stereo_slot != -1 || a.identity < -1
          || a.fixed_charge < 0 || a.fixed_charge > (observation ? 3 : 1)
          || a.match_mask < 0 || (a.match_mask & ~I(63)))
        throw Failure(1, "invalid wildcard contract");
      count(a.baseline_lp, "lone-pair baseline");
      if (a.left.lp + a.baseline_lp > 1000000000
          || a.right.lp + a.baseline_lp > 1000000000)
        throw Failure(1, "restored lone-pair population exceeds representation limit");
      if (a.role == 4) throw Failure(1, "stereo wildcard applications are not compiled");
      if (a.role >= 0 ? (!persistent || a.left.element != 0 || a.right.element != 0)
                      : (((a.left.flags & 1) && a.left.element == 0)
                         || ((a.right.flags & 1) && a.right.element == 0)))
        throw Failure(1, "wildcard roles require persistent symbolic atoms");
      for (auto d : {a.elements, a.charges, a.radicals, a.bond_orders})
        if (d.count < -1 || d.count == 0 || d.offset < 0 || d.offset > r.n_domains
            || (d.count > 0 && d.count > r.n_domains - d.offset))
          throw Failure(1, "invalid wildcard domain");
    }
    std::map<Key, bool> rule_pairs;
    for (I i = 0; i < r.n_rule_edges; ++i) {
      const auto &b = r.rule_edges[i];
      if (b.u < 0 || b.v < 0 || b.u >= r.n_rule_atoms || b.v >= r.n_rule_atoms
          || b.u == b.v || rule_pairs.count(key(b.u, b.v))) throw Failure(1, "rule must be simple");
      rule_pairs[key(b.u, b.v)] = true; bond_valid(b.left); bond_valid(b.right);
      if (b.min_pi2 < -1 || (b.min_pi2 >= 0 && (b.min_pi2 % 2 || b.min_pi2 > 1000000000)))
        throw Failure(1, "invalid relative pi demand");
      for (int side = 0; side < 2; ++side) {
        auto state = side ? b.right : b.left;
        auto u = side ? r.rule_atoms[b.u].right : r.rule_atoms[b.u].left;
        auto v = side ? r.rule_atoms[b.v].right : r.rule_atoms[b.v].left;
        if (state.order2 && (!(u.flags & 1) || !(v.flags & 1)))
          throw Failure(1, "an observed bond needs two observed endpoints");
      }
      if ((r.rule_atoms[b.u].role >= 0 || r.rule_atoms[b.v].role >= 0)
          && !equal(b.left, b.right)) throw Failure(1, "wildcard boundaries must be read-only");
      if (b.left.order2) { source[b.u][b.v] = &b; source[b.v][b.u] = &b; }
    }
    for (I p = 0; p < r.n_rule_atoms; ++p) {
      const auto &a = r.rule_atoms[p];
      if (a.role < 0) continue;
      if (a.owner >= 0 && !source[p].count(a.owner))
        throw Failure(1, "a wildcard owner must be adjacent");
      if ((a.role == 1 || a.role == 2)
          && (source[p].size() != 1 || a.owner < 0))
        throw Failure(1, "an attachment port requires exactly one declared owner edge");
    }
    std::vector<bool> visited(r.n_rule_atoms, false);
    for (I start = 0; start < r.n_rule_atoms; ++start) if (!visited[start] && (r.rule_atoms[start].left.flags & 1)) {
      std::vector<I> component{start}; visited[start] = true;
      for (size_t j = 0; j < component.size(); ++j)
        for (const auto &entry : source[component[j]]) if (!visited[entry.first]) {
          visited[entry.first] = true; component.push_back(entry.first);
        }
      std::sort(component.begin(), component.end()); components.push_back(component);
    }
    if (components.empty()) {
      if (observation && r.n_rule_atoms == 0) {
        accept(); out.stats.applications = static_cast<I>(out.applications.size()); return;
      }
      throw Failure(1, "rule needs a nonempty observed input");
    }
    for (I p = 0; p < r.n_rule_atoms; ++p)
      for (I h = 0; h < r.n_atoms; ++h)
        if (host[h].size() >= source[p].size() && node_matches(p, h)) domains[p].push_back(h);
    embeddings.resize(components.size()); component_images.resize(components.size());
    chosen_images.resize(components.size()); disappearing.resize(components.size(), observation);
    for (size_t c = 0; c < components.size(); ++c)
      for (I p : components[c]) disappearing[c] = disappearing[c] && !(r.rule_atoms[p].right.flags & 1);
    if (first_only) {
      std::vector<I> nodes;
      rule_component.resize(r.n_rule_atoms);
      for (size_t c = 0; c < components.size(); ++c) for (I p : components[c]) {
        nodes.push_back(p); rule_component[p] = c;
      }
      std::vector<std::vector<I>> unused;
      embed_component(nodes, 0, unused);
      if (out.rejected_completions) throw Failure(3, "all full observation bindings were rejected");
      return;
    }
    for (size_t c = 0; c < components.size(); ++c) {
      embedding_component = c; represented.assign(host_sizes.size(), false);
      auto order = components[c];
      std::stable_sort(order.begin(), order.end(), [&](I a, I b) {
        return domains[a].size() < domains[b].size();
      });
      embed_component(order, 0, embeddings[c]);
      for (const auto &embedding : embeddings[c]) {
        std::vector<I> image;
        for (I p : components[c]) image.push_back(embedding[p]);
        std::sort(image.begin(), image.end()); component_images[c].push_back(image);
      }
    }
    std::vector<I> chosen;
    I n = static_cast<I>(components.size());
    for (I k = n; k >= (r.partial ? 1 : n); --k) subsets(0, k, chosen);
    if (out.applications.empty() && out.rejected_completions)
      throw Failure(3, "all partial observation completions were rejected");
    out.stats.applications = static_cast<I>(out.applications.size());
  }
};
}

extern "C" {
int skp_abi(void) { return 1; }
static SkpResult *execute_request(const SkpRequest *request, bool observed, bool first = false) {
  auto result = new (std::nothrow) SkpResult;
  if (!result) return nullptr;
  auto start = std::chrono::steady_clock::now();
  try {
    if (!request) throw Failure(1, "null request");
    Kernel(*request, *result, observed, first).run();
  } catch (const Found &) {
    result->search_complete = false;
    result->stats.applications = static_cast<I>(result->applications.size());
  } catch (const Failure &e) {
    result->status = e.status; result->error = e.what(); result->applications.clear();
    result->stats.applications = 0; result->search_complete = false;
  } catch (const std::exception &e) {
    result->status = 1; result->error = e.what(); result->applications.clear();
    result->stats.applications = 0; result->search_complete = false;
  } catch (...) {
    result->status = 1; result->error = "unknown native failure"; result->applications.clear();
    result->stats.applications = 0; result->search_complete = false;
  }
  result->stats.elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::steady_clock::now() - start).count();
  return result;
}
SkpResult *skp_execute(const SkpRequest *request) { return execute_request(request, false); }
SkpResult *skp_execute_observed(const SkpRequest *request) { return execute_request(request, true); }
SkpResult *skp_execute_observed_first(const SkpRequest *request) { return execute_request(request, true, true); }
int skp_search_complete(const SkpResult *r) { return r && r->search_complete; }
int64_t skp_rejected_completion_count(const SkpResult *r) { return r ? r->rejected_completions : 0; }
int skp_status(const SkpResult *r) { return r ? r->status : 1; }
const char *skp_error(const SkpResult *r) { return r ? r->error.c_str() : "native allocation failed"; }
const SkpStats *skp_stats(const SkpResult *r) { return r ? &r->stats : nullptr; }
int64_t skp_application_count(const SkpResult *r) { return r ? static_cast<I>(r->applications.size()) : 0; }
const int64_t *skp_mapping(const SkpResult *r, I i) {
  if (!r || i < 0 || i >= static_cast<I>(r->applications.size())) return nullptr;
  return r->applications[i].mapping.data();
}
const SkpProductAtom *skp_atoms(const SkpResult *r, I i, I *n) {
  if (!r || !n || i < 0 || i >= static_cast<I>(r->applications.size())) return nullptr;
  const auto &v = r->applications[i].atoms; *n = static_cast<I>(v.size()); return v.data();
}
const SkpProductEdge *skp_edges(const SkpResult *r, I i, I *n) {
  if (!r || !n || i < 0 || i >= static_cast<I>(r->applications.size())) return nullptr;
  const auto &v = r->applications[i].edges; *n = static_cast<I>(v.size()); return v.data();
}
void skp_free(SkpResult *r) { delete r; }
}
