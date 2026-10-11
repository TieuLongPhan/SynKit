"""Explicit-incidence mass-action models for the chiral-feedback pilot.

All dimensional choices here are the dimensionless choices of the research
plan/reference code. A repeated reactant contributes x**stoichiometry; no
combinatorial 1/2 is inserted into ligation rates.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from itertools import product

import numpy as np


@dataclass(frozen=True)
class Channel:
    id: str
    reactants: dict[str, int]
    products: dict[str, int]
    rate: float
    module: str
    expression: str
    reverse: str | None = None
    external_reactants: tuple[str, ...] = ()
    external_products: tuple[str, ...] = ()


class Network:
    def __init__(self, name, species, mirror, channels, feed, loss, internal, metadata):
        self.name = name
        self.species = tuple(species)
        self.mirror = np.asarray(mirror, dtype=int)
        if not np.array_equal(self.mirror[self.mirror], np.arange(len(species))):
            raise ValueError("The mirror must be an involution")
        if any(r.rate < 0 or not np.isfinite(r.rate) for r in channels):
            raise ValueError("Rates must be finite and nonnegative")
        # Exactly zero coefficients are absent from structural support.
        self.channels = tuple(r for r in channels if r.rate > 0)
        self.feed = np.broadcast_to(feed, (len(species),)).astype(float).copy()
        self.loss = np.broadcast_to(loss, (len(species),)).astype(float).copy()
        self.internal = tuple(internal)
        self.metadata = metadata
        self.minus = np.zeros((len(species), len(self.channels)), dtype=int)
        self.plus = np.zeros_like(self.minus)
        index = {s: i for i, s in enumerate(species)}
        for j, r in enumerate(self.channels):
            for s, count in r.reactants.items():
                self.minus[index[s], j] = count
            for s, count in r.products.items():
                self.plus[index[s], j] = count
        self.net = self.plus - self.minus
        self.rates = np.asarray([r.rate for r in self.channels])
        self.q_even, self.q_odd = symmetry_bases(self.mirror)
        self._mirror_channels = []
        for j in range(len(self.channels)):
            matches = np.flatnonzero(
                np.all(self.minus == self.minus[self.mirror, j, None], axis=0)
                & np.all(self.plus == self.plus[self.mirror, j, None], axis=0)
                & np.isclose(self.rates, self.rates[j], rtol=0, atol=0)
            )
            if len(matches) != 1:
                raise ValueError(f"Missing or ambiguous mirror of {self.channels[j].id}")
            self._mirror_channels.append(int(matches[0]))
        if not (np.array_equal(self.feed, self.feed[self.mirror])
                and np.array_equal(self.loss, self.loss[self.mirror])):
            raise ValueError("Feed and loss must respect the mirror")

    def flux(self, x):
        return self.rates * np.prod(np.asarray(x)[:, None] ** self.minus, axis=0)

    def rhs(self, t, x):
        return self.net @ self.flux(x) + self.feed - self.loss * x

    def jacobian(self, t, x):
        # Differentiate monomials directly, including at zero concentration.
        x = np.asarray(x)
        dv = np.zeros((len(self.channels), len(self.species)), dtype=x.dtype)
        for i in range(len(x)):
            js = np.flatnonzero(self.minus[i])
            powers = self.minus[:, js].copy()
            powers[i] -= 1
            dv[js, i] = (self.rates[js] * self.minus[i, js]
                         * np.prod(x[:, None] ** powers, axis=0))
        return self.net @ dv - np.diag(self.loss)

    def to_record(self):
        rows = []
        for j, r in enumerate(self.channels):
            item = asdict(r)
            item["mirror"] = self.channels[self._mirror_channels[j]].id
            item["reverse_present"] = any(c.id == r.reverse for c in self.channels)
            item["explicit_catalysts"] = {
                s: min(r.reactants[s], r.products[s])
                for s in r.reactants.keys() & r.products.keys()
            }
            rows.append(item)
        return {
            "name": self.name, "species": list(self.species),
            "mirror": self.mirror.tolist(), "internal_for_productivity": list(self.internal),
            "feed": self.feed.tolist(), "loss": self.loss.tolist(),
            "channels": rows, "S_minus": self.minus.tolist(),
            "S_plus": self.plus.tolist(), "S": self.net.tolist(),
            "metadata": self.metadata,
        }


def symmetry_bases(mirror):
    n = len(mirror)
    even, odd = [], []
    for i, j in enumerate(mirror):
        if i > j:
            continue
        e = np.zeros(n)
        if i == j:
            e[i] = 1
        else:
            e[i] = e[j] = 1 / np.sqrt(2)
            o = e.copy()
            o[j] *= -1
            odd.append(o)
        even.append(e)
    return np.column_stack(even), np.column_stack(odd)


MODULES = ("matched", "mismatched", "homoligation", "heteroligation")


@dataclass(frozen=True)
class ParentParameters:
    mechanism: str = "constructive"
    u: float = 0.0
    k1: float = 3.0
    k2: float = 0.75
    a1: float = 1.0
    a2: float = 4.0
    kmon: float = 0.01
    kdim: float = 0.0
    phi: float = 0.1
    total_feed: float = 1.0
    kr: float = 0.0

    @property
    def ab(self):
        if self.mechanism == "constructive":
            return 1.0, self.kmon
        if self.mechanism == "destructive":
            return self.kmon, 1.0
        raise ValueError(self.mechanism)

    @property
    def inlet(self):
        if self.mechanism == "constructive":
            return np.array([self.total_feed, 0, 0, 0, 0, 0, 0.])
        return np.array([0, self.total_feed/2, self.total_feed/2, 0, 0, 0, 0.])


def parent(p=ParentParameters(), modules=MODULES):
    modules = frozenset(modules)
    if not modules <= frozenset(MODULES):
        raise ValueError("Unknown intervention module")
    a, b = p.ab
    channels = []

    def pair(name, lhs, rhs, forward, backward, module, expression):
        if module in ("background", "matched", "mismatched"):
            forward_expression, backward_expression = expression+"*a", expression+"*b"
        elif module.endswith("ligation"):
            forward_expression, backward_expression = expression, expression+"*kdim"
        else:
            forward_expression = backward_expression = expression
        channels.append(Channel(name+"_f", lhs, rhs, forward, module,
                                forward_expression, name+"_b"))
        channels.append(Channel(name+"_b", rhs, lhs, backward, module,
                                backward_expression, name+"_f"))

    for mon, same, opposite in (("L", "LL", "DD"), ("D", "DD", "LL")):
        pair("background_"+mon, {"P": 1}, {mon: 1}, p.u*a, p.u*b,
             "background", "u")
        for module, catalyst, rate in (("matched", same, p.k1),
                                       ("mismatched", opposite, p.k2)):
            if module in modules:
                pair(module+"_"+mon, {"P": 1, catalyst: 1}, {mon: 1, catalyst: 1},
                     rate*a, rate*b, module, "k1" if module == "matched" else "k2")
        if "homoligation" in modules:
            pair("homo_"+mon, {mon: 2}, {same: 1}, p.a1, p.a1*p.kdim,
                 "homoligation", "a1")
    if "heteroligation" in modules:
        for dimer in ("LD", "DL"):
            pair("hetero_"+dimer, {"L": 1, "D": 1}, {dimer: 1},
                 p.a2, p.a2*p.kdim, "heteroligation", "a2")
    if p.kr > 0:
        pair("racemization", {"L": 1}, {"D": 1}, p.kr, p.kr, "racemization", "kr")
    return Network(p.mechanism, ("P", "L", "D", "LL", "LD", "DL", "DD"),
                   (0, 2, 1, 6, 5, 4, 3), channels, p.phi*p.inlet, p.phi,
                   ("L", "D", "LL", "LD", "DL", "DD"),
                   {"parameters": asdict(p), "modules": sorted(modules),
                    "P_role": "dynamic precursor; excluded from internal productivity rows",
                    "hydrolysis": "LD and DL each yield L+D",
                    "environment": "feed and uniform dilution fixed within a support comparison"})


def parent_supports():
    for bits in product((False, True), repeat=len(MODULES)):
        yield tuple(m for m, bit in zip(MODULES, bits) if bit)


def control_a():
    channels = []
    for s in ("L", "D"):
        channels.extend([
            Channel("growth_"+s, {s: 1}, {s: 2}, 2, "growth", "2",
                    external_reactants=("A",)),
            Channel("quadratic_"+s, {s: 2}, {s: 1}, 1, "removal", "1",
                    external_products=("W",)),
        ])
    return Network("productive_stable", ("L", "D"), (1, 0), channels, 1, 1,
                   ("L", "D"), {"boundary": "A chemostatted at unit activity; input/loss excluded from core search"})


def control_b(eps=0.0, kr=0.0):
    channels = []
    for mon, same, opposite in (("L", "U", "V"), ("D", "V", "U")):
        channels.extend([
            Channel("ligation_"+mon, {mon: 2}, {same: 1}, 1, "ligation", "h=1"),
            Channel("destruction_"+mon, {mon: 1, opposite: 1}, {opposite: 1},
                    6, "destruction", "c=6", "restoration_"+mon,
                    external_products=("W",)),
            Channel("restoration_"+mon, {opposite: 1}, {opposite: 1, mon: 1},
                    eps, "destruction", "epsilon", "destruction_"+mon,
                    external_reactants=("W",)),
        ])
    if kr > 0:
        channels.extend([
            Channel("rac_L", {"L": 1}, {"D": 1}, kr, "racemization", "kr", "rac_D"),
            Channel("rac_D", {"D": 1}, {"L": 1}, kr, "racemization", "kr", "rac_L"),
        ])
    return Network("selective_removal", ("L", "D", "U", "V"), (1, 0, 3, 2),
                   channels, (9, 9, 0, 0), 1, ("L", "D", "U", "V"),
                   {"epsilon": eps, "kr": kr, "h": 1, "c": 6, "q": 9,
                    "boundary": "W at unit activity when restoration is enabled; input/loss excluded"})


def reference_initial(ee=1e-4):
    l, d = 1+ee, 1-ee
    x = np.array([1, l, d, l*l, l*d, d*l, d*d])
    return x / (x[0]+x[1]+x[2]+2*np.sum(x[3:]))
