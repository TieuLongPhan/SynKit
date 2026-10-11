"""Chiral instability of all module supports over a rate domain.

The two reference points of ``run.py`` are replaced by log-uniform samples
of the rate coefficients. For each sample and each of the 16 supports the
racemic states are the roots of the exact quartic, and a support is counted
as chirally unstable when some positive racemate has alpha_+ < 0 < alpha_-.
For every draw the smallest unstable supports are integrated from a small
odd seed to test the nonlinear phenotype.

Two scans are run. The base scan varies the six coefficients of the
published model. The extended scan also varies background conversion,
reverse ligation and racemization. Run from the repository root::

    PYTHONDONTWRITEBYTECODE=1 python Experiment/Autocatalyic/parameter_domain.py
"""
from __future__ import annotations

import json
import os
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from analysis import paired_trajectories, racemic_parent_states, stability
from models import MODULES, ParentParameters, parent, parent_supports

HERE = Path(__file__).resolve().parent
RESULTS = HERE/"results"
PAPER = HERE.parents[1]/"paper"/"autocatalytic"
SEED = 1729
SAMPLES = 2000
SLOWEST_GROWTH = 1e-4  # slower instabilities are counted but not integrated
BASE = {"k1": (0.1, 100.0), "k2": (0.1, 100.0), "a1": (0.1, 10.0),
        "a2": (0.4, 40.0), "phi": (0.01, 1.0), "kmon": (1e-4, 1e-1)}
EXTENDED = dict(BASE, u=(1e-4, 1e-1), kdim=(1e-4, 1.0), kr=(1e-4, 1.0))
SUPPORTS = list(parent_supports())


def bits(modules):
    return "".join("1" if m in modules else "0" for m in MODULES)


def draw(rng, mechanism, domain):
    values = {k: float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
              for k, (lo, hi) in domain.items()}
    return ParentParameters(mechanism=mechanism, **values)


def unstable_racemate(p, modules):
    """A positive racemate with a clean chiral instability, None, or 'failed'."""
    try:
        states = racemic_parent_states(p, modules)
    except RuntimeError:
        return "failed"
    model = parent(p, modules)
    for x in states:
        if x[1] <= 1e-9:
            continue
        info = stability(model, x)
        if info["alpha_even"] < -1e-8 and info["alpha_odd"] > 1e-8:
            return x, info["alpha_odd"]
    return None


def confirm(p, modules, x, odd):
    if odd < SLOWEST_GROWTH:
        return "slow"
    try:
        pair, *_ = paired_trajectories(parent(p, modules), x, horizon=max(4000., 30/odd))
    except (RuntimeError, ValueError):
        return "unresolved"
    plus = pair["plus"]
    ok = (plus["alpha_full"] < -1e-8 and abs(plus["metrics"]["ee_monomer"] or 0) > 1e-3
          and pair["paired_trajectory_mirror_error"] < 1e-6)
    return "confirmed" if ok else "other endpoint"


def one_draw(p):
    passing, failed = {}, 0
    for modules in SUPPORTS:
        found = unstable_racemate(p, modules)
        if found == "failed":
            failed += 1
        elif found:
            passing[modules] = found
    minimal = [m for m in passing if not any(set(o) < set(m) for o in passing)]
    return {"passing": [bits(m) for m in passing], "minimal": [bits(m) for m in minimal],
            "outcomes": [confirm(p, m, *passing[m]) for m in minimal], "failed": failed}


def scan(mechanism, domain, rng, pool):
    draws = [draw(rng, mechanism, domain) for _ in range(SAMPLES)]
    results = pool.map(one_draw, draws, chunksize=10)
    needed = "1.1." if mechanism == "constructive" else ".11."
    positions = [i for i, c in enumerate(needed) if c == "1"]
    unstable, minimal, outcomes = Counter(), Counter(), Counter()
    for r in results:
        unstable.update(r["passing"])
        minimal.update(r["minimal"])
        outcomes.update(r["outcomes"])
    return {"mechanism": mechanism, "samples": SAMPLES,
            "samples_with_unstable_support": sum(bool(r["passing"]) for r in results),
            "unstable_by_support": dict(sorted(unstable.items())),
            "minimal_by_support": dict(sorted(minimal.items())),
            "unresolved_quartic_evaluations": sum(r["failed"] for r in results),
            "unstable_supports_missing_required_modules": sum(
                n for key, n in unstable.items() if any(key[i] != "1" for i in positions)),
            "nonlinear_outcomes": dict(outcomes)}


def percent(count, total):
    return f"{100*count/max(1, total):.1f}"


def macros(prefix, con, des):
    def base(record, key):
        return percent(record["minimal_by_support"].get(key, 0),
                       record["samples_with_unstable_support"])
    outcomes = Counter(con["nonlinear_outcomes"])+Counter(des["nonlinear_outcomes"])
    integrated = sum(n for k, n in outcomes.items() if k != "slow")
    return {
        f"{prefix}ConUnstable": percent(con["samples_with_unstable_support"], SAMPLES),
        f"{prefix}DesUnstable": percent(des["samples_with_unstable_support"], SAMPLES),
        f"{prefix}ConMinimalBase": base(con, "1010"),
        f"{prefix}DesMinimalBase": base(des, "0110"),
        f"{prefix}Violations": con["unstable_supports_missing_required_modules"]
        + des["unstable_supports_missing_required_modules"],
        f"{prefix}Checks": integrated,
        f"{prefix}Confirmed": outcomes["confirmed"],
        f"{prefix}Slow": outcomes["slow"],
        f"{prefix}Unresolved": outcomes["unresolved"]+outcomes["other endpoint"],
    }


def main():
    rng = np.random.default_rng(SEED)
    records = {}
    with Pool(int(os.environ.get("AUTOCAT_WORKERS", "8"))) as pool:
        for name, domain in (("base", BASE), ("extended", EXTENDED)):
            records[name] = {m: scan(m, domain, rng, pool)
                             for m in ("constructive", "destructive")}
            for record in records[name].values():
                print(name, json.dumps(record), flush=True)
    (RESULTS/"parameter_domain.json").write_text(json.dumps(
        {"seed": SEED, "domains": {"base": BASE, "extended": EXTENDED},
         "slowest_integrated_growth": SLOWEST_GROWTH,
         "records": records["base"], "extended_records": records["extended"]},
        indent=2)+"\n")
    values = {"DomainSamples": SAMPLES}
    values.update(macros("Domain", *records["base"].values()))
    values.update(macros("Wide", *records["extended"].values()))
    lines = ["% Generated by Experiment/Autocatalyic/parameter_domain.py; do not edit."]
    lines += [rf"\newcommand{{\{k}}}{{{v}}}" for k, v in values.items()]
    (PAPER/"domain_results.tex").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    main()
