# Bidirectional rule-replay benchmark

This experiment reproduces forward and backward rule replay for both the
Lewis-labelled graph (LLG, `tuple`) and legacy `typesGH` representations in
the LWG paper workflow:

```bash
conda run -n synkit python \
  Experiment/Lewis/rule_replay/benchmark.py
```

The runner writes separate tuple and `typesGH` summaries, compressed case-level
evidence, and a path- and timing-independent `results.json` that can be
committed upstream. Complete enumeration is the default: there is no
per-direction timeout and no embedding cap. Use `--record-ids` to select a
focused audit; `--case-timeout` and `--embedding-threshold` are optional
diagnostic ceilings only. Both representations use the minimal changed-edge
reaction center; unchanged internal context edges are not added implicitly.

The default `structural` policy deduplicates mappings by attributed rule
automorphisms and products by attributed ITS isomorphism. WL colours and label
inventories are rejection filters; equality is checked by an exact certificate
or VF2. The `deferred` policy is available for output-set checks and is not used
for the runtime comparison. Both policies must produce the same standardized
reaction set.

Reports distinguish raw serialized reactions, duplicates removed after
standardization, per-case unique reactions, and the global unique set.
Standardization removes atom mapping, canonicalizes both sides, and sorts
components before duplicate removal.

## Historical MØD baseline

`mod_benchmark.py` reproduces the rule-application path retained in SynKit
1.0, using the optional MØD engine and the legacy atom/bond GML projection:

```bash
conda run -n synkit python \
  Experiment/Lewis/rule_replay/mod_benchmark.py \
  --limit 10 \
  --output-dir Experiment/Lewis/Runs/mod-pilot
```

The `bt` strategy tries strict component participation before relaxed
application. The `comp` and `all` strategies expose the other historical
paths. Reports include the MØD version, adapter identity, stage timings,
endpoint recovery, derivation counts, and standardized output counts.

The MØD run is a runtime and endpoint-recovery baseline. Its GML input contains
ordinary atom and bond labels, not the full LLG electron state.

Record 886 is the relative lone-pair regression:

```bash
conda run -n synkit python \
  Experiment/Lewis/rule_replay/benchmark.py \
  --record-ids 886 \
  --directions forward \
  --output-dir Experiment/Lewis/rule_replay/Data/relative-lone-pair
```

Records 12272, 12602, 13898, and 32345 cover high-multiplicity regressions
without a timeout or embedding cap.

Plot the graph-rewriting population comparison; runtime is intentionally
excluded:

```bash
conda run -n synkit python \
  Experiment/Lewis/rule_replay/plot.py
```

The paired-dot figure reports LLG change relative to the atom-bond graph for
forward/reverse mapping populations and unique standardized reactions. Exact
counts are printed beside the points. The script writes PDF/PNG outputs here
and refreshes `paper/lwg/fig/graph_rewriting_comparison.png`.
