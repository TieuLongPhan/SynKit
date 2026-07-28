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
diagnostic ceilings only. Record 886 is the relative lone-pair regression:

```bash
conda run -n synkit python \
  Experiment/Lewis/rule_replay/benchmark.py \
  --record-ids 886 \
  --directions forward \
  --output-dir Experiment/Lewis/rule_replay/Data/relative-lone-pair
```

After symmetry-safe matcher and product-clustering optimization, records 12272,
12602, 13898, and 32345 recover all 16 tested representation/direction
combinations without either ceiling.

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
