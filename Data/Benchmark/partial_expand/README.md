# Minimal partial-expansion benchmark

This directory compares minimal partial atom-mapping expansion. The general
corpus contains 39,732 records with an independent fully mapped `smart`
reference. The radical corpus contains 5,426 records without an independent
full-AAM reference, so it reports valid-completion coverage instead of mapping
accuracy.

The methods are:

- `synkit`: `ITSExpand.expand_rsmi` in the current `synkit` environment;
- `gm`: historical GranMapache expansion;
- `rb1`: historical `PartialAAMs.extend`;
- `rb2`: historical `PartialAAMs.extend_g`.

Run SynKit five times:

```bash
conda run -n synkit python \
  Data/Benchmark/partial_expand/run_expansion_5x_synkit.py
```

Use the same generation-only boundary for the radical corpus:

```bash
conda run -n synkit python \
  Data/Benchmark/partial_expand/run_expansion_5x_synkit.py \
  --suite radical
```

Only radical attribute transport is enabled during radical generation. All
completion, constitution, and radical-state checks run after the generation
timer.

Run the three historical methods in the reconstructed `aam` environment:

```bash
conda run -n synkit python \
  Data/Benchmark/partial_expand/run_expansion_5x_external_aam.py
```

Generate the two-panel comparison with the single plot script:

```bash
conda run -n synkit python \
  Data/Benchmark/partial_expand/plot_general_radical_comparison.py
```

Panel A reports general-corpus generation time. Panel B reports general ITS
accuracy and radical valid-completion coverage; radical runtime is deliberately
omitted because its required attribute transport is not directly comparable to
the normal path. External radical coverage uses the audited aggregate retained
in `sprint/SS_LOG.md`.

Use `--limit 10 --repetitions 1 --output-dir /tmp/partial-expand-pilot`
for a pilot. Existing outputs are protected unless `--force` is supplied.

Reproduce forward and backward rule replay for both graph representations:

```bash
conda run -n synkit python \
  Data/Benchmark/partial_expand/benchmark_bidirectional_replay.py
```

The runner writes separate tuple and `typesGH` summaries plus compressed
case-level evidence. Complete enumeration is the default: there is no
per-direction timeout and no embedding cap. Use `--record-ids` to select a
focused audit; `--case-timeout` and `--embedding-threshold` are optional
diagnostic ceilings only. After symmetry-safe matcher and product-clustering
optimization, records 12272, 12602, 13898, and 32345 recover all 16 tested
representation/direction combinations without either ceiling.

Plot the graph-rewriting population comparison (runtime is intentionally
excluded):

```bash
conda run -n synkit python \
  Data/Benchmark/partial_expand/plot_graph_rewriting_comparison.py
```

The paired-dot figure reports LLG change relative to the atom--bond graph for
forward/reverse mapping populations and unique standardized reactions. Exact
counts are printed beside the points. The script writes PDF/PNG benchmark
outputs and refreshes `paper/lwg/fig/graph_rewriting_comparison.png`.
