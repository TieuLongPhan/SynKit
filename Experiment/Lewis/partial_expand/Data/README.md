# Minimal partial-expansion benchmark

This experiment compares minimal partial atom-mapping expansion. The general
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
  Experiment/Lewis/partial_expand/repeat_synkit.py
```

Use the same generation-only boundary for the radical corpus:

```bash
conda run -n synkit python \
  Experiment/Lewis/partial_expand/repeat_synkit.py \
  --suite radical
```

Only radical attribute transport is enabled during radical generation. All
completion, constitution, and radical-state checks run after the generation
timer.

Run the three historical methods in the reconstructed `aam` environment:

```bash
conda run -n synkit python \
  Experiment/Lewis/partial_expand/repeat_external.py
```

Generate the two-panel comparison:

```bash
conda run -n synkit python \
  Experiment/Lewis/partial_expand/plot.py
```

Panel A reports general-corpus generation time. Panel B reports general ITS
accuracy and radical valid-completion coverage; radical runtime is deliberately
omitted because its required attribute transport is not directly comparable to
the normal path. External radical coverage uses the audited aggregate retained
in `sprint/SS_LOG.md`.

Use `--limit 10 --repetitions 1 --output-dir /tmp/partial-expand-pilot`
for a pilot. Existing outputs are protected unless `--force` is supplied.
Retained outputs are stored in this directory.
