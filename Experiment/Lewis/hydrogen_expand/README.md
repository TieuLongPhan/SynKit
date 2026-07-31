# Hydrogen-extension comparison

This experiment uses the 109-reaction `hydrogen.pkl.gz` corpus from the
PartialAAMs analysis. It runs sequentially with one process and forces common
numeric libraries to one thread.

```bash
bash Experiment/Lewis/hydrogen_expand/run_comparison.sh
```

Use `--limit 3 --repetitions 1` for a pilot. Results are written under
`Experiment/Lewis/Runs` by default.

After changing only HExtend, reuse a completed reference run instead of
rerunning the six external reference columns:

```bash
bash Experiment/Lewis/hydrogen_expand/run_comparison.sh \
  --reuse-reference-dir Experiment/Lewis/Runs/<completed-run>
```

The runner compares hydrogen-extension implementations on the same 104
reactions accepted by the published analysis:

- `hextend_legacy` uses SynKit's historical same-permutation candidate
  enumeration, followed by full-ITS classification.
- `hextend_new` uses provenance-aware hydrogen-transfer enumeration, followed
  by the same full-ITS classification used for the reference comparison. Its
  RC-invariant and hydrogen-distance signatures are used only as
  necessary-condition prefilters before the final full-graph isomorphism
  decision.
- `method_a` enumerates every hydrogen permutation and classifies the complete
  ITS graphs.
- `method_b` computes heavy-atom ITS automorphisms and uses them as anchors for
  co-extension.
- Both reference methods run separately as `AN-gm` (native GranMapache stable
  extension), `RB-gm` (anchor-relabeling GranMapache), and `RB-nx`
  (anchor-relabeling NetworkX), reproducing the six columns of Table 2.

The active environment runs HExtend and `RB-nx`. The launcher uses the `aam`
Conda environment for the two GranMapache backends; override it with
`--gmapache-env NAME` if needed.

Method A and Method B are not aliases for PartialAAMs' GM, RB1, and RB2. Those
methods complete partial atom maps and belong to `partial_expand`; they are not
part of this hydrogen-extension runtime comparison.

The five source records rejected by the reference filters are retained in
`excluded-records.json`. The aggregate records class-count agreement between
each HExtend implementation and Method A, plus Table-2-style mean and
population-standard-deviation timings grouped by unmatched hydrogen count.
