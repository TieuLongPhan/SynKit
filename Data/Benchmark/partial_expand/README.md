# Partial expansion and rule-replay benchmarks

This directory separates two experiments that answer different questions.

1. **Partial-AAM expansion** reconstructs a complete mapped reaction from the
   supplied reaction centre and one endpoint. Existing atom maps provide the
   correspondence, so this step performs no new matching-morphism search.
2. **Rule replay** extracts the reaction-centre rule and then applies it to an
   unmapped endpoint. This experiment includes matching, rewriting, and
   serialization and times those stages independently. Recovery requires a
   complete generated reaction to equal the reference after both are normalized
   with `Standardize.fit(remove_aam=True, ignore_stereo=True)`; it is not merely
   a product-side or reaction-centre check.

Rule replay uses disclosed ceilings of 10,000 raw embeddings and five wall
seconds per direction and case. A case that exceeds either ceiling is reported
as a threshold error or timeout; it is not silently truncated. These guards
prevent an unbounded symmetric embedding/application population from blocking
the corpus-level report.

`ITSExpand` is an operational, map-anchored reconstruction with the shape of a
DPO completion. The implementation should not be described as an exact
categorical pushout/pushout-complement construction unless the relevant
universal property and DPO gluing conditions are established separately.

## Data and protocols

- General expansion uses all 39,732 records in
  `Data/Benchmark/benchmark.json.gz`; `smart` is the fully mapped reference and
  `partial` is the masked input.
- Radical expansion uses all 5,426 source records, but these have no independent
  fully mapped reference. It therefore reports guarded completion gates rather
  than a misleading AAMValidator ITS self-comparison.
- Full-corpus rule application uses every general reaction in both the
  electron-aware Lewis-labelled graph (`tuple`) and legacy atom-bond graph
  (`typesGH`) representations. The radical corpus is replayed only as `tuple`,
  because `typesGH` has no side-specific lone-pair, radical, or valence-electron
  annotations. Hydrogen policy, automorphism handling, ceilings, and recovery
  checks are identical.

Generated candidates and machine reports belong in `results/`, which is
ignored by Git. Commands assume the repository root as the working directory.

```bash
conda run -n synkit python \
  Data/Benchmark/partial_expand/run_full_benchmarks.py
```

This single command runs SynKit, GM, RB1 (`PartialAAMs.extend`), and RB2
(`PartialAAMs.extend_g`) completion on both corpora, followed by forward and
inverse replay for general/tuple, general/`typesGH`, and radical/tuple. It
expects the frozen external sources at the paths disclosed by
`benchmark_expansion_comparison.py`. Use `--limit N --force` for a pilot and
`--only expansion` or `--only replay` to run one experiment family. Put pilots
in a separate directory, for example `--results /tmp/synkit-benchmark-pilot`,
so `--force` cannot overwrite completed full reports.
When all five reports are present, the runner also writes
`results/full-benchmark-summary.json`, including matched successful-case
solution totals for the two general replay representations.

The rule is extracted with `core=True` and implicit-H normalization. Both
general representations retain mapped-H edits as reaction-centre before/after
`hcount` transitions. `explicit_h=False` therefore avoids expanding spectator
hydrogen nodes without dropping reacting-hydrogen information.

Use `--limit N` for a pilot. General expansion generation and evaluation are
separate so the generated mapping is assessed against the independent
`smart` reference by the current `AAMValidator` (`ITS` and `RC`, constitutional
profile).

## Five-repeat expansion comparison

The paper comparison uses AAM accuracy without requiring source-map anchors to
be retained.  Anchor preservation remains a separate reported property.  Mean
generation time uses every attempted input, including failed attempts, so an
implementation is not made artificially faster by early exceptions.

Run SynKit five times on both complete datasets in the current environment:

```bash
conda run -n synkit python \
  Data/Benchmark/partial_expand/run_expansion_5x_synkit.py
```

Run GM, RB1 (`extend`), and RB2 (`extend_g`) five times on both datasets.  The
parent/current environment performs the common evaluation, while generation is
re-launched inside the isolated `aam` environment:

```bash
conda run -n synkit python \
  Data/Benchmark/partial_expand/run_expansion_5x_external_aam.py
```

The reconstructed historical generator stack is SynKit 0.0.6, PartialAAMs
commit `008173e`, and GranMapache commit `4c8f292`. These are the final versions
available before the 5 May 2025 CSV run. The later PartialAAMs commit `edfbcbec`
contains that CSV but not the source/dependency state that generated it.

Each runner writes individual JSON reports and compressed case records plus an
`aggregate.json` containing five-run means and sample standard deviations.
Use `--limit 10 --repetitions 1 --output-dir /tmp/partial-aam-pilot` for a quick
pilot. Existing outputs are protected unless `--force` is supplied.
