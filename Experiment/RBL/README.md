# RBL paired reconstruction benchmark

The runtime API is available as
`from synkit.Synthesis.RBL import RBLEngine`. Its serializable
`engine.result` mapping carries the versioned schema identifier
`synkit.rbl-result/2`.

## Dataset construction

`build_dataset.py` joins `USPTO_50K.csv` to `../../Data/smart.json.gz` by the
zero-based source row stored as `R-id`. It writes `uspto_50k_rbl.json.gz` with:

- `R_id`: `R_<CSV row index>`
- `raw`: canonical incomplete reaction, without AAM or stereo
- `complete`: canonical completed reaction, without AAM or stereo
- `aam`: mapped completed reaction used to extract the rule

Both `raw` and `complete` are produced by `synkit.Standardize` with
`remove_aam=True`, `ignore_stereo=True`, and `remove_invalid=False`. The full
mapped ITS is hydrogen-completed exhaustively; only one equivariant RC class is
accepted, and that completed stereo-free mapped serialization is stored in
`aam`. The builder also requires element- and net-formal-charge-balanced ground
truth. Of 34,393 mapped sources, two records (`R_45326` and `R_45874`) fail the
current RDKit parser and 13 have charge-invalid ground truth, leaving 34,378
validated paired records. No source is hydrogen-ambiguous.
`validate_dataset.py` performs an independent artifact audit.

## Inputs and provenance

`Data/smart.json.gz` is tracked, but the source `USPTO_50K.csv` is not
redistributed by this repository. Place the source beside this README before
building. The audited local inputs and deterministic output are:

- `USPTO_50K.csv`: 50,017 physical lines, SHA-256
  `1d69b90a299bd255b00342ccd15c3bc11d6c3047a3fecefaec539c3d04168dc9`;
- `Data/smart.json.gz`: SHA-256
  `45d52cb916f193661c24038ca7b579b5b99d8d34ada31f510b044e59710b6638`;
- `uspto_50k_rbl.json.gz`: SHA-256
  `31b4fabd32549b4a8dfd222d56ea29bef215bef64951884773da05b53cdcd05e`.

The builder records input/output hashes and byte sizes in its report, allowing
each generated artifact to be checked against the audited inputs.

## Benchmark protocol

Run the default benchmark with:

```bash
python Experiment/RBL/benchmark.py
```

The benchmark output carries explicit schema identifiers for the top-level
report (`synkit.rbl-benchmark/1`), each record
(`synkit.rbl-benchmark-record/1`), and each method evaluation
(`synkit.rbl-evaluation/1`). Consumers should reject unsupported major schema
versions instead of inferring a layout from file names.

The harness parses the full mapped tuple ITS, applies `HComplete`, and keeps a
rule only when exhaustive hydrogen-transfer enumeration yields one equivariant
RC class. It then adapts that unique RC through
`SynRule(implicit_h=False, format="tuple")` and applies it to `raw`. Explicit
mapped H vertices carry hydrogen transfers; an unchanged `hcount` pair such as
`(1, 1)` remains local matching context. Relative electron-resource pairs are
still normalized (for example lone-pair `(3, 2)` becomes `(1, 0)`). A solve is
counted only when a candidate is exactly equal to `complete` after the same
stereo-free, AAM-free standardization. Each method/record runs in a child
process, so `--timeout` is a hard wall-time bound even inside RDKit or SciPy.
The report also binds the dataset SHA-256, every selected row's semantic input
digest, Git commit and dirty state, Python/SynKit/RDKit versions, platform,
thread environment, and exact method configuration. Per-method records retain
the engine's search status, completeness, incomplete reasons, policy, strict
acceptance boundary, and stage timings.

Search profiles have distinct scopes. `fast_track` uses quick replay and
resolved non-wildcard outputs without MCS or fusion. `fast_fusion` adds the
bounded component-MCS fallback. `early_stop`, `full`, and `verified` retain
their wider scopes. `verified` enumerates all admitted typed partial overlaps,
uses exact typed port matching and categorical pushouts, applies strict
component/conservation acceptance, and emits replayable
`synkit.rbl-proof/2` certificates. Any typed-overlap state/result/time limit is
reported as `INCOMPLETE`; only a limit-free exhausted universe can report
`PROVED_NONE`.

The historical command name `scan_verified_mcs.py` is retained, but it now
runs the all-typed verified policy and records that authoritative scope in its
artifact. A resumable smoke or full scan can be run with:

```bash
python Experiment/RBL/scan_verified_mcs.py --workers 16 --timeout 30
```

## Fast-track baseline

Run fusion-free fast track over every paired record with:

```bash
python Experiment/RBL/scan_fast_track.py --workers 16
```

`fast_track_all.json.gz` contains every record not exactly reconstructed.
`fast_track_all_summary.json` contains aggregate counts and 100 evenly spaced
examples from that unsolved set. The retained 29,604/34,378 (86.113%) result
is a historical `SynRule(implicit_h=True)` compatibility baseline: 2,272
records produced no candidate and 2,502 produced one balanced non-exact
candidate. This is a historical compatibility baseline, not the current
explicit-H result. A new run uses `SynRule(implicit_h=False)` and records that
adapter in both the row audit and summary. Aggregate counts are valid only
after the full scan completes.

## Complete-input retry

Retry the 4,774 raw-input failures with the canonical `complete` reaction as
the RBL input using:

```bash
python Experiment/RBL/retry_complete.py --workers 16
```

This remains fusion-free. The mapped AAM is hydrogen-completed before RC
extraction, and mapped candidate SMILES are validated with explicit H atoms
preserved. All 4,774 records have exactly one exhaustive HComplete RC class;
none is rejected as ambiguous. Exact success still requires the generated
reaction to equal `complete`; merely producing a candidate does not count. The
retry solves 4,749/4,774 (99.476%), with 25 no-candidate records and no
non-exact candidates.

The explicit-H rebuild is stored in `complete_retry_explicit_h.json.gz`, with
aggregate results in `complete_retry_explicit_h_summary.json`. It solves all
4,774 complete-input records exactly by fast track, invokes no MCS/fusion, and
reports zero rule-normalization violations. The previous 25-record residual is
classified in `complete_retry_residual_audit.json`: 12 Mg/Zn charge-transition
records are excluded from the non-charge inspection by policy, while all 13
remaining records now replay exactly. The two records `R_33854` and `R_44480`
represent duplicate chemistry.

## Artifact audits

`complete_retry_hcomplete.json.gz` includes an element and formal-charge audit
of the 2,502 prior non-exact candidates. Element balance counts implicit
hydrogen by adding RDKit hydrogens before counting atom symbols. Charge balance
means that the sum of formal charges is identical on both sides; it is
reported separately from both sides being neutral. All 2,502 `complete`
references and all 2,502 prior generated candidates are element- and
charge-balanced.

Run the stricter complete-input classification in an environment containing
the project dependencies:

```bash
python Experiment/RBL/classify_complete_retry.py
```

This accepts an exact result only when it both equals canonical `complete` and
passes element plus net-formal-charge conservation. Of the 4,774 safe
raw-input failures, 4,749 are exact and balanced on complete-input replay, 25
have no candidate, and there are no non-exact or unbalanced candidates.
