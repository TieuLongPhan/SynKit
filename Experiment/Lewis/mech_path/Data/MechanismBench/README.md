# MechanismBench

`Experiment/Lewis/mech_path/Data/MechanismBench/` is the canonical on-branch location for the three reviewed
MechanismBench partitions. The former large `polar.csv` and `radical.csv`
source pools are not vendored; all release cases and their review provenance
are embedded in the JSON manifests below.

The public, executable MechanismBench layout has exactly three partitions:

- `radical.json` — 80 chemistry-reviewed, macro-balanced radical mechanisms.
  Every case embeds its correction history, strict forward/reverse replay, and
  unmapped rule-reapplication evidence.
- `polar.json` — 80 reviewed two-electron mechanisms selected deterministically
  from SynEPD's 1,915-record `polar.json` source pool. The manifest records the
  source SHA-256, CC BY 4.0 provenance, eight top-level POLAR strata, and strict
  replay/reapplication evidence for every selected record. The full source pool
  is intentionally not copied here.
- `stereo.json` — 80 positive transformations: the original 40 reaction-SMILES
  cases, four reviewed non-tetrahedral/atrop rewrite fixtures, seven Phase 2R
  electron-flow × stereo fixtures, 21 reviewed native-descriptor rewrites, and
  eight reviewed final promotions. Its eight negative assertions are retained
  as corruption/specification fixtures and do not count as positive cases.

The compact nine-case rewrite conformance suite is development-only data at
`Test/Synthesis/Reactor/fixtures/small_rewrite_conformance.json`. It deliberately
does not define a MechanismBench partition.

The current stereo baseline is 80 independently reviewed positives. Negative
fixtures remain outside that count.

Raw candidate pools, audit summaries, and release-owner checklists do not belong
in this directory. They should be regenerated from source data or kept as project
documentation rather than presented as benchmark cases.

Generate the current replay/corruption/runtime evidence with:

```bash
python Experiment/Lewis/mech_path/evidence.py \
  --output Experiment/Lewis/mech_path/Data/MechanismBench/evidence/mechanismbench_evidence.json
```

The report currently covers typed `MechanismRecord` fixtures. It records the
reaction-SMILES, non-tetrahedral, and promoted native-descriptor rewrite
fixtures that remain outside that shared replay representation instead of
assigning them fabricated metrics.

The manuscript's arrow-mechanism benchmark remains 160 cases: 80 polar plus 80
radical. The evidence report's `typed_replay_cases` total is 167 because the
same executable path also covers seven typed electron-flow/stereo extension
cases. These counts are reported separately as
`paper_arrow_mechanism_cases` and `typed_stereo_extension_cases`.
