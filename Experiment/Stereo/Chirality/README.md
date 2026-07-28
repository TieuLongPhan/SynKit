# Molecular chirality

This track keeps three whole-molecule questions together:

- `published.py`: the ACS 258-record supplied-stereo binary benchmark;
- `exact_mirror.py`: exact configured-stereograph mirror audit;
- `global_certificate_benchmark.py`: separately named exact-plus-global
  necessary-chirality protocol;
- `relations.py`: designed pairwise stereoisomer relations.

The ACS exact audit preserves every explicit source `@`/`@@` atom
configuration and compares that supplied configured stereograph with its
mirror under chemical identity. Alternative resonance/Kekule drawings are
enumerated exactly, while genuine bond-order differences remain distinct.
Additional perceived but undeclared loci remain unconstrained; they are not
promoted to configurations and do not block the supplied-input comparison.
Strict completeness checking remains available through the library
classifier's default mode.

The VS215/216 resonance regressions and the VS170/VS300 task-boundary
differences are collected in
`../Data/Chirality/exact_acs_mirror_inspection.tsv`.
Their identity/support causes and the VS300 mirror witness are reviewed in
`../Data/Chirality/exact_acs_mirror_disagreement_review.md`.

`vs300_diagnostic.py` makes the remaining VS300 boundary executable. It
records both input contracts, audits the exact original-to-mirror atom
mapping, and evaluates a nine-atom topology-completion positive, its
configured enantiomer pair, and an achiral near-miss. The diagnostic is an
experiment and regression oracle; it does not change production
classification.

The production boundary is exposed by
`synkit.Chem.Molecule.global_stereo.analyze_global_stereo_support`. It returns
an orientation-unspecified `GlobalStereoCertificate`: topology may prove
necessary chirality but cannot select a configured enantiomer. A fixed
`FrameworkStereo` requires explicit authorized orientation provenance.

Run:

```bash
python Experiment/Stereo/Chirality/run.py relations
python Experiment/Stereo/Chirality/run.py acs --case-timeout 5
python Experiment/Stereo/Chirality/vs300_diagnostic.py \
  --output /tmp/vs300-diagnostic.json \
  --fixtures /tmp/vs300-fixtures.json
python Experiment/Stereo/Chirality/global_certificate_benchmark.py \
  --output /tmp/exact-plus-global-acs.json
```

CIP and RotA are excluded from global chirality accuracy because their source
labels answer local CIP-assignment and positive axial-locus questions.
