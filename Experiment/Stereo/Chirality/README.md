# Molecular chirality

This track keeps three whole-molecule questions together:

- `published.py`: the ACS 258-record supplied-stereo binary benchmark;
- `exact_mirror.py`: exact configured-stereograph mirror audit;
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

Run:

```bash
python Experiment/Stereo/Chirality/run.py relations
python Experiment/Stereo/Chirality/run.py acs --case-timeout 5
```

CIP and RotA are excluded from global chirality accuracy because their source
labels answer local CIP-assignment and positive axial-locus questions.
