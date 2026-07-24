# Molecular chirality

This track keeps three whole-molecule questions together:

- `published.py`: the ACS 258-record supplied-stereo binary benchmark;
- `exact_mirror.py`: exact configured-stereograph mirror audit;
- `relations.py`: designed pairwise stereoisomer relations.

Run:

```bash
python Experiment/Stereo/Chirality/run.py relations
python Experiment/Stereo/Chirality/run.py acs --case-timeout 5
```

CIP and RotA are excluded from global chirality accuracy because their source
labels answer local CIP-assignment and positive axial-locus questions.
