# Mechanism-path experiment

This directory owns the mechanism-path benchmark assets.

- `evidence.py` regenerates replay, corruption, and timing evidence for the
  reviewed MechanismBench partitions.
- `audit.py` reconstructs the complete PMechDB and RMechDB corpora, repairs
  repeated endpoint-map labels before guarded expansion, applies the reviewed
  radical arrow corrections, and writes an aggregate summary plus unresolved
  row IDs.
- `Data/MechanismBench` contains the reviewed polar, radical, and stereo
  manifests plus retained evidence.
- `Data/reconstruction_audit` contains the rerun's empty polar failure list,
  the one unresolved radical ID, and an identifier-only radical arrow-review
  table. It intentionally contains no source reactions, conditions, or other
  source metadata.

The manuscript's corpus-audit panel is
`paper/lwg/fig/mechanism_pathway.tex`. Its displayed 95,888/95,888 polar and
5,425/5,426 radical counts come from `Data/reconstruction_audit`, so it belongs
to this experiment even though the TikZ presentation remains with the paper.

Regression tests remain under `Test`.
