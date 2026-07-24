# Mechanism-path experiment

This directory owns the mechanism-path benchmark assets.

- `evidence.py` regenerates replay, corruption, and timing evidence for the
  reviewed MechanismBench partitions.
- `audit.py` reconstructs the complete PMechDB and RMechDB corpora and writes
  an aggregate summary plus failure row IDs.
- `Data/MechanismBench` contains the reviewed polar, radical, and stereo
  manifests plus retained evidence.
- `Data/reconstruction_audit` contains the 3,274 polar duplicate-map failures
  and 10 radical invalid-flow failure IDs from the 101,314-case construction
  run. It intentionally contains no source reactions or metadata.

The manuscript's corpus-audit panel is
`paper/lwg/fig/mechanism_pathway.tex`. Its displayed 92,614/95,888 polar and
5,416/5,426 radical counts come from `Data/reconstruction_audit`, so it belongs
to this experiment even though the TikZ presentation remains with the paper.

Regression tests remain under `Test`.
