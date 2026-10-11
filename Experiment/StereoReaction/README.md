# Reaction-stereo experiments

This directory is the executable experiment root for the reaction layer of
SynKit stereochemistry. It is intentionally separate from
`Experiment/Stereo`, which covers molecular perception, canonicalization, and
whole-molecule chirality.

## Contents

- `workflows.py` exercises exact and generic stereo-rule extraction, branching
  and correlated outcomes, proof-bearing composition and interchange,
  stepwise mechanism replay, and a substituted four-electron path with exact
  ring-closure reversal.
- `../Lewis/mech_path/Data/MechanismBench/stereo.json` is the reviewed
  reaction-stereo baseline.
- `../Lewis/mech_path/Data/MechanismBench/stereo_couplings.json` contains the
  reviewed `SYN`/`ANTI` coupling cases.
- `../Lewis/mech_path/Data/MechanismBench/electrocyclic_machinery.json`
  contains the conrotatory/disrotatory machinery cases.
- `validation.py` runs the bounded reaction-stereo
  serialization and graph-sidecar workload.
- `paper_audit.py` records the current focused test selection, original fixture
  inventories, migration comparisons, source hashes and raw outputs.
- `witness_basis_audit.py` enumerates the six native finite frame models and
  checks the normalizer criterion and basis-aware production replay.
- `evidence/` retains the historical reaction-stereo closeout and bounded
  validation records used by the audits. Their contents and checksums are
  preserved from the original runs.

The MechanismBench files retain their canonical paths because they are frozen,
checksum-bound evidence shared with the mechanism experiments. They are linked
here rather than copied or reclassified.

Run the public workflow from the repository root:

```bash
python Experiment/StereoReaction/workflows.py
python Experiment/StereoReaction/validation.py
python Experiment/StereoReaction/paper_audit.py
python Experiment/StereoReaction/witness_basis_audit.py
python -m pytest -q Test/Graph/Stereo Test/Graph/ITS
python -m pytest -q Test/Rule Test/Synthesis/Reactor Test/Mechanism
```

The canonical manuscript, supporting information, figures and retained audit
reports are in `paper/stereo_rxn/`. Rebuild the local review package with
`python paper/stereo_rxn/tools/build_review_package.py`. The package and its
receipts distinguish current focused tests, historical full-suite results,
finite algebra and chemical development records. No submission is performed.

The electrocyclic verifier checks a supplied annotation. It verifies the
thermal/photochemical 4π/6π conrotatory/disrotatory matrix, correlated terminal
motion, ring direction, mapped support, reversal, and tamper boundaries. It
does not attach strict molecular stereo descriptors to the termini: replay
derives relative inward-neighbor/substituent frames, canonicalizes their
order, and stores canonical signed before-to-after neighbor changes plus the
frame parity needed to recover physical rotation. It does not predict
activation conditions, torquoselectivity, or a preferred pathway.
