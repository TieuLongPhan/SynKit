# Reaction-stereo experiments

This directory is the executable experiment root for the reaction layer of
SynKit stereochemistry. It is intentionally separate from
`Experiment/Stereo`, which covers molecular perception, canonicalization, and
whole-molecule chirality.

## Contents

- `workflows.py` exercises exact and generic stereo-rule extraction, branching
  and correlated outcomes, proof-bearing composition and interchange,
  stepwise mechanism replay, and a substituted Figure 11 4π path with exact
  ring-closure reversal.
- `../Lewis/mech_path/Data/MechanismBench/stereo.json` is the reviewed
  reaction-stereo baseline.
- `../Lewis/mech_path/Data/MechanismBench/stereo_couplings.json` contains the
  reviewed `SYN`/`ANTI` coupling cases.
- `../Lewis/mech_path/Data/MechanismBench/electrocyclic_machinery.json`
  contains the Figure 11 conrotatory/disrotatory machinery cases.
- `validation.py` runs the bounded reaction-stereo
  serialization and graph-sidecar workload.

The MechanismBench files retain their canonical paths because they are frozen,
checksum-bound evidence shared with the mechanism experiments. They are linked
here rather than copied or reclassified.

Run the public workflow from the repository root:

```bash
python Experiment/StereoReaction/workflows.py
python Experiment/StereoReaction/validation.py
python -m pytest -q Test/Graph/Stereo Test/Graph/ITS
python -m pytest -q Test/Rule Test/Synthesis/Reactor Test/Mechanism
```

The electrocyclic verifier checks a supplied annotation. It verifies the
thermal/photochemical 4π/6π conrotatory/disrotatory matrix, correlated terminal
motion, ring direction, mapped support, reversal, and tamper boundaries. It
does not attach strict molecular stereo descriptors to the termini: replay
derives relative inward-neighbor/substituent frames, canonicalizes their
order, and stores canonical signed before-to-after neighbor changes plus the
frame parity needed to recover physical rotation. It does not predict
activation conditions, torquoselectivity, or a preferred pathway.
