# Global chirality and stereoisomer relations

This is the benchmark stage after local canonicalization.

```bash
python Experiment/Stereo/Global/run.py relations
python Experiment/Stereo/Global/run.py acs --case-timeout 5
```

The two tasks must remain separate:

1. `relations` is a nine-case designed conformance matrix covering identical
   stereographs, enantiomers, diastereomers, meso mirror identity,
   constitutionally different structures, and incomplete stereo input.
2. `acs` compares exact whole-stereograph mirror identity with the 258 manual
   ACS chiral/achiral labels. It reports definitive coverage, accuracy among
   definitive cases, accuracy over all records, outcome counts, and timing.

CIP is not included because its labels are local CIP assignments rather than
whole-molecule chirality truth. RotA is not included because it supplies
positive axial loci without configured handedness or achiral controls.

Run the relation conformance task first. Then run ACS with a declared per-case
timeout and retain nondefinitive outcomes in the denominator for the strict
accuracy value.
