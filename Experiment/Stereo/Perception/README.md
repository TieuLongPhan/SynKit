# Stereo perception

This track measures local evidence rather than whole-molecule chirality:

- `Data/Perception/perception_conformance_cases.json`: designed data contract
  covering four information states across ten families;
- `conformance.py`: task-aware runner that scores connectivity perception,
  RDKit shape/carrier adapters, and formal sidecar contracts after removing
  configuration evidence;
- `family_executors.py`: executable contracts for non-tetrahedral configured
  adapters and formal helical/planar sidecars;
- `stereo_elements.py`: configuration-neutral carrier inventory;
- `cip_labels.py`: independent local CIP-label assignment;
- `axis_loci.py`: typed RotA axial-locus detection.
- `full_detection.py`: exhaustive carrier detection over all 1,208 source
  rows under both configuration-erased and
  oriented-neighbor-frame-retained settings.

Reports must state whether the source provides configuration, handedness,
stability, negatives, and whole-molecule truth. RotA is positive-only; CIP
labels are local.

Run the designed check with:

```bash
Experiment/Stereo/run_experiments.sh perception-conformance
```

Run the exhaustive empirical benchmark with an integrity-matched checkout of
the non-vendored CIP Validation Suite:

```bash
Experiment/Stereo/run_experiments.sh perception-full \
  --cip-path /path/to/cip-validation-suite-compounds.smi
```

The full report keeps the three tasks separate. ACS scores recovery of
configured loci present in its source SMILES because its manual truth is
whole-molecule only. RotA uses family-typed projections for bond axes,
phosphacumulenes, adjacent chiral-atom pairs, spiro centers, and spiral
chains; it recovers all 698 positive annotations and all 15 expanded cumulene
paths. CIP scores all 1,252 recommended local positions, including breakdowns
by carrier class, descriptor case, and stereo-unit category. Setting 1 keeps
configuration only as oriented neighbor frames at other local elements.
Setting 2 erases every configuration before prediction. Neither setting uses
CIP or canonical local descriptors, and the focal center's own frame never
participates in its detection. Reverse atom renumbering checks both settings,
carrier detection, and RotA projections.
