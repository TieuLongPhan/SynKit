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
- `cip_input_contract.py`: SMILES identifiability and optional pinned-3D
  orientation audit;
- `axis_loci.py`: typed RotA axial-locus detection;
- `rota_synthetic_negatives.py`: exact-symmetry constitutional negative
  controls;
- `full_detection.py`: exhaustive carrier detection over all 1,208 source
  rows under both configuration-erased and
  oriented-neighbor-frame-retained settings.

Reports must state whether the source provides configuration, handedness,
stability, negatives, and whole-molecule truth. RotA is positive-only; CIP
labels are local.

Run the designed check directly with:

```bash
python Experiment/Stereo/Perception/conformance.py
```

Run the exhaustive empirical benchmark with an integrity-matched checkout of
the non-vendored CIP Validation Suite:

```bash
python Experiment/Stereo/Perception/full_detection.py \
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

The synthetic RotA report adds 24 negative controls: 13 axis-like cases are
rejected with explicit axis-fixed automorphisms, and 11 contain no supported
axis topology. All pass reverse atom renumbering. This establishes
constitutional specificity but does not supply experimental barrier,
stability, or handedness truth.

The CIP input-contract audit proves that the supplied SMILES alone cannot
identify all 300 configured records: VS010 and VS011 are byte-identical but
carry opposite helical labels, so the deterministic ceiling is 299. The
external pinned ``compounds_3d.sdf`` distinguishes the pair with an oriented
coordinate witness and therefore makes the input identifiable. The validated
coordinate path now configures 2/2 helical and 5/5 CT4 records, and
unambiguous constitutional transport configures 4/7 AT records. Fully
material-framed coordinate evidence also completes the VS144 cumulene.
Completing the remaining labels still requires the unresolved AT mappings,
multi-unit Rule 4b, the remaining hierarchical digraph cases, and the other
TH3/TH5 projections.
