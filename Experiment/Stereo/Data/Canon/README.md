# Stereo canonicalization benchmark

The primary benchmark holds the molecular graph fixed and exhaustively
permutes only the local references of configured stereo elements.

## Primary local-permutation runner

```bash
python Experiment/Stereo/Canonicalization/run.py internal
python Experiment/Stereo/Canonicalization/run.py acs --jobs 4
python Experiment/Stereo/Canonicalization/run.py cip \
  --cip-path /path/to/audited/compounds.smi --jobs 4
```

Every run regenerates the stereo-only input inventory, writes a detailed JSON
report, and writes a compact CSV table. The table has one row per tested
configuration. Extracted ACS/CIP descriptors contribute their configured
class; source-recovered CIP supports contribute both formal classes.
It records:

- stereo family and configuration index;
- configured elements in the containing graph;
- expected and checked local permutations;
- passing and failing permutations;
- number of distinct canonical configured stereographs;
- canonical digest, accuracy status, and wall time.

Public records are independent. Use `--jobs` to process a bounded number in
parallel; each exact canonicalization retains the declared per-call timeout.
The default remains one worker for reproducibility on memory-limited hosts.

For molecules with multiple configured elements, one element is permuted at a
time while all other elements and the molecular graph remain fixed. A later
joint Cartesian-product experiment can be reported separately.

The internal task exhausts 940 raw local arrangements:

| Family | Raw arrangements | Configurations | Arrangements per configuration |
| --- | ---: | ---: | ---: |
| Tetrahedral | 24 | 2 | 12 |
| Square planar | 24 | 3 | 8 |
| Trigonal bipyramidal | 120 | 20 | 6 |
| Octahedral | 720 | 30 | 24 |
| E/Z planar bond | 8 | 2 | 4 |
| Atropisomeric bond | 8 | 2 | 4 |
| Cumulene axis | 8 | 2 | 4 |
| Extended cis/trans cumulene | 8 | 2 | 4 |
| Helical | 4 | 2 | 2 |
| Planar chirality | 16 | 2 | 8 |
| **Total** | **940** | **67** | — |

The current exhaustive result is 940/940 passing representations, producing
exactly 67 distinct canonical configured stereographs with no collision.

ACS contributes 228 configured structures. CIP contributes 298 records with a
source stereo-unit tag, but structure-dependent execution requires the exact
audited external `compounds.smi`; it is not redistributed here.

For CIP records tagged `CT4`, RDKit supplies the molecular topology but not
the directional extended-cumulene configuration. The runner therefore
perceives every odd-bond cumulene support, enumerates its two
`ExtendedCisTransStereo` classes, checks all four equivalent terminal-frame
representations of each class, and requires the two class certificates to be
distinct. This is a canonicalization test of both possible configurations;
it does not claim recovery of the source E/Z orientation.

The same source-guided procedure covers otherwise unextractable `TH`, `CT`,
`AT`, `TH3`, `TH5`, and `HE` supports. It uses the source tag only to select
the stereo family and carrier, enumerates both formal configurations, and
keeps all other recovered supports fixed. Opposite formal configurations may
legitimately share one certificate when whole-graph symmetry makes the
marked support non-stereogenic; those reductions are reported as
configuration-class collapses, not canonicalization failures.

For the two `HE` records, the recommended labels report positions 19 and 26.
Their unique seven-atom shortest path defines one coupled helical carrier.
The runner enumerates both handedness classes without using the reported
`P`/`M` direction. Recovery fails closed if the endpoints are absent,
out-of-range, differently labelled, or connected by multiple shortest paths.

The summary keeps extraction and testability separate:
`records_with_rdkit_extracted_configured_stereo` and
`rdkit_direct_extraction_coverage` count only descriptors recovered directly
from RDKit. `records_with_source_enumerated_stereo` counts source-guided
supports, while `records_with_testable_stereo_support` and
`stereo_support_coverage` report their testable union.

## Secondary whole-graph relabeling

Whole-graph atom renumbering is a separate robustness test:

```bash
python Experiment/Stereo/Canonicalization/atom_relabel.py acs \
  --permutations 100

python Experiment/Stereo/Canonicalization/atom_relabel.py rota \
  --permutations 100
```

Use `--exhaustive-atom-relabelings` only for explicitly selected small
molecules. It is not the local stereo-permutation benchmark. RotA belongs only
to this support-invariance tier because its source structures generally do
not encode fixed axial handedness.
