# Stereo canonicalization benchmarks

The primary benchmark removes supplied configuration, holds the molecular
constitution fixed, and exhaustively enumerates every raw local reference
ordering of each perceived carrier. The global A/B/C benchmark then evaluates
mixed-family Cartesian products with all carriers configured simultaneously.

## 1. Configuration-free exhaustive local canonicalization

```bash
python Experiment/Stereo/Canonicalization/configuration_free_local.py internal \
  --output /tmp/internal.json --table /tmp/internal.csv
python Experiment/Stereo/Canonicalization/configuration_free_local.py acs \
  --jobs 8 --output /tmp/acs.json --table /tmp/acs.csv
python Experiment/Stereo/Canonicalization/configuration_free_local.py cip \
  --jobs 8 --cip-path /path/to/audited/compounds.smi \
  --output /tmp/cip.json --table /tmp/cip.csv
python Experiment/Stereo/Canonicalization/configuration_free_local.py rota \
  --jobs 8 --output /tmp/rota.json --table /tmp/rota.csv
```

`Experiment/Stereo/benchmark.sh` runs all four datasets. Each report records:

- the configuration-free carrier and its family;
- every raw local permutation and its theoretical formal class;
- exact certificate collapse within each formal class;
- exact separation between formal classes unless a whole-graph automorphism
  identifies them;
- timeout, parse, and canonicalization failures.

Carriers are evaluated independently in this local protocol. The A/B/C
protocol below is the separate joint Cartesian-product experiment.

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

The complete retained result is:

| Source | Rows | Carriers | Raw representations | Theoretical local classes | Observed global classes | Symmetry quotients | Failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Internal ten-family fixtures | 10 | 10 | 940 | 67 | 67 | 0 | 0 |
| ACS | 258 | 1,062 | 23,808 | 2,124 | 1,838 | 286 | 0 |
| CIP | 300 | 1,118 | 24,632 | 2,236 | 1,941 | 295 | 0 |
| RotA | 650 | 1,286 | 17,088 | 2,572 | 2,039 | 533 | 0 |
| **Total** | **1,218** | **3,476** | **66,468** | **6,999** | **5,885** | **1,114** | **0** |

All 3,476 carriers and all 66,468 raw representations complete within the
10-second per-canonicalization budget. A symmetry quotient is an exact
whole-graph identification of opposite local formal classes, not a failure.
The CIP structures remain an audited external input and are not redistributed.

## 2. Selective global-by-local permutations

`Canonicalization/global_local.py` combines exhaustive local-class coverage
with whole-graph atom relabeling. The Cartesian product is deliberately
selective: all atom relabelings are used only for small designed fixtures;
larger fixtures and remaining local classes use fixed-seed samples. This
covers every stereo family, both configurations, symmetric/mirror-fixed
classes, variable path/plane supports, and multi-element graphs without
pretending that factorial enumeration of every public molecule is useful.

The focused smoke gate is:

```bash
python Experiment/Stereo/Canonicalization/atom_relabel.py global-local \
  --family tetrahedral --permutations 2 --class-relabelings 1 \
  --exhaustive-max-atoms 0
```

Use `--exhaustive-atom-relabelings` only on explicitly selected small
fixtures. The report states which tiers were exhaustive and which were
sampled.

The retained selective run covers all 10 families and 67 configuration
classes, rechecks all 940 raw local arrangements, performs 201 per-class
global relabelings and 528 representative relabelings, and has zero failures
or timeouts:

```text
global_local_canonicalization_report.json
```

## 3. Configuration-free mixed-family global A/B/C matrix

The A/B/C protocol factorizes each carrier's raw representations into formal
local classes, then canonicalizes their joint product. Raw A additionally
enumerates the complete raw product and proves that it collapses onto the same
formal tuples and global classes:

| Gate | Cases | Enumerated assignments | Invariant formal tuples | Global classes | Timeouts/failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| Formal A | 10/10 | 48/48 | 48/48 | 38 | 0/0 |
| Raw A | 10/10 | 2,048/2,048 | 48/48 | 38 | 0/0 |
| Formal B | 11/11 | 192/192 | 192/192 | 95 | 0/0 |
| Formal C | 9/9 | 1,920/1,920 | 1,920/1,920 | 752 | 0/0 |

Formal A/B/C therefore cover 30 mixed-family cases and 2,160 joint formal
assignments. The largest C case represents a hypothetical raw product of
782,757,789,696 local encodings.

The complete runner finishes with full raw B (992,256 assignments) and only
raw C/VS146 (2,359,296 assignments). Other raw C cases remain excluded because
they range up to 782.76 billion assignments. Raw B/C require explicit
`--allow-expensive-raw` authorization at the Python CLI.

## 4. Public whole-graph relabeling

Whole-graph atom renumbering is a separate robustness test:

```bash
python Experiment/Stereo/Canonicalization/atom_relabel.py acs \
  --permutations 100

python Experiment/Stereo/Canonicalization/atom_relabel.py rota \
  --permutations 100
```

For public corpora this remains a separate robustness tier. RotA belongs only
to support invariance because its source structures generally do not encode
fixed axial handedness.

## 5. Multi-element composition

`multi_element.py` covers designed two- and three-element graphs, all binary
local assignments, enantiomer/diastereomer/meso relations, descriptor order,
selected atom relabelings, mirror closure, and selected public multi-centre
records. Keep this result separate from the single-element local-permutation
accuracy.
