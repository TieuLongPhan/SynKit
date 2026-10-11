# Stereo benchmarks

Stereo is split by scientific question. Results from different tracks must
not be pooled into one accuracy number.

```text
Experiment/Stereo/
├── Canonicalization/  representation invariance and exact certificates
├── Chirality/         whole-molecule mirror identity and pair relations
├── Perception/        carrier detection and local CIP-label assignment
├── ChemicalValidation/ independently frozen sourced geometry and symmetry
├── Diagnostics/       cross-backend comparisons; not headline accuracy
├── Data/              sources, licenses, frozen reports, and result tables
├── datasets.py        shared integrity-checked dataset loading
├── reproduce.py       bounded/full/explicit raw-stress protocols
└── benchmark.sh       historical long pipeline (includes expensive raw stress)
```

## Data used by the paper

`paper/stereo_graph/results/manifest.json` selects the reports behind every
number in the manuscript. Only these `Data/Revision` directories are kept:

| Directory | Role |
| --- | --- |
| `CleanReleaseFull/` | The full run from a clean checkout; source of 34 of the 37 selected reports and of the cost and registration plots |
| `PaperEvidence/` | Chemical selection manifests, RDKit representation fixtures, environment locks and the archived earlier paper revisions |
| `Protocols/` | Topology-completion report selected by the manifest |
| `CIPPublished/` | Publisher CIP inputs read by `cip_sources.py` |
| `Chemical/`, `Performance/`, `ProtocolReconciled/`, `Historical/` | Reports read by `chemical_results.py` and by `Test/Benchmark/Stereo` |

Intermediate reruns that the final run superseded (`Bounded*`,
`ArchiveReaderBounded`, `CleanReleaseBounded`, `ProtocolIsotopeCorrected`,
`PublisherCoordinates` and the earlier `Revision/Canonicalization` reports)
were removed from the working tree. They remain in Git history at commit
`8faacf2a`.

## Revision reproduction

The current revision entry point defaults to bounded correctness validation:

```bash
python -m Experiment.Stereo.reproduce --tier bounded --jobs 2 \
  --output-dir Experiment/Stereo/Data/Revision/Bounded
```

The full implemented protocol suite requires the checksum-pinned external CIP
inputs. It writes fresh reports without overwriting the retained development
reports:

```bash
python -m Experiment.Stereo.reproduce --tier full --jobs 2 \
  --cip-path /tmp/cip-validation-suite-compounds.smi \
  --coordinate-path /tmp/cip-validation-suite-compounds-3d.sdf \
  --output-dir Experiment/Stereo/Data/Revision/Full
```

This suite covers the independent catalogue, revised local/formal global gates,
three separate ACS procedures, two naming inputs, overlap and maintained
perception protocols, controlled performance, registration, sourced chemical
annotation and evaluation, and the frozen RotA structural annotation and
production comparison.
Stage manifests retain commands, checksums, source hashes, refusals and elapsed
time. A failed required stage exits unsuccessfully; a stage timeout terminates
its worker process group. `--stage-timeout` bounds each stage, not the entire
suite. Results from a truncated suite are explicitly incomplete.

Use `--tier protocols` with the same pinned inputs to regenerate only the
six perception reports and relation matrix. This tier retains renumbering
checks and records each report's full Python source hashes. External input
files are checksummed before execution and must remain unchanged. Write
outputs outside the checkout when a clean working-tree record is needed.

Raw B and raw C/VS146 stress are separately selected with `--tier raw-stress
--allow-expensive-raw`. They are absent from the default and full tiers. The
historical `benchmark.sh` still includes those expensive final stages and is
retained for replay of the old protocol.

Current naming reports use `synkit.cip-native-validation/4`: SMILES-only and
family-conditioned coordinate-assisted results are different protocols. The
SMILES RDKit comparator is matched to the former. The latter includes a
separate RDKit result derived from the same SDF, with exact unanimous atom
transport and its own limitations; its family-projection policy still differs.
All emitted, correct, wrong and missing labels and exact-record denominators
are retained. This is local naming validation, not molecular chirality accuracy.

`manuscript_tables.py` reads a checksummed manifest with explicit report scopes
and expectations, refuses inconsistent required correctness or denominators,
and writes a JSON/CSV numerical ledger and TeX macros. Historical figures are
not automatically promoted to current values.

## Sourced chemical validation

`ChemicalValidation/annotation.py` verifies unchanged MACE XYZ bytes, source
donor-slot conventions, tetrahedral tags and coordinate witnesses. It derives
the proper octahedral poses independently and constructs complete bounded
configuration partitions. The annotation artifact is frozen before production
codes are evaluated. `ChemicalValidation/evaluate.py` independently replays
those partitions, compares complete certificate bytes for the sourced states,
coordinate and production mirrors, and atom renumberings, and tests all 48
sourced structural negatives against replayed constitutional witnesses.

```bash
python -m Experiment.Stereo.ChemicalValidation.annotation \
  --output Experiment/Stereo/Data/Revision/Chemical/independent_annotation.json
python -m Experiment.Stereo.ChemicalValidation.evaluate \
  --annotation Experiment/Stereo/Data/Revision/Chemical/independent_annotation.json \
  --output Experiment/Stereo/Data/Revision/Chemical/production_validation.json
python -m Experiment.Stereo.ChemicalValidation.rota_annotation \
  --output Experiment/Stereo/Data/Revision/Chemical/rota_structural_annotation.json
python -m Experiment.Stereo.ChemicalValidation.rota_evaluate \
  --annotation Experiment/Stereo/Data/Revision/Chemical/rota_structural_annotation.json \
  --output Experiment/Stereo/Data/Revision/Chemical/rota_production_validation.json
```

The 40 coordination inputs cover 20 constitutions. Two exceed the frozen
geometric distortion limit; all 38 certified inputs and all 48 sourced
negatives pass the initial production comparison. These are source-generated
initial configurations and structural symmetry controls, without experimental
stability labels. The separate 60-case RotA annotation covers axis supports,
adjacent tetrahedral pairs and spiro centers. It records aromatic ring-edge
supports separately from ordinary single-bond axes and keeps terminal ligand
equivalence separate from global configured orientation. These are automated
structural checks of a development sample, not independent human adjudication
or a new chemical holdout.

## Benchmark matrix

The quantities below describe retained historical reports. They are being
reproduced under the revised contracts; their invariance and multiplicity gates
alone do not establish soundness of the complete global partition. New local
and global reports use schema revision 2 and require the independent oracle.

| Track | Question | Historical report evidence | Claim boundary |
| --- | --- | --- | --- |
| Configuration-free local canonicalization | Do all raw local orders collapse to the exact formal quotient? | 66,468/66,468 representations and 3,476/3,476 carriers over 1,218 rows | One carrier at a time on a fixed constitution |
| Mixed-family global A/B/C | Do simultaneous carrier configurations compose under whole-graph symmetry? | Formal A/B/C 2,160/2,160; raw A 2,048/2,048; zero failures/timeouts | Full raw B and raw C/VS146 run as final stress stages |
| Global-by-local canonicalization | Does the certificate survive atom relabeling for every local class? | 67 classes, 201 class relabelings, 528 representative relabelings, zero failures/timeouts | Selective Cartesian product |
| Multi-element composition | Do several configured elements compose without collisions? | Six designed two/three-element graphs plus selected public records | Separate from single-element accuracy |
| Molecular chirality | Is the configured molecule identical to its mirror? | ACS 258-case published benchmark plus exact-mirror audit | ACS supplies valid binary truth |
| Stereoisomer relation | Are pairs identical, enantiomeric, diastereomeric, constitutionally different, or incomplete? | Nine designed conformance cases | Designed matrix |
| Stereo perception | Which carriers exist and what local labels can be assigned? | RotA axial loci, CIP labels, multi-family element inventory | Not global chirality |

## Why global-by-local is selective

The local representation group is small and chemically meaningful, so it is
exhausted. Whole-graph atom relabeling grows as \(n!\), so a full product over
every public molecule would mostly repeat the same invariant at enormous
cost. The protocol therefore:

1. covers both configurations and every supported stereo family;
2. exhausts all atom relabelings for explicitly small fixtures;
3. samples deterministic relabelings for larger fixtures and every remaining
   local class;
4. includes symmetric/mirror-fixed, variable path/plane, virtual-reference,
   and multi-element scenarios;
5. records exhaustive and sampled tiers separately.

These tests address representation invariance. The independent orbit protocol
below checks separation and the exact quotient under constitutional symmetry.

## Independent assignment partitions

`Canonicalization/independent_orbits.py` derives local rotations from geometry,
enumerates attributed constitutional automorphisms preserving a frozen carrier
inventory, and constructs global assignment orbits directly. It compares whole
partitions with production certificates and checks mirrors and atom renumbering.
Every merged assignment retains an explicit atom map. The reference model does
not call production canonical forms, shape tables, stereo equality, encoders or
transport methods. Constructors and raw input fields are shared as adapters.

The catalogue contains 33 constructed fixtures with 279 assignments and the 30
public formal A/B/C cases with 2,160 assignments. It includes all eleven configured
families, coupled frameworks, carrier exchange, virtual ports, and path/plane
symmetry. These are prospective mathematical checks conditional on the declared
inventory; they do not validate carrier perception or create a chemical holdout.

Run the bounded pilot with five cases and all seven mutation controls:

```bash
python Experiment/Stereo/Canonicalization/independent_orbits.py \
  --case tetrahedral:distinct \
  --case tetrahedral:two:meso \
  --case planar_bond:distinct \
  --case framework:coupled:opposite \
  --case mixed:tetrahedral+octahedral:distinct \
  --output Experiment/Stereo/Data/Canonicalization/Independent/pilot.json
```

Run the full declared catalogue:

```bash
python Experiment/Stereo/Canonicalization/independent_orbits.py \
  --include-public-abc --cip-path /path/to/audited/compounds.smi --jobs 2 \
  --output Experiment/Stereo/Data/Canonicalization/Independent/full_report.json
```

Reference caps are 1,024 assignments, 50,000 constitutional automorphisms and
10 seconds per reference case. Each production certificate has a separate
5-second default limit. An incomplete oracle never passes. Production peak RSS
is checked against 1 GiB between computations; a native call can temporarily
overshoot. Full certificates are interned in a temporary SQLite store using
full-byte equality, including when indexing hashes collide. This avoids retaining
several copies of large certificates in memory. Reported hashes are indices.
The command fails if a partition, mirror, renumbering or mutation gate fails,
or if its inspected sources change during execution.

## Additional stereo work worth keeping

The next independent benchmark tracks should be:

1. **Reaction stereo transport:** retention, inversion, creation, deletion,
   mapped hydrogen/lone-pair references, and fail-closed non-invertible
   `UNSPECIFIED` effects under forward and reverse rule replay.
2. **Interchange round trips:** RDKit/GML/JSON import-export identity for every
   descriptor family, including unsupported/fail-closed cases.
3. **Experimental axis stability:** the maintained 24-case synthetic negative
   suite now covers constitutional specificity. Barrier, isolation-timescale,
   stability, and handedness truth still require an external annotated source
   because RotA is positive-only.
4. **Scalability evidence:** runtime and peak memory versus atom count,
   automorphism count, stereo-element count, and local-class product size,
   with timeout coverage reported explicitly.

Keep these as separate reports. Canonicalization correctness, chirality
accuracy, perception recall, reaction replay, and performance answer different
questions.

Regression tests remain under `Test`; executable research protocols live here.
Third-party sources and their notices remain colocated under `Data`.

Run every maintained report with the single benchmark script. It finishes with
full raw B and the selected raw C case VS146. Those final stress stages can
take several days:

```bash
CIP_FILE=/path/to/audited/compounds.smi \
CIP_3D_FILE=/path/to/audited/compounds_3d.sdf \
  bash Experiment/Stereo/benchmark.sh
```

If either CIP path is omitted or does not match its pinned checksum, the script
downloads the corresponding validation file to `/tmp`. The structures remain
external and are not redistributed by SynKit. Run the benchmark from an
environment containing SynKit's dependencies. Case-parallel tasks use 16
workers by default; override that with `SYNKIT_BENCHMARK_JOBS`.
