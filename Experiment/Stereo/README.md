# Stereo benchmarks

Stereo is split by scientific question. Results from different tracks must
not be pooled into one accuracy number.

```text
Experiment/Stereo/
├── Canonicalization/  representation invariance and exact certificates
├── Chirality/         whole-molecule mirror identity and pair relations
├── Perception/        carrier detection and local CIP-label assignment
├── Diagnostics/       cross-backend comparisons; not headline accuracy
├── Data/              sources, licenses, frozen reports, and result tables
├── datasets.py        shared integrity-checked dataset loading
└── benchmark.sh       single complete benchmark entry point
```

## Benchmark matrix

| Track | Question | Current evidence | Claim boundary |
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

This is stronger and more interpretable than calling a small random corpus
“global exhaustive.”

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
