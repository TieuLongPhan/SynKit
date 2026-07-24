# Stereo experiments

This top-level package contains executable stereo benchmark protocols together
with their source datasets and retained evidence. Regression tests remain
under `Test`.

```text
Experiment/Stereo/
├── Data/             datasets, metadata, reports, and result tables
├── acs_molecular_chirality.py
│                     ACS whole-molecule benchmark
├── datasets.py       audited dataset loading and integrity checks
├── *_comparison.py   cross-backend diagnostic experiments
├── cip_native.py     native CIP assignment experiment
├── rota_loci.py      RotA locus experiment
├── stereo_elements.py carrier/extraction experiment
├── Canonicalization/  fixed-graph configured-stereo experiments
└── Global/            whole-molecule mirror and relation experiments
```

The collocated data registry is:

```text
Experiment/Stereo/Data/
├── ACS-StereoMolGraph/  audited source data and metadata
├── CIPValidationSuite/  external-source metadata
├── ChiralFinder-RotA/   audited source data and metadata
├── Canon/               canonicalization reports and tables
└── Global/              global experiment reports
```

The `Data` directory has no Python modules, so retained reports do not become
importable package content. The central manifest and each licensed source stay
beside the runners that consume them.

The next canonicalization protocol is intentionally divided into four result
tiers:

1. single-element exhaustive frame enumeration;
2. homogeneous multi-element focal enumeration;
3. mixed-family focal enumeration;
4. bounded joint Cartesian enumeration.

ACS and CIP establish carrier/support provenance. Their supplied orientation
is not used as the enumerated configuration truth in these four tiers.
