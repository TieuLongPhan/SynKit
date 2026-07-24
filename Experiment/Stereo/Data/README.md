# Stereo benchmark registry

This directory keeps external molecule-stereo datasets together while
preserving their different scientific tasks and license boundaries. Datasets
must not be pooled into one accuracy number merely because they contain
stereochemistry.

Generated and retained evidence is grouped by task:

```text
Canonicalization/  local, global-by-local, and multi-element reports
Chirality/         exact mirror and stereoisomer-relation reports
Perception/        CIP label, carrier, and axial-locus reports
Diagnostics/       historical or task-mismatched backend comparisons
```

## Registered datasets

| Dataset | Records | Actual task | Vendored | License |
| --- | ---: | --- | --- | --- |
| ACS StereoMolGraph validation enantiomer | 258 | Global whole-molecule chiral/achiral classification | Yes | CC BY-NC 4.0 |
| ChiralFinder RotA | 650 | Positive axial-locus detection over 2D structures / 3D conformers | Yes | MIT |
| CIP Validation Suite | 300 | Local CIP descriptor assignment across multiple stereo-unit types | No | No redistribution license found |

Only the ACS dataset supports both whole-molecule protocols as accuracy
benchmarks directly:

1. classify the supplied stereochemical input against its manual binary label;
2. remove atom/bond stereo, enumerate supported completions, and report the
   four-state configuration-aware outcome.

The supplied-stereo report compares SynKit with the immutable published
StereoMolGraph, RDKit-SMILES, InChI, and chython columns. These are the
publisher's recorded backend results, not claims about later live releases.
The frozen result is 258/258 for SynKit and published StereoMolGraph and
235/258 for published RDKit-SMILES.

The exact Version 2 configured-stereograph engine is evaluated separately.
The primary canonicalization benchmark holds each graph fixed and
canonicalizes all 940 local-reference permutations across ten stereo
families and 67 configurations. All 940 representations pass, producing
exactly 67 distinct canonical configured stereographs with no cross-
configuration collision. These are synthetic proof fixtures, not public
chemical data. Whole-graph atom relabelling remains a separate secondary
robustness test.

The global benchmark is also separate. Its nine-case pairwise relation matrix
is 9/9. Across all 258 ACS records with a 2-second case budget, the Version 2
mirror path is definitive on 169 and correct on 151 of those, with 8
incomplete and 81 timeout outcomes. This audit must not be conflated with the
ACS-specialized topology-completion classifier's 258/258 result.

Multi-element composition is frozen separately in
`Canonicalization/multi_element_canonicalization_report.json`. Six designed two- and
three-element graphs exhaust 32 binary local assignments, which quotient to 31
global classes: 13 enantiomer pairs, 64 diastereomer pairs, and five
mirror-fixed classes. All class mirrors close, and six representative
descriptor-order, relabelling, and double-mirror checks pass without timeout.
The ACS inventory contains 164/258 records with at least two supplied
configured elements (maximum 25). Seven selected public records containing 18
elements in total pass 14 deterministic atom-renumbering checks and 7/7 exact
mirror comparisons against the ACS global labels.

The same report extracts task-scoped multi-locus manifests without inventing
new truth labels. RotA has 40 records carrying 88 annotated loci; all 40 retain
the existing renumbering-invariance result, while only 15 recover every
annotation and 10 have an exact predicted locus set. CIP has 153 records with
multiple reference \(R/S\) positions and 116 with at least two configurations
attached by the current detector; all 116 retain renumbering invariance. CIP
structures are not copied into the report and structure-dependent reruns
remain unavailable unless the audited external checkout is supplied.

After stereo removal, “configuration-dependent” means at least one enumerated
completion is chiral and at least one is achiral. It is a definitive property
of the underspecified constitution, not a failed prediction. At the 256-isomer
cap, VS226, VS265, VS266, and VS268 are incomplete because only one class was
observed without exhausting their theoretical assignment spaces. VS226 is
fully resolvable as necessarily achiral by raising its local cap to 1,024; the
other three have theoretical upper bounds from 4,194,304 to 33,554,432 and
need stronger symmetry reduction or substantially more search.

RotA is positive-only and its Excel SMILES do not encode the conformational
atropisomer. Its atom-pair labels are chiral-axis loci, not global molecular
chirality labels. Although 175 rows contain some RDKit-recognized atom/bond
stereo and change under generic stereo removal, that information does not turn
the axis labels into global binary truth. The CIP suite similarly provides
local descriptor labels, not global chiral/achiral truth.

The earlier `benchmark_report.json` preserves the pre-enhancement diagnostic
run over all three datasets with a 256-isomer cap and a declared 10-second
per-case budget.
RotA's 650 supplied inputs produce 108 chiral and 542 achiral global outputs;
after removal, 43 are configuration-dependent, 538 necessarily achiral, 67
necessarily chiral, one unsupported, and the 119-atom RotA-0293 constitution
times out. CIP's 300 supplied inputs produce 175 chiral and 125 achiral global
outputs; after removal, the four-state counts are 68, 94, 134, and 4 in the
same order. These are diagnostic distributions with undefined reference
accuracy. They must not be pooled with ACS or presented as performance on the
datasets' actual axial-locus and CIP-assignment tasks.

The manifest records exact sources, revisions, hashes, licenses, protocol and
diagnostic status, and redistribution decisions. The external CIP structures
are read only from an integrity-checked checkout; its frozen result contains
aggregates and case identifiers, not redistributed structures.

## Historical live three-backend comparison

`Diagnostics/backend_comparison_report.json` compares SynKit, the publisher's live RDKit-
SMILES procedure, and live StereoMolGraph revision
`2189f610f23eaaf992e2e01a12ea4d0532496601`. All 1,208 inputs completed in
both supplied-stereo and stereo-removed settings for all three backends with no
parse failure or timeout. Its RotA and CIP SynKit columns used an exploratory
method that assigned one arbitrary orientation to each potential cumulene or
biaryl axis. Those columns are retained only as historical diagnostics and
must not be cited as current classification or accuracy. The report carries a
machine-readable ``claim_status`` retraction.

Here A/C means binary achiral/chiral output. On ACS supplied stereo, SynKit is
258/258, the live RDKit method is 235/258 and exactly reproduces the published
RDKit column, while this live StereoMolGraph revision is 254/258. Its four
differences from the published StereoMolGraph column are VS246, VS247, VS248,
and VS299. After removal, apparent agreement with the original ACS label is
223/258 for SynKit, 94/258 for RDKit, and 220/258 for StereoMolGraph. This is
not recovery accuracy because the input configuration has been erased.

RDKit's result after removal is uniformly achiral by construction: its mirror
procedure only inverts retained tetrahedral atom tags, and none remain. SynKit
now keeps topology-derived cumulene and biaryl candidates as orientation-
unspecified ``PotentialStereoLocus`` evidence. They do not enter binary mirror
comparison and cannot recover an erased stereoisomer.

RotA remains a typed positive axial-locus task, not a 650-case global-binary
reference. The provisional 300-case CIP binary curation is also not an
independent benchmark. The former SynKit RotA ``488/650`` and CIP ``298/300``
claims are retracted. CIP is scored only through SynKit's independent local
perception, ranking, and label-projection layers, never through its global
mirror classifier.

For the CIP suite's native local-label task, RDKit's
`rdCIPLabeler.AssignCIPLabels` exactly reproduces 245/300 complete atom-numbered
label sets (81.67%). Across 1,252 reference labels its micro recall is 90.18%
and precision is 99.82%; one frozen pass takes 0.710 s (2.37 ms/input), while
11 independent passes have a 0.682 s median. StereoMolGraph represents
relative configurations but exposes no CIP assignment API, so its native CIP
accuracy is N/A. SynKit's incremental independent assigner exactly reproduces
155/300 complete label sets (51.67%). It recovers 843/1,252 reference labels
with 67.33% recall and 97.80% precision. Rule 2 uses exact nuclide masses
rather than integer mass numbers. RDKit supplies parsed relative stereo
descriptors at the input boundary, but its ``_CIPCode``/``_CIPRank`` values are
not inputs to SynKit ranking or projection. The 145 non-exact rows are frozen
as 15 unsupported-class, 27 missing-orientation, 101 ranking-defect, and 2
label-projection primary limitations in ``cip_native_report.json``.

The current SynKit typed detector is now scored against RotA's supplied
undirected atom-pair and expanded-path annotations. It recovers 378/698 loci
(54.15% recall) while emitting 763 candidates (49.54% annotation precision),
including 313/347 biaryl, 51/51 C--N heterobiaryl, and 13/15 allene-like loci.
All 650 records are invariant under reverse atom renumbering. The detector does
not recover the annotated C--B, nonbiaryl, chiral-atom-pair, or spiral families.
Because RotA is positive-only and its 2D input generally lacks orientation,
``rota_locus_report.json`` is evidence for typed locus scope only, not
handedness, stability, true-negative specificity, or global chirality.
