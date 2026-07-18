# Stereo benchmark registry

This directory keeps external molecule-stereo datasets together while
preserving their different scientific tasks and license boundaries. Datasets
must not be pooled into one accuracy number merely because they contain
stereochemistry.

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

## Live three-backend comparison

`backend_comparison_report.json` compares SynKit, the publisher's live RDKit-
SMILES procedure, and live StereoMolGraph revision
`2189f610f23eaaf992e2e01a12ea4d0532496601`. All 1,208 inputs completed in
both supplied-stereo and stereo-removed settings for all three backends with no
parse failure or timeout. SMILES parsing and StereoMolGraph explicit-H
preparation are outside timing.

| Dataset / setting | SynKit A/C (s) | RDKit A/C (s) | StereoMolGraph A/C (s) |
| --- | --- | --- | --- |
| ACS supplied | 94/164 (2.177) | 111/147 (0.042) | 90/168 (2.031) |
| ACS removed | 103/155 (1.033) | 258/0 (0.034) | 98/160 (2.045) |
| RotA supplied | 162/488 (5.404) | 563/87 (0.194) | 541/109 (108.111) |
| RotA removed | 173/477 (5.701) | 650/0 (0.178) | 555/95 (107.555) |
| CIP supplied | 108/192 (1.978) | 136/164 (0.041) | 123/177 (2.998) |
| CIP removed | 117/183 (1.261) | 300/0 (0.039) | 132/168 (2.574) |

Here A/C means binary achiral/chiral output. On ACS supplied stereo, SynKit is
258/258, the live RDKit method is 235/258 and exactly reproduces the published
RDKit column, while this live StereoMolGraph revision is 254/258. Its four
differences from the published StereoMolGraph column are VS246, VS247, VS248,
and VS299. After removal, apparent agreement with the original ACS label is
223/258 for SynKit, 94/258 for RDKit, and 220/258 for StereoMolGraph. This is
not recovery accuracy because the input configuration has been erased.

RDKit's result after removal is uniformly achiral by construction: its mirror
procedure only inverts retained tetrahedral atom tags, and none remain. SynKit
and StereoMolGraph instead add provisional complete-topology orientations and
therefore produce binary outputs, but those outputs still cannot reconstruct
the erased stereoisomer. The separate four-state SynKit assessment is the
appropriate configuration-aware interpretation.

The live comparison additionally supplies explicit derived binary references.
RotA assigns all 650 positive axial examples `Chiral`; this yields positive
recall, not balanced accuracy. On supplied/removed input, SynKit is
488/650 (75.08%)/477/650 (73.38%), RDKit is 87/650 (13.38%)/0/650, and live
StereoMolGraph is 109/650 (16.77%)/95/650 (14.62%). SynKit's gain comes from
generic molecule-only even-cumulene and biaryl-axis topology completion. The
remaining 162 supplied cases include other axial/spiral classes and
rigidity-dependent axes; the classifier still does not emit RotA's native
atom-pair/chain locus labels.

The CIP binary reference has 194 chiral and 106 achiral cases. It combines 258
published ACS manual labels, seven independent atropisomer labels from the MIT-
licensed experiment notebook, and 35 new expert-curated labels. The last tier
is provisional and requires independent chemistry review before publication.
On supplied input SynKit/RDKit/StereoMolGraph score 298/300 (99.33%), 264/300
(88.00%), and 271/300 (90.33%). SynKit's only two misses are the enantiomeric
helicene records VS010 and VS011: their identical SMILES retain neither
helicity nor an RDKit stereo locus. After removal the methods score 263/300
(87.67%), 106/300 (35.33%), and 236/300 (78.67%); removed-label agreement still
does not reconstruct the erased stereoisomer.

For the CIP suite's native local-label task, RDKit's
`rdCIPLabeler.AssignCIPLabels` exactly reproduces 245/300 complete atom-numbered
label sets (81.67%). Across 1,252 reference labels its micro recall is 90.18%
and precision is 99.82%; one frozen pass takes 0.710 s (2.37 ms/input), while
11 independent passes have a 0.682 s median. StereoMolGraph represents
relative configurations but exposes no CIP assignment API, so its native CIP
accuracy is N/A. SynKit currently has no independent full CIP assigner; using
RDKit properties inside a SynKit graph would not constitute a separate method.

None of the three adapters emits RotA axial atom-pair/chain loci, so native
RotA accuracy remains N/A. SynKit now uses candidate biaryl and cumulene axes
internally for global mirror classification, but that is not the same output
task and does not establish a rotational barrier from 2D connectivity.
