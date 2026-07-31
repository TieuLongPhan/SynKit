# Perception evidence

These reports answer local tasks:

- `perception_conformance_cases.json`: designed pre-benchmark data contract;
- `stereo_element_report.json`: carrier/configuration attachment inventory;
- `cip_native_report.json`: local CIP-label assignment;
- `rota_locus_report.json`: positive axial-locus detection.
- `full_detection_report.json`: all records and usable annotations from ACS,
  RotA, and CIP under both configuration-erased and
  oriented-neighbor-frame-retained detection.

They are not whole-molecule chirality accuracy.

## Exhaustive three-dataset result

`full_detection_report.json` evaluates all 1,208 input rows without sampling:
258 ACS rows, 650 RotA rows, and 300 CIP rows. The ACS rows are also an exact
ID-and-SMILES subset of CIP, so the report explicitly refuses to pool an
accuracy across datasets. The retained CIP section contains identifiers,
labels, stereo-unit categories, and detected carriers, but not the
non-vendored source structures.

The report preserves both broad candidates and constitutionally confirmed
carriers. Symmetry-related candidates therefore remain inspectable without
being promoted to confirmed stereocenters. Dataset-level annotations are
exhaustive within their source contracts: ACS configured atom/bond loci,
RotA endpoint and expanded-path labels, and CIP recommended local positions.

The retained two-setting comparison uses broad typed carrier recovery:

| Dataset/task | Local frames retained | Additional loci | Configuration erased | Additional loci |
| --- | ---: | ---: | ---: | ---: |
| ACS supplied configured loci | 1,007/1,007 (100.00%) | 109 | 743/1,007 (73.78%) | 52 |
| RotA positive loci | 698/698 (100.00%) | 1,562 | 698/698 (100.00%) | 1,562 |
| RotA expanded cumulene paths | 15/15 (100.00%) | 0 | 15/15 (100.00%) | 0 |
| CIP local positions | 1,252/1,252 (100.00%) | 57 | 969/1,252 (77.40%) | 19 |

All 1,208 row-level reverse-renumbering checks pass independently in both
settings, with no parse failures, detection errors, or timeouts. For RotA,
the 1,562 additional broad candidates reflect an intentionally broad,
positive-only axis inventory and are not specificity errors. Confirmed RotA
carriers recover the same 698 references with 843 additional projections.
CIP configuration-erased recovery is 71.12% tetrahedral and 100.00% for both
planar and axial references. Two helicene-like topology carriers recover all
four formerly missed helical endpoint positions without assigning handedness
or stability.

The strict uppercase/lowercase split is 90.13% versus 13.46%. The separate
neighbor-assisted tier raises tetrahedral recovery from 697/980 to 980/980
and recovers all 208 lowercase positions. It converts each supplied
neighboring configuration directly into an oriented local-neighbor frame,
never into a CIP label, rank, canonical descriptor, or resolved-center
marker. For each focal center, its own frame is removed. The exact
center-fixed graph automorphisms are then restricted by the remaining local
frames; the center is stereogenic exactly when none induces an odd
permutation of its four slots.

The retained result is a conditional local question: all other supplied
orientations are held fixed. It has 57 additional tetrahedral candidates not
named by the recommended-position list, so its 95.65% annotation precision is
reported as list agreement, not as a specificity estimate. ACS similarly has
109 additional candidates; its manual annotation is global chirality plus
configured source loci, not complete local negative truth.

Every erased-input miss has been structurally audited. ACS has 264 and CIP
has 283; all 547 are carbon tetrahedral carriers with an odd
constitution-preserving center symmetry, rather than unsupported topologies.
All are recovered when supplied neighboring frames are retained. ACS divides
into 134 acyclic and 130 ring centers; CIP divides into 137 acyclic and 146
ring centers. The available neighboring evidence is predominantly
tetrahedral, with smaller planar groups and one CIP cumulene-dependent case.

## Designed conformance data

`perception_conformance_cases.json` contains 40 cases: one configured
positive, unconfigured positive, negative near-miss, and
unsupported/ambiguous case for each of ten stereo families.

The family scope is explicit:

| Scope | Families |
| --- | --- |
| Current topology detector | tetrahedral, double bond, cumulene axis, extended cis/trans, atrop axis, helicene-like helical path |
| Configured adapter only | square planar, trigonal bipyramidal, octahedral |
| Sidecar only | general formal helical support, planar chirality |

SMILES cases are parseable designed inputs. Helical and planar-chirality
records use formal graphs and carry `chemical_validation: false`; they define
future API behavior, not empirical chemical truth. No record asserts
configurational stability or whole-molecule chirality.

The integrity tests validate schema, case balance, structure parsing, support
references, descriptor construction, and fail-closed reason codes. They do
not execute the perception engine or calculate a score.

## Current conformance result

`Perception/conformance.py` executes all three declared contracts:

- connectivity perception for 20 records;
- configured RDKit adapters for 12 records;
- formal sidecar support/restoration for eight records.

All 40 records are scored and none are hidden as not applicable. The retained
result is 40/40 with no execution errors. Configuration is deliberately not
scored: orientation evidence is erased or ignored before detection. All 20
emitted structured-reason checks pass.

The unconfigured square-planar, trigonal-bipyramidal, and octahedral inputs
declare their geometry as `@SP`, `@TB`, and `@OH` without a permutation
number. The adapter therefore knows the typed carrier and its neighbors while
correctly preserving the unknown configuration as `parity=None`. Bare
connectivity without a shape declaration would remain geometry-ambiguous.

For records that contain a configured input, the runner removes tetrahedral
and bond orientation, reduces numbered non-tetrahedral tags to their
shape-only forms, and ignores sidecar parity. Configured and unconfigured
positive strata therefore exercise the same carrier decision.

Exact axis-fixed graph automorphisms separate constitutionally confirmed
carriers from broad candidates with symmetry-related terminal references.

Broad symmetry-related candidates remain in the perception inventory for
diagnostics; they are not promoted to confirmed carriers. This classification
does not assert configurational stability, including for atropisomeric axes.
