# Live backend disagreement review

This review covers the SynKit/StereoMolGraph disagreements in
`backend_comparison_report.json`. A disagreement is not automatically a SynKit
failure. ACS supplies published global truth. RotA and CIP now also have
derived binary references, but RotA is positive-only and 35 CIP labels remain
provisional expert curation rather than independent published truth.

## Case-level conclusions

| Cases | SynKit status after enhancement | Evidence and conclusion |
| --- | --- | --- |
| VS175, VS177, VS178, VS179, VS183, VS186 | fixed | Isotope is now part of molecular identity, so isotope-defined tetrahedral chirality is no longer collapsed. |
| VS079, VS141, VS144, VS166, VS287 | fixed | An even consecutive-double-bond path supplies a molecule-only extended-tetrahedral axis. RDKit discards the input `@` marker, but connectivity still identifies the axis and exact mirror isomorphism adjudicates its global effect. Achiral controls VS231, VS232, and VS243 remain achiral. |
| VS023, VS055, VS057, VS073, VS086, VS158 | fixed | Non-aromatic bonds joining aromatic atoms now receive provisional biaryl-axis probes. The detector identifies stereogenicity from topology but deliberately does not claim a rotational-barrier or native RotA-locus prediction. |
| VS010, VS011 | still unsupported | These opposite helicene records parse to exactly the same canonical SMILES and RDKit exposes no potential stereo locus or chiral tag. A molecule-only topology/helicity model is needed; assigning either handedness from this input would be fabricated. |
| VS180, VS181, VS182, VS187 and stripped VS119 | live SMG limitation | Live StereoMolGraph constructs atoms from element symbols and drops isotope mass. SynKit retains the isotope distinction. |
| VS246, VS247, VS248, VS299 | live SMG regression | The immutable ACS manual and published StereoMolGraph columns label all four achiral. SynKit and RDKit agree; the tested live StereoMolGraph revision changed them to chiral. |

The supplied CIP derived-binary error count therefore falls from 19 to two:
six isotope, five cumulene, and six atropisomer cases are resolved, leaving
only the helicene enantiomer pair. This is global chiral/achiral scoring, not
CIP-label assignment; SynKit still does not assign the expected `M/P` labels.

## Accuracy that is scientifically defined

| Dataset/task | SynKit | RDKit | StereoMolGraph |
| --- | --- | --- | --- |
| ACS supplied global binary | 258/258 | 235/258 | 254/258 live; 258/258 published column |
| ACS removed versus original label | 223/258 apparent agreement | 94/258 apparent agreement | 220/258 apparent agreement |
| RotA derived positive-only binary, supplied | 488/650 (75.08%) | 87/650 (13.38%) | 109/650 (16.77%) |
| RotA derived positive-only binary, removed | 477/650 (73.38%) | 0/650 | 95/650 (14.62%) |
| CIP derived binary, supplied | 298/300 (99.33%) | 264/300 (88.00%) | 271/300 (90.33%) |
| CIP derived binary, removed | 263/300 (87.67%) | 106/300 (35.33%) | 236/300 (78.67%) |
| RotA native axial-locus detection | N/A: no axis detector | N/A: no axis detector | N/A: no axis detector from these SMILES |
| CIP native local descriptor assignment | N/A: no independent full assigner | 245/300 exact label sets (81.67%) | N/A: no CIP-label API |

The ACS removed numbers are not recovery accuracy because stereo information
was erased. RDKit CIP micro label recall is 90.18% and precision is 99.82%
over 1,252 reference labels. Most missing label sets involve helical,
atropisomeric, extended tetrahedral/cis-trans, pseudoasymmetric, or later
globally stereogenic validation cases.

RDKit exact-record coverage by every stereo-unit tag present in a row is:

| Unit tag | Exact rows / rows carrying tag |
| --- | ---: |
| TH | 214/249 |
| CT | 54/65 |
| HE | 0/2 |
| AT | 0/7 |
| CT4 | 0/5 |
| TH3 | 0/8 |
| TH5 | 0/2 |

Mixed-unit rows contribute to every tag they carry, and exactness requires the
entire expected label set. The zeroes therefore expose unsupported extended,
helical, and atropisomeric assignment rather than parser failure.

RotA is a positive-only axis corpus. Reporting the fraction classified globally
chiral as “accuracy” would conflate a 2D constitution with a specified
atropisomer and would supply no true-negative information.

## Efficiency

| Dataset / setting | SynKit | RDKit-SMILES | StereoMolGraph |
| --- | ---: | ---: | ---: |
| ACS supplied | 2.177 s | 0.042 s | 2.031 s |
| ACS removed | 1.033 s | 0.034 s | 2.045 s |
| RotA supplied | 5.404 s | 0.194 s | 108.111 s |
| RotA removed | 5.701 s | 0.178 s | 107.555 s |
| CIP supplied | 1.978 s | 0.041 s | 2.998 s |
| CIP removed | 1.261 s | 0.039 s | 2.574 s |

RDKit-SMILES is fastest because it only inverts retained tetrahedral tags and
compares strings; after removal it returns every molecule achiral. It is not an
equivalent complete-topology method. Against live StereoMolGraph, SynKit is
approximately equal/slower on ACS supplied but 1.98x faster after removal,
20.0x/18.9x faster on RotA, and 1.52x/2.04x faster on CIP for
supplied/removed settings.

RDKit native CIP assignment takes 0.710 s in the frozen report (2.37 ms per
input); eleven independent passes have a 0.682 s median.

## Required implementation follow-up

1. Add a topology-aware helicene locus model and an input representation that
   can retain or derive helicity. VS010/VS011 prove that ordinary SMILES alone
   cannot distinguish opposite `M/P` configurations.
2. Do not use direct binary `stereo_complete` classification for stripped
   input. VS229 proves that a provisional orientation can fall outside the
   valid enumerated completion population. Use the four-state assessment.
3. Extend the four-state enumerator to include candidate cumulene and biaryl
   axes instead of applying one provisional orientation. Binary completion is
   accurate on supplied CIP input but is not a proof over erased assignments.
4. Implement task-specific RotA atom-pair/chain output plus a calibrated
   rotational-stability model. Global mirror classification and a topological
   candidate axis are not substitutes for the native task.
5. Implement or integrate a full independent CIP assigner before claiming
   SynKit CIP accuracy; do not relabel RDKit-derived CIP properties as an
   independent SynKit result.
